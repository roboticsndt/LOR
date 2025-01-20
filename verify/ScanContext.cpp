#include <ctime>
#include <cassert>
#include <cmath>
#include <utility>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <memory>
#include <iostream>

#include <Eigen/Dense>

#include "ScanContext.hpp"

using namespace Eigen;
using namespace nanoflann;

using std::cout;
using std::endl;
using std::make_pair;

using std::atan2;
using std::cos;
using std::sin;

static float rad2deg(float radians)
{
    return radians * 180.0 / M_PI;
}

static float deg2rad(float degrees)
{
    return degrees * M_PI / 180.0;
}

template <typename Scalar, int R, int C>
static Eigen::Matrix<Scalar, R, C> circshift(const Eigen::Matrix<Scalar, R, C> &_mat, int _num_shift)
{
    // shift columns to right direction
    assert(_num_shift >= 0);

    if (_num_shift == 0)
    {
        Eigen::Matrix<Scalar, R, C> shifted_mat(_mat);
        return shifted_mat; // Early return
    }

    Eigen::Matrix<Scalar, R, C> shifted_mat = Eigen::Matrix<Scalar, R, C>::Zero(_mat.rows(), _mat.cols());
    for (int col_idx = 0; col_idx < _mat.cols(); col_idx++)
    {
        int new_location = (col_idx + _num_shift) % _mat.cols();
        shifted_mat.col(new_location) = _mat.col(col_idx);
    }

    return shifted_mat;

} // circshift

template <typename Scalar, int R, int C>
std::vector<float> eig2stdvec(const Eigen::Matrix<Scalar, R, C> &_eigmat)
{
    std::vector<float> vec(_eigmat.data(), _eigmat.data() + _eigmat.size());
    return vec;
} // eig2stdvec

template <typename Scalar, int R, int C>
static double distDirectSC(const Eigen::Matrix<Scalar, R, C> &_sc1, const Eigen::Matrix<Scalar, R, C> &_sc2)
{
    int num_eff_cols = 0; // i.e., to exclude all-nonzero sector
    double sum_sector_similarity = 0;
    for (int col_idx = 0; col_idx < _sc1.cols(); col_idx++)
    {
        auto col_sc1 = _sc1.col(col_idx);
        auto col_sc2 = _sc2.col(col_idx);

        if (col_sc1.norm() == 0 | col_sc2.norm() == 0)
            continue; // don't count this sector pair.

        double sector_similarity = col_sc1.dot(col_sc2) / (col_sc1.norm() * col_sc2.norm());

        sum_sector_similarity = sum_sector_similarity + sector_similarity;
        num_eff_cols = num_eff_cols + 1;
    }

    double sc_sim = sum_sector_similarity / num_eff_cols;
    return 1.0 - sc_sim;

} // distDirectSC

template <typename Scalar, int R, int C>
static int fastAlignUsingVkey(const Eigen::Matrix<Scalar, R, C> &_vkey1, const Eigen::Matrix<Scalar, R, C> &_vkey2)
{
    int argmin_vkey_shift = 0;
    double min_veky_diff_norm = 10000000;
    for (int shift_idx = 0; shift_idx < _vkey1.cols(); shift_idx++)
    {
        MatrixXd vkey2_shifted = circshift(_vkey2, shift_idx);

        MatrixXd vkey_diff = _vkey1 - vkey2_shifted;

        double cur_diff_norm = vkey_diff.norm();
        if (cur_diff_norm < min_veky_diff_norm)
        {
            argmin_vkey_shift = shift_idx;
            min_veky_diff_norm = cur_diff_norm;
        }
    }

    return argmin_vkey_shift;

} // fastAlignUsingVkey

template <typename Scalar, int R, int C>
static Eigen::Vector<Scalar, R> makeRingkeyFromScancontext(const Eigen::Matrix<Scalar, R, C> &_desc)
{
    /*
     * summary: rowwise mean vector
     */
    Eigen::Vector<Scalar, R> invariant_key;
    for (int row_idx = 0; row_idx < R; row_idx++)
    {
        const auto &curr_row = _desc.row(row_idx);
        invariant_key(row_idx, 0) = curr_row.mean();
    }

    return invariant_key;
} // SCManager::makeRingkeyFromScancontext

template <typename Scalar, int R, int C>
static Eigen::RowVector<Scalar, C> makeSectorkeyFromScancontext(const Eigen::Matrix<Scalar, R, C> &_desc)
{
    /*
     * summary: columnwise mean vector
     */
    Eigen::RowVector<Scalar, C> variant_key;
    for (int col_idx = 0; col_idx < _desc.cols(); col_idx++)
    {
        variant_key(0, col_idx) = _desc.col(col_idx).mean();
    }

    return variant_key;
} // SCManager::makeSectorkeyFromScancontext

template <typename Scalar, int R, int C>
static std::pair<double, int> distanceBtnScanContext(const Eigen::Matrix<Scalar, R, C> &_sc1, const Eigen::Matrix<Scalar, R, C> &_sc2)
{
    // 1. fast align using variant key (not in original IROS18)
    auto vkey_sc1 = makeSectorkeyFromScancontext(_sc1);
    auto vkey_sc2 = makeSectorkeyFromScancontext(_sc2);
    int argmin_vkey_shift = fastAlignUsingVkey(vkey_sc1, vkey_sc2);

    const int SEARCH_RADIUS = round(0.5 * 0.1 * _sc1.cols()); // a half of search range
    std::vector<int> shift_idx_search_space{argmin_vkey_shift};
    for (int ii = 1; ii < SEARCH_RADIUS + 1; ii++)
    {
        shift_idx_search_space.push_back((argmin_vkey_shift + ii + _sc1.cols()) % _sc1.cols());
        shift_idx_search_space.push_back((argmin_vkey_shift - ii + _sc1.cols()) % _sc1.cols());
    }
    std::sort(shift_idx_search_space.begin(), shift_idx_search_space.end());

    // 2. fast columnwise diff
    int argmin_shift = 0;
    double min_sc_dist = 10000000;
    for (int num_shift : shift_idx_search_space)
    {
        auto sc2_shifted = circshift(_sc2, num_shift);
        double cur_sc_dist = distDirectSC(_sc1, sc2_shifted);
        if (cur_sc_dist < min_sc_dist)
        {
            argmin_shift = num_shift;
            min_sc_dist = cur_sc_dist;
        }
    }

    return make_pair(min_sc_dist, argmin_shift);

} // distanceBtnScanContext

void ScanContext::add(const Matrix &desc)
{
    auto ring_key = makeRingkeyFromScancontext(desc);
    polarcontexts_.push_back(desc);
    polarcontext_invkeys_.push_back(ring_key);
    polarcontext_vkeys_.push_back(makeSectorkeyFromScancontext(desc));
    polarcontext_invkeys_mat_.push_back(eig2stdvec(ring_key));
}

std::pair<int, Eigen::Isometry3d> ScanContext::loop(const Matrix &desc) const
{
    int loop_id{-1}; // init with -1, -1 means no loop (== LeGO-LOAM's variable "closestHistoryFrameID")

    auto curr_key = eig2stdvec(makeRingkeyFromScancontext(desc)); // current observation (query)
    auto curr_desc = desc;           // current observation (query)

    /*
     * step 1: candidates from ringkey tree_
     */
    if (polarcontext_invkeys_mat_.size() < NUM_EXCLUDE_RECENT + 1)
    {
        return {-1, Eigen::Isometry3d::Identity()}; // Early return
    }

    // tree_ reconstruction (not mandatory to make everytime)
    if (tree_making_period_conter % TREE_MAKING_PERIOD_ == 0) // to save computation cost
    {
        polarcontext_invkeys_to_search_.clear();
        polarcontext_invkeys_to_search_.assign(polarcontext_invkeys_mat_.begin(), polarcontext_invkeys_mat_.end() - NUM_EXCLUDE_RECENT);

        polarcontext_tree_.reset();
        polarcontext_tree_ = std::make_unique<InvKeyTree>(PC_NUM_RING /* dim */, polarcontext_invkeys_to_search_, 10 /* max leaf */);
        // tree_ptr_->index->buildIndex(); // inernally called in the constructor of InvKeyTree (for detail, refer the nanoflann and KDtreeVectorOfVectorsAdaptor)
    }
    tree_making_period_conter = tree_making_period_conter + 1;

    if (polarcontext_tree_ == nullptr)
    {
        return {-1, Eigen::Isometry3d::Identity()}; // Early return
    }

    double min_dist = 10000000; // init with somthing large
    int nn_align = 0;
    int nn_idx = 0;

    // knn search
    std::vector<size_t> candidate_indexes(NUM_CANDIDATES_FROM_TREE);
    std::vector<float> out_dists_sqr(NUM_CANDIDATES_FROM_TREE);

    nanoflann::KNNResultSet<float> knnsearch_result(NUM_CANDIDATES_FROM_TREE);
    knnsearch_result.init(&candidate_indexes[0], &out_dists_sqr[0]);
    polarcontext_tree_->index->findNeighbors(knnsearch_result, &curr_key[0] /* query */, nanoflann::SearchParameters(10));

    /*
     *  step 2: pairwise distance (find optimal columnwise best-fit using cosine distance)
     */
    for (int candidate_iter_idx = 0; candidate_iter_idx < NUM_CANDIDATES_FROM_TREE; candidate_iter_idx++)
    {
        Matrix polarcontext_candidate = polarcontexts_[candidate_indexes[candidate_iter_idx]];
        std::pair<double, int> sc_dist_result = distanceBtnScanContext(curr_desc, polarcontext_candidate);

        double candidate_dist = sc_dist_result.first;
        int candidate_align = sc_dist_result.second;

        if (candidate_dist < min_dist)
        {
            min_dist = candidate_dist;
            nn_align = PC_NUM_SECTOR - candidate_align;

            nn_idx = candidate_indexes[candidate_iter_idx];
        }
    }

    loop_id = nn_idx;

    if (loop_id != -1)
    {
        printf("min_dist: %f, nn_align: %d, nn_idx: %d\n", min_dist, nn_align, nn_idx);
        Eigen::Isometry3d loop_transform = Eigen::Isometry3d::Identity();
        loop_transform.translation() = Eigen::Vector3d(0, 0, 0);
        loop_transform.rotate(Eigen::AngleAxisd(nn_align * M_PI * 2.0 / PC_NUM_SECTOR, Eigen::Vector3d::UnitZ()));

        return make_pair(loop_id, loop_transform);
    }

    return make_pair(-1, Eigen::Isometry3d::Identity());
}

void ScanContext::save_state(std::filesystem::path p) const
{
    FILE *fp = fopen(p.c_str(), "wb");

    char magic[16] = "ScanContext";

    // write magic
    fwrite(magic, sizeof(char), 16, fp);

    // object size
    size_t sz = polarcontexts_.size();

    // write object size
    fwrite(&sz, sizeof(size_t), 1, fp);

    // write objects
    for (size_t i = 0; i < sz; i++)
    {
        const Matrix &obj = polarcontexts_[i];
        fwrite(obj.data(), sizeof(double), obj.size(), fp);
    }

    fclose(fp);
}

void ScanContext::load_state(std::filesystem::path p)
{
    FILE *fp = fopen(p.c_str(), "rb");

    char magic[16];
    if (1 != fread(magic, 16, 1, fp))
    {
        printf("Error: magic read failed\n");
        fclose(fp);
        return;
    }

    assert(strcmp(magic, "ScanContext") == 0);

    // read object size
    size_t sz;
    if (1 != fread(&sz, sizeof(size_t), 1, fp))
    {
        printf("Error: object size read failed\n");
        fclose(fp);
        return;
    }

    // read objects
    polarcontexts_.resize(sz);
    for (size_t i = 0; i < sz; i++)
    {
        Matrix &obj = polarcontexts_[i];
        if (obj.size() != fread(obj.data(), sizeof(double), obj.size(), fp))
        {
            printf("Error: object read failed\n");
            fclose(fp);
            return;
        }
    }

    fclose(fp);

    // re-calc invkeys and vkeys
    polarcontext_invkeys_.clear();
    polarcontext_vkeys_.clear();
    polarcontext_invkeys_mat_.clear();

    for (size_t i = 0; i < sz; i++)
    {
        auto ring_key = makeRingkeyFromScancontext(polarcontexts_[i]);
        auto ikey = eig2stdvec(ring_key);
        polarcontext_invkeys_.push_back(ring_key);
        polarcontext_vkeys_.push_back(makeSectorkeyFromScancontext(polarcontexts_[i]));
        polarcontext_invkeys_mat_.push_back(ikey);
        polarcontext_invkeys_to_search_.push_back(ikey);
    }

    // re-make tree
    polarcontext_tree_.reset();
    polarcontext_tree_ = std::make_unique<InvKeyTree>(PC_NUM_RING /* dim */, polarcontext_invkeys_to_search_, 10 /* max leaf */);

    // reset counter
    tree_making_period_conter = 1;
}

// } // namespace SC2