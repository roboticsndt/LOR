#ifndef __SCAN_CONTEXT_H__
#define __SCAN_CONTEXT_H__

#include <Eigen/Dense>
#include <memory>
#include <filesystem>
#include "kdnanoflann.h"

static inline float xy2theta(float _x, float _y)
{
    if (_x >= 0 & _y >= 0)
        return (180 / M_PI) * atan(_y / _x);

    if (_x < 0 & _y >= 0)
        return 180 - ((180 / M_PI) * atan(_y / (-_x)));

    if (_x < 0 & _y < 0)
        return 180 + ((180 / M_PI) * atan(_y / _x));

    if (_x >= 0 & _y < 0)
        return 360 - ((180 / M_PI) * atan((-_y) / _x));

    return 0.0f;
}

struct ScanContext
{
    static constexpr double LIDAR_HEIGHT = 2.0; // lidar height : add this for simply directly using lidar scan in the lidar local coord (not robot base coord) / if you use robot-coord-transformed lidar scans, just set this as 0.

    static constexpr int PC_NUM_RING = 20;        // 20 in the original paper (IROS 18)
    static constexpr int PC_NUM_SECTOR = 60;      // 60 in the original paper (IROS 18)
    static constexpr double PC_MAX_RADIUS = 80.0; // 80 meter max in the original paper (IROS 18)
    static constexpr double PC_UNIT_SECTORANGLE = 360.0 / double(PC_NUM_SECTOR);
    static constexpr double PC_UNIT_RINGGAP = PC_MAX_RADIUS / double(PC_NUM_RING);

    // tree
    static constexpr int NUM_EXCLUDE_RECENT = 50;       // simply just keyframe gap, but node position distance-based exclusion is ok.
    static constexpr int NUM_CANDIDATES_FROM_TREE = 10; // 10 is enough. (refer the IROS 18 paper)

    // loop thres
    static constexpr double SC_DIST_THRES = 0.315; // empirically 0.1-0.2 is fine (rare false-alarms) for 20x60 polar context (but for 0.15 <, DCS or ICP fit score check (e.g., in LeGO-LOAM) should be required for robustness)
    // const double SC_DIST_THRES = 0.5; // 0.4-0.6 is good choice for using with robust kernel (e.g., Cauchy, DCS) + icp fitness threshold / if not, recommend 0.1-0.15

    // config
    static constexpr int TREE_MAKING_PERIOD_ = 50; // i.e., remaking tree frequency, to avoid non-mandatory every remaking, to save time cost / if you want to find a very recent revisits use small value of it (it is enough fast ~ 5-50ms wrt N.).

    using Matrix = Eigen::Matrix<double, PC_NUM_RING, PC_NUM_SECTOR>;
    using desc_type = Matrix;

    template <typename Input, typename Pose>
    Matrix transform(Input &&_scan_down, Pose &&p) const
    {
        size_t num_pts_scan_down = _scan_down.size();

        // main
        const int NO_POINT = -1000;
        Matrix desc = NO_POINT * Matrix::Ones(PC_NUM_RING, PC_NUM_SECTOR);

        float azim_angle, azim_range; // wihtin 2d plane
        int ring_idx, sctor_idx;
        for (int pt_idx = 0; pt_idx < num_pts_scan_down; pt_idx++)
        {
            auto x = _scan_down[pt_idx].x;
            auto y = _scan_down[pt_idx].y;
            auto z = _scan_down[pt_idx].z + LIDAR_HEIGHT; // naive adding is ok (all points should be > 0).

            // xyz to ring, sector
            azim_range = sqrt(x * x + y * y);
            azim_angle = xy2theta(x, y);

            // if range is out of roi, pass
            if (azim_range > PC_MAX_RADIUS)
                continue;

            ring_idx = std::max(std::min(PC_NUM_RING, int(ceil((azim_range / PC_MAX_RADIUS) * PC_NUM_RING))), 1);
            sctor_idx = std::max(std::min(PC_NUM_SECTOR, int(ceil((azim_angle / 360.0) * PC_NUM_SECTOR))), 1);

            // taking maximum z
            if (desc(ring_idx - 1, sctor_idx - 1) < z) // -1 means cpp starts from 0
                desc(ring_idx - 1, sctor_idx - 1) = z; // update for taking maximum value at that bin
        }

        // reset no points to zero (for cosine dist later)
        for (int row_idx = 0; row_idx < desc.rows(); row_idx++)
            for (int col_idx = 0; col_idx < desc.cols(); col_idx++)
                if (desc(row_idx, col_idx) == NO_POINT)
                    desc(row_idx, col_idx) = 0;

        return desc;
    }

    std::pair<int, Eigen::Isometry3d> loop(const Matrix &desc) const;

    void add(const Matrix &desc);

    void save_state(std::filesystem::path path) const;

    void load_state(std::filesystem::path path);

private:
    // data
    // std::vector<double> polarcontexts_timestamp_; // optional.
    std::vector<Matrix> polarcontexts_;
    std::vector<Eigen::Vector<double, PC_NUM_RING>> polarcontext_invkeys_;
    std::vector<Eigen::RowVector<double, PC_NUM_SECTOR>> polarcontext_vkeys_;

    using KeyMat = std::vector<std::vector<float>>;
    KeyMat polarcontext_invkeys_mat_;

    using InvKeyTree = KDTreeVectorOfVectorsAdaptor<KeyMat, float>;

    mutable int tree_making_period_conter = 0;
    mutable std::shared_ptr<InvKeyTree> polarcontext_tree_;
    mutable KeyMat polarcontext_invkeys_to_search_;
};

#endif // __SCAN_CONTEXT_H__