#ifndef __LOAM_H__
#define __LOAM_H__

#include <algorithm>
#include <vector>
#include <cmath>
#include <atomic>
#include <chrono>
#include <Eigen/Core>
#include <nanoflann.hpp>
#include <pcl/common/transforms.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>

#ifndef __TRANSFORM_HPP__
#define __TRANSFORM_HPP__

#include <Eigen/Dense>

template<typename Scalar>
struct _Transform
{
    Scalar x = 0.0f, y = 0.0f, z = 0.0f;
    Scalar roll = 0.0f, pitch = 0.0f, yaw = 0.0f;

    template<typename Scalar2>
    _Transform<Scalar2> cast() const
    {
        _Transform<Scalar2> tr;

        tr.x = (Scalar2)x;
        tr.y = (Scalar2)y;
        tr.z = (Scalar2)z;
        tr.roll = (Scalar2)roll;
        tr.pitch = (Scalar2)pitch;
        tr.yaw = (Scalar2)yaw;

        return tr;
    }
};

using Transform = _Transform<double>;
using Transformf = _Transform<float>;

template<typename Scalar>
using Isometry3 = Eigen::Transform<Scalar, 3, Eigen::Isometry>;

template<typename Scalar>
static inline Isometry3<Scalar> to_eigen(const _Transform<Scalar> &tr)
{

    using Matrix = Eigen::Matrix4<Scalar>;
    // T = s.Matrix([[1, 0, 0, tx], [0, 1, 0, ty], [0, 0, 1, tz], [0, 0, 0, 1]])
    // Rx = s.Matrix([[1, 0, 0, 0], [0, s.cos(rx), -s.sin(rx), 0], [0, s.sin(rx), s.cos(rx), 0], [0, 0, 0, 1]])
    // Ry = s.Matrix([[s.cos(ry), 0, s.sin(ry), 0], [0, 1, 0, 0], [-s.sin(ry), 0, s.cos(ry), 0], [0, 0, 0, 1]])
    // Rz = s.Matrix([[s.cos(rz), -s.sin(rz), 0, 0], [s.sin(rz), s.cos(rz), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    // Transformation matrix
    // M = T * Rz * Ry * Rx

    Matrix m = Matrix::Identity();
    m(0, 3) = tr.x;
    m(1, 3) = tr.y;
    m(2, 3) = tr.z;

    Scalar cr = std::cos(tr.roll);
    Scalar sr = std::sin(tr.roll);
    Scalar cp = std::cos(tr.pitch);
    Scalar sp = std::sin(tr.pitch);
    Scalar cy = std::cos(tr.yaw);
    Scalar sy = std::sin(tr.yaw);

    m(0, 0) = cp * cy;
    m(0, 1) = cy * sp * sr - cr * sy;
    m(0, 2) = sr * sy + cr * cy * sp;
    m(1, 0) = cp * sy;
    m(1, 1) = cr * cy + sp * sr * sy;
    m(1, 2) = cr * sp * sy - cy * sr;
    m(2, 0) = -sp;
    m(2, 1) = cp * sr;
    m(2, 2) = cr * cp;

    return Isometry3<Scalar>(m);
}

template <typename Scalar>
static inline _Transform<Scalar> from_eigen(const Eigen::Matrix4<Scalar> &m)
{
    _Transform<Scalar> tr;
    tr.x = m(0, 3);
    tr.y = m(1, 3);
    tr.z = m(2, 3);
    tr.pitch = std::atan2(-m(2, 0), std::sqrt(m(0, 0) * m(0, 0) + m(1, 0) * m(1, 0)));

    Scalar c = std::cos(tr.pitch);
    tr.yaw = std::atan2(m(1, 0) / c, m(0, 0) / c);
    tr.roll = std::atan2(m(2, 1) / c, m(2, 2) / c);

    return tr;
}

template <typename Scalar, int Options>
static inline _Transform<Scalar> from_eigen(const Eigen::Transform<Scalar, 3, Options> &m) {
    return from_eigen(m.matrix());
}

static inline Eigen::Isometry3d to_isometry(Eigen::Matrix4d m)
{
    // check if it is a valid transformation matrix
    auto R = m.block<3, 3>(0, 0);

    if((R * R.transpose() - Eigen::Matrix3d::Identity()).norm() > 1e-10) {
        printf("Invalid rotation matrix\n");
    }

    // normalize the rotation matrix
    Eigen::Quaterniond q(R);
    q.normalize();
    m.block<3, 3>(0, 0) = q.toRotationMatrix();
    return Eigen::Isometry3d(m);
}


using jacobian = Eigen::Matrix<double, 6, 1>;

struct coeff
{
    float px, py, pz;
    float x, y, z;
    float b;
    float s;
};

struct jacobian_g
{
    double srx, crx, sry, cry, srz, crz;
};

static inline void init_jacobian_g(jacobian_g &g, const Transform &t)
{
    g.srx = sin(t.roll);
    g.crx = cos(t.roll);
    g.sry = sin(t.pitch);
    g.cry = cos(t.pitch);
    g.srz = sin(t.yaw);
    g.crz = cos(t.yaw);
}

static inline jacobian J(const coeff &c, const jacobian_g &g)
{
    jacobian j;
    j(3) = (c.py * (g.srx * g.srz + g.sry * g.crx * g.crz) + c.pz * (g.srz * g.crx - g.srx * g.sry * g.crz)) * c.x +
           (c.py * (g.sry * g.srx * g.crx - g.srx * g.crz) - c.pz * (g.srx * g.sry * g.srz + g.crx * g.crz)) * c.y +
           (c.py * g.crx * g.cry - c.pz * g.srx * g.cry) * c.z;

    j(4) = (-c.px * g.sry * g.crz + c.py * g.srx * g.cry * g.crz + c.pz * g.crx * g.cry * g.crz) * c.x +
           (-c.px * g.sry * g.srz + c.py * g.srx * g.srz * g.cry + c.pz * g.srz * g.crx * g.cry) * c.y +
           (-c.px * g.cry - c.py * g.srx * g.sry - c.pz * g.sry * g.crx) * c.z;

    j(5) = (-c.px * g.srz * g.cry + c.py * (-g.sry * g.srx * g.srz - g.crx * g.crz) + c.pz * (-g.sry * g.srz * g.crx + g.srx * g.crz)) * c.x +
           (c.px * g.cry * g.crz + c.py * (g.sry * g.srx * g.crz - g.srz * g.crx) + c.pz * (g.sry * g.crx * g.crz + g.srx * g.srz)) * c.y;

    j(0) = c.x;
    j(1) = c.y;
    j(2) = c.z;
    return j;
}

static inline jacobian J2(const coeff &c) {
    jacobian j;
    j(0) = c.x;
    j(1) = c.y;
    j(2) = c.z;
    j(3) = -c.pz * c.y + c.py * c.z;
    j(4) = c.pz * c.x - c.px  * c.z;
    j(5) = -c.py * c.x + c.px * c.y;
    return j;
}

template <typename V>
auto p2(V v) -> decltype(v * v)
{
    return v * v;
}

template<typename Scalar>
static inline void save_pose(const Eigen::Matrix4<Scalar> &tr, const char *filename)
{
    FILE * f = fopen(filename, "w");
    if(f == nullptr) {
        return;
    }

    _Transform<Scalar> t = from_eigen(tr);
    if constexpr (std::is_same<Scalar, float>::value) {
        fprintf(f, "%f,%f,%f,%f,%f,%f\n", t.x, t.y, t.z, t.roll, t.pitch, t.yaw);
    } else {
        fprintf(f, "%lf,%lf,%lf,%lf,%lf,%lf\n", t.x, t.y, t.z, t.roll, t.pitch, t.yaw);
    }

    fclose(f);
}

template<typename Scalar, int Options>
static inline void save_pose(const Eigen::Transform<Scalar, 3, Options> &tr, const char *filename)
{
    save_pose(tr.matrix(), filename);
}

template<typename Scalar>
static inline Isometry3<Scalar> load_pose(const char *filename)
{
    FILE * f = fopen(filename, "r");
    if(f == nullptr) {
        return Isometry3<Scalar>::Identity();
    }

    _Transform<Scalar> t;
    int result = fscanf(f, "%lf,%lf,%lf,%lf,%lf,%lf\n", &t.x, &t.y, &t.z, &t.roll, &t.pitch, &t.yaw);

    fclose(f);

    if(result != 6) {
        printf("Failed to load pose, use identity\n");
        return Isometry3<Scalar>::Identity();
    }

    return to_eigen(t);
}

template <typename Cloud>
static inline void deskew(Cloud &cloud, Eigen::Matrix4d matrix1, Eigen::Matrix4d matrix2)
{
    double time1 = cloud[0].time;
    double time2 = cloud[cloud.size() - 1].time;

    Eigen::Matrix4d between = matrix1.inverse() * matrix2;
    Transform tr = from_eigen(between);

    constexpr double eps = 1e-5;
    if (fabsf(tr.x) < eps &&
        fabsf(tr.y) < eps &&
        fabsf(tr.z) < eps &&
        fabsf(tr.roll) < eps &&
        fabsf(tr.pitch) < eps &&
        fabsf(tr.yaw) < eps)
    {
        return ;
    }
 
    for (auto &p : cloud)
    {
        double ratio = (p.time - time1) / (time2 - time1);

        Transform tr2 = {
            tr.x * ratio,
            tr.y * ratio,
            tr.z * ratio,
            tr.roll * ratio,
            tr.pitch * ratio,
            tr.yaw * ratio,
        };

        auto m = to_eigen(tr2);

        Eigen::Vector3d v(p.x, p.y, p.z);

        v = m * v;

        p.x = v.x();
        p.y = v.y();
        p.z = v.z();
    }

}

#endif // __TRANSFORM_HPP__

template <typename point_type>
struct array_adaptor
{
    using num_t = decltype(point_type::x);
    typedef array_adaptor<point_type> self_t;
    typedef typename nanoflann::metric_L2::template traits<num_t, self_t>::distance_t metric_t;
    typedef nanoflann::KDTreeSingleIndexAdaptor<metric_t, self_t, 3, size_t> index_t;

    index_t *index = nullptr; //! The kd-tree index for the user to call its methods as usual with any other FLANN index.
    /// Constructor: takes a const ref to the vector of vectors object with the data points
    array_adaptor(const point_type *array = nullptr, size_t count = 0, const int leaf_max_size = 10) 
        : m_data(array), length(count)
    {
        if(array != nullptr) {
            index = new index_t(3, *this, nanoflann::KDTreeSingleIndexAdaptorParams(leaf_max_size));
            index->buildIndex();
        }
    }

    

    ~array_adaptor() {
        if (index != nullptr) {
            delete index;
        }
    }

    const point_type *m_data = nullptr;
    size_t length = 0;

    /** Query for the \a num_closest closest points to a given point (entered as query_point[0:dim-1]).
     *  Note that this is a short-cut method for index->findNeighbors().
     *  The user can also call index->... methods as desired.*/
    template <typename query_point_type>
    inline void query(const query_point_type &query_point, const size_t num_closest, size_t *out_indices, num_t *out_distances_sq) const
    {
        num_t val[3] = {query_point.x, query_point.y, query_point.z};
        nanoflann::KNNResultSet<num_t, size_t> resultSet(num_closest);
        resultSet.init(out_indices, out_distances_sq);
        index->findNeighbors(resultSet, val, nanoflann::SearchParameters());
    }

    template <typename Container, typename query_point_type>
    inline Container radius(const query_point_type &query_point, const num_t radius) const
    {
        num_t val[3] = {query_point.x, query_point.y, query_point.z};
        std::vector<nanoflann::ResultItem<size_t, num_t>> ret_matches;
        nanoflann::SearchParameters params(0.0f, false);
        index->radiusSearch(val, radius * radius, ret_matches, params);

        Container ret;
        ret.reserve(ret_matches.size());

        for (auto &match : ret_matches)
        {
            ret.push_back(m_data[match.first]);
        }

        return ret;
    }

    /** @name Interface expected by KDTreeSingleIndexAdaptor
     * @{ */

    const self_t &derived() const
    {
        return *this;
    }

    self_t &derived()
    {
        return *this;
    }

    // Must return the number of data points
    inline size_t kdtree_get_point_count() const
    {
        return length;
    }

    // Returns the dim'th component of the idx'th point in the class:
    inline num_t kdtree_get_pt(const size_t idx, const size_t dim) const
    {
        switch (dim)
        {
        case 0:
            return m_data[idx].x;
        case 1:
            return m_data[idx].y;
        case 2:
            return m_data[idx].z;
        }
        return 0;
    }

    // Optional bounding-box computation: return false to default to a standard bbox computation loop.
    //   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it again.
    //   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
    template <class BBOX>
    bool kdtree_get_bbox(BBOX & /*bb*/) const
    {
        return false;
    }
}; // end of KDTreeVectorOfVectorsAdaptor

template <typename T>
struct array_view
{
    T *__ptr;
    size_t __size;

    array_view(T *begin, size_t size) : __ptr(begin), __size(size) {}

    T &operator[](size_t i)
    {
        return __ptr[i];
    }

    const T &operator[](size_t i) const
    {
        return __ptr[i];
    }

    T *begin()
    {
        return __ptr;
    }

    T *end()
    {
        return __ptr + __size;
    }

    const T *begin() const
    {
        return __ptr;
    }

    const T *end() const
    {
        return __ptr + __size;
    }

    size_t size() const
    {
        return __size;
    }

    bool empty() const
    {
        return __size == 0;
    }

    T *data()
    {
        return __ptr;
    }
};

struct smoothness_t
{
    float value;
    size_t ind;

    bool operator<(const smoothness_t &other) const
    {
        return value < other.value;
    }
};

struct ranged_points
{
    size_t range_offsets[64];

    template <typename value_type>
    array_view<value_type> ring_span(int ring_id, value_type *points)
    {
        assert(ring_id < 64 && ring_id >= 0);
        if (ring_id == 0)
        {
            return array_view(points, range_offsets[0]);
        }
        return array_view(points + range_offsets[ring_id - 1], range_offsets[ring_id] - range_offsets[ring_id - 1]);
    }
};

struct maybe_type {
    float x, y, z, intensity;
};

template <typename point_type>
struct cloud_featured
{
    std::vector<point_type> corner_points;
    std::vector<point_type> surface_points;
};

template <typename point_type>
static inline void downsample_surf(std::vector<point_type> &surface_points)
{
    auto selected = surface_points.begin();
    for (auto i = selected + 1; i != surface_points.end(); i++)
    {
        if (p2(i->x - selected->x) + p2(i->y - selected->y) + p2(i->z - selected->z) < 4.0)
        {
            continue;
        }
        *++selected = std::move(*i);
    }
    surface_points.erase(selected + 1, surface_points.end());
}


template <typename point_type>
static inline void transform_point(point_type &p, const Eigen::Isometry3d &t)
{
    Eigen::Vector3d v(p.x, p.y, p.z);
    v = t * v;
    p.x = v.x();
    p.y = v.y();
    p.z = v.z();
}

template <typename point_type, typename In>
static inline point_type as(const In& value) {
    point_type p;
    p.x = value.x;
    p.y = value.y;
    p.z = value.z;
    p.intensity = value.intensity;
    

    return p;
}

template<typename point_type, typename point_type_in>
inline void get_features(const point_type_in* begin, const point_type_in* end, const size_t* ranges,
                         cloud_featured<point_type>& features) {

    constexpr size_t H_SCAN = 1800;
    constexpr float edgeThreshold = 1.0;
    constexpr float surfThreshold = 0.1;

    ranged_points points;

    std::vector<float> range(std::distance(begin, end));
    std::vector<int> cols(std::distance(begin, end));
    // since points is arranged by time, we can use std::stable_partition to split the points into
    // rings

    const point_type_in* ring_start = begin;
    size_t ring_id = 0;
    while(ring_start != end && ring_id < 64) {
        const point_type_in* ring = ring_start + ranges[ring_id];
        points.range_offsets[ring_id] = ring - begin;
        ring_id++;
        ring_start = ring;
    }

    // now we have the points arranged by rings, we can calculate the features
    // for each ring
    std::vector<float> curvature(std::distance(begin, end));
    std::vector<smoothness_t> smoothness(std::distance(begin, end));

    for(const point_type_in* i = begin; i != end; i++) {
        float d = i->x * i->x + i->y * i->y + i->z * i->z;
        range[i - begin] = std::sqrt(d);

        float angle = std::atan2(i->x, i->y) * 180.0f / M_PI;
        int columnIdn = -round((angle - 90.0f) / (360.0f / H_SCAN)) + H_SCAN / 2;
        if(columnIdn >= H_SCAN)
            columnIdn -= H_SCAN;

        if(columnIdn < 0 || columnIdn >= H_SCAN)
            columnIdn = 0;
        cols[i - begin] = columnIdn;
    }

    std::vector<bool> neighbor_picked(std::distance(begin, end), false);
    std::vector<bool> flag(std::distance(begin, end), false);

    size_t cloudSize = curvature.size();
    for(int i = 5; i < cloudSize - 5; i++) {
        float curv = range[i - 5] + range[i - 4] + range[i - 3] + range[i - 2] + range[i - 1] -
            range[i] * 10 + range[i + 1] + range[i + 2] + range[i + 3] + range[i + 4] +
            range[i + 5];

        curvature[i] = curv * curv;
        smoothness[i].value = curvature[i];
        smoothness[i].ind = i;
    }

    // mark occluded points and parallel beam points
    for(size_t i = 5; i < cloudSize - 6; ++i) {
        // occluded points
        float depth1 = range[i];
        float depth2 = range[i + 1];
        int diff = std::abs(cols[i + 1] - cols[i]);

        if(diff < 10) {
            // 10 pixel diff in range image
            if(depth1 - depth2 > 0.3) {
                neighbor_picked[i - 5] = true;
                neighbor_picked[i - 4] = true;
                neighbor_picked[i - 3] = true;
                neighbor_picked[i - 2] = true;
                neighbor_picked[i - 1] = true;
                neighbor_picked[i] = true;
            } else if(depth2 - depth1 > 0.3) {
                neighbor_picked[i + 1] = true;
                neighbor_picked[i + 2] = true;
                neighbor_picked[i + 3] = true;
                neighbor_picked[i + 4] = true;
                neighbor_picked[i + 5] = true;
                neighbor_picked[i + 6] = true;
            }
        }
        // parallel beam
        float diff1 = std::abs(range[i - 1] - range[i]);
        float diff2 = std::abs(range[i + 1] - range[i]);

        if(diff1 > 0.02 * range[i] && diff2 > 0.02 * range[i])
            neighbor_picked[i] = true;
    }

    for(int i = 0; i < ring_id; i++) {
        auto cloud_span = points.ring_span(i, begin);
        auto smoothness_span = points.ring_span(i, smoothness.data());

        for(int j = 0; j < 6; j++) {

            int sp = (cloud_span.size() * j) / 6;
            int ep = (cloud_span.size() * (j + 1)) / 6 - 1;

            if(sp >= ep)
                continue;

            std::sort(smoothness_span.begin() + sp, smoothness_span.begin() + ep);

            int largestPickedNum = 0;
            for(int k = ep; k >= sp; k--) {
                int ind = smoothness_span[k].ind;
                if(neighbor_picked[ind] == false && curvature[ind] > edgeThreshold &&
                   range[ind] > 2.0f) {
                    largestPickedNum++;
                    if(largestPickedNum <= 20) {
                        flag[ind] = true;
                        features.corner_points.push_back(as<point_type>(begin[ind]));
                    } else {
                        break;
                    }

                    neighbor_picked[ind] = true;
                    for(int l = 1; l <= 5; l++) {
                        int columnDiff = std::abs(int(cols[ind + l] - cols[ind + l - 1]));
                        if(columnDiff > 10)
                            break;
                        neighbor_picked[ind + l] = true;
                    }
                    for(int l = -1; l >= -5; l--) {
                        int columnDiff = std::abs(int(cols[ind + l] - cols[ind + l + 1]));
                        if(columnDiff > 10)
                            break;
                        neighbor_picked[ind + l] = true;
                    }
                }
            }

            for(int k = sp; k <= ep; k++) {
                int ind = smoothness_span[k].ind;
                if(neighbor_picked[ind] == false && curvature[ind] < surfThreshold &&
                   range[ind] > 2.0f) {

                    flag[ind] = false;
                    neighbor_picked[ind] = true;

                    for(int l = 1; l <= 5; l++) {

                        int columnDiff = std::abs(cols[ind + l] - cols[ind + l - 1]);
                        if(columnDiff > 10)
                            break;

                        neighbor_picked[ind + l] = true;
                    }
                    for(int l = -1; l >= -5; l--) {

                        int columnDiff = std::abs(cols[ind + l] - cols[ind + l + 1]);
                        if(columnDiff > 10)
                            break;

                        neighbor_picked[ind + l] = true;
                    }
                }
            }

            for(int k = sp; k <= ep; k++) {
                if(!flag[smoothness_span[k].ind]) {
                    features.surface_points.push_back(as<point_type>(cloud_span[k]));
                }
            }
        }
    }
}

template<typename point_type, typename Container>
cloud_featured<point_type> feature_velodyne(const Container& cloud, 
                                  float drop_radius = 2.0f) {
    cloud_featured<point_type> feature;

    size_t ring_count[64] = { 0 };
    auto cond = [drop_radius](auto&& p) {
        auto d = sqrtf(p2(p.x) + p2(p.y) + p2(p.z));
        return d > drop_radius && d < 100.0;
    };

    for(auto& p: cloud) {
        if(p.ring < 64 && cond(p))
            ring_count[p.ring]++;
    }

    size_t ring_offset[64] = { 0 };
    for(int i = 1; i < 64; i++) {
        ring_offset[i] = ring_offset[i - 1] + ring_count[i - 1];
    }

    std::vector<typename Container::value_type> points(ring_offset[63] + ring_count[63]);
    for(auto& p: cloud) {
        if(p.ring < 64 && cond(p))
            points[ring_offset[p.ring]++] = p;
    }

    get_features(points.data(), points.data() + points.size(), ring_count, feature);
    downsample_surf(feature.surface_points);
    return feature;
}

struct searched_line
{
    float nx, ny, nz;
    float cx, cy, cz;
    bool ok;
};

template <typename point_type, typename query_point_type>
static inline searched_line search_line(const array_adaptor<point_type> &tree, query_point_type point)
{

    size_t pointSearchInd[5];
    float pointSearchSqDis[5];

    tree.query(point, 5, pointSearchInd, pointSearchSqDis);
    Eigen::Matrix3f A1 = Eigen::Matrix3f::Zero();

    if (pointSearchSqDis[4] < 16.0)
    {
        float cx = 0, cy = 0, cz = 0;
        for (int j = 0; j < 5; j++)
        {
            cx += tree.m_data[pointSearchInd[j]].x;
            cy += tree.m_data[pointSearchInd[j]].y;
            cz += tree.m_data[pointSearchInd[j]].z;
        }
        cx /= 5;
        cy /= 5;
        cz /= 5;

        float a11 = 0, a12 = 0, a13 = 0, a22 = 0, a23 = 0, a33 = 0;
        for (int j = 0; j < 5; j++)
        {
            float ax = tree.m_data[pointSearchInd[j]].x - cx;
            float ay = tree.m_data[pointSearchInd[j]].y - cy;
            float az = tree.m_data[pointSearchInd[j]].z - cz;

            a11 += ax * ax;
            a12 += ax * ay;
            a13 += ax * az;
            a22 += ay * ay;
            a23 += ay * az;
            a33 += az * az;
        }
        a11 /= 5;
        a12 /= 5;
        a13 /= 5;
        a22 /= 5;
        a23 /= 5;
        a33 /= 5;

        A1(0, 0) = a11;
        A1(0, 1) = a12;
        A1(0, 2) = a13;
        A1(1, 0) = a12;
        A1(1, 1) = a22;
        A1(1, 2) = a23;
        A1(2, 0) = a13;
        A1(2, 1) = a23;
        A1(2, 2) = a33;

        Eigen::EigenSolver<Eigen::Matrix3f> es(A1);
        Eigen::Vector3f D1 = es.eigenvalues().real();
        Eigen::Matrix3f V1 = es.eigenvectors().real();

        if (D1(0) > 3 * D1(1))
        {
            searched_line line;
            line.nx = V1(0, 0);
            line.ny = V1(1, 0);
            line.nz = V1(2, 0);
            line.cx = cx;
            line.cy = cy;
            line.cz = cz;
            line.ok = true;
            return line;
        }
    }
    searched_line line;
    line.ok = false;
    return line;
}

template <typename point_type>
static inline coeff line_coeff(const searched_line &line, const point_type &p)
{
    coeff c;
    if (!line.ok)
    {
        c.px = p.x;
        c.py = p.y;
        c.pz = p.z;
        c.x = 0;
        c.y = 0;
        c.z = 0;
        c.b = 0;
        c.s = 0;
        return c;
    }

    float x0 = p.x;
    float y0 = p.y;
    float z0 = p.z;

    float x1 = line.cx + 0.1 * line.nx;
    float y1 = line.cy + 0.1 * line.ny;
    float z1 = line.cz + 0.1 * line.nz;
    float x2 = line.cx - 0.1 * line.nx;
    float y2 = line.cy - 0.1 * line.ny;
    float z2 = line.cz - 0.1 * line.nz;

    float a012 = sqrt(((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) * ((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) + ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1)) * ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1)) + ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1)) * ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1)));

    float l12 = sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2));

    float la = ((y1 - y2) * ((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) + (z1 - z2) * ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1))) / a012 / l12;

    float lb = -((x1 - x2) * ((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) - (z1 - z2) * ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1))) / a012 / l12;

    float lc = -((x1 - x2) * ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1)) + (y1 - y2) * ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1))) / a012 / l12;

    float ld2 = a012 / l12;

    float s = (4 - 0.9 * fabs(ld2)) / 4.0f;

    c.x = s * la;
    c.y = s * lb;
    c.z = s * lc;
    c.b = s * ld2;
    c.s = s;

    c.px = p.x;
    c.py = p.y;
    c.pz = p.z;

    return c;
}

struct plane
{
    float a, b, c, d;
    bool ok;
};

template <typename point_type>
static inline plane search_plane(const array_adaptor<point_type> &tree, const point_type &p)
{
    size_t pointSearchInd[5];
    float pointSearchSqDis[5];

    tree.query(p, 5, pointSearchInd, pointSearchSqDis);

    Eigen::Matrix<float, 5, 3> matA0;
    Eigen::Matrix<float, 5, 1> matB0;
    Eigen::Vector3f matX0;

    matA0.setZero();
    matB0.fill(-1);
    matX0.setZero();

    if (pointSearchSqDis[4] < 9.0)
    {
        for (int j = 0; j < 5; j++)
        {
            matA0(j, 0) = tree.m_data[pointSearchInd[j]].x;
            matA0(j, 1) = tree.m_data[pointSearchInd[j]].y;
            matA0(j, 2) = tree.m_data[pointSearchInd[j]].z;
        }

        matX0 = matA0.colPivHouseholderQr().solve(matB0);

        float pa = matX0(0, 0);
        float pb = matX0(1, 0);
        float pc = matX0(2, 0);
        float pd = 1;

        float ps = sqrt(pa * pa + pb * pb + pc * pc);
        pa /= ps;
        pb /= ps;
        pc /= ps;
        pd /= ps;

        bool planeValid = true;
        for (int j = 0; j < 5; j++)
        {
            if (fabs(pa * tree.m_data[pointSearchInd[j]].x +
                     pb * tree.m_data[pointSearchInd[j]].y +
                     pc * tree.m_data[pointSearchInd[j]].z + pd) > 0.2)
            {
                planeValid = false;
                break;
            }
        }

        plane pl;
        pl.a = pa;
        pl.b = pb;
        pl.c = pc;
        pl.d = pd;

        pl.ok = planeValid;
        return pl;
    }

    plane pl;
    pl.ok = false;
    return pl;
}

template <typename point_type>
static inline coeff plane_coeff(const plane &pl, const point_type &p)
{
    if (pl.ok)
    {
        float pd2 = pl.a * p.x + pl.b * p.y + pl.c * p.z + pl.d;

        float s = 1 - fabs(pd2) / 3.0f; // sqrt(sqrt(p.x * p.x + p.y * p.y + p.z * p.z));
        coeff c;

        c.x = s * pl.a;
        c.y = s * pl.b;
        c.z = s * pl.c;
        c.b = s * pd2;
        c.s = s;
        c.px = p.x;
        c.py = p.y;
        c.pz = p.z;

        return c;
    }

    coeff c;
    c.s = 0;
    return c;
}

using MatrixA = Eigen::Matrix<float, Eigen::Dynamic, 6>;
using VectorB = Eigen::Matrix<float, Eigen::Dynamic, 1>;

size_t remove_zero_rows(MatrixA &A, VectorB& b )
{
    size_t row = 0;
    for (size_t i = 0; i < A.rows(); i++)
    {
        if(b(i) != 0.0) {
            if(row != i) {
                A.row(row) = A.row(i);
                b(row) = b(i);
            }
            row++;
        }
    }

    return row;
}

template <typename source_point_type, typename target_point_type>
static inline Transform __LM_iteration_v1(const cloud_featured<source_point_type> &source, array_adaptor<target_point_type> &corner,
                                       array_adaptor<target_point_type> &surf, MatrixA &A, VectorB &b, const Transform &initial_guess, float *loss = nullptr)
{
    Eigen::Isometry3d transform = to_eigen(initial_guess);

    jacobian_g      g;
    init_jacobian_g(g, initial_guess);

    size_t corner_size = source.corner_points.size();
    size_t surf_size = source.surface_points.size();
    if (loss)
    {
        *loss = 0;
    }

    float loss_value = 0;

#pragma omp parallel for reduction(+ : loss_value)
    for (size_t i = 0; i < corner_size + surf_size; ++i)
    {
        coeff c;
        c.s = 0;
        if (i < corner_size)
        {
            source_point_type p2 = source.corner_points[i];
            transform_point(p2, transform);

            searched_line sl = search_line(corner, p2);
            if (sl.ok)
            {
                c = line_coeff(sl, p2);
            }
        }
        else
        {
            int idx = i - corner_size;
            source_point_type p2 = source.surface_points[idx];
            transform_point(p2, transform);

            plane sp = search_plane(surf, p2);
            if (sp.ok)
            {
                c = plane_coeff(sp, p2);
            }
        }

        if (c.s < 0.1f)
        {
            A.row(i).setZero();
            b(i) = 0;
            continue;
        }
        jacobian j = J(c, g);
        A.row(i) = j.cast<float>().transpose();
        b(i) = -c.b;

        loss_value += fabsf(c.b);
    }

    size_t row = remove_zero_rows(A, b);
    if(row < 100) {
        if (loss)
        {
            loss[0] = 1000;
        }
        return initial_guess;
    }
    

    auto A_block = A.topRows(row);
    auto b_block = b.topRows(row);

    Eigen::Matrix<float, 6, 6> ATA = A_block.transpose() * A_block;
    Eigen::Matrix<float, 6, 1> ATb = A_block.transpose() * b_block;

    Eigen::Matrix<float, 6, 1> x = ATA.householderQr().solve(ATb);

    Transform delta;
    delta.x = x(0, 0) + initial_guess.x;
    delta.y = x(1, 0) + initial_guess.y;
    delta.z = x(2, 0) + initial_guess.z;
    delta.roll = x(3, 0) + initial_guess.roll;
    delta.pitch = x(4, 0) + initial_guess.pitch;
    delta.yaw = x(5, 0) + initial_guess.yaw;  

    if (loss)
    {
        loss[0] = loss_value / row;
    }

    return delta;
}


template <typename source_point_type, typename target_point_type>
static inline Transform __LM_iteration_v2(const cloud_featured<source_point_type> &source, array_adaptor<target_point_type> &corner,
                                       array_adaptor<target_point_type> &surf, MatrixA &A, VectorB &b, const Transform &initial_guess, float *loss = nullptr)
{
    Eigen::Isometry3d transform = to_eigen(initial_guess);

    size_t corner_size = source.corner_points.size();
    size_t surf_size = source.surface_points.size();
    if (loss)
    {
        *loss = 0;
    }

    float loss_value = 0;

#pragma omp parallel for reduction(+ : loss_value)
    for (size_t i = 0; i < corner_size + surf_size; ++i)
    {
        coeff c;
        c.s = 0;
        if (i < corner_size)
        {
            source_point_type p2 = source.corner_points[i];
            transform_point(p2, transform);

            searched_line sl = search_line(corner, p2);
            if (sl.ok)
            {
                c = line_coeff(sl, p2);
            }
        }
        else
        {
            int idx = i - corner_size;
            source_point_type p2 = source.surface_points[idx];
            transform_point(p2, transform);

            plane sp = search_plane(surf, p2);
            if (sp.ok)
            {
                c = plane_coeff(sp, p2);
            }
        }

        if (c.s < 0.1f)
        {
            A.row(i).setZero();
            b(i) = 0;
            continue;
        }

        jacobian j = J2(c);

        A.row(i) = j.cast<float>().transpose();
        b(i) = -c.b;

        loss_value += fabsf(c.b);
    }

    size_t row = remove_zero_rows(A, b);
    if(row < 100) {
        if (loss)
        {
            loss[0] = 1000;
        }
        return initial_guess;
    }
    

    auto A_block = A.topRows(row);
    auto b_block = b.topRows(row);

    Eigen::Matrix<float, 6, 6> ATA = A_block.transpose() * A_block;
    Eigen::Matrix<float, 6, 1> ATb = A_block.transpose() * b_block;

    Eigen::Matrix<float, 6, 1> x = ATA.householderQr().solve(ATb);

    Transform delta;
    delta.x = x(0, 0);
    delta.y = x(1, 0);
    delta.z = x(2, 0);
    delta.roll = x(3, 0);
    delta.pitch = x(4, 0);
    delta.yaw = x(5, 0);

    transform = to_eigen(delta) * transform;
    delta = from_eigen(transform);

    if (loss)
    {
        loss[0] = loss_value / row;
    }

    return delta;
}


template <typename _Call>
struct Caller
{
    _Call call;
    Caller(_Call &&call) : call(std::forward<_Call>(call)) {}
    template <typename... Args>
    auto operator()(Args &&...args)
    {
        return call(std::forward<Args>(args)...);
    }
};

template <>
struct Caller<nullptr_t>
{
    explicit Caller(nullptr_t) {}
    template <typename... Args>
    void operator()(Args &&...args) {}
};

template <bool use_v1, typename source_point_type, typename target_point_type, typename _Call = nullptr_t>
static Transform LM(const cloud_featured<source_point_type> &source, const cloud_featured<target_point_type> &target,
                    const Transform &initial_guess = Transform(), float *loss = nullptr, _Call &&call = nullptr)
{
    array_adaptor<target_point_type> corner(target.corner_points.data(), target.corner_points.size());
    array_adaptor<target_point_type> surf(target.surface_points.data(), target.surface_points.size());

    MatrixA A(source.corner_points.size() + source.surface_points.size(), 6);
    VectorB b(source.corner_points.size() + source.surface_points.size());

    Transform result = initial_guess;
    Caller<_Call> caller(std::forward<_Call>(call));

    for (int iter = 0; iter < 50; ++iter)
    {
        Transform u ;
        if constexpr (use_v1)
            u = __LM_iteration_v1(source, corner, surf, A, b, result, loss);
        else
            u = __LM_iteration_v2(source, corner, surf, A, b, result, loss);

        float deltaR = sqrtf(p2(u.roll - result.roll) + p2(u.pitch - result.pitch) + p2(u.yaw - result.yaw));
        float deltaT = sqrtf(p2(u.x - result.x) + p2(u.y - result.y) + p2(u.z - result.z));
        result = u;
        if (deltaR < 0.00005 && deltaT < 0.0005)
        {
            break;
        }
        caller(result);
    }

    return result;
}

struct LOAMarg {
    float loss_threshold = 0.12f;
};

static inline void save(const cloud_featured<maybe_type>& cloud, size_t index) {
    char filename[256];
    sprintf(filename, "cloud_corner_%zu.pcd", index);
    pcl::PointCloud<pcl::PointXYZI> corner;
    for(auto& p: cloud.corner_points) {
        pcl::PointXYZI pt;
        pt.x = p.x;
        pt.y = p.y;
        pt.z = p.z;
        pt.intensity = p.intensity;
        corner.push_back(pt);
    }

    pcl::io::savePCDFileBinary(filename, corner);

    sprintf(filename, "cloud_surface_%zu.pcd", index);
    pcl::PointCloud<pcl::PointXYZI> surface;
    for(auto& p: cloud.surface_points) {
        pcl::PointXYZI pt;
        pt.x = p.x;
        pt.y = p.y;
        pt.z = p.z;
        pt.intensity = p.intensity;
        surface.push_back(pt);
    }
    pcl::io::savePCDFileBinary(filename, surface);
}

template<bool use_v1>
struct loam_verify_v {
    template<typename InputType>
    std::pair<Eigen::Isometry3d, double> verify(
        const InputType& source, const InputType& target, const Eigen::Isometry3d& guess) {
        auto source_feature = feature_velodyne<maybe_type>(source);
        auto target_feature = feature_velodyne<maybe_type>(target);
        float loss = 0.0f;

        auto result = LM<use_v1>(source_feature, target_feature, from_eigen(guess), &loss);

        return { to_eigen(result), loss };
    }
};

using loam_verify_v1 = loam_verify_v<true>;
using loam_verify_v2 = loam_verify_v<false>;

#endif
