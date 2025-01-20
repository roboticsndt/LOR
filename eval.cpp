#include <stdio.h>
#include <Eigen/Dense>
#include <map>
#include <sophus/se3.hpp>
#include <set>

struct Tum
{
    double timestamp;
    Sophus::SE3d pose;
};

static std::vector<Tum> load_tum(const char *path)
{
    std::vector<Tum> poses;
    FILE *f = fopen(path, "r");
    if (f == nullptr)
    {
        printf("Failed to open file %s\r\n", path);
        return poses;
    }
    double timestamp;
    double x, y, z, qx, qy, qz, qw;
    while (fscanf(f, "%lf %lf %lf %lf %lf %lf %lf %lf", &timestamp, &x, &y, &z, &qx, &qy, &qz, &qw) == 8)
    {
        Eigen::Vector3d t(x, y, z);
        Eigen::Quaterniond q(qw, qx, qy, qz);
        poses.push_back({timestamp, Sophus::SE3d(q, t)});
    }

    fclose(f);
    return poses;
}

struct loop_info
{
    size_t self_id;
    int history_id;
    float score;
    Sophus::SE3d transform;
};

template <typename Scalar>
struct _Transform
{
    Scalar x = 0.0f, y = 0.0f, z = 0.0f;
    Scalar roll = 0.0f, pitch = 0.0f, yaw = 0.0f;

    template <typename Scalar2>
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

template <typename Scalar>
static inline Eigen::Matrix4<Scalar> to_eigen(const _Transform<Scalar> &tr)
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

    return m;
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
static inline _Transform<Scalar> from_eigen(const Eigen::Transform<Scalar, 3, Options> &m)
{
    return from_eigen(m.matrix());
}

using loop_set = std::set<std::pair<size_t, size_t>>;

static loop_set generate_loop_set(const std::vector<Tum>& tum) {
    constexpr size_t exclude = 50;

    loop_set loops;

    for(size_t i = exclude; i < tum.size(); i++) {
        for(size_t j = 0; j < i - exclude; j++) {
            auto p0 = tum[i].pose.translation();
            auto p1 = tum[j].pose.translation();

            if((p0 - p1).norm() < 3.0) {
                loops.insert({i, j});
            }
        }
    }

    return loops;
};

static std::vector<loop_info> load_loop_var(std::string log, std::string matrix)
{
    std::vector<loop_info> data;
    FILE *f_log = fopen(log.c_str(), "r");
    if (f_log == nullptr)
    {
        printf("Failed to open loop log file\n");
        return {};
    }

    FILE *f_matrix = fopen(matrix.c_str(), "r");
    if (f_matrix == nullptr)
    {
        printf("Failed to open loop matrix file\n");
        return {};
    }

    // id history_id score
    // id x y z roll pitch yaw

    int matrix_id, id, history_id;
    float score;

    Transform t;

    while (fscanf(f_log, "%d %d %f", &id, &history_id, &score) == 3)
    {
        if (fscanf(f_matrix, "%d %lf %lf %lf %lf %lf %lf", &matrix_id, &t.x, &t.y, &t.z, &t.roll, &t.pitch, &t.yaw) != 7)
        {
            printf("Failed to read matrix file\n");
            return {};
        }

        if (matrix_id != id)
        {
            printf("Matrix id mismatch\n");
            return {};
        }

        loop_info info;
        info.history_id = history_id;
        info.score = score;
        info.transform = Sophus::SE3d(to_eigen(t));
        info.self_id = id;
        data.push_back(info);
    }

    fclose(f_log);
    fclose(f_matrix);

    return data;
}

struct eval_info {
    struct eval_inx {
        size_t self_id;
        size_t history_id;
        double score;
        double loss;
    };

    std::vector<eval_inx> losses;
    double average_loss;
    double average_tr_loss;
    double average_rot_loss;
};

template<typename V>
static V scale(V in) {
    V out = in;
    out(0) *= 0.1f;
    out(1) *= 0.1f;
    out(2) *= 0.1f;

    return out;
}

static eval_info eval(loop_set loops, std::vector<loop_info> &infos, const std::vector<Tum> &poses)
{
    double all_loss = 0.0f;
    double tr_loss = 0.0f;
    double rot_loss = 0.0f;
    size_t count = 0;
    eval_info info;

    for(size_t i = 0; i < infos.size(); i++) {
        std::pair<size_t, size_t> p = {infos[i].self_id, infos[i].history_id};

        /*if(loops.count(p) == 0) {
            continue;
        }*/

        Sophus::SE3d pose = poses[infos[i].self_id].pose;
        Sophus::SE3d history = poses[infos[i].history_id].pose;
        Sophus::SE3d transform = history.inverse() * pose;
        Sophus::SE3d estimate = infos[i].transform;

        Sophus::SE3d between = transform.inverse() * estimate;

        double loss = scale(between.log()).squaredNorm();

        double this_tr_loss = between.translation().norm();
        double this_rot_loss = std::acos((between.rotationMatrix().trace() - 1) / 2.0);

        info.losses.push_back({
            (size_t)infos[i].self_id,
            (size_t)infos[i].history_id,
            (double)infos[i].score,
            loss});

        if (loss < 1.0f) {
            all_loss += loss;            
            tr_loss += this_tr_loss;
            rot_loss += this_rot_loss;
            count++;
        }
    }
    
    if(count == 0) {
        info.average_loss = 0.0f;
        info.average_tr_loss = 0.0f;
        info.average_rot_loss = 0.0f;
    } else {
        info.average_loss = all_loss / count;
        info.average_tr_loss = tr_loss / count;
        info.average_rot_loss = rot_loss / count;
    }

    return info;
}

static void save_eval_info(const eval_info &info, const char *path)
{
    FILE *f = fopen(path, "w");
    if(f == nullptr) {
        printf("Failed to open file %s\n", path);
        return;
    }

    for(auto& l: info.losses) {
        fprintf(f, "%zd %zd %f %f\n", l.self_id, l.history_id, l.score, l.loss);
    }

    printf("Average loss: %f(%zu)\n", info.average_loss, info.losses.size());   
    printf("Average translation loss: %f\n", info.average_tr_loss);
    printf("Average rotation loss: %f\n", info.average_rot_loss);
    fclose(f);
}

static void save_loop_set(const loop_set &loops, const char *path)
{
    FILE *f = fopen(path, "w");
    if(f == nullptr) {
        printf("Failed to open file %s\n", path);
        return;
    }

    std::map<size_t, std::vector<size_t>> data;

    for(auto& l: loops) {
        data[l.first].push_back(l.second);
    }

    for(size_t i = 0; i < data.size(); i++) {
        fprintf(f, "%zu", i);
        for(auto& l: data[i]) {
            fprintf(f, " %zu", l);
        }
        fprintf(f, "\n");
    }

    fclose(f);
}

int main(int argc, char **argv)
{
    if(argc < 5) {
        printf("Usage: %s <loop_log> <loop_matrix> <tum> <output> [loop_set]\n", argv[0]);
        return 0;
    }

    std::string loop_log = argv[1];
    std::string loop_matrix = argv[2];
    std::string tum = argv[3];

    auto poses = load_tum(tum.c_str());
    auto infos = load_loop_var(loop_log, loop_matrix);
    auto loops = generate_loop_set(poses);
    auto info = eval(loops, infos, poses);
    save_eval_info(info, argv[4]);
    
    if(argc > 5) {
        save_loop_set(loops, argv[5]);
    }

    return 0;
}