#include <string>
#include <vector>
#include <filesystem>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>

#include <verify/loam.hpp>
#include <verify/ScanContext.hpp>

struct XYZI
{
    float x;
    float y;
    float z;
    float intensity;
};

struct PointXYZIR
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;
    uint16_t ring;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZIR,
                                  (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(uint16_t, ring, ring))



static inline bool str_is_number(const char *str)
{
    while (*str)
    {
        if (*str < '0' || *str > '9')
            return false;
        str++;
    }
    return true;
}

static bool numberic_compare(const std::string &a, const std::string &b)
{
    std::filesystem::path p1(a);
    std::filesystem::path p2(b);

    return std::stoi(p1.stem()) < std::stoi(p2.stem());
}

static bool list_all_bin_files(const char *path, std::vector<std::string> &files)
{
    for (auto &p : std::filesystem::directory_iterator(path))
    {
        if (p.is_regular_file())
        {
            if (!str_is_number(p.path().stem().c_str()))
            {
                printf("Ignore file: %s\r\n", p.path().string().c_str());
                continue;
            }
            if (p.path().extension() == ".bin")
                files.push_back(p.path().string());
            else
            {
                printf("Ignore file: %s\r\n", p.path().string().c_str());
            }
        }
    }

    std::sort(files.begin(), files.end(), numberic_compare);
    return true;
}

static bool list_all_pcd_files(const char *path, std::vector<std::string> &files)
{
    for (auto &p : std::filesystem::directory_iterator(path))
    {
        if (p.is_regular_file())
        {
            if (!str_is_number(p.path().stem().c_str()))
            {
                printf("Ignore file: %s\r\n", p.path().string().c_str());
                continue;
            }
            if (p.path().extension() == ".pcd")
                files.push_back(p.path().string());
            else
            {
                printf("Ignore file: %s\r\n", p.path().string().c_str());
            }
        }
    }

    std::sort(files.begin(), files.end(), numberic_compare);
    return true;
}

static bool load_from_bin(const char *path, std::vector<XYZI> &cloud)
{
    FILE *f = fopen(path, "rb");
    if (!f)
    {
        printf("Open file %s failed\r\n", path);
        return false;
    }

    fseek(f, 0, SEEK_END);
    size_t size = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (size % (4 * sizeof(float)) != 0)
    {
        printf("Invalid data in file %s\r\n", path);
        fclose(f);
        return false;
    }

    size_t num_points = size / (4 * sizeof(float));
    cloud.resize(num_points);

    size_t counter = fread(cloud.data(), sizeof(XYZI), num_points, f);
    if (counter != num_points)
    {
        printf("Read file %s failed\r\n", path);
        fclose(f);
        return false;
    }

    fclose(f);
    return true;
}

struct KLoader
{
    std::vector<std::string> files;

    KLoader(const char *path)
    {
        list_all_bin_files(path, files);
        if (files.empty())
        {
            printf("No files found in %s\r\n", path);
            return;
        }
    }

    static inline size_t get_rings(const XYZI *points, size_t count)
    {
        bool neg = false;
        for (size_t i = 0; i < count; i++)
        {
            float yaw = atan2f(points[i].y, points[i].x);
            if (yaw < 0)
                neg = true;
            if (yaw > 0 && neg && i > 1100)
                return i;
        }
        return count;
    }

    static void calculate_ringid(const std::vector<XYZI> &points, std::vector<PointXYZIR, Eigen::aligned_allocator<PointXYZIR>> &result)
    {
        size_t offset = 0;
        size_t ring_index = 0;
        size_t count = points.size();
        result.resize(count);

        for (int i = 0; i < 64 && offset < count; i++)
        {
            size_t next_ring = get_rings(points.data() + offset, count - offset);
            for (size_t j = offset; j < offset + next_ring; j++)
            {
                result[j].x = points[j].x;
                result[j].y = points[j].y;
                result[j].z = points[j].z;
                result[j].intensity = points[j].intensity * 255;
                result[j].ring = ring_index;
            }
            offset += next_ring;
            ring_index++;
        }

        assert(offset == count);
        // assert(ring_index == 64);
    }

    pcl::PointCloud<PointXYZIR>::Ptr load(size_t i) const
    {
        pcl::PointCloud<PointXYZIR>::Ptr cloud(new pcl::PointCloud<PointXYZIR>);
        std::vector<XYZI> data;

        if (i >= files.size())
        {
            printf("Index out of range\r\n");
            return cloud;
        }

        load_from_bin(files[i].c_str(), data);

        cloud->resize(data.size());
        calculate_ringid(data, cloud->points);
        return cloud;
    }

    size_t size() const
    {
        return files.size();
    }
};

struct PLoader
{
    std::vector<std::string> files;

    PLoader(const char *path)
    {
        list_all_pcd_files(path, files);
        if (files.empty())
        {
            printf("No files found in %s\r\n", path);
            return;
        }
    }

    pcl::PointCloud<PointXYZIR>::Ptr load(size_t i) const
    {
        pcl::PointCloud<PointXYZIR>::Ptr cloud(new pcl::PointCloud<PointXYZIR>);
        pcl::io::loadPCDFile(files[i], *cloud);
        return cloud;
    }

    size_t size() const
    {
        return files.size();
    }
};

template <typename PointType>
void save_map(const std::string &path,
              const std::vector<typename pcl::PointCloud<PointType>::Ptr> &all_clouds,
              const std::vector<size_t> &key_frame_id,
              const std::vector<Eigen::Isometry3d> &poses)
{
    std::filesystem::path p(path);
    if (!std::filesystem::exists(p))
    {
        std::filesystem::create_directories(p);
    }

    std::filesystem::path full_cloud = p / "full_cloud.pcd";
    printf("Saving full cloud to %s\r\n", full_cloud.string().c_str());
    pcl::PointCloud<PointType> full_cloud_data;
    for (size_t i = 0; i < key_frame_id.size(); i++)
    {
        size_t id = key_frame_id[i];
        pcl::PointCloud<PointType> transformed;
        pcl::transformPointCloud(*all_clouds[id], transformed, poses[id].matrix().cast<float>());
        full_cloud_data += transformed;
    }

    pcl::io::savePCDFileBinary(full_cloud.string(), full_cloud_data);

    std::filesystem::path poses_file = p / "poses.txt";

    FILE *f = fopen(poses_file.string().c_str(), "w");
    for (size_t i = 0; i < poses.size(); i++)
    {
        Eigen::Quaterniond q(poses[i].linear());
        Eigen::Vector3d t = poses[i].translation();
        fprintf(f, "%f %f %f %f %f %f %f %f\n", key_frame_id[i] * 0.1f, t.x(), t.y(), t.z(), q.x(), q.y(), q.z(), q.w());
    }

    fclose(f);
}

template<typename LoaderType, typename VerifyType>
double run(std::string loader_var, std::string output_path, std::string matrix_path) {
    LoaderType loader(loader_var.c_str());
    VerifyType verify;

    ScanContext looper;

    std::vector<pcl::PointCloud<PointXYZIR>::Ptr> all_clouds;
    FILE* f = fopen(output_path.c_str(), "w");
    FILE* mat = fopen(matrix_path.c_str(), "w");

    auto now = std::chrono::system_clock::now();
    for (size_t i = 0; i < loader.size(); i++)
    {
        pcl::PointCloud<PointXYZIR>::Ptr cloud = loader.load(i);
        all_clouds.push_back(cloud);

        auto desc = looper.transform(*cloud, Eigen::Isometry3d::Identity());
        auto [id, pose] = looper.loop(desc);
        looper.add(desc);

        if(id < 0) {
            printf("No loop found for frame %zu\r\n", i);
            continue;
        }

        auto [result, loss] = verify.verify(*cloud, *all_clouds[id], pose);

        Transform tr = from_eigen(result);

        fprintf(f, "%zu %d %f\n", i, id, loss);
        fprintf(mat, "%zu %f %f %f %f %f %f\n", i, tr.x, tr.y, tr.z, tr.roll, tr.pitch, tr.yaw);
        printf("Frame %zu %d loss %f\n", i, id, loss);
        fflush(f);
        fflush(stdout);
    }

    auto end = std::chrono::system_clock::now();
    fclose(f);

    return std::chrono::duration_cast<std::chrono::milliseconds>(end - now).count();
}

template<typename VerifyType>
static double consume_loader(const std::string loader_type, const std::string& loader_var, const std::string& output_path, const std::string& matrix_path) {
    if(loader_type == "bin") {
        return run<KLoader, VerifyType>(loader_var, output_path, matrix_path);
    } else if(loader_type == "pcd") {
        return run<PLoader, VerifyType>(loader_var, output_path, matrix_path);
    } else {
        printf("Invalid loader type %s\r\n", loader_type.c_str());
        return 0;
    }
}

static double consume_verify(const std::string& verify_type, const std::string& loader_type, 
    const std::string& loader_var, const std::string& output_path, const std::string& matrix_path) {
    if(verify_type == "loamv1") {
        return consume_loader<loam_verify_v1>(loader_type, loader_var, output_path, matrix_path);
    } else if(verify_type == "loamv2") {
        return consume_loader<loam_verify_v2>(loader_type, loader_var, output_path, matrix_path);
    } else {
        printf("Invalid verify type %s\r\n", verify_type.c_str());
        return 0;
    }
}

int main(int argc, char **argv)
{
    if (argc < 6)
    {
        printf("Usage: %s <verify_type> <loader_type> <loader_var> <output_path> <matrix>\r\n", argv[0]);
        return 0;
    }

    double time = consume_verify(argv[1], argv[2], argv[3], argv[4], argv[5]);
    printf("Time consumed[%s, %s]: %f ms\r\n", argv[1], argv[2], time);
    return 0;
}