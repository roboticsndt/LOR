# LOR : LiDAR Odometry-based Registration
### This is the repo for IJCAI submission #2005.

A LiDAR odometry-based registration method is proposed for loop closure detection.

![ours](doc/2025-01-17T12_35_08.898Z-139132.gif)
The GIF illustrates the iterative process. The source point cloud is shown in white and blue, while the target point cloud is depicted in green and red. The different colors within each point cloud represent edge features and planar features.

The code has been tested in the following environment.
|ubuntu| pcl | nanoflann |
| -- | -- | -- | 
| 24.04 | 1.14.0 | 1.5.4 |

## Self-collected dataset

The Campus "08", "43", "51", "59" sequences can be available at https://www.kaggle.com/datasets/anacondaspyder/self-collected-dataset.

## Prepare the data

Save the data in a folder, naming each file according to the frame number, starting from 0.

If the files are in PCD format, the points in each file should include the fields: x, y, z, intensity, and ring.

If the files are in the same bin format as KITTI, the point clouds should also be ordered in the same way as KITTI.This is to calculate the ring information of the point cloud.


## Compile

1. Install dependences
``` shell
sudo apt install libpcl-dev libnanoflann-dev
```
2. Clone the sophus library
``` shell
git clone https://github.com/strasdat/Sophus.git
```
Copy the sophus folder into the corresponding folder within this repository.

3. Compile
Create a new folder, change to the corresponding directory, and execute cmake & make.
``` shell
mkdir build-release
cd build-release
cmake ..
make -j
```

4. run the code

``` shell
./path_to_executable loamv2 <bin|pcd> /path/to/data/folder /path/to/output/result.txt /path/to/matrix_output.txt
```

To evaluate the results, you can use PRcurve1.py to calculate the PR curve, use eval to generate the evaluation data.
For this, you will need to prepare an additional file containing the ground truth poses for each frame. see src/eval.cpp for details.


### Acknowledgement 
LOR is based on LOAM(J. Zhang and S. Singh.) and lio-sam(Tixiao Shan).
