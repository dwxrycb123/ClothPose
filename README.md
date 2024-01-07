# ClothPose
This repository contains codes for paper: <br>ClothPose: A Real-world Benchmark for Visual Analysis of Garment Pose via An Indirect Recording Solution, ICCV2023

For barrier function computation, continuous collision detection (CCD) and linear system solving, our codes are based on [IPC repository](https://github.com/ipc-sim/IPC). 

# Install 
## Build From Source
We need to install tbb first: 
```bash 
# for ubuntu users 
sudo apt-get install libtbb-dev
```

Clone this repository and build the library using: 
```bash
git clone https://github.com/dwxrycb123/ClothPose.git --recursive --depth 1 
cd ClothPose
cmake -Bbuild -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 16
```

We can run test programs using the following command: 
```bash 
./build/test_optimizer 
python ./test/test_pybind.py
```
The output for `./build/test_optimizer` is an obj file `fit_init.obj` in the folder `./output`. You can visualize it by using Meshlab or other softwares. 

## Use Prebuilt Python Binding 
For `cpython-310-x86_64-linux-gnu` (Python 3.10, `x86_64` architecture, linux), we have also provided a prebuilt Python library in `./prebuilt/py_clothpose`. 

# Usage 
The basic usages of the library are shown in `./test/test_optimizer.cpp` (for C++ interfaces) and `./test/test_pybind.py` (for Python binding interfaces). 

These two examples first load the rest mesh of a dress from `./resources/init/rest_repaired.obj`, then use a coarse mesh (located in `./resources/init/ClothMesh_156.obj`) for guidance to fit the rest mesh onto a real-world point cloud (located in `./resources/init/init.ply`). 


# Dataset 
Coming soon 

# Citation
Please cite the following paper if you find this repository useful: 
```
@inproceedings{xu2023clothpose,
  title={Clothpose: A real-world benchmark for visual analysis of garment pose via an indirect recording solution},
  author={Xu, Wenqiang and Du, Wenxin and Xue, Han and Li, Yutong and Ye, Ruolin and Wang, Yan-Feng and Lu, Cewu},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={58--68},
  year={2023}
}
```
