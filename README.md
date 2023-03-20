# yolov5-cpp
Script that monitors folders asynchronously, infers with `C++ YOLOv5` when additional images are detected, and stores the detected images and confidence scores.

## Build Requirements
- OpenCV 4.5.4+
- CMake-gui 3.2+
- Windows

## Quick Usage
1. 
```
git clone https://github.com/ta-ke-inf/yolov5-cpp.git
cd ./yolov5-cpp/yolov5-cpp
mkdir image result result_img
yolov5-cpp.exe
```
2. Copy `test.png` to the `image` folder you just created.

