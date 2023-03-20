# yolov5-img-monitor
![result](https://user-images.githubusercontent.com/115391575/226294371-ec9ecd2f-7722-4ec5-bd90-677832166226.png)

Script that monitors folders asynchronously, infers with `C++ YOLOv5` when additional images are detected, and stores the detected images and confidence scores.

## Build Requirements
- OpenCV 4.5.4+
- CMake-gui 3.2+
- Windows

## How to convert Pytorch".pt" to ONNX".onnx"
Please see https://github.com/ultralytics/yolov5
```
git clone https://github.com/ultralytics/yolov5  # clone
cd yolov5
pip install -r requirements.txt  # install
python export.py --weights yolov5s.pt --include onnx # convert
```
## Quick Usage
1. Start asynchronous monitoring
```
git clone https://github.com/ta-ke-inf/yolov5-cpp.git
cd ./yolov5-cpp/yolov5-cpp
mkdir image result result_img
yolov5-cpp.exe
```
2. Copy `test.png` to the `image` folder you just created.

