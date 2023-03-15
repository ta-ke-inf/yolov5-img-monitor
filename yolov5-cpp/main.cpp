#include <iostream>
#include <opencv2/opencv.hpp>


int main()
{
    // 表示するときのウィンドウ名
    const char* windowName = "image";

    // ユーザーの名前は自分のものに変更してください
    cv::Mat img = cv::imread("C:\\Users\\hirom\\Desktop\\test.png");

    cv::imshow(windowName, img);

    cv::waitKey(0);

    return 0;
}