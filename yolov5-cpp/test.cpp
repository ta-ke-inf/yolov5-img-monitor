#include <iostream>
#include <opencv2/opencv.hpp>


int main()
{
    // �\������Ƃ��̃E�B���h�E��
    const char* windowName = "image";

    // ���[�U�[�̖��O�͎����̂��̂ɕύX���Ă�������
    cv::Mat img = cv::imread("C:\\Users\\hirom\\Desktop\\test.png");

    cv::imshow(windowName, img);

    cv::waitKey(0);

    return 0;
}