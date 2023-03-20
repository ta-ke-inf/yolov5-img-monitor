#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <stdio.h>
#include<locale.h>
#include<tchar.h>
#include<vector>
#include<conio.h>
#include <Windows.h>
#include <string>


using namespace std;
using namespace cv;
using namespace dnn;
//using namespace cuda;

std::vector<std::string> load_class_list()
{
    std::vector<std::string> class_list;
    std::ifstream ifs("classes.txt");
    std::string line;
    while (getline(ifs, line))
    {
        class_list.push_back(line);
    }
    return class_list;
}

void load_net(cv::dnn::Net& net, bool is_cuda)
{
    auto result = cv::dnn::readNet("yolov5s.onnx");
    if (is_cuda)
    {
        std::cout << "Attempty to use CUDA\n";
        result.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        result.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
    }
    else
    {
        std::cout << "Running on CPU\n";
        result.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        result.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }
    net = result;
}

const std::vector<cv::Scalar> colors = { cv::Scalar(255, 255, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 255), cv::Scalar(255, 0, 0) };

const float INPUT_WIDTH = 640.0;
const float INPUT_HEIGHT = 640.0;
const float SCORE_THRESHOLD = 0.2;
const float NMS_THRESHOLD = 0.4;
const float CONFIDENCE_THRESHOLD = 0.4;

struct Detection
{
    int class_id;
    float confidence;
    cv::Rect box;
};


cv::Mat format_yolov5(const cv::Mat& source) {
    int col = source.cols;
    int row = source.rows;
    int _max = MAX(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(cv::Rect(0, 0, col, row)));
    return result;
}


void detect(cv::Mat& image, cv::dnn::Net& net, std::vector<Detection>& output, const std::vector<std::string>& className) {
    cv::Mat blob;

    auto input_image = format_yolov5(image);

    cv::dnn::blobFromImage(input_image, blob, 1. / 255., cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(), true, false);
    net.setInput(blob);
    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    float x_factor = input_image.cols / INPUT_WIDTH;
    float y_factor = input_image.rows / INPUT_HEIGHT;

    float* data = (float*)outputs[0].data;

    const int dimensions = 85;
    const int rows = 25200;

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (int i = 0; i < rows; ++i) {

        float confidence = data[4];
        if (confidence >= CONFIDENCE_THRESHOLD) {

            float* classes_scores = data + 5;
            cv::Mat scores(1, className.size(), CV_32FC1, classes_scores);
            cv::Point class_id;
            double max_class_score;
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
            if (max_class_score > SCORE_THRESHOLD) {

                confidences.push_back(confidence);

                class_ids.push_back(class_id.x);

                float x = data[0];
                float y = data[1];
                float w = data[2];
                float h = data[3];
                int left = int((x - 0.5 * w) * x_factor);
                int top = int((y - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);
                boxes.push_back(cv::Rect(left, top, width, height));
            }

        }

        data += 85;

    }

    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, nms_result);
    for (int i = 0; i < nms_result.size(); i++) {
        int idx = nms_result[i];
        Detection result;
        result.class_id = class_ids[idx];
        result.confidence = confidences[idx];
        result.box = boxes[idx];
        output.push_back(result);
    }
}

char* ConvertFromUnicode(const WCHAR* pszWchar)
{
	int		nLen;
	char* pszChar;

	//char�ɕK�v�ȕ������̎擾
	nLen = ::WideCharToMultiByte(CP_THREAD_ACP, 0, pszWchar, -1, NULL, 0, NULL, NULL);
	pszChar = new char[nLen];
	if (pszChar == NULL)
		return	NULL;

	//�ϊ�
	nLen = ::WideCharToMultiByte(CP_THREAD_ACP, 0, pszWchar, wcslen(pszWchar) + 1, pszChar, nLen, NULL, NULL);
	if (nLen)
		return	pszChar;

	delete	pszChar;

	return	NULL;
}


void Evaluation_image(std::vector<WCHAR> fileName)
{
	FILE* fp;
	errno_t csv_error;
	string sourceFileName;
	char FileName[256];
	char outputCsvName[256] = "./result/result.txt";
	char outputImgName[256] = "./result_img/result.png";

    std::vector<std::string> class_list = load_class_list();
	std::vector<std::string> detected_class_list;
	std::vector<float> confidence_list;

	sourceFileName = ConvertFromUnicode(&fileName[0]);
	sprintf_s(FileName, 256, "./image/%s", sourceFileName);
	cv::Mat img = cv::imread(FileName);
	if (img.empty()) {
		printf("�摜���ǂݍ��߂܂���ł���\n");
		return;
	}
    
    //bool is_cuda = argc > 1 && strcmp(argv[1], "cuda") == 0;
    bool is_cuda = false;

    cv::dnn::Net net;
    load_net(net, is_cuda);

	std::vector<Detection> output;
	detect(img, net, output, class_list);

	int detections = output.size();

	for (int i = 0; i < detections; ++i)
	{

		auto detection = output[i];
		auto box = detection.box;
		auto classId = detection.class_id;
		const auto color = colors[classId % colors.size()];

		auto confidence = detection.confidence;

		cv::rectangle(img, box, color, 3);

		cv::rectangle(img, cv::Point(box.x, box.y - 20), cv::Point(box.x + box.width, box.y), color, cv::FILLED);
		cv::putText(img, class_list[classId].c_str(), cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));

		//std::cout << classId << std::endl;
		std::cout <<"class: "<< class_list[classId] << ", classID: " << classId << ", confidence: " << confidence << std::endl;
		detected_class_list.push_back(class_list[classId]);
		confidence_list.push_back(confidence);
	}

	//sprintf_s(outputCsvName, 256, "./result/result-%s.txt", sourceFileName);	//�o��CSV�t�@�C����
	
	if ((csv_error = fopen_s(&fp, outputCsvName, "w")) != 0) { // GPS_sakurai (3) : GPS���g��������pCSV�t�@�C����
		printf("hantei file open error!!\n");
		exit(EXIT_FAILURE);	/* (3)�G���[�̏ꍇ�͒ʏ�A�ُ�I������*/
	}

	for (int i = 0; i < detected_class_list.size(); i++) {

		if (i == detected_class_list.size() - 1) {
			std::fprintf(fp, "%s\n", detected_class_list[i]);
		}
		else {
			std::fprintf(fp, "%s, ", detected_class_list[i]);
		}
	}

	for (int i = 0; i < confidence_list.size(); i++) {

		if (i == confidence_list.size() - 1) {
			std::fprintf(fp, "%lf\n", confidence_list[i]);
		}
		else {
			std::fprintf(fp, "%lf, ", confidence_list[i]);
		}
	}
	//std::fprintf(fp, "%d,%lf,%d,%lf", normal, Pnormal, abnormal, Pabnormal);
	std::fclose(fp);

	//sprintf_s(outputImgName, 256, "./result_img/result-%s", sourceFileName);	//�o��CSV�t�@�C����
	cv::imwrite(outputImgName, img);
	cv::imshow("output", img);
	cv::waitKey(1);

	return;
}


// �G���[�̕\��
static void ShowError(LPCTSTR msg)
{
	DWORD errcode = GetLastError();
	std::wprintf(_T("%s errorcode: %lx\r\n"), msg, errcode);
}

// �L�[���͂̃`�F�b�N
static inline bool CheckQuitKey()
{
	return _kbhit() && (_getch() == 'q');
}

// ���C���G���g��
int main(int argc, char** argv)
{
	// �R���\�[���o�͂���{��\��
	setlocale(LC_ALL, "");

	// �I�v�V���������̒l��ێ�����
	LPCTSTR pDir = _T("./image");
	size_t bufsiz = 0;
	int waittime = 0;
	bool hasError = false;


	//if (argc == 2) {
	//	sprintf_s(modelBinary, 256, "yolov2-tiny_yamamoto_%s.weights", argv[1]);
	//}
	// �����̉��
	if (argc > 2) {
		_TCHAR** pArg = (_TCHAR**)argv[2];
		while (*pArg) {
			if (_tcsicmp(_T("/b"), *pArg) == 0) {
				// �o�b�t�@�T�C�Y
				pArg++;
				if (*pArg) {
					bufsiz = _ttol(*pArg);
				}

			}
			else if (_tcsicmp(_T("/w"), *pArg) == 0) {
				// �E�F�C�g����
				pArg++;
				if (*pArg) {
					waittime = _ttoi(*pArg);
				}

			}
			else if (**pArg != '/') {
				// �Ď���f�B���N�g��
				pDir = *pArg;
				break;

			}
			else {
				_ftprintf(stderr, _T("�s���Ȉ���: %s\r\n"), *pArg);
				hasError = true;
			}

			pArg++;
		}

	}

	if (hasError) {
		return 2;
	}

	if (bufsiz <= 0) {
		bufsiz = 1024 * 8;
	}
	if (waittime <= 0) {
		waittime = 0;
	}

	std::wprintf(_T("�Ď���: %s\r\n�o�b�t�@�T�C�Y: %ld\r\n���[�v���̑҂�����: %ld\r\n"),
		pDir, bufsiz, waittime);


	// �Ώۂ̃f�B���N�g�����Ď��p�ɃI�[�v������.
	// ���L�f�B���N�g���g�p�A�Ώۃt�H���_���폜��
	// �񓯊�I/O�g�p
	HANDLE hDir = CreateFile(
		pDir, // �Ď���
		FILE_LIST_DIRECTORY,
		FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
		NULL,
		OPEN_EXISTING,
		FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OVERLAPPED, // ReadDirectoryChangesW�p
		NULL
	);
	if (hDir == INVALID_HANDLE_VALUE) {
		ShowError(_T("CreateFile�ł̎��s"));
		return 1;
	}

	// �Ď����� (FindFirstChangeNotification�Ɠ���)
	DWORD filter =
		FILE_NOTIFY_CHANGE_FILE_NAME |  // �t�@�C�����̕ύX
		FILE_NOTIFY_CHANGE_DIR_NAME |  // �f�B���N�g�����̕ύX
		FILE_NOTIFY_CHANGE_ATTRIBUTES |  // �����̕ύX
		FILE_NOTIFY_CHANGE_SIZE |  // �T�C�Y�̕ύX
		FILE_NOTIFY_CHANGE_LAST_WRITE;    // �ŏI�������ݓ����̕ύX

	// �ύX���ꂽ�t�@�C���̃��X�g���L�^���邽�߂̃o�b�t�@.
	// �ŏ���ReadDirectoryChangesW�̒ʒm���玟��ReadDirectoryChangesW�܂ł�
	// �ԂɕύX���ꂽ�t�@�C���̏����i�[�ł��邾���̃T�C�Y���K�v.
	// �o�b�t�@�I�[�o�[�Ƃ��Ă��t�@�C���ɕύX�������������Ƃ͊��m�ł��邪�A
	// �Ȃɂ��ύX���ꂽ���͒ʒm�ł��Ȃ��B
	std::vector<unsigned char> buf(bufsiz);
	void* pBuf = &buf[0];

	// �񓯊�I/O�̊����ҋ@�p, �蓮���Z�b�g���[�h�B
	// �ύX�ʒm�̃C�x���g����ƃL�����Z�������̃C�x���g�����
	// 2�̃C�x���g�\�[�X�����邽�߃C�x���g�̗��ꂪ�\�z�ł���
	// �������Z�b�g�C�x���g�ɂ���̂͊댯�B
	HANDLE hEvent = CreateEvent(NULL, TRUE, FALSE, NULL);

	// �ύX�ʒm���󂯎��Â��郋�[�v
	for (;;) {
		// Sleep���g���A�C�x���g�n���h�����O���x���P�[�X���G�~�����[�g����.
		// ���[�v2��ڈȍ~�Ȃ��Sleep�\�����Ƀt�@�C����ύX���Ă��A
		// �ύX���ǐՂ���Ă��邱�Ƃ��m�F�ł���.
		// �܂��o�b�t�@�T�C�Y�𒴂���قǂɂ�������̃t�@�C����ύX�����
		// �o�b�t�@�I�[�o�[���m�F�ł���B
		const int mx = waittime * 10;
		for (int idx = 0; idx < mx; idx++) {
			std::wprintf(_T("sleep... %d/%d \r"), idx + 1, mx);
			Sleep(100);
		}
		std::wprintf(_T("\r\nstart.\r\n"));

		// �C�x���g�̎蓮���Z�b�g
		ResetEvent(hEvent);

		// �񓯊�I/O
		// ���[�v���ł̂ݎg�p�E�ҋ@����̂ŁA����(�X�^�b�N)�ɒu���Ă����S.
		OVERLAPPED olp = { 0 };
		olp.hEvent = hEvent;

		// �ύX���Ď�����.
		// ����Ăяo�����ɃV�X�e�����w��T�C�Y�Ńo�b�t�@���m�ۂ��A�����ɕύX���L�^����.
		// �����ʒm����V�X�e���͕ύX��ǐՂ��Ă���A�㑱��ReadDirectoryChangeW��
		// �Ăяo���ŁA�O��ʒm�ォ��̕ύX���܂Ƃ߂Ď󂯎�邱�Ƃ��ł���.
		// �o�b�t�@�����ӂꂽ�ꍇ�̓T�C�Y0�ŉ������Ԃ����.
		if (!ReadDirectoryChangesW(
			hDir,   // �Ώۃf�B���N�g��
			pBuf,   // �ʒm���i�[����o�b�t�@
			bufsiz, // �o�b�t�@�T�C�Y
			TRUE,   // �T�u�f�B���N�g����Ώۂɂ��邩?
			filter, // �ύX�ʒm���󂯎��t�B���^
			NULL,   // (���ʃT�C�Y, �񓯊��Ȃ̂Ŗ��g�p)
			&olp,   // �񓯊�I/O�o�b�t�@
			NULL    // (�������[�`��, ���g�p)
		)) {
			// �J�n�ł��Ȃ������ꍇ�̃G���[
			ShowError(_T("ReadDirectoryChangesW�ł̎��s"));
			break;
		}

		// �����ҋ@���[�v (q�L�[�œr���I��)
		bool quit;
		while (!(quit = CheckQuitKey())) {
			// �ύX�ʒm�܂�
			DWORD waitResult = WaitForSingleObject(hEvent, 500); // 0.5�b�҂�
			if (waitResult != WAIT_TIMEOUT) {
				// �ύX�ʒm���������ꍇ (�C�x���g���V�O�i����ԂɂȂ����ꍇ)
				break;
			}
			// �҂��󂯕\��
			std::wprintf(_T("."));
		}
		std::wprintf(_T("\r\n"));

		if (quit) {
			// �r���I������Ȃ�񓯊�I/O�����~���A
			// Overlapped�\���̂��V�X�e�����g��Ȃ��Ȃ�܂őҋ@����K�v������.
			CancelIo(hDir);
			WaitForSingleObject(hEvent, INFINITE);
			break;
		}

		// �񓯊�I/O�̌��ʂ��擾����.
		DWORD retsize = 0;
		if (!GetOverlappedResult(hDir, &olp, &retsize, FALSE)) {
			// ���ʎ擾�Ɏ��s�����ꍇ
			ShowError(_T("GetOverlappedResult�ł̎��s"));
			break;
		}

		// �ύX�ʒm���R���\�[���Ƀ_���v����.
		std::wprintf(_T("returned size=%ld\r\n"), retsize);

		if (retsize == 0) {
			// �ԋp�T�C�Y�A0�Ȃ�΃o�b�t�@�I�[�o�[������
			std::wprintf(_T("buffer overflow!!\r\n"));

		}
		else {
			// �ŏ��̃G���g���Ɉʒu�t����
			FILE_NOTIFY_INFORMATION* pData =
				reinterpret_cast<FILE_NOTIFY_INFORMATION*>(pBuf);

			// �G���g���̖����܂ŌJ��Ԃ�
			for (;;) {
				// �A�N�V�����^�C�v���Ǖ����ɕϊ�
				TCHAR pActionMsg[] = _T("Others");
				//TCHAR* pActionMsg;
				//wprintf_s(_T("UNKNOWN"), pActionMsg);

				switch (pData->Action) {
				case FILE_ACTION_ADDED:
					::_tcscpy_s(pActionMsg, 256 * sizeof(TCHAR), _T("Added"));
					//pActionMsg[] =  _T("Added") ;
					//wprintf_s(_T("Added"), pActionMsg);
					break;
				default:
					break;
					//case FILE_ACTION_REMOVED:
					//	//pActionMsg = _T("Removed");
					//	wprintf_s(_T("Removed"), pActionMsg);
					//	break;
					//case FILE_ACTION_MODIFIED:
					//	//pActionMsg = _T("Modified");
					//	wprintf_s(_T("Modified"), pActionMsg);
					//	break;
					//case FILE_ACTION_RENAMED_OLD_NAME:
					//	//pActionMsg = _T("Rename Old");
					//	wprintf_s(_T("Rename Old"), pActionMsg);
					//	break;
					//case FILE_ACTION_RENAMED_NEW_NAME:
					//	//pActionMsg = _T("Rename New");
					//	wprintf_s(_T("Rename New"), pActionMsg);
					//	break;
				}

				// �t�@�C�����̓k���I�[����Ă��Ȃ��̂�
				// ��������I�[�����Ă���.
				DWORD lenBytes = pData->FileNameLength; // �������ł͂Ȃ�, �o�C�g��
				std::vector<WCHAR> fileName(lenBytes / sizeof(WCHAR) + 1); // �k���I�[�p��+1
				std::memcpy(&fileName[0], pData->FileName, lenBytes);

				// �A�N�V�����ƑΏۃt�@�C����\��.
				// (�t�@�C�����͎w��f�B���N�g������̑��΃p�X�Œʒm�����.)
				std::wprintf(_T("[%s]<%s>\r\n"), pActionMsg, &fileName[0]);

				if (pData->Action == FILE_ACTION_ADDED)	Evaluation_image(fileName);	//Evaluation_video(fileName)

				if (pData->NextEntryOffset == 0) {
					// ���̃G���g���͖���
					break;
				}
				// ���̃G���g���̈ʒu�܂ňړ�����. (���݃A�h���X����̑��΃o�C�g��)
				pData = reinterpret_cast<FILE_NOTIFY_INFORMATION*>(
					reinterpret_cast<unsigned char*>(pData) + pData->NextEntryOffset);
			}
		}
	}

	// �n���h���̉��
	CloseHandle(hEvent);
	CloseHandle(hDir);

	return 0;
}