#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    cv::VideoCapture cap("../videos/seame.mp4");

    if (!cap.isOpened()) {
        std::cout << "Could not open or find the video!" << std::endl;
        return -1;
    }

    cv::Mat frame, gray, edges;

    while (true) {
        cap >> frame;

        if (frame.empty()) {
            break;
        }

        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        cv::Canny(gray, edges, 100, 200);

        cv::imshow("Edge Detection", edges);

        if (cv::waitKey(30) == 'q') {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
