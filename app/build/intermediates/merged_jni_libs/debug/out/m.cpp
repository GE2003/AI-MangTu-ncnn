#include <cmath>
#include <utility>
#include <vector>
#include <string>
#include <iostream>
#include "yolo.h"
#include <opencv2/opencv.hpp>
// mediapipe环境还没配


std::map<int, std::string> label_mapping = {
    {0, "眼和触角"},
    {1, "头"},
    {2, "壳"},
    {3, "身体"}
};


double vector_2d_angle(std::pair<double, double> v1, std::pair<double, double> v2) {
    double v1_x = v1.first;
    double v1_y = v1.second;
    double v2_x = v2.first;
    double v2_y = v2.second;

    try {
        double angle_ = std::acos(
            (v1_x * v2_x + v1_y * v2_y) / (((v1_x * v1_x + v1_y * v1_y) * 0.5) * ((v2_x * v2_x + v2_y * v2_y) * 0.5)));
        angle_ = std::fmod(std::abs(angle_), 360.0);
        if (angle_ > 180.0) {
            angle_ = 65535.0;
        }
        return angle_;
    }
    catch (...) {
        return 65535.0;
    }
}


std::vector<double> hand_angle(std::vector<std::vector<int>> hand) {
    std::vector<double> angle_list;

    auto calculate_angle = [](std::pair<int, int> v1, std::pair<int, int> v2) {
        double v1_x = v1.first;
        double v1_y = v1.second;
        double v2_x = v2.first;
        double v2_y = v2.second;

        try {
            double angle_ = std::acos(
                (v1_x * v2_x + v1_y * v2_y) / (((v1_x * v1_x + v1_y * v1_y) * 0.5) * ((v2_x * v2_x + v2_y * v2_y) * 0.5)));
            angle_ = std::fmod(std::abs(angle_), 360.0);
            if (angle_ > 180.0) {
                angle_ = 65535.0;
            }
            return angle_;
        }
        catch (...) {
            return 65535.0;
        }
    };

    angle_list.push_back(calculate_angle({ hand[0][0] - hand[2][0], hand[0][1] - hand[2][1] }, { hand[3][0] - hand[4][0], hand[3][1] - hand[4][1] }));
    angle_list.push_back(calculate_angle({ hand[0][0] - hand[6][0], hand[0][1] - hand[6][1] }, { hand[7][0] - hand[8][0], hand[7][1] - hand[8][1] }));
    angle_list.push_back(calculate_angle({ hand[0][0] - hand[10][0], hand[0][1] - hand[10][1] }, { hand[11][0] - hand[12][0], hand[11][1] - hand[12][1] }));
    angle_list.push_back(calculate_angle({ hand[0][0] - hand[14][0], hand[0][1] - hand[14][1] }, { hand[15][0] - hand[16][0], hand[15][1] - hand[16][1] }));
    angle_list.push_back(calculate_angle({ hand[0][0] - hand[18][0], hand[0][1] - hand[18][1] }, { hand[19][0] - hand[20][0], hand[19][1] - hand[20][1] }));

    return angle_list;
}


std::string h_gesture(std::vector<double> angle_list) {
    double thr_angle = 65.0;
    double thr_angle_s = 49.0;
    std::string gesture_str;

    if (std::find(angle_list.begin(), angle_list.end(), 65535.0) == angle_list.end()) {
        if (angle_list[0] > 5 && angle_list[1] < thr_angle_s && angle_list[2] > thr_angle &&
            angle_list[3] > thr_angle && angle_list[4] > thr_angle) {
            gesture_str = "one";
        }
        else if (angle_list[0] < thr_angle_s && angle_list[1] < thr_angle_s && angle_list[2] < thr_angle_s &&
            angle_list[3] < thr_angle_s && angle_list[4] < thr_angle_s) {
            gesture_str = "five";
        }
        else {
            gesture_str = "other";
        }
    }

    return gesture_str;
}



void processDetectionResults(const cv::Mat& rgb, std::vector<Object>& objects, float prob_threshold, float nms_threshold, std::vector<std::tuple<float, float, float, float, int>>& detectionInfo) {
    int detectionResult = Yolo::detect(rgb, objects, prob_threshold, nms_threshold);
    if (detectionResult == 0) {
        for (const auto& obj : objects) {
            float x0 = obj.rect.x;
            float y0 = obj.rect.y;
            float x1 = obj.rect.x + obj.rect.width;
            float y1 = obj.rect.y + obj.rect.height;
            int label = obj.label;
            detectionInfo.push_back(std::make_tuple(x0, y0, x1, y1, label));
        }
    }
}


void speak(const std::string& text) {
    // 调用安卓语音
}


void checkIfPointInDetectionBox(int cx, int cy, const std::vector<std::tuple<float, float, float, float, int>>& detectionInfo, const std::map<int, std::string>& label_mapping) {
    for (const auto& detection : detectionInfo) {
        float x0 = std::get<0>(detection);
        float y0 = std::get<1>(detection);
        float x1 = std::get<2>(detection);
        float y1 = std::get<3>(detection);
        int label = std::get<4>(detection);
        if (cx > x0 && cx < x1 && cy > y0 && cy < y1) {
            speak(label_mapping[label]);
        }
    }
}







int main()
{
    mediapipe::Hands hands;
    mediapipe::Hands::Options options;
    options.set_static_image_mode(false);
    options.set_max_num_hands(1);
    options.set_min_detection_confidence(0.75);
    options.set_min_tracking_confidence(0.75);
    hands.Initialize(options);

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        return -1;
    }

    std::string previous_gesture;

    Yolo yolo;
    AAssetManager* mgr = nullptr;
    const char* modeltype = "ncnn";
    int target_size = 640;
    const float mean_vals[3] = { 0.75, 0.75, 0.75 };
    const float norm_vals[3] = { 0.75, 0.75, 0.75 };
    bool use_gpu = false;
    int result = yolo.load(mgr, modeltype, target_size, mean_vals, norm_vals, use_gpu);
    if (result != 0) {
        return -1;
    }

    while (true) {
        cv::Mat frame;
        cap >> frame;

        std::vector<Object> objects;
        float prob_threshold = 0.75;
        float nms_threshold = 0.75;
        int detect_result = yolo.detect(frame, objects, prob_threshold, nms_threshold);
        if (detect_result != 0) {
            continue;
        }

        int draw_result = yolo.draw(frame, objects);
        if (draw_result != 0) {
            continue;
        }

        std::vector<std::tuple<float, float, float, float, int>> detectionInfo;
        std::thread detectionThread(processDetectionResults, frame, std::ref(objects), prob_threshold, nms_threshold, std::ref(detectionInfo));
        detectionThread.join();

        cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
        mediapipe::Hands::Results results = hands.process(frame);
        cv::cvtColor(frame, frame, cv::COLOR_RGB2BGR);

        if (!results.multi_hand_landmarks.empty()) {
            for (const auto& hand_landmarks : results.multi_hand_landmarks) {
                for (int id = 0; id < hand_landmarks.landmark_size(); id++) {
                    const auto& lm = hand_landmarks.landmark(id);
                    if (id == 8) {
                        int h = frame.rows;
                        int w = frame.cols;
                        int cx = static_cast<int>(lm.x * w);
                        int cy = static_cast<int>(lm.y * h);
                        cv::circle(frame, cv::Point(cx, cy), 5, cv::Scalar(225, 0, 0), cv::FILLED);
                    }
                }

                std::vector<std::pair<int, int>> hand_local;

                for (int i = 0; i < 21; i++) {
                    int x = static_cast<int>(hand_landmarks.landmark(i).x * frame.cols);
                    int y = static_cast<int>(hand_landmarks.landmark(i).y * frame.rows);
                    hand_local.push_back(std::make_pair(x, y));
                }

                if (!hand_local.empty()) {
                    std::vector<double> angle_list = hand_angle(hand_local);
                    std::string gesture_str = h_gesture(angle_list);
                    if (gesture_str == "one" && previous_gesture != gesture_str) {
                        std::thread detectionThread(checkIfPointInDetectionBox, cx, cy, detectionInfo, label_mapping);
                        detectionThread.join();
                    }
                }
                previous_gesture = gesture_str;
            }
        }

        cv::imshow("MangTu", frame);

        if (cv::waitKey(1) & 0xFF == 'q') {
            cv::destroyAllWindows();
            break;
        }
    }

    cap.release();
    return 0;
}
