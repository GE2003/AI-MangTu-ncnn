

#include <android/asset_manager_jni.h>
#include <android/native_window_jni.h>
#include <android/native_window.h>
#include <android/log.h>
#include <numeric>
#include <jni.h>
#include <string>
#include <vector>
#include <cmath>
#include <utility>

// Platform and Benchmark headers
#include <platform.h>
#include <benchmark.h>
#include <future> // 用于异步调用
#include "yolo.h"
#include "hand.h"
#include "speak_lib.h"
#include "ndkcamera.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

static int draw_unsupported(cv::Mat& rgb) {
    const char text[] = "unsupported";

    int baseLine = 0;
    cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 1.0, 1, &baseLine);

    int y = (rgb.rows - label_size.height) / 2;
    int x = (rgb.cols - label_size.width) / 2;

    cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)), cv::Scalar(255, 255, 255), -1);
    cv::putText(rgb, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 0));

    return 0;
}

static float fps_history[10] = {0.f}; // Move outside to avoid reallocation
static int draw_fps(cv::Mat& rgb) {
    // resolve moving average
    float avg_fps = 0.f;
    static double t0 = 0.f;

    double t1 = ncnn::get_current_time();
    if (t0 == 0.f) {
        t0 = t1;
        return 0;
    }

    float fps = 1000.f / (t1 - t0);
    t0 = t1;
    // Move the FPS history update logic outside
    std::rotate(std::begin(fps_history), std::begin(fps_history) + 1, std::end(fps_history));
    fps_history[0] = fps;
    if (fps_history[9] == 0.f) {
        return 0;
    }
    avg_fps = std::accumulate(std::begin(fps_history), std::end(fps_history), 0.f) / 10.f;

    char text[32];
    sprintf(text, "FPS=%.2f", avg_fps);

    int baseLine = 0;
    cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

    int y = 0;
    int x = rgb.cols - label_size.width;

    cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)), cv::Scalar(255, 255, 255), -1);
    cv::putText(rgb, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));

    return 0;
}

// 创建2个对象
static Yolo* g_yolo = nullptr;
static ncnn::Mutex lock;

static Hand* g_hand = nullptr;
static ncnn::Mutex lock2;

static std::vector<Object> objectsA;
static std::vector<PalmObject> objectsB;

class MyNdkCamera : public NdkCameraWindow {
public:
    virtual void on_image_render(cv::Mat& rgb) const;
};

//void MyNdkCamera::on_image_render(cv::Mat& rgb) const {
//    {
//        ncnn::MutexLockGuard g(lock);
//        if (g_yolo) {
//            g_yolo->detect(rgb, objectsA);
//            g_yolo->draw(rgb, objectsA);
//        } else {
//            draw_unsupported(rgb);
//        }
//    }
//
//    {
//        ncnn::MutexLockGuard g(lock2);
//        if (g_hand) {
//            g_hand->detect(rgb, objectsB);
//            g_hand->draw(rgb, objectsB);
//        } else {
//            draw_unsupported(rgb);
//        }
//    }
//
//    draw_fps(rgb);
//}

void MyNdkCamera::on_image_render(cv::Mat& rgb) const {
    // 检测手掌（异步）
    std::future<void> hand_detection = std::async(std::launch::async, [&]() {
        ncnn::MutexLockGuard g(lock);
        if (g_hand) {
            g_hand->detect(rgb, objectsB);
        }
    });

    // 检测物体
    {
        ncnn::MutexLockGuard g(lock);
        if (g_yolo) {
            g_yolo->detect(rgb, objectsA);
            g_yolo->draw(rgb, objectsA);
        } else {
            draw_unsupported(rgb);
        }
    }

    // 等待手掌检测完成后绘制
    hand_detection.wait();
    if (g_hand) {
        ncnn::MutexLockGuard g(lock);
        g_hand->draw(rgb, objectsB);
    } else {
        draw_unsupported(rgb);
    }

    draw_fps(rgb);
}

static MyNdkCamera* g_camera = nullptr;

extern "C" {

JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved) {
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "JNI_OnLoad");
    g_camera = new MyNdkCamera;
    return JNI_VERSION_1_4;
}

JNIEXPORT void JNI_OnUnload(JavaVM* vm, void* reserved) {
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "JNI_OnUnload");

    {
        ncnn::MutexLockGuard g(lock);
        delete g_yolo;
        g_yolo = nullptr;
    }

    {
        ncnn::MutexLockGuard g(lock2);
        delete g_hand;
        g_hand = nullptr;
    }

    delete g_camera;
    g_camera = nullptr;
}

// public native boolean loadModel(AssetManager mgr, int modelid, int cpugpu);
JNIEXPORT jboolean JNICALL Java_com_tencent_yolov8ncnn_Yolov8Ncnn_loadModel(JNIEnv* env, jobject thiz, jobject assetManager, jint modelid, jint cpugpu) {
    if (modelid < 0 || modelid > 6 || cpugpu < 0 || cpugpu > 1) {
        return JNI_FALSE;
    }

    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "loadModel %p", mgr);

    const char* modeltypes[] = {
            // "yolov8n",
            // "yolov8s",
            "model"
    };

    const int target_sizes[] = {
            320,
            // 320,
            // 320,
    };

    const float mean_vals[][3] = {
            {103.53f, 116.28f, 123.675f},
            // {103.53f, 116.28f, 123.675f},
            // {103.53f, 116.28f, 123.675f},
    };

    const float norm_vals[][3] = {
            { 1 / 255.f, 1 / 255.f, 1 / 255.f },
            // { 1 / 255.f, 1 / 255.f, 1 / 255.f },
            // { 1 / 255.f, 1 / 255.f, 1 / 255.f },
    };

    const char* modeltype = modeltypes[(int)modelid];
    int target_size = target_sizes[(int)modelid];
    bool use_gpu = (int)cpugpu == 1;

    // reload
    {
        ncnn::MutexLockGuard g(lock);

        if (use_gpu && ncnn::get_gpu_count() == 0) {
            // no gpu
            delete g_yolo;
            g_yolo = nullptr;
        } else {
            if (!g_yolo) {
                g_yolo = new Yolo;
            }
            g_yolo->load(mgr, modeltype, target_size, mean_vals[(int)modelid], norm_vals[(int)modelid], use_gpu);
        }
    }

    return JNI_TRUE;
}
JNIEXPORT jboolean JNICALL Java_com_tencent_yolov8ncnn_Yolov8Ncnn_loadModel2(JNIEnv* env, jobject thiz, jobject assetManager, jint modelid, jint cpugpu)
{
    if (modelid < 0 || modelid > 6 || cpugpu < 0 || cpugpu > 1)
    {
        return JNI_FALSE;
    }

    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "loadModel %p", mgr);

    const char* modeltypes[] =
    {
        "palm-full"
    };

    const int target_sizes[] =
    {
        192,
        192
    };

    const float mean_vals[][3] =
    {
        {0.f,0.f,0.f},
        {0.f,0.f,0.f},
    };

    const float norm_vals[][3] =
    {
        {1 / 255.f, 1 / 255.f, 1 / 255.f},
        {1 / 255.f, 1 / 255.f, 1 / 255.f},
    };

    const char* modeltype = modeltypes[(int)modelid];
    int target_size = target_sizes[(int)modelid];
    bool use_gpu = (int)cpugpu == 1;

    // reload
    {
        ncnn::MutexLockGuard g(lock2);

        if (use_gpu && ncnn::get_gpu_count() == 0)
        {
            // no gpu
            delete g_hand;
            g_hand = 0;
        }
        else
        {
            if (!g_hand)
                g_hand = new Hand;
            g_hand->load(mgr, modeltype, target_size, mean_vals[(int)modelid], norm_vals[(int)modelid], use_gpu);
        }
    }

    return JNI_TRUE;
}

//JNIEXPORT jboolean JNICALL Java_com_tencent_yolov8ncnn_Yolov8Ncnn_loadModel2(JNIEnv* env, jobject thiz, jobject assetManager, jint modelid, jint cpugpu) {
//    if (modelid < 0 || modelid > 6 || cpugpu < 0 || cpugpu > 1) {
//        return JNI_FALSE;
//    }
//
//    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);
//    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "loadModel %p", mgr);
//
//    const char* modeltypes[] = {"palm-full"};
//    const int target_sizes[] = {192, 192};
//    const float mean_vals[][3] = {{0.f, 0.f, 0.f}, {0.f, 0.f, 0.f}};
//    const float norm_vals[][3] = {{1 / 255.f, 1 / 255.f, 1 / 255.f}, {1 / 255.f, 1 / 255.f, 1 / 255.f}};
//
//    const char* modeltype = modeltypes[(int)modelid];
//    int target_size = target_sizes[(int)modelid];
//    bool use_gpu = (int)cpugpu == 1;
//
//    // reload
//    {
//        ncnn::MutexLockGuard g(lock2);
//        if (use_gpu && ncnn::get_gpu_count() == 0) {
//            // no gpu
//            delete g_hand;
//            g_hand = nullptr;
//        } else {
//            if (!g_hand) {
//                g_hand = new Hand;
//            }
//            g_hand->load(mgr, modeltype, target_size, mean_vals[(int)modelid], norm_vals[(int)modelid], use_gpu);
//        }
//    }
//
//    return JNI_TRUE;
//}

// 判断点是否在矩形内
bool pointInsideRect(const cv::Point2f& point, const cv::Rect_<float>& rect) {
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "Point Coordinates: (%f, %f)", point.x, point.y);
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "rect Coordinates: (%f, %f)", rect.x, rect.y);
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "rect width: (%f, %f)", rect.width, rect.height);
    return (point.x  >= rect.x && point.x  <= rect.x + rect.width  && point.y  >= rect.y && point.y  <= rect.y + rect.height );
}

// 获取检测框的标签值
const char* getLabelForRect(int label) {
    static const char* class_names[] = {
            "QTtoubeibu", "QTchibang", "QTshenti", "CYfuyanzu", "CYfubu", "CYchibang",
            "HDyanchujiao", "HDchibang", "WZkouqi", "WZfuyan", "WZzu", "WZchibang",
            "WZshenti", "MYtoubu", "MYxiongbu", "MYfubu", "MYchujiao", "MYzu",
            "KDtoubu", "KDweiba", "PXyan", "PXqianzi", "PXfubu", "PXfuzhi",
            "QWqvgan", "QWqianzhi", "QWhouzhi", "TLtoubu", "TLfubu", "TLqianzu",
            "TLzhongzu", "TLhouzu", "WGtou", "WGke", "WGweiba", "WGzu",
            "XSqvgan", "XSqianzu", "XSzhongzu", "XShouzu", "XSchujiao", "XSweisi",
            "YHCtou", "YHCshenti", "YHCguang", "YHCzu", "ZZshenti", "ZZzu",
            "BSLtoubu", "BSLqvgan", "BSLweibu", "BHtou", "BHshenti", "BHweiba",
            "BHsizhi", "EYtou", "EYshenti", "EYsizhi", "QEhui", "QEtou",
            "QEshenti", "QEchi", "QEzu", "SYshenti", "SYwei", "SYqi",
            "MFtou", "MFchibang", "MFxiongbu", "MFfubu", "WNtou", "WNke",
            "WNshenti"
    };



    if (label >= 0 && label < sizeof(class_names) / sizeof(class_names[0])) {
        return class_names[label];
    }
    return "Unknown";
}
bool flag = true;
JNIEXPORT jstring JNICALL Java_com_tencent_yolov8ncnn_Yolov8Ncnn_checkPointIfInDetectBox(JNIEnv *env, jobject thiz) {
    std::vector<CachedPosition> cached_positions = getCachedPositions();
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "Fetched %zu cached positions", cached_positions.size());
    bool fingerInAnyBox = false;

//    for (const auto& handObj : objectsB) {
//        fingerInAnyBox = false;
//        std::string labelString = "手指不在检测框";  // 默认返回值
//
//        // 查询每个缓存的位置，判断手指是否在里面
//        for (const auto& cached_pos : cached_positions) {
//            for (const auto& skeleton : handObj.skeleton) {
//                if (pointInsideRect(skeleton, cached_pos.rect)) {
//                 //   __android_log_print(ANDROID_LOG_DEBUG, "ncnn-cache", "Finger landmark coordinates: (%f, %f)", landmark.x, landmark.y);
//                    labelString = getLabelForRect(cached_pos.label);  // 获取匹配标签
//                    fingerInAnyBox = true;
//                    break;
//                }
//            }
//        }
//        if (fingerInAnyBox) {
//            return env->NewStringUTF(labelString.c_str());  // 返回检测到的标签
//        }
//    }
    fingerInAnyBox = false;
    std::string labelString = "手指不在检测框";  // 默认返回值
    for (const auto &handObj: objectsB) {
        for (const auto &skeleton: handObj.skeleton) {
            for (const auto &yoloObj: objectsA) {
                const Object &real_time_obj = yoloObj;
                //置信度大于70%
                if (real_time_obj.prob>0.45){
                    if (pointInsideRect(skeleton, real_time_obj.rect)) {
                        // Finger landmark is inside the detection box, get the label and return to Java layer
                        __android_log_print(ANDROID_LOG_DEBUG, "ncnn",
                                            "Finger landmark coordinates: (%f, %f)",
                                            skeleton.x, skeleton.y);
                        const char *label = getLabelForRect(yoloObj.label);
                        // __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "Detected label: %s", label);
                        fingerInAnyBox = true;
                        return env->NewStringUTF(label);
                        break;
                    }
                }
            }
//            for (const auto &cached_pos: cached_positions) {
//                    if (pointInsideRect(skeleton, cached_pos.rect)&&!fingerInAnyBox) {
//                        //   __android_log_print(ANDROID_LOG_DEBUG, "ncnn-cache", "Finger landmark coordinates: (%f, %f)", landmark.x, landmark.y);
//                        labelString = getLabelForRect(cached_pos.label);  // 获取匹配标签
//                        fingerInAnyBox = true;
//                        return env->NewStringUTF(labelString.c_str());  // 返回检测到的标签
//                        break;
//                    }
//                }
//
//                fingerInAnyBox= false;
        }
    }


        // 查询每个缓存的位置，判断手指是否在里面

//            for (const auto& handObj : objectsB) {
//                for (const auto &cached_pos: cached_positions) {
//                    for (const auto &skeleton: handObj.skeleton) {
//                        if (pointInsideRect(skeleton, cached_pos.rect)) {
//                            //   __android_log_print(ANDROID_LOG_DEBUG, "ncnn-cache", "Finger landmark coordinates: (%f, %f)", landmark.x, landmark.y);
//                            labelString = getLabelForRect(cached_pos.label);  // 获取匹配标签
//                            fingerInAnyBox = true;
//                            return env->NewStringUTF(labelString.c_str());  // 返回检测到的标签
//                            break;
//                        }
//                    }
//                }
//            }
//    for (const auto &cached_pos: cached_positions) {
//    for (const auto& handObj : objectsB) {
//            for (const auto &skeleton: handObj.skeleton) {
//                if (pointInsideRect(skeleton, cached_pos.rect)) {
//                    //   __android_log_print(ANDROID_LOG_DEBUG, "ncnn-cache", "Finger landmark coordinates: (%f, %f)", landmark.x, landmark.y);
//                    labelString = getLabelForRect(cached_pos.label);  // 获取匹配标签
//                    fingerInAnyBox = true;
//                    return env->NewStringUTF(labelString.c_str());  // 返回检测到的标签
//                    break;
//                }
//            }
//        }
//    }


//        if (fingerInAnyBox) {
//            return env->NewStringUTF(labelString.c_str());  // 返回检测到的标签
//        }

    std::string labelres = "";  // 默认返回值
    if (objectsA.size()<1&&objectsB.size()==0){
        labelres="0";
        flag= true;
    } else if (objectsA.size()>=3&&flag){
        labelres="1";
        flag= false;
    }

    return env->NewStringUTF(labelres.c_str());
}


// 检查手指是否在检测框内
//JNIEXPORT jstring JNICALL Java_com_tencent_yolov8ncnn_Yolov8Ncnn_checkPointIfInDetectBox(JNIEnv *env, jobject thiz) {
//    std::vector<CachedPosition> cached_positions = getCachedPositions();
//    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "Fetched %zu cached positions", cached_positions.size());
//
//    for (const auto& handObj : objectsB) {
//        bool fingerInAnyBox = false;
//        std::string labelString = "手指不在检测框";  // 默认返回值
//        for (const auto& landmark : handObj.skeleton) {
//        // 查询每个缓存的位置，判断手指是否在里面
//
//
//        for (const auto& cached_pos : cached_positions) {
//                if (pointInsideRect(landmark, cached_pos.rect)) {
//                    __android_log_print(ANDROID_LOG_DEBUG, "ncnn-cache", "Finger landmark coordinates: (%f, %f)", landmark.x, landmark.y);
//                    labelString = getLabelForRect(cached_pos.label);  // 获取匹配标签
//                    fingerInAnyBox = true;
//                    break;
//                }
//            }
//            if (fingerInAnyBox) {
//                return env->NewStringUTF(labelString.c_str());  // 返回检测到的标签
//            }
//
//
//            for (const auto &yoloObj: objectsA) {
//                const Object &real_time_obj = yoloObj;
//                // Check if finger landmark is inside the detection box for both cached and real-time coordinates
////                __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "方框坐标: (%f, %f)",
////                                    real_time_obj.rect.x, real_time_obj.rect.y);
////                __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "手指坐标: (%f, %f)",
////                                    landmark.x, landmark.y);
////                __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "方框坐标标签 %s",
////                                    getLabelForRect(yoloObj.label));
//                //置信度大于70%
//                    if (pointInsideRect(landmark, real_time_obj.rect)) {
//                        // Finger landmark is inside the detection box, get the label and return to Java layer
////                        __android_log_print(ANDROID_LOG_DEBUG, "ncnn",
////                                            "Finger landmark coordinates: (%f, %f)",
////                                            landmark.x, landmark.y);
//
//                        const char *label = getLabelForRect(yoloObj.label);
//                        // __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "Detected label: %s", label);
//
//                        return env->NewStringUTF(label);
//                    }
//                }
//
//        }
//
//
//    }
//
//    // 如果所有手指均不在任何检测框内
//    return env->NewStringUTF("手指不在检测框");
//}


//JNIEXPORT jstring JNICALL Java_com_tencent_yolov8ncnn_Yolov8Ncnn_checkPointIfInDetectBox(JNIEnv* env, jobject thiz) {
//    // Iterate over hand objects and YOLO objects
//   //先识别出框，再根据框判断手
//
//
//    for (const auto &handObj: objectsB) {
//        for (const auto &landmark: handObj.skeleton) {
//            for (const auto &yoloObj: objectsA) {
//                // Get object position for both cached and real-time coordinates
//                std::vector<CachedPosition> cached_positions = getCachedPositions();
//                const Object &real_time_obj = yoloObj;
//                // Check if finger landmark is inside the detection box for both cached and real-time coordinates
//                __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "方框坐标: (%f, %f)",
//                                    real_time_obj.rect.x, real_time_obj.rect.y);
//                __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "手指坐标: (%f, %f)",
//                                    landmark.x, landmark.y);
//                __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "方框坐标标签 %s",
//                                    getLabelForRect(yoloObj.label));
//                //置信度大于70%
//                if (real_time_obj.prob>0.4){
//                    if (pointInsideRect(landmark, real_time_obj.rect)) {
//                        // Finger landmark is inside the detection box, get the label and return to Java layer
//                        __android_log_print(ANDROID_LOG_DEBUG, "ncnn",
//                                            "Finger landmark coordinates: (%f, %f)",
//                                            landmark.x, landmark.y);
//
//                        const char *label = getLabelForRect(yoloObj.label);
//                        // __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "Detected label: %s", label);
//
//                        return env->NewStringUTF(label);
//                    }
//                }
//            }
//        }
//    }
//
//    // Finger landmark is not inside any detection box, return a default value
//    return env->NewStringUTF("手指不在检测框");
//}


// 打开摄像头
JNIEXPORT jboolean JNICALL Java_com_tencent_yolov8ncnn_Yolov8Ncnn_openCamera(JNIEnv* env, jobject thiz, jint facing) {
    if (facing < 0 || facing > 1)
        return JNI_FALSE;

    // 打印Android日志
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "openCamera %d", facing);
    g_camera->open((int)facing);
    return JNI_TRUE;
}

// 关闭摄像头
JNIEXPORT jboolean JNICALL Java_com_tencent_yolov8ncnn_Yolov8Ncnn_closeCamera(JNIEnv* env, jobject thiz) {
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "closeCamera");
    g_camera->close();
    return JNI_TRUE;
}

// 设置输出窗口
JNIEXPORT jboolean JNICALL Java_com_tencent_yolov8ncnn_Yolov8Ncnn_setOutputWindow(JNIEnv* env, jobject thiz, jobject surface) {
    ANativeWindow* win = ANativeWindow_fromSurface(env, surface);
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "setOutputWindow %p", win);
    g_camera->set_window(win);
    return JNI_TRUE;
}

} // extern "C"
