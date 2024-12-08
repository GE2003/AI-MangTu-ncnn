
#include "hand.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <deque>
#include "cpu.h"

// Calculate scale for anchor generation
static float calculate_scale(float min_scale, float max_scale, int stride_index, int num_strides) {
    if (num_strides == 1)
        return (min_scale + max_scale) * 0.5f;
    else
        return min_scale + (max_scale - min_scale) * 1.0f * stride_index / (num_strides - 1.0f);
}

// Generate anchors
static void generate_anchors(std::vector<Anchor>& anchors, const AnchorsParams& anchor_params) {
    int last_same_stride_layer = 0;
    for (int layer_id = 0; layer_id < anchor_params.strides.size(); layer_id = last_same_stride_layer) {
        std::vector<float> anchor_height, anchor_width, aspect_ratios, scales;

        while (last_same_stride_layer < static_cast<int>(anchor_params.strides.size()) &&
               anchor_params.strides[last_same_stride_layer] == anchor_params.strides[layer_id]) {

            float scale = calculate_scale(anchor_params.min_scale, anchor_params.max_scale, last_same_stride_layer, anchor_params.strides.size());
            for (float aspect_ratio : anchor_params.aspect_ratios) {
                aspect_ratios.push_back(aspect_ratio);
                scales.push_back(scale);
            }

            float scale_next = (last_same_stride_layer == static_cast<int>(anchor_params.strides.size()) - 1) ? 1.0f
                                                                                                              : calculate_scale(anchor_params.min_scale, anchor_params.max_scale, last_same_stride_layer + 1, anchor_params.strides.size());
            scales.push_back(std::sqrt(scale * scale_next));
            aspect_ratios.push_back(1.0);

            ++last_same_stride_layer;
        }

        for (size_t i = 0; i < aspect_ratios.size(); ++i) {
            float ratio_sqrts = std::sqrt(aspect_ratios[i]);
            anchor_height.push_back(scales[i] / ratio_sqrts);
            anchor_width.push_back(scales[i] * ratio_sqrts);
        }

        int feature_map_height = static_cast<int>(std::ceil(1.0f * anchor_params.input_size_height / anchor_params.strides[layer_id]));
        int feature_map_width = static_cast<int>(std::ceil(1.0f * anchor_params.input_size_width / anchor_params.strides[layer_id]));

        for (int y = 0; y < feature_map_height; ++y) {
            for (int x = 0; x < feature_map_width; ++x) {
                for (size_t anchor_id = 0; anchor_id < anchor_height.size(); ++anchor_id) {
                    Anchor new_anchor;
                    float x_center = (x + anchor_params.anchor_offset_x) * 1.0f / feature_map_width;
                    float y_center = (y + anchor_params.anchor_offset_y) * 1.0f / feature_map_height;
                    new_anchor.x_center = x_center;
                    new_anchor.y_center = y_center;
                    new_anchor.w = anchor_width[anchor_id];
                    new_anchor.h = anchor_height[anchor_id];
                    anchors.push_back(new_anchor);
                }
            }
        }
    }
}

// Create SSD anchors
static void create_ssd_anchors(int input_w, int input_h, std::vector<Anchor>& anchors) {
    AnchorsParams anchor_params;
    anchor_params.num_layers = 4;
    anchor_params.min_scale = 0.1484375;
    anchor_params.max_scale = 0.75;
    anchor_params.input_size_height = 192;
    anchor_params.input_size_width = 192;
    anchor_params.anchor_offset_x = 0.5f;
    anchor_params.anchor_offset_y = 0.5f;
    anchor_params.strides = {8, 16, 16, 16};
    anchor_params.aspect_ratios = {1.0};
    generate_anchors(anchors, anchor_params);
}

// Sigmoid function
static inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

// Decode bounds using anchors
static int decode_bounds(std::list<DetectRegion>& region_list, float score_thresh, int input_img_w, int input_img_h, float* scores_ptr, float* bboxes_ptr, std::vector<Anchor>& anchors) {
    int i = 0;
    for (const auto& anchor : anchors) {
        float score = sigmoid(scores_ptr[i]);
        if (score > score_thresh) {
            float* p = bboxes_ptr + (i * 18);

            float cx = p[0] / input_img_w + anchor.x_center;
            float cy = p[1] / input_img_h + anchor.y_center;
            float w = p[2] / input_img_w;
            float h = p[3] / input_img_h;

            cv::Point2f topleft = { cx - w * 0.5f, cy - h * 0.5f };
            cv::Point2f btmright = { cx + w * 0.5f, cy + h * 0.5f };

            DetectRegion region = { score, topleft, btmright, {} };

            for (int j = 0; j < 7; ++j) {
                float lx = p[4 + (2 * j) + 0];
                float ly = p[4 + (2 * j) + 1];
                lx = (lx + anchor.x_center * input_img_w) / static_cast<float>(input_img_w);
                ly = (ly + anchor.y_center * input_img_h) / static_cast<float>(input_img_h);
                region.landmarks[j] = { lx, ly };
            }

            region_list.push_back(region);
        }
        ++i;
    }
    return 0;
}

// Calculate Intersection over Union (IoU)
static float calc_intersection_over_union(const DetectRegion& region0, const DetectRegion& region1) {
    // 获取 region0 的边界坐标
    float sx0 = region0.topleft.x;
    float sy0 = region0.topleft.y;
    float ex0 = region0.btmright.x;
    float ey0 = region0.btmright.y;

    // 获取 region1 的边界坐标
    float sx1 = region1.topleft.x;
    float sy1 = region1.topleft.y;
    float ex1 = region1.btmright.x;
    float ey1 = region1.btmright.y;

    // 确定一个区域的最小和最大坐标
    float xmin0 = std::min(sx0, ex0);
    float ymin0 = std::min(sy0, ey0);
    float xmax0 = std::max(sx0, ex0);
    float ymax0 = std::max(sy0, ey0);

    float xmin1 = std::min(sx1, ex1);
    float ymin1 = std::min(sy1, ey1);
    float xmax1 = std::max(sx1, ex1);
    float ymax1 = std::max(sy1, ey1);

    // 计算每个区域的面积
    float area0 = (ymax0 - ymin0) * (xmax0 - xmin0);
    float area1 = (ymax1 - ymin1) * (xmax1 - xmin1);

    // 检查区域面积是否为零
    if (area0 <= 0 || area1 <= 0)
        return 0.0f;

    // 计算交叉区域的最小和最大坐标
    float intersect_xmin = std::max(xmin0, xmin1);
    float intersect_ymin = std::max(ymin0, ymin1);
    float intersect_xmax = std::min(xmax0, xmax1);
    float intersect_ymax = std::min(ymax0, ymax1);

    // 计算交叉区域的面积
    float intersect_area = std::max(intersect_ymax - intersect_ymin, 0.0f) *
                           std::max(intersect_xmax - intersect_xmin, 0.0f);

    // 计算重叠度
    return intersect_area / (area0 + area1 - intersect_area);
}

// Non-Maximum Suppression (NMS)
static int non_max_suppression(std::list<DetectRegion>& region_list, std::list<DetectRegion>& region_nms_list, float iou_thresh) {
    region_list.sort([](const DetectRegion& v1, const DetectRegion& v2) { return v1.score > v2.score; });

    for (const auto& region_candidate : region_list) {
        bool ignore_candidate = false;

        for (const auto& region_nms : region_nms_list) {
            float iou = calc_intersection_over_union(region_candidate, region_nms);
            if (iou >= iou_thresh) {
                ignore_candidate = true;
                break;
            }
        }

        if (!ignore_candidate) {
            region_nms_list.push_back(region_candidate);
            if (region_nms_list.size() >= 5)
                break;
        }
    }

    return 0;
}

// Normalize angle to radians
static float normalize_radians(float angle) {
    return angle - 2 * M_PI * std::floor((angle - (-M_PI)) / (2 * M_PI));
}

// Compute rotation
static void compute_rotation(DetectRegion& region) {
    float x0 = region.landmarks[0].x;
    float y0 = region.landmarks[0].y;
    float x1 = region.landmarks[2].x;
    float y1 = region.landmarks[2].y;

    float target_angle = M_PI * 0.5f;
    float rotation = target_angle - std::atan2(-(y1 - y0), x1 - x0);

    region.rotation = normalize_radians(rotation);
}

// Rotate vector
void rot_vec(cv::Point2f& vec, float rotation) {
    float sx = vec.x;
    float sy = vec.y;

    vec.x = sx * std::cos(rotation) - sy * std::sin(rotation);
    vec.y = sx * std::sin(rotation) + sy * std::cos(rotation);
}

// Compute detection to region of interest (ROI)
void compute_detect_to_roi(DetectRegion& region, int target_size, PalmObject& palm) {
    float width = region.btmright.x - region.topleft.x;
    float height = region.btmright.y - region.topleft.y;
    float palm_cx = region.topleft.x + width * 0.5f;
    float palm_cy = region.topleft.y + height * 0.5f;
    float hand_cx;
    float hand_cy;
    float rotation = region.rotation;
    float shift_x = 0.0f;
    float shift_y = -0.5f;

    if (rotation == 0.0f) {
        hand_cx = palm_cx + (width * shift_x);
        hand_cy = palm_cy + (height * shift_y);
    } else {
        float dx = (width * shift_x) * std::cos(rotation) - (height * shift_y) * std::sin(rotation);
        float dy = (width * shift_x) * std::sin(rotation) + (height * shift_y) * std::cos(rotation);
        hand_cx = palm_cx + dx;
        hand_cy = palm_cy + dy;
    }

    float long_side = std::max(width, height);
    width = long_side;
    height = long_side;
    float hand_w = width * 2.6f;
    float hand_h = height * 2.6f;

    palm.hand_cx = hand_cx;
    palm.hand_cy = hand_cy;
    palm.hand_w = hand_w;
    palm.hand_h = hand_h;

    float dx = hand_w * 0.5f;
    float dy = hand_h * 0.5f;

    palm.hand_pos[0] = { -dx, -dy };
    palm.hand_pos[1] = { +dx, -dy };
    palm.hand_pos[2] = { +dx, +dy };
    palm.hand_pos[3] = { -dx, +dy };

    for (int i = 0; i < 4; ++i) {
        rot_vec(palm.hand_pos[i], rotation);
        palm.hand_pos[i].x += hand_cx;
        palm.hand_pos[i].y += hand_cy;
    }

    for (int i = 0; i < 7; ++i) {
        palm.landmarks[i] = region.landmarks[i];
    }

    palm.score = region.score;
}


// 包装检测结果
static void pack_detect_result(std::vector<DetectRegion>& detect_results, std::list<DetectRegion>& region_list, const int& target_size, std::vector<PalmObject>& palmlist) {
    for (auto& region : region_list) {
        compute_rotation(region);

        PalmObject palm;
        compute_detect_to_roi(region, target_size, palm);

        palmlist.push_back(palm);
        detect_results.push_back(region);
    }
}

// Hand 类的构造函数
Hand::Hand() {
    blob_pool_allocator.set_size_compare_ratio(0.f);
    workspace_pool_allocator.set_size_compare_ratio(0.f);
}

// 加载模型
int Hand::load(AAssetManager* mgr, const char* modeltype, int _target_size, const float* _mean_vals, const float* _norm_vals, bool use_gpu) {
    blazepalm_net.clear();
    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    blazepalm_net.opt = ncnn::Option();

#if NCNN_VULKAN
    blazepalm_net.opt.use_vulkan_compute = use_gpu;
#endif

    blazepalm_net.opt.num_threads = ncnn::get_big_cpu_count();
    blazepalm_net.opt.blob_allocator = &blob_pool_allocator;
    blazepalm_net.opt.workspace_allocator = &workspace_pool_allocator;

    char parampath[256];
    char modelpath[256];
    sprintf(parampath, "%s-op.param", modeltype);
    sprintf(modelpath, "%s-op.bin", modeltype);

    if (blazepalm_net.load_param(mgr, parampath) != 0 || blazepalm_net.load_model(mgr, modelpath) != 0) {
        return -1; // return error if load fails
    }

    // 双模型
    if (landmark.load(mgr, "hand_lite-op") != 0) {
        return -1; // return error if landmark load fails
    }
    // 手动调整模型名称来加载完整模型（可选）
    // if (landmark.load(mgr, "hand_full-op") != 0) {
    //     return -1; // return error if landmark load fails
    // }

    target_size = _target_size;
    std::copy(_mean_vals, _mean_vals + 3, mean_vals);
    std::copy(_norm_vals, _norm_vals + 3, norm_vals);

    // 重新生成锚点
    anchors.clear();
    create_ssd_anchors(target_size, target_size, anchors);

    return 0;
}
std::deque<cv::Point2f> finger_positions; // 全局变量存储最近几帧中的位置

cv::Point2f average_position() {
    cv::Point2f sum(0, 0);
    for (const auto& pos : finger_positions) {
        sum += pos;
    }
    return cv::Point2f(sum.x / finger_positions.size(), sum.y / finger_positions.size());
}
// 手指跟踪检测
int Hand::detect(const cv::Mat& rgb, std::vector<PalmObject>& objects, float prob_threshold, float nms_threshold) {
    int width = rgb.cols;
    int height = rgb.rows;
    int w = width, h = height;
    float scale = std::min(static_cast<float>(target_size) / w, static_cast<float>(target_size) / h);
    w = static_cast<int>(w * scale);
    h = static_cast<int>(h * scale);

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_RGB, width, height, w, h);

    int wpad = target_size - w;
    int hpad = target_size - h;

    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 0.f);

    const float norm_vals[3] = { 1 / 255.f, 1 / 255.f, 1 / 255.f };
    in_pad.substract_mean_normalize(0, norm_vals);

    ncnn::Extractor ex = blazepalm_net.create_extractor();

    ncnn::Mat cls, reg;
    ex.input("input", in_pad);
    ex.extract("cls", cls);
    ex.extract("reg", reg);

    float* scores = (float*)cls.data;
    float* bboxes = (float*)reg.data;

    std::list<DetectRegion> region_list, region_nms_list;
    std::vector<DetectRegion> detect_results;

    decode_bounds(region_list, prob_threshold, target_size, target_size, scores, bboxes, anchors);
    non_max_suppression(region_list, region_nms_list, nms_threshold);
    objects.clear();
    pack_detect_result(detect_results, region_nms_list, target_size, objects);

    for (auto& object : objects) {
        for (int j = 0; j < 4; ++j) {
            object.hand_pos[j].x = (object.hand_pos[j].x * target_size - (wpad / 2)) / scale;
            object.hand_pos[j].y = (object.hand_pos[j].y * target_size - (hpad / 2)) / scale;
        }

        for (int j = 0; j < 7; ++j) {
            object.landmarks[j].x = (object.landmarks[j].x * target_size - (wpad / 2)) / scale;
            object.landmarks[j].y = (object.landmarks[j].y * target_size - (hpad / 2)) / scale;
        }

        cv::Point2f srcPts[4] = { object.hand_pos[0], object.hand_pos[1], object.hand_pos[2], object.hand_pos[3] };
        cv::Point2f dstPts[4] = { cv::Point2f(0, 0), cv::Point2f(224, 0), cv::Point2f(224, 224), cv::Point2f(0, 224) };

        cv::Mat trans_mat = cv::getAffineTransform(srcPts, dstPts);
        cv::warpAffine(rgb, object.trans_image, trans_mat, cv::Size(224, 224), 1, 0);

        cv::Mat trans_mat_inv;
        cv::invertAffineTransform(trans_mat, trans_mat_inv);

        if (landmark.detect(object.trans_image, trans_mat_inv, object.skeleton) == -1) {
            return -1; // return error code if detection fails
        }
    }
// 计算平均位置来平滑
    for (auto& object : objects) {
        if (object.skeleton.size() > 8) {
            finger_positions.push_back(object.skeleton[8]); // 第8个点假设为食指尖点
            if (finger_positions.size() > 5) { // 限制队列大小为5
                finger_positions.pop_front();
            }
            object.skeleton[8] = average_position();
        }
    }
    return 0;
}

// 绘制检测结果
int Hand::draw(cv::Mat& rgb, const std::vector<PalmObject>& objects) {
    for (const auto& object : objects) {
        // 我们可以只绘制第8点，也就是假设这是食指点的标记。
        int finger_tip_index = 8;

        cv::Scalar color_fingers(255, 0, 0); // 红色表示食指尖点
        __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "手指hand坐标测试: (%f, %f)", object.skeleton[finger_tip_index].x, object.skeleton[finger_tip_index].y);
        // 绘制食指尖点
        if (!object.skeleton.empty()) {
            cv::circle(rgb, object.skeleton[finger_tip_index], 4, color_fingers, -1);
        }

        // 其他绘制逻辑（可选）

        // 如需绘制其他骨架点
//        for (int j = 0; j < object.skeleton.size(); j++) {
//            if (j != 8) { // 如果不等于食指点
//                cv::circle(rgb, object.skeleton[j], 2, cv::Scalar(0, 255, 0), -1); // 绿色表示其他点
//            }
//        }

        // 绘制手掌周围的矩形框
//         for (int j = 0; j < 4; ++j) {
//             cv::line(rgb, object.hand_pos[j], object.hand_pos[(j+1)%4], cv::Scalar(0, 255, 0), 2);
//         }
    }

    return 0;
}
