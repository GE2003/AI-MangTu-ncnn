// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#ifndef YOLO_H
#define YOLO_H

#include <opencv2/core/core.hpp>
#include <net.h>
#include <unordered_map>

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

struct CachedPosition {
    int label;
    cv::Rect_<float> rect;
};

static std::vector<CachedPosition> cached_positions;

std::vector<CachedPosition>& getCachedPositions();

// 缓存多个对象的部位识别框
//static std::unordered_map<int, std::vector<CachedPosition>> cached_positions;

// 获取缓存的识别框集合
// 修改后的 getCachedPositions 方法
//inline std::vector<CachedPosition>& getCachedPositions(int label) {
//    if (cached_positions.find(label) != cached_positions.end()) {
//        return cached_positions[label];
//    } else {
//        return cached_positions[label];  // 自动插入新元素
//    }
//}




//void addCachedPosition(const CachedPosition& new_pos) {
//    // 如果缓存已经有4个元素，移除最先加入的元素
//    if (cached_positions.size() >= 4) {
//        cached_positions.erase(cached_positions.begin());  // 移除第一个元素
//    }
//    cached_positions.push_back(new_pos);  // 添加新的元素
//}

static void clearCachedPosition()
{
    cached_positions.clear();
}

struct GridAndStride
{
    int grid0;
    int grid1;
    int stride;
};

class Yolo
{
public:
    Yolo();

    int load(const char* modeltype, int target_size, const float* mean_vals, const float* norm_vals, bool use_gpu = false);

    int load(AAssetManager* mgr, const char* modeltype, int target_size, const float* mean_vals, const float* norm_vals, bool use_gpu = false);

    int detect(const cv::Mat& rgb, std::vector<Object>& objects, float prob_threshold = 0.4f, float nms_threshold = 0.5f);

    int draw(cv::Mat& rgb, const std::vector<Object>& objects);

    // 新增清空缓存的方法
//    void clearCache()
//    {
//        clearCachedPositions();
//    }

private:
    ncnn::Net yolo;
    int target_size;
    float mean_vals[3];
    float norm_vals[3];
    ncnn::UnlockedPoolAllocator blob_pool_allocator;
    ncnn::PoolAllocator workspace_pool_allocator;
};

#endif // NANODET_H
