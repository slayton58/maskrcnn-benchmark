#pragma once

#include <ATen/ATen.h>
#include <THC/THC.h>

#include <vector>

/**
 * Generate boxes associated to topN pre-NMS scores
 */
std::vector<at::Tensor> GeneratePreNMSUprightBoxes(
    const int num_images,
    const int A,
    const int H,
    const int W,
    at::Tensor& sorted_indices, // topK sorted pre_nms_topn indices
    at::Tensor& sorted_scores,  // topK sorted pre_nms_topn scores [N, A, H, W]
    at::Tensor& bbox_deltas,    // [N, A*4, H, W] (full, unsorted / sliced)
    at::Tensor& anchors,        // input (full, unsorted, unsliced)
    at::Tensor& image_shapes,   // (h, w) of images
    const int pre_nms_nboxes,
    const int feature_stride,
    const int rpn_min_size,
    const float bbox_xform_clip_default,
    const bool correct_transform_coords);
