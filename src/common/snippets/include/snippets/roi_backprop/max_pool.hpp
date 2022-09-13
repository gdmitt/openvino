// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/op/gather.hpp>
#include <openvino/core/partial_shape.hpp>

namespace ov {
namespace op {
namespace v1 {
template <class ShapeType>
void roi_backprop(
        const MaxPool* op,
        const std::vector<ShapeType>& input_shapes,
        std::vector<ShapeType>& roi_shapes,
        std::vector<ov::Shape>& strides) {
    NODE_VALIDATION_CHECK(op, (input_shapes.size() == 1ul) && (roi_shapes.size() == 1));
    NODE_VALIDATION_CHECK(op, input_shapes[0].rank() == roi_shapes[0].rank());

    // TODO: roi backpropagation: dynamic shape can be used
    // TODO: roi backpropagation: params are ignored

    const auto kernel = op->get_kernel();
    auto roi_shape = roi_shapes[0];
    auto shape_offset = input_shapes[0].size() - (kernel.size() + 2ul);
    for (auto i = 2ul; i < roi_shape.size() - shape_offset; ++i) {
        roi_shape[i] = roi_shape[i] * kernel[i - 2ul];
    }
    roi_shapes[0] = roi_shape;

    const auto op_strides = op->get_strides();
    auto strides0 = strides[0];
    auto strides0_offset = input_shapes[0].size() - (op_strides.size() + 2ul);
    for (auto i = 2ul; i < strides0.size() - strides0_offset; ++i) {
        strides0[i] = strides0[i] * op_strides[i - 2ul];
    }
    strides[0] = strides0;
}

}  // namespace v1
}  // namespace op
}  // namespace ov