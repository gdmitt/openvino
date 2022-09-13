// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/op/gather.hpp>
#include <openvino/core/partial_shape.hpp>

namespace ov {
namespace op {
namespace v8 {
template <class ShapeType>
void roi_backprop(
        const Gather* op,
        const std::vector<ShapeType>& input_shapes,
        std::vector<ShapeType>& roi_shapes,
        std::vector<ov::Shape>& strides) {
   NODE_VALIDATION_CHECK(op, input_shapes.size() == 3);

    roi_shapes.resize(3ul);

    auto& roi_data = roi_shapes[0];
    auto& data_shape = input_shapes[0];
    if (auto constant = ov::as_type_ptr<ov::opset1::Constant>(op->get_input_node_shared_ptr(2))) {
        const auto axis_value = constant->cast_vector<int64_t>()[0];
        roi_data = ov::Shape(data_shape.size(), 1);
        roi_data[axis_value] = data_shape[axis_value];
    } else {
        roi_data = data_shape;
    }

    roi_shapes[1] = input_shapes[1];
    roi_shapes[2] = input_shapes[2];
}

}  // namespace v8
}  // namespace op
}  // namespace ov