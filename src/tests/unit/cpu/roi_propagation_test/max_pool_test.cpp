// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <openvino/op/ops.hpp>
#include <openvino/opsets/opset1.hpp>

#include <snippets/roi_backprop/roi_backprop.hpp>
#include <openvino/pass/visualize_tree.hpp>

namespace {
std::shared_ptr<ov::Model> get_model(
        const ov::Strides& strides,
        const ov::Shape& pads_begin,
        const ov::Shape& pads_end,
        const ov::Shape& kernel) {
    auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{1, 3, 32, 32});

    auto max_pool = std::make_shared<ov::opset1::MaxPool>(
            input,
            strides,
            pads_begin,
            pads_end,
            kernel,
            ov::op::RoundingType::CEIL);

    auto result = std::make_shared<ov::opset1::Result>(max_pool);

    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{ input });
}

std::shared_ptr<ov::op::v0::Parameter> get_parameter(const std::shared_ptr<ov::Model>& model) {
    return *model->get_parameters().begin();
}
} // namespace

TEST(ROI_Backprop, MaxPoolTest_3_2) {
    const auto model = get_model(ov::Strides{3, 2}, {0, 0}, {0, 0}, ov::Shape{3, 2});

    ov::pass::VisualizeTree("svg/max_pool_test.svg").run_on_model(model);

    auto map = ov::snippets::get_roi_from_function(model, {{1ul, 1ul, 1ul, 1ul}});
    const auto& actual_roi = map[get_parameter(model).get()];

    auto expected_shapes = std::vector<ov::PartialShape>{{1ul, 1ul, 3ul, 2ul}};
    EXPECT_EQ(actual_roi.shapes, expected_shapes);

    auto expected_strides = std::vector<ov::Shape>{{1ul, 1ul, 3ul, 2ul}};
    EXPECT_EQ(actual_roi.strides, expected_strides);
}

TEST(ROI_Backprop, MaxPoolTest_1_1) {
    const auto model = get_model(ov::Strides{1, 13}, {0, 0}, {0, 0}, ov::Shape{3, 2});

    ov::pass::VisualizeTree("svg/max_pool_test.svg").run_on_model(model);

    auto map = ov::snippets::get_roi_from_function(model, {{1ul, 1ul, 1ul, 1ul}});
    const auto& actual_roi = map[get_parameter(model).get()];

    auto expected_shapes = std::vector<ov::PartialShape>{{1ul, 1ul, 3ul, 2ul}};
    EXPECT_EQ(actual_roi.shapes, expected_shapes);

    auto expected_strides = std::vector<ov::Shape>{{1ul, 1ul, 3ul, 2ul}};
    EXPECT_EQ(actual_roi.strides, expected_strides);
}
