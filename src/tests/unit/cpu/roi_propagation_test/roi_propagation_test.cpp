// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <openvino/op/ops.hpp>

#include <snippets/roi_backprop/roi_backprop.hpp>
#include <openvino/pass/visualize_tree.hpp>

#include <openvino/opsets/opset8.hpp>
#include <openvino/opsets/opset1.hpp>

std::vector<ov::PartialShape> get_default_start_roi(const std::shared_ptr<ov::Model>& m) {
    std::vector<ov::PartialShape> start_roi;
    for (const auto& result : m->get_results()) {
        start_roi.emplace_back(ov::Shape(result->get_input_partial_shape(0).size(), 1));
    }
    return start_roi;
}

ov::snippets::roi_map dump_roi_for_model(const std::shared_ptr<ov::Model>& m, const std::vector<ov::PartialShape>& start_roi = {}) {
    const auto map = [&]() {
        if (start_roi.empty()) {
            const auto default_roi = get_default_start_roi(m);
            return ov::snippets::get_roi_from_function(m, default_roi);
        } else {
            return ov::snippets::get_roi_from_function(m, start_roi);
        }
    }();

    for (const auto& elem : map) {
        std::stringstream ss;
        if (!elem.second.shapes.empty()) {
            for (const auto& roi : elem.second.shapes) {
                ss << roi << ", ";
            }
            elem.first->get_rt_info()["ROI"] = ss.str();
        }
    }
    ov::pass::VisualizeTree("C://models//test.roi").run_on_model(m);

    return map;
}

TEST(ROI_Backprop, TestTwoGathers) {
    auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{1, 3, 10, 10});
    auto constant = ov::opset1::Constant::create(ov::element::f32, {}, {2.f});
    auto mul = std::make_shared<ov::opset1::Multiply>(input, constant);

    auto indices_1 = ov::opset1::Constant::create(ov::element::i32, {5}, {0, 1, 2, 3, 4});
    auto indices_2 = ov::opset1::Constant::create(ov::element::i32, {7}, {0, 1, 2, 3, 4, 5, 6});

    auto axis_value_1 = ov::opset1::Constant::create(ov::element::i32, {}, {2});
    auto axis_value_2 = ov::opset1::Constant::create(ov::element::i32, {}, {3});

    auto gather_1 = std::make_shared<ov::opset8::Gather>(mul, indices_1, axis_value_1);
    auto gather_2 = std::make_shared<ov::opset8::Gather>(mul, indices_2, axis_value_2);

    auto result_1 = std::make_shared<ov::opset1::Result>(gather_1);
    auto result_2 = std::make_shared<ov::opset1::Result>(gather_2);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result_1, result_2}, ov::ParameterVector{input});

    auto map = dump_roi_for_model(model);
    auto expected_roi = std::vector<ov::PartialShape>{{1, 1, 10, 10}};
    EXPECT_EQ(map[input.get()].shapes, expected_roi);
}

TEST(ROI_Backprop, DifferentROIForEachInput) {
    auto input_1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{1, 3, 10, 10});
    auto constant = ov::opset1::Constant::create(ov::element::f32, {}, {2.f});
    auto mul = std::make_shared<ov::opset1::Multiply>(input_1, constant);

    auto indices = ov::opset1::Constant::create(ov::element::i32, {5}, {0, 1, 2, 3, 4});
    auto input_2 = std::make_shared<ov::opset1::Parameter>(ov::element::i32, ov::PartialShape{1});
    auto add_const = ov::opset1::Constant::create(ov::element::i32, {1}, {-1});
    auto add = std::make_shared<ov::opset1::Add>(input_2, add_const);

    auto gather = std::make_shared<ov::opset8::Gather>(mul, indices, add);

    auto result = std::make_shared<ov::opset1::Result>(gather);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input_1, input_2});

    auto map = dump_roi_for_model(model);
    auto expected_roi = std::vector<ov::PartialShape>{{1, 3, 10, 10}};
    EXPECT_EQ(map[input_1.get()].shapes, expected_roi);
}

TEST(ROI_Backprop, TestTwoOutsWithDifferentStartROIs) {
    auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{1, 3, 10, 10});
    auto constant = ov::opset1::Constant::create(ov::element::f32, {}, {2.f});
    auto mul = std::make_shared<ov::opset1::Multiply>(input, constant);

    auto clamp_1 = std::make_shared<ov::opset1::Clamp>(mul, 0, 6);
    auto clamp_2 = std::make_shared<ov::opset1::Clamp>(mul, 0, 6);

    auto result_1 = std::make_shared<ov::opset1::Result>(clamp_1);
    auto result_2 = std::make_shared<ov::opset1::Result>(clamp_2);

    auto model = std::make_shared<ov::Model>(ov::ResultVector{result_1, result_2}, ov::ParameterVector{input});

    auto map = dump_roi_for_model(model, {{1, 1, 1, 5}, {1, 1, 1, 10}});
    auto expected_roi = std::vector<ov::PartialShape>{{1, 1, 1, 10}};
    EXPECT_EQ(map[input.get()].shapes, expected_roi);
}