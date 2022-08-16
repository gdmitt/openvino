// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "concatenate_constants_function.hpp"
#include "common_test_utils/data_utils.hpp"
#include <snippets/snippets_isa.hpp>
#include "function_helper.hpp"

namespace ov {
namespace test {
namespace snippets {

std::shared_ptr<ov::Model> ConcatenateConstantsFunction::get(
    const ngraph::Shape& inputShape,
    const element::Type inputType,
    const std::vector<ngraph::Shape>& constantShapes,
    const std::vector<std::shared_ptr<ngraph::Node>>& prerequisites) {
    assert(constantShapes.size() == 2ul);

    const auto parameter = std::make_shared<ngraph::opset1::Parameter>(inputType, inputShape);
    parameter->set_friendly_name("parameter");

    auto parent = FunctionHelper::applyPrerequisites(parameter, prerequisites);

    auto generate_values = [](const ngraph::Shape& shape, const float initial_value = 0.f) {
        std::vector<float> multiply_values(shape_size(shape));
        for (auto i = 0; i < multiply_values.size(); ++i) {
            multiply_values[i] = static_cast<float>(initial_value + i);
        }
        return multiply_values;
    };

    const auto multiply_value = ngraph::opset1::Constant::create(
            element::f32,
            constantShapes[0],
            generate_values(constantShapes[0], 2.f));
    parent = std::make_shared<ngraph::opset1::Multiply>(parent, multiply_value);

    const auto add_value = ngraph::opset1::Constant::create(
            element::f32,
            constantShapes[1],
            generate_values(constantShapes[1], shape_size(constantShapes[0]) + 2.f));
    parent = std::make_shared<ngraph::opset1::Add>(parent, add_value);

    const auto result = std::make_shared<ngraph::opset1::Result>(parent);
    result->set_friendly_name("result");

    auto function = std::make_shared<ngraph::Function>(
            ngraph::ResultVector{ result },
            ParameterVector{ parameter },
            "SplitFunction");
    function->validate_nodes_and_infer_types();

    return function;
}

}  // namespace snippets
}  // namespace test
}  // namespace ov
