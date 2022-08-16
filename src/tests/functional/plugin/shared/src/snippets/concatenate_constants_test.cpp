
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/concatenate_constants_test.hpp"

#include <memory>
#include <tuple>

#include "ngraph_ops/type_relaxed.hpp"
#include "concatenate_constants_function.hpp"
#include "function_helper.hpp"

namespace LayerTestsDefinitions {

std::string ConcatenateConstantsTest::getTestCaseName(testing::TestParamInfo<testsParams> obj) {
    std::ostringstream result;
    auto values = std::get<0>(obj.param);
    const size_t input_branch = std::get<1>(obj.param);
    values.inputShape[0] = input_branch;
    const auto input_type = std::get<2>(obj.param);
    const auto operations_number = std::get<3>(obj.param);
    const auto targetDevice = std::get<4>(obj.param);

    result << "IS=" << CommonTestUtils::vec2str(values.inputShape) << "_";
    result << "netPRC=" << input_type << "_";
    result << "D=" << targetDevice << "_";
    result << "IN=" << input_type << "_";
    result << "NN1=" << operations_number.first;
    result << "NN2=" << operations_number.second;
    for (auto i = 0; i < values.constantShapes.size(); ++i) {
        result << "_SH" << i << "=" << values.constantShapes[i];
    }
    return result.str();
}

void ConcatenateConstantsTest::SetUp() {
    // not initialized by default
    abs_threshold = 0.01;
    rel_threshold = 0.01;

    auto& testsParams = this->GetParam();

    auto values = std::get<0>(testsParams);

    auto input_batch = std::get<1>(testsParams);
    values.inputShape[0] = input_batch;
    const auto input_type = std::get<2>(testsParams);
    targetDevice = std::get<4>(testsParams);

    init_input_shapes({{values.inputShape, {values.inputShape}}});

    function = ov::test::snippets::ConcatenateConstantsFunction::get(
        {values.inputShape},
        input_type,
        values.constantShapes,
        ov::test::snippets::FunctionHelper::makePrerequisitesOriginal());
}

void ConcatenateConstantsTest::run() {
    SubgraphBaseTest::run();

    const auto operations_number = std::get<3>(GetParam());
    ref_num_nodes = operations_number.first;
    ref_num_subgraphs = operations_number.second;

    validateNumSubgraphs();
}

TEST_P(ConcatenateConstantsTest, CompareWithRefImpl) {
    run();
};

}  // namespace LayerTestsDefinitions
