
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/max_pool_test.hpp"

#include <memory>
#include <tuple>

#include "ngraph_ops/type_relaxed.hpp"
#include "max_pool_function.hpp"

namespace LayerTestsDefinitions {

std::string MaxPoolTest::getTestCaseName(testing::TestParamInfo<testsParams> obj) {
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
    result << "P_S=" << CommonTestUtils::vec2str(values.params.strides) << "_";
    result << "P_PB=" << CommonTestUtils::vec2str(values.params.pads_begin) << "_";
    result << "P_PE=" << CommonTestUtils::vec2str(values.params.pads_end) << "_";
    result << "P_K=" << CommonTestUtils::vec2str(values.params.kernel) << "_";
    result << "NN1=" << operations_number.first;
    result << "NN2=" << operations_number.second;
    for (auto i = 0; i < values.constantShapes.size(); ++i) {
        result << "_SH" << i << "=" << values.constantShapes[i];
    }
    return result.str();
}

void MaxPoolTest::SetUp() {
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

    function = ov::test::snippets::MaxPoolFunction::get(
            {values.inputShape},
            input_type,
            {
                values.prerequisites_params.strides,
                values.prerequisites_params.pads_begin,
                values.prerequisites_params.pads_end,
                values.prerequisites_params.kernel
            },
            {
                values.params.strides,
                values.params.pads_begin,
                values.params.pads_end,
                values.params.kernel
            },
            values.constantShapes);

    ngraph::pass::VisualizeTree("svg/test.original.svg").run_on_model(function);
}

void MaxPoolTest::run() {
    SubgraphBaseTest::run();

    const auto operations_number = std::get<3>(GetParam());
    ref_num_nodes = operations_number.first;
    ref_num_subgraphs = operations_number.second;

    validateNumSubgraphs();
}

TEST_P(MaxPoolTest, CompareWithRefImpl) {
    run();
};

}  // namespace LayerTestsDefinitions
