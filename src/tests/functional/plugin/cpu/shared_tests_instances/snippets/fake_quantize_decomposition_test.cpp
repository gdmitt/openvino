// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "snippets/fake_quantize_decomposition_test.hpp"

using namespace LayerTestsDefinitions;
using namespace ngraph;

namespace {

namespace decompositionIgnore {
const std::vector<TestValues> testValuesDecomposition = {
    {
        ov::element::f32,
        ngraph::Shape{1, 3, 16, 16},
        ov::element::f32,
        1.f,
        {{1, 3, 16, 16}, {1, 3, 16, 16}, {1, 3, 16, 16}, {1, 3, 16, 16}}
    }
};

std::vector<std::pair<std::shared_ptr<Node>, std::pair<std::string, std::string>>> operations = {
    {std::make_shared<ngraph::opset1::Parameter>(), {"Subgraph", "fakeQuantize/DequantizationSubtract,fakeQuantize"}},
};

INSTANTIATE_TEST_SUITE_P(
    smoke_Snippets,
    FakeQuantizeDecompositionTest,
    ::testing::Combine(
        ::testing::ValuesIn(testValuesDecomposition),
        ::testing::ValuesIn(operations),
        ::testing::Values(std::pair<size_t, size_t>{13, 1}),
        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
    FakeQuantizeDecompositionTest::getTestCaseName);
}  // namespace decompositionIgnore

std::vector<std::pair<std::shared_ptr<Node>, std::pair<std::string, std::string> >> operations = {
        {std::make_shared<opset1::Abs>(), {"Subgraph", "Abs,fakeQuantize"}},
        {std::make_shared<opset1::Clamp>(), {"Subgraph", "Clamp,fakeQuantize"}},
        {std::make_shared<opset1::Floor>(), {"Subgraph", "Floor,fakeQuantize"}},
        {std::make_shared<opset1::Ceiling>(), {"Subgraph", "Ceiling,fakeQuantize"}},
        {std::make_shared<opset1::Elu>(), {"Subgraph", "Elu,fakeQuantize"}},
        {std::make_shared<opset1::Erf>(), {"Subgraph", "Erf,fakeQuantize"}},
        {std::make_shared<opset1::Exp>(), {"Subgraph", "Exp,fakeQuantize"}},
        {std::make_shared<opset1::LogicalNot>(), {"Subgraph", "LogicalNot,fakeQuantize"}},
        {std::make_shared<opset1::Negative>(), {"Subgraph", "Negative,fakeQuantize"}},
        {std::make_shared<opset1::Relu>(), {"Subgraph", "fakeQuantize"}},
        {std::make_shared<opset5::Round>(), {"Subgraph", "Round,fakeQuantize"}},
        {std::make_shared<opset1::Sigmoid>(), {"Subgraph", "Sigmoid,fakeQuantize"}},
        {std::make_shared<opset1::Tanh>(), {"Subgraph", "Tanh,fakeQuantize"}},
        {std::make_shared<ngraph::op::v0::Gelu>(), {"Subgraph", "Gelu,fakeQuantize"}},
        {std::make_shared<ngraph::op::v7::Gelu>(), {"Subgraph", "Gelu,fakeQuantize"}},
        {std::make_shared<ngraph::op::v4::HSwish>(), {"Subgraph", "HSwish,fakeQuantize"}},
};

namespace decompositionInSubgraphConstBatch1 {
const std::vector<TestValues> testValuesDecomposition = {
    {
        ov::element::f32,
        ngraph::Shape{1, 3, 16, 16},
        ov::element::f32,
        1.f,
        {{}, {}, {}, {}},
    }
};

INSTANTIATE_TEST_SUITE_P(
    smoke_Snippets,
    FakeQuantizeDecompositionTest,
    ::testing::Combine(
        ::testing::ValuesIn(testValuesDecomposition),
        ::testing::ValuesIn(operations),
        ::testing::Values(std::pair<size_t, size_t>{4, 1}),
        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
    FakeQuantizeDecompositionTest::getTestCaseName);
}  // namespace decompositionInSubgraphConstBatch1


namespace decompositionInSubgraphConstBatch3 {
const std::vector<TestValues> testValuesDecomposition = {
    {
        ov::element::f32,
        ngraph::Shape{1, 3, 16, 16},
        ov::element::f32,
        1.f,
        {{}, {}, {1, 3, 1, 1}, {1, 3, 1, 1}}
    },
    {
        ov::element::f32,
        ngraph::Shape{1, 3, 16, 16},
        ov::element::f32,
        1.f,
        {{1, 3, 1, 1}, {1, 3, 1, 1}, {}, {}}
    },
    {
        ov::element::f32,
        ngraph::Shape{1, 3, 16, 16},
        ov::element::f32,
        1.f,
        {{1, 3, 1, 1}, {1, 3, 1, 1}, {1, 3, 1, 1}, {1, 3, 1, 1}}
    },
};

INSTANTIATE_TEST_SUITE_P(
    smoke_Snippets,
    FakeQuantizeDecompositionTest,
    ::testing::Combine(
            ::testing::ValuesIn(testValuesDecomposition),
            ::testing::ValuesIn(operations),
            ::testing::Values(std::pair<size_t, size_t>{5, 1}),
            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
    FakeQuantizeDecompositionTest::getTestCaseName);
}  // namespace decompositionInSubgraphConstBatch3

namespace legacyFuse {
const std::vector<TestValues> testValuesLegacyFuse = {
    {
        ov::element::f32,
        ngraph::Shape{1, 3, 16, 16},
        ov::element::f32,
        1.f,
        {{1, 3, 1, 1}, {1, 3, 1, 1}, {}, {}}
    },
    {
        ov::element::f32,
        ngraph::Shape{1, 3, 16, 16},
        ov::element::f32,
        1.f,
        {{}, {}, {1, 3, 1, 1}, {1, 3, 1, 1}}
    },
    {
        ov::element::f32,
        ngraph::Shape{1, 3, 16, 16},
        ov::element::f32,
        1.f,
        {{}, {}, {}, {}}
    },
    {
        ov::element::f32,
        ngraph::Shape{1, 3, 16, 16},
        ov::element::f32,
        1.f,
        {{1, 3, 1, 1}, {1, 3, 1, 1}, {1, 3, 1, 1}, {1, 3, 1, 1}}
    },
};

std::vector<std::pair<std::shared_ptr<Node>, std::pair<std::string, std::string>>> operations = {
    {std::make_shared<opset1::Convolution>(), {"Convolution", "Convolution,fakeQuantize"}},
};

INSTANTIATE_TEST_SUITE_P(
    smoke_Snippets,
    FakeQuantizeDecompositionTest,
    ::testing::Combine(
        ::testing::ValuesIn(testValuesLegacyFuse),
        ::testing::ValuesIn(operations),
        ::testing::Values(std::pair<size_t, size_t>{6, 0}), //Pooling + 2 * Reorder + Convolution
        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
    FakeQuantizeDecompositionTest::getTestCaseName);

}  // namespace legacyFuse

}  // namespace
