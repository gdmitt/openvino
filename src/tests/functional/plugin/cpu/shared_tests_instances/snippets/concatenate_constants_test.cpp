// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "snippets/concatenate_constants_test.hpp"

using namespace LayerTestsDefinitions;
using namespace ngraph;


namespace concatenateConstantsInSubgraph {
const std::vector<ConcatenateConstantsTestValues> testValues = {
    {
        {1, 3, 16, 16},
        {{1, 3, 1, 1}, {1, 3, 1, 1}},
    },
    {
        {1, 3, 16, 16},
        {{1, 3, 1, 1}, {}},
    },
    {
        {1, 3, 16, 16},
        {{}, {1, 3, 1, 1}},
    },
    {
        {1, 10, 16, 16},
        {{1, 10, 1, 1}, {1, 10, 1, 1}},
    },
    {
        {1, 10, 16, 16},
        {{1, 10, 1, 1}, {}},
    },
    {
        {1, 10, 16, 16},
        {{}, {1, 10, 1, 1}},
    }
};

std::vector<size_t> input_batches = { 1ul, 2ul };

std::vector<ov::element::Type> input_types = { ov::element::f32 };

INSTANTIATE_TEST_SUITE_P(
    smoke_Snippets,
    ConcatenateConstantsTest,
    ::testing::Combine(
        ::testing::ValuesIn(testValues),
        ::testing::ValuesIn(input_batches),
        ::testing::ValuesIn(input_types),
        ::testing::Values(std::pair<size_t, size_t>{5, 1}),
        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
    ConcatenateConstantsTest::getTestCaseName);
}  // namespace concatenateConstantsInSubgraph


namespace concatenateConstantsIgnoredConstBatch1 {
const std::vector<ConcatenateConstantsTestValues> testValues = {
    {
        {1, 3, 16, 16},
        {{1, 3, 16, 16}, {1, 3, 16, 16}},
    },
    {
        {1, 10, 16, 16},
        {{1, 10, 16, 16}, {1, 10, 16, 16}},
    },
};

std::vector<size_t> input_batches = { 1ul, 2ul };

std::vector<ov::element::Type> input_types = { ov::element::f32 };

INSTANTIATE_TEST_SUITE_P(
        smoke_Snippets,
        ConcatenateConstantsTest,
        ::testing::Combine(
                ::testing::ValuesIn(testValues),
                ::testing::ValuesIn(input_batches),
                ::testing::ValuesIn(input_types),
                ::testing::Values(std::pair<size_t, size_t>{6, 1}),
                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ConcatenateConstantsTest::getTestCaseName);
}  // namespace concatenateConstantsIgnoredConstBatch1


namespace concatenateConstantsIgnoredConstBatch2 {
const std::vector<ConcatenateConstantsTestValues> testValues = {
    {
            {2, 3, 16, 16},
            {{2, 3, 16, 16}, {2, 3, 16, 16}},
    },
    {
            {2, 10, 16, 16},
            {{2, 10, 16, 16}, {2, 10, 16, 16}},
    },
};

std::vector<size_t> input_batches = { 2ul };

std::vector<ov::element::Type> input_types = { ov::element::f32 };

INSTANTIATE_TEST_SUITE_P(
        smoke_Snippets,
        ConcatenateConstantsTest,
        ::testing::Combine(
                ::testing::ValuesIn(testValues),
                ::testing::ValuesIn(input_batches),
                ::testing::ValuesIn(input_types),
                ::testing::Values(std::pair<size_t, size_t>{6, 1}),
                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ConcatenateConstantsTest::getTestCaseName);
}  // namespace concatenateConstantsIgnoredConstBatch2