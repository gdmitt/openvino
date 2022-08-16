// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/builders.hpp"
#include "shared_test_classes/base/snippets_test_utils.hpp"

namespace LayerTestsDefinitions {

class ConcatenateConstantsTestValues {
public:
    ngraph::Shape inputShape;
    std::vector<ngraph::Shape> constantShapes;
};

typedef std::tuple<
    ConcatenateConstantsTestValues, // test values
    size_t,                         // branches
    ov::element::Type,              // input_type,
    std::pair<size_t, size_t>,      // number of nodes
    std::string                     // target device
> testsParams;

class ConcatenateConstantsTest : public testing::WithParamInterface<testsParams>, virtual public ov::test::SnippetsTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<testsParams> obj);

protected:
    void SetUp() override;

    void run() override;
};

}  // namespace LayerTestsDefinitions
