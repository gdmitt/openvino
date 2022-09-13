// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/ngraph.hpp"

namespace ov {
namespace test {
namespace snippets {

class MaxPoolFunction {
public:
    struct Params {
        ov::Strides strides;
        ov::Shape pads_begin;
        ov::Shape pads_end;
        ov::Shape kernel;
    };

    static std::shared_ptr<ov::Model> get(
            const ngraph::Shape& inputShape,
            const element::Type inputType,
            const Params& prerequisites_params,
            const Params& params,
            const std::vector<ngraph::Shape>& fakeQuantizeShapes);
};

}  // namespace snippets
}  // namespace test
}  // namespace ov
