// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/concatenate_constants.hpp"

#include <memory>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/pass/constant_folding.hpp>

#include "transformations/utils/utils.hpp"
#include "snippets/op/subgraph.hpp"
#include "snippets/itt.hpp"

NGRAPH_RTTI_DEFINITION(ngraph::snippets::pass::ConcatenateConstants, "Snippets::ConcatenateConstants", 0);

namespace ngraph {
namespace snippets {
namespace pass {
ConcatenateConstants::ConcatenateConstants() {
    const auto wrapper = ngraph::pattern::wrap_type<ngraph::snippets::op::Subgraph>();

    ngraph::graph_rewrite_callback callback = [this](pattern::Matcher& m) {
        OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::ConcatenateConstants");

        const auto subgraph = ngraph::as_type_ptr<ngraph::snippets::op::Subgraph>(m.get_match_root());
        if (transformation_callback(subgraph)) {
            return false;
        }

        const auto body = subgraph->get_body();

        const size_t axis = 0;
        std::vector<size_t> split_lengths;

        std::unordered_map<std::string, std::shared_ptr<opset1::Constant>> constant_input_ids;
        OutputVector previous_inputs;
        OutputVector outputs;
        std::unordered_map<std::string, std::shared_ptr<opset1::Parameter>> parameter_input_ids;
        std::unordered_map<std::string, size_t> index_by_briendly_name;
        std::string concatenated_constant_friendly_name;
        bool concatenated_constant_was_added = false;
        ngraph::element::Type input_precision;
        size_t d2 = -1;

        for (auto target_input : subgraph->inputs()) {
            auto source_output = target_input.get_source_output();
            auto constant = as_type_ptr<opset1::Constant>(source_output.get_node()->shared_from_this());

            auto check_constant = [](const std::shared_ptr<opset1::Constant>& constant, const size_t d2) -> bool {
                if (constant->get_element_type() != ngraph::element::f32) {
                    return false;
                }

                const auto shape = constant->get_shape();
                const auto size = shape.size();
                if (shape.empty() || (size == 1ul) || (ngraph::shape_size(shape) == 1ul)) {
                    return true;
                }

                if ((shape[0] != 1ul) || ((d2 != -1) && (size >= 2ul) && (shape[1] != d2))) {
                    return false;
                }

                for (auto i = 1ul; i <= (size - 2ul); ++i) {
                    if (shape[size - i] != 1ul) {
                        return false;
                    }
                }

                return true;
            };

            if ((constant != nullptr) && check_constant(constant, d2)) {
                if (!concatenated_constant_was_added) {
                    concatenated_constant_friendly_name = constant->get_friendly_name();
                    concatenated_constant_was_added = true;
                    input_precision = constant->output(0).get_element_type();
                }
                const auto& shape = constant->output(0).get_shape();
                if ((d2 == -1) && (shape.size() >= 2ul)) {
                    d2 = shape[1];
                }
                split_lengths.push_back(shape[axis]);

                constant->output(0).remove_target_input(target_input);

                constant_input_ids.emplace(constant->get_friendly_name(), constant);
                outputs.push_back(constant->output(0));
                index_by_briendly_name[constant->get_friendly_name()] = target_input.get_index() - 1ul;
                continue;
            } else {
                previous_inputs.push_back(source_output);
            }
        }

        assert(outputs.size() == constant_input_ids.size());
        assert(split_lengths.size() == constant_input_ids.size());
        assert(index_by_briendly_name.size() == constant_input_ids.size());

        if (constant_input_ids.size() < 2ul) {
            return false;
        }

        OutputVector folding_outputs(1);
        auto concat = std::make_shared<ngraph::opset1::Concat>(outputs, axis);
        if (!concat->constant_fold(folding_outputs, outputs)) {
            throw ov::Exception("constants were not concatenated");
        }
        if (folding_outputs.size() != 1ul) {
            throw ov::Exception("unexpected constant folding results count");
        }
        const auto result = ov::as_type_ptr<opset1::Constant>(folding_outputs[0].get_node_shared_ptr());
        if (result == nullptr) {
            throw ov::Exception("constants were not concatenated");
        }

        auto& rt_info = result->get_rt_info();
        rt_info.insert({"concatenated", true});

        result->set_friendly_name(concatenated_constant_friendly_name);
        previous_inputs.push_back(result);

        auto parameter = std::make_shared<opset1::Parameter>(input_precision, result->output(0).get_partial_shape());
        parameter->set_friendly_name(concatenated_constant_friendly_name);

        assert(std::all_of(split_lengths.begin(), split_lengths.end(), [&split_lengths](const size_t value) { return split_lengths[0] == value; }));
        const std::shared_ptr<Node> split = std::dynamic_pointer_cast<Node>(std::make_shared<opset1::Split>(
            parameter,
            std::make_shared<opset1::Constant>(element::i32, Shape{}, std::vector<size_t>{axis}),
            split_lengths.size()));

        body->validate_nodes_and_infer_types();

        auto parameters = body->get_parameters();
        for (auto i = 0; i < parameters.size(); ++i) {
            auto tmp_parameter = parameters[i];
            auto it = constant_input_ids.find(tmp_parameter->get_friendly_name());
            if (it == constant_input_ids.end()) {
                continue;
            }

            auto indexIt = index_by_briendly_name.find(tmp_parameter->get_friendly_name());
            if (indexIt == index_by_briendly_name.end()) {
                throw ngraph_error("parameter was not mapped");
            }

            for (auto input : tmp_parameter->output(0).get_target_inputs()) {
                auto index = indexIt->second;
                input.replace_source_output(split->output(index));
            }

            body->remove_parameter(tmp_parameter);
        }
        body->add_parameters({ parameter });
        body->validate_nodes_and_infer_types();

        auto new_subgraph = std::make_shared<ngraph::snippets::op::Subgraph>(previous_inputs, body);
        replace_node(subgraph, new_subgraph);
        copy_runtime_info(subgraph, new_subgraph);

        return true;
    };

    const auto m = std::make_shared<ngraph::pattern::Matcher>(wrapper, "snippets::pass::ConcatenateConstants");
    this->register_matcher(m, callback);
}

} // namespace pass
} // namespace snippets
} // namespace ngraph