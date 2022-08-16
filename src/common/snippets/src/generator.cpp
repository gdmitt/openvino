// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/generator.hpp"
#include "snippets/pass/assign_registers.hpp"
#include "snippets/pass/vector_to_scalar.hpp"
#include "snippets/pass/insert_load_store.hpp"
#include "snippets/op/tile.hpp"
#include "snippets/op/kernel.hpp"
#include <snippets/itt.hpp>

#include <ngraph/pass/manager.hpp>

namespace {
void set_should_post_increment(std::shared_ptr<ov::Model>& m) {
    std::unordered_map<std::shared_ptr<ngraph::Node>, std::vector<std::shared_ptr<ngraph::snippets::op::Load>>> loads_by_split;
    std::unordered_map<std::shared_ptr<ngraph::snippets::op::Load>, size_t> ordered_loads;
    size_t order = 0;
    for (auto n : m->get_ordered_ops()) {
        if (auto load = ngraph::as_type_ptr<ngraph::snippets::op::Load>(n)) {
            if (auto split = ngraph::as_type_ptr<ngraph::opset1::Split>(n->get_input_node_shared_ptr(0))) {
                ordered_loads.emplace(load, order);
                order++;

                auto get_nodes = [&]() -> std::vector<std::shared_ptr<ngraph::snippets::op::Load>> & {
                    auto it = loads_by_split.find(split);
                    if (it != loads_by_split.end()) {
                        return it->second;
                    }

                    loads_by_split[split] = {};
                    return loads_by_split[split];
                };

                const auto index = n->get_input_source_output(0).get_index();
                auto &nodes = get_nodes();
                if (nodes.size() <= index) {
                    nodes.resize(index + 1);
                }
                nodes[index] = load;
            }
        }
    }

    for (auto loads_it : loads_by_split) {
        std::shared_ptr<ngraph::snippets::op::Load> latest;
        size_t latest_order;
        for (auto load : loads_it.second) {
            const auto order_it = ordered_loads.find(load);
            if (order_it == ordered_loads.end()) {
                throw ov::Exception("order for node `" + load->get_friendly_name() + "` was not found");
            }

            order = order_it->second;
            if ((latest == nullptr) || (order > latest_order)) {
                latest = load;
                latest_order = order;
            }
        }

        if (latest == nullptr) {
            throw ov::Exception("latest node was not found");
        }
        latest->should_post_increment = true;
    }
}
} // namespace

auto ngraph::snippets::getRegisters(std::shared_ptr<ngraph::Node>& n) -> ngraph::snippets::RegInfo {
    OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::getRegisters")
    auto rt = n->get_rt_info();

    // ToDo: change to reg_t
    std::vector<size_t> rin, rout;

    auto it_rt = rt.find("reginfo");
    if (it_rt != rt.end()) {
        for (auto reg : it_rt->second.as<std::vector<size_t>>()) {
            rout.push_back(reg);
        }
    }

    for (const auto& input : n->inputs()) {
        auto rt = input.get_source_output().get_node_shared_ptr()->get_rt_info();
        auto it_rt = rt.find("reginfo");
        if (it_rt != rt.end()) {
            for (auto& reg : it_rt->second.as<std::vector<size_t>>()) {
                rin.push_back(reg);
            }
        }
    }
    return std::make_pair(rin, rout);
}

ngraph::snippets::code ngraph::snippets::Generator::generate(std::shared_ptr<ov::Model>& m,
                                                             const void* compile_params) const {
    OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::Generator::generate")
    if (!target->is_supported())
        throw ngraph_error("unsupported architecture for code genration");

    auto params = m->get_parameters();
    auto results = m->get_results();
    auto in = params.size();
    auto out = results.size();

    OV_ITT_TASK_CHAIN(GENERATE, ngraph::pass::itt::domains::SnippetsTransform, "Snippets::Generator", "::VectorTile")
    // vector tile
    std::vector<AllocatedEmitter> lowered;

    set_should_post_increment(m);

    for (auto n : m->get_ordered_ops()) {
        lowered.emplace_back(std::make_pair(target->get(n->get_type_info())(n), ngraph::snippets::getRegisters(n)));
    }
    OV_ITT_TASK_NEXT(GENERATE, "::ScalarTile")

    // scalar tile
    auto m_scalar = ov::clone_model(*m.get());
    ngraph::pass::Manager mng;
    mng.register_pass<ngraph::snippets::pass::ReplaceLoadsWithScalarLoads>();
    mng.register_pass<ngraph::snippets::pass::ReplaceStoresWithScalarStores>();
    mng.run_passes(m_scalar);
    OV_ITT_TASK_NEXT(GENERATE, "::ScalarTile_get")
    std::vector<AllocatedEmitter> scalar_lowered;
    for (auto n : m_scalar->get_ordered_ops()) {
        scalar_lowered.emplace_back(std::make_pair(target->get(n->get_type_info())(n), ngraph::snippets::getRegisters(n)));
    }
    OV_ITT_TASK_NEXT(GENERATE, "::Tiles1D")
    // wrapping into tiles1D
    const auto& vector_tile = std::make_shared<ngraph::snippets::op::Tile>(lowered);
    const auto& vector_region = std::make_pair(target->get(ngraph::snippets::op::Tile::get_type_info_static())(vector_tile),
                                   std::make_pair(std::vector<size_t>{target->get_lanes()}, std::vector<size_t>{}));
    const auto& scalar_tile = std::make_shared<ngraph::snippets::op::Tile>(scalar_lowered);
    const auto& scalar_region = std::make_pair(target->get(ngraph::snippets::op::Tile::get_type_info_static())(scalar_tile),
                    std::make_pair(std::vector<size_t>{1}, std::vector<size_t>{}));

    OV_ITT_TASK_NEXT(GENERATE, "::Tiles2D")
    // wrapping into tiles2D
    auto tile_scheduler = std::make_shared<ngraph::snippets::op::TileScheduler>(vector_region, scalar_region);
    tile_scheduler->compile_params = compile_params;
    const auto& tile_scheduler_region = std::make_pair(target->get(ngraph::snippets::op::TileScheduler::get_type_info_static())(tile_scheduler),
                                                       std::make_pair(std::vector<size_t>({in, out, target->get_lanes()}), std::vector<size_t>{}));

    OV_ITT_TASK_NEXT(GENERATE, "::EmitCode")
    // emission
    auto tiles2DKernel = std::make_shared<ngraph::snippets::op::Kernel>(std::vector<AllocatedEmitter> {tile_scheduler_region});
    tiles2DKernel->compile_params = compile_params;
    std::shared_ptr<Emitter> kernel = target->get(ngraph::snippets::op::Kernel::get_type_info_static())(tiles2DKernel);
    kernel->emit_code({in, out}, {});
    OV_ITT_TASK_NEXT(GENERATE, "::EmitData")
    lowered.insert(lowered.end(), scalar_lowered.begin(), scalar_lowered.end());
    for (auto& op : lowered) {
        op.first->emit_data();
    }
    OV_ITT_TASK_NEXT(GENERATE, "::GetSnippet")
    return target->get_snippet();
}
