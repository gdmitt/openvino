// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "nodes/reference.h"
#include "nodes/shapeof.h"
#include "nodes/batch_to_space.h"
#include "nodes/multiclass_nms.hpp"
#include "nodes/adaptive_pooling.h"
#include "nodes/conv.h"
#include "nodes/roi_align.h"
#include "nodes/lrn.h"
#include "nodes/generic.h"
#include "nodes/experimental_detectron_roifeatureextractor.h"
#include "nodes/eltwise.h"
#include "nodes/reorg_yolo.h"
#include "nodes/pooling.h"
#include "nodes/transpose.h"
#include "nodes/grn.h"
#include "nodes/interpolate.h"
#include "nodes/experimental_detectron_detection_output.h"
#include "nodes/roll.h"
#include "nodes/fake_quantize.h"
#include "nodes/embedding_segments_sum.h"
#include "nodes/region_yolo.h"
#include "nodes/matmul.h"
#include "nodes/detection_output.h"
#include "nodes/reverse_sequence.h"
#include "nodes/pad.h"
#include "nodes/ctc_greedy_decoder_seq_len.h"
#include "nodes/reshape.h"
#include "nodes/psroi_pooling.h"
#include "nodes/memory.hpp"
#include "nodes/bin_conv.h"
#include "nodes/gather_elements.h"
#include "nodes/experimental_detectron_priorgridgenerator.h"
#include "nodes/tile.h"
#include "nodes/mathematics.h"
#include "nodes/normalize.h"
#include "nodes/proposal.h"
#include "nodes/tensoriterator.h"
#include "nodes/fullyconnected.h"
#include "nodes/extract_image_patches.h"
#include "nodes/ctc_loss.h"
#include "nodes/reorder.h"
#include "nodes/gather_nd.h"
#include "nodes/shuffle_channels.h"
#include "nodes/bucketize.h"
#include "nodes/space_to_depth.h"
#include "nodes/concat.h"
#include "nodes/softmax.h"
#include "nodes/space_to_batch.h"
#include "nodes/select.h"
#include "nodes/topk.h"
#include "nodes/broadcast.h"
#include "nodes/matrix_nms.h"
#include "nodes/mvn.h"
#include "nodes/gather.h"
#include "nodes/scatter_update.h"
#include "nodes/gather_tree.h"
#include "nodes/def_conv.h"
#include "nodes/embedding_bag_offset_sum.h"
#include "nodes/deconv.h"
#include "nodes/roi_pooling.h"
#include "nodes/range.h"
#include "nodes/split.h"
#include "nodes/one_hot.h"
#include "nodes/log_softmax.h"
#include "nodes/strided_slice.h"
#include "nodes/dft.h"
#include "nodes/non_max_suppression.h"
#include "nodes/convert.h"
#include "nodes/rnn.h"
#include "nodes/experimental_detectron_topkrois.h"
#include "nodes/cum_sum.h"
#include "nodes/depth_to_space.h"
#include "nodes/input.h"
#include "nodes/experimental_detectron_generate_proposals_single_image.h"
#include "nodes/embedding_bag_packed_sum.h"
#include "nodes/reduce.h"
#include "nodes/if.h"
#include "nodes/ctc_greedy_decoder.h"
#include "nodes/non_zero.h"
#include "nodes/color_convert.h"
#include "nodes/subgraph.h"
#include "nodes/priorbox.h"
#include "nodes/priorbox_clustered.h"

namespace ov {
namespace intel_cpu {

#define INTEL_CPU_NODE(__prim, __type) \
    registerNodeIfRequired(intel_cpu, __prim, __type, NodeImpl<__prim>)

Node::NodesFactory::NodesFactory()
    : Factory("NodesFactory") {
    using namespace node;
    INTEL_CPU_NODE(Generic, Type::Generic);
    INTEL_CPU_NODE(CumSum, Type::CumSum);
    INTEL_CPU_NODE(Convolution, Type::Convolution);
    INTEL_CPU_NODE(SpaceToBatch, Type::SpaceToBatch);
    INTEL_CPU_NODE(Lrn, Type::Lrn);
    INTEL_CPU_NODE(BatchToSpace, Type::BatchToSpace);
    INTEL_CPU_NODE(NormalizeL2, Type::NormalizeL2);
    INTEL_CPU_NODE(If, Type::If);
    INTEL_CPU_NODE(Proposal, Type::Proposal);
    INTEL_CPU_NODE(Broadcast, Type::Broadcast);
    INTEL_CPU_NODE(ExperimentalDetectronTopKROIs, Type::ExperimentalDetectronTopKROIs);
    INTEL_CPU_NODE(Reorder, Type::Reorder);
    INTEL_CPU_NODE(BinaryConvolution, Type::BinaryConvolution);
    INTEL_CPU_NODE(MatrixNms, Type::MatrixNms);
    INTEL_CPU_NODE(AdaptivePooling, Type::AdaptivePooling);
    INTEL_CPU_NODE(Pooling, Type::Pooling);
    INTEL_CPU_NODE(Reduce, Type::Reduce);
    INTEL_CPU_NODE(Eltwise, Type::Eltwise);
    INTEL_CPU_NODE(SoftMax, Type::Softmax);
    INTEL_CPU_NODE(EmbeddingBagPackedSum, Type::EmbeddingBagPackedSum);
    INTEL_CPU_NODE(Input, Type::Input);
    INTEL_CPU_NODE(Input, Type::Output);
    INTEL_CPU_NODE(MemoryInput, Type::MemoryInput);
    INTEL_CPU_NODE(MemoryOutput, Type::MemoryOutput);
    INTEL_CPU_NODE(Tile, Type::Tile);
    INTEL_CPU_NODE(DFT, Type::DFT);
    INTEL_CPU_NODE(GatherTree, Type::GatherTree);
    INTEL_CPU_NODE(SpaceToDepth, Type::SpaceToDepth);
    INTEL_CPU_NODE(FullyConnected, Type::FullyConnected);
    INTEL_CPU_NODE(CTCGreedyDecoder, Type::CTCGreedyDecoder);
    INTEL_CPU_NODE(Transpose, Type::Transpose);
    INTEL_CPU_NODE(DeformableConvolution, Type::DeformableConvolution);
    INTEL_CPU_NODE(ReorgYolo, Type::ReorgYolo);
    INTEL_CPU_NODE(EmbeddingSegmentsSum, Type::EmbeddingSegmentsSum);
    INTEL_CPU_NODE(Select, Type::Select);
    INTEL_CPU_NODE(ShapeOf, Type::ShapeOf);
    INTEL_CPU_NODE(ExperimentalDetectronGenerateProposalsSingleImage, Type::ExperimentalDetectronGenerateProposalsSingleImage);
    INTEL_CPU_NODE(ReverseSequence, Type::ReverseSequence);
    INTEL_CPU_NODE(FakeQuantize, Type::FakeQuantize);
    INTEL_CPU_NODE(NonMaxSuppression, Type::NonMaxSuppression);
    INTEL_CPU_NODE(ExperimentalDetectronPriorGridGenerator, Type::ExperimentalDetectronPriorGridGenerator);
    INTEL_CPU_NODE(GatherND, Type::GatherND);
    INTEL_CPU_NODE(LogSoftmax, Type::LogSoftmax);
    INTEL_CPU_NODE(PSROIPooling, Type::PSROIPooling);
    INTEL_CPU_NODE(RNN, Type::RNNCell);
    INTEL_CPU_NODE(RNN, Type::RNNSeq);
    INTEL_CPU_NODE(CTCLoss, Type::CTCLoss);
    INTEL_CPU_NODE(Split, Type::Split);
    INTEL_CPU_NODE(DetectionOutput, Type::DetectionOutput);
    INTEL_CPU_NODE(GatherElements, Type::GatherElements);
    INTEL_CPU_NODE(CTCGreedyDecoderSeqLen, Type::CTCGreedyDecoderSeqLen);
    INTEL_CPU_NODE(Bucketize, Type::Bucketize);
    INTEL_CPU_NODE(ExperimentalDetectronROIFeatureExtractor, Type::ExperimentalDetectronROIFeatureExtractor);
    INTEL_CPU_NODE(Math, Type::Math);
    INTEL_CPU_NODE(MultiClassNms, Type::MulticlassNms);
    INTEL_CPU_NODE(Convert, Type::Convert);
    INTEL_CPU_NODE(EmbeddingBagOffsetSum, Type::EmbeddingBagOffsetsSum);
    INTEL_CPU_NODE(Roll, Type::Roll);
    INTEL_CPU_NODE(Pad, Type::Pad);
    INTEL_CPU_NODE(Reshape, Type::Reshape);
    INTEL_CPU_NODE(MVN, Type::MVN);
    INTEL_CPU_NODE(MatMul, Type::MatMul);
    INTEL_CPU_NODE(ScatterUpdate, Type::ScatterUpdate);
    INTEL_CPU_NODE(ScatterUpdate, Type::ScatterElementsUpdate);
    INTEL_CPU_NODE(ScatterUpdate, Type::ScatterNDUpdate);
    INTEL_CPU_NODE(Interpolate, Type::Interpolate);
    INTEL_CPU_NODE(ROIPooling, Type::ROIPooling);
    INTEL_CPU_NODE(TensorIterator, Type::TensorIterator);
    INTEL_CPU_NODE(Concat, Type::Concatenation);
    INTEL_CPU_NODE(ExtractImagePatches, Type::ExtractImagePatches);
    INTEL_CPU_NODE(OneHot, Type::OneHot);
    INTEL_CPU_NODE(ExperimentalDetectronDetectionOutput, Type::ExperimentalDetectronDetectionOutput);
    INTEL_CPU_NODE(ROIAlign, Type::ROIAlign);
    INTEL_CPU_NODE(ShuffleChannels, Type::ShuffleChannels);
    INTEL_CPU_NODE(DepthToSpace, Type::DepthToSpace);
    INTEL_CPU_NODE(Deconvolution, Type::Deconvolution);
    INTEL_CPU_NODE(Gather, Type::Gather);
    INTEL_CPU_NODE(RegionYolo, Type::RegionYolo);
    INTEL_CPU_NODE(Range, Type::Range);
    INTEL_CPU_NODE(TopK, Type::TopK);
    INTEL_CPU_NODE(StridedSlice, Type::StridedSlice);
    INTEL_CPU_NODE(GRN, Type::GRN);
    INTEL_CPU_NODE(NonZero, Type::NonZero);
    INTEL_CPU_NODE(Snippet, Type::Subgraph);
    INTEL_CPU_NODE(ColorConvert, Type::ColorConvert);
    INTEL_CPU_NODE(PriorBox, Type::PriorBox);
    INTEL_CPU_NODE(PriorBoxClustered, Type::PriorBoxClustered);
}

#undef INTEL_CPU_NODE

}   // namespace intel_cpu
}   // namespace ov
