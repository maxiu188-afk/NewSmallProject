#include <torch/extension.h>

torch::Tensor w4a8_grouped_int32_cuda(
    torch::Tensor activations, torch::Tensor packed_weights, int64_t group_size);

torch::Tensor w4a8_grouped_int32(
    torch::Tensor activations, torch::Tensor packed_weights, int64_t group_size) {
  TORCH_CHECK(activations.is_cuda(), "activations must be a CUDA tensor");
  TORCH_CHECK(packed_weights.is_cuda(), "packed_weights must be a CUDA tensor");
  TORCH_CHECK(activations.is_contiguous(), "activations must be contiguous");
  TORCH_CHECK(packed_weights.is_contiguous(), "packed_weights must be contiguous");
  TORCH_CHECK(activations.scalar_type() == torch::kInt8, "activations must have dtype int8");
  TORCH_CHECK(packed_weights.scalar_type() == torch::kUInt8, "packed_weights must have dtype uint8");
  TORCH_CHECK(activations.dim() == 2, "activations must have shape [tokens, in_features]");
  TORCH_CHECK(packed_weights.dim() == 2, "packed_weights must have shape [out_features, in_features / 2]");
  TORCH_CHECK(activations.size(1) == packed_weights.size(1) * 2,
              "packed weight width must represent activation in_features");
  TORCH_CHECK(group_size > 0 && group_size % 2 == 0, "group_size must be positive and even");
  TORCH_CHECK(activations.size(1) % group_size == 0,
              "in_features must be divisible by group_size");
  TORCH_CHECK(activations.size(1) % 32 == 0,
              "W4A8 kernel requires in_features divisible by 32");
  return w4a8_grouped_int32_cuda(activations, packed_weights, group_size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("w4a8_grouped_int32", &w4a8_grouped_int32,
             "W4A8 grouped integer accumulators (CUDA)");
}
