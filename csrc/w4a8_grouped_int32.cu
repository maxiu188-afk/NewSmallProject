#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <cstdint>

namespace {

constexpr int kThreads = 256;

__global__ void w4a8_grouped_int32_kernel(
    const int8_t* activations,
    const uint8_t* packed_weights,
    int32_t* output,
    int64_t tokens,
    int64_t out_features,
    int64_t in_features,
    int64_t group_size,
    int64_t groups_per_row) {
  const int64_t token = blockIdx.x;
  const int64_t out_feature = blockIdx.y;
  const int64_t group = blockIdx.z;
  const int64_t group_start = group * group_size;
  const int64_t activation_offset = token * in_features;
  const int64_t weight_offset = out_feature * (in_features / 2);

  int32_t partial = 0;
  for (int64_t feature = threadIdx.x; feature < group_size; feature += blockDim.x) {
    const int64_t input_index = group_start + feature;
    const uint8_t packed = packed_weights[weight_offset + input_index / 2];
    const uint8_t nibble = input_index % 2 == 0 ? packed & 0x0F : packed >> 4;
    const int32_t weight = nibble >= 8 ? static_cast<int32_t>(nibble) - 16 : static_cast<int32_t>(nibble);
    partial += weight * static_cast<int32_t>(activations[activation_offset + input_index]);
  }

  __shared__ int32_t reduction[kThreads];
  reduction[threadIdx.x] = partial;
  __syncthreads();
  for (int stride = kThreads / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      reduction[threadIdx.x] += reduction[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    output[(token * out_features + out_feature) * groups_per_row + group] = reduction[0];
  }
}

}  // namespace

torch::Tensor w4a8_grouped_int32_cuda(
    torch::Tensor activations, torch::Tensor packed_weights, int64_t group_size) {
  const int64_t tokens = activations.size(0);
  const int64_t in_features = activations.size(1);
  const int64_t out_features = packed_weights.size(0);
  const int64_t groups_per_row = in_features / group_size;
  auto output = torch::empty(
      {tokens, out_features, groups_per_row},
      torch::TensorOptions().device(activations.device()).dtype(torch::kInt32));

  const dim3 blocks(tokens, out_features, groups_per_row);
  w4a8_grouped_int32_kernel<<<blocks, kThreads, 0, at::cuda::getDefaultCUDAStream()>>>(
      activations.data_ptr<int8_t>(),
      packed_weights.data_ptr<uint8_t>(),
      output.data_ptr<int32_t>(),
      tokens,
      out_features,
      in_features,
      group_size,
      groups_per_row);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
