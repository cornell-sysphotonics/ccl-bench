// torchtitan/components/gin/cuda/gin_all_gather.cpp
#include <torch/extension.h>

// Exposed CUDA launcher (could be gin_lsa_all_gather_cuda directly)
void gin_all_gather_into_tensor_cuda(
    torch::Tensor out_bytes,
    torch::Tensor local_in_bytes,
    torch::Tensor peer_ptrs_u64,
    int64_t world_size,
    int64_t rank);

// Higher-level wrapper that accepts arbitrary dtypes and flattens to bytes.
void gin_all_gather_into_tensor(
    torch::Tensor out,
    torch::Tensor local_in,
    torch::Tensor peer_ptrs_u64,
    int64_t world_size,
    int64_t rank) {
  TORCH_CHECK(out.is_cuda() && local_in.is_cuda(),
              "out and local_in must be CUDA");
  TORCH_CHECK(out.is_contiguous() && local_in.is_contiguous(),
              "out and local_in must be contiguous");
  TORCH_CHECK(peer_ptrs_u64.is_cuda(), "peer_ptrs_u64 must be CUDA");
  TORCH_CHECK(peer_ptrs_u64.scalar_type() == torch::kUInt64,
              "peer_ptrs_u64 must be uint64");

  auto out_bytes      = out.view(torch::kUInt8);
  auto local_in_bytes = local_in.view(torch::kUInt8);

  gin_all_gather_into_tensor_cuda(
      out_bytes, local_in_bytes, peer_ptrs_u64, world_size, rank);
}

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//   m.def("all_gather", &gin_all_gather_into_tensor, "GIN LSA all_gather");
// }
