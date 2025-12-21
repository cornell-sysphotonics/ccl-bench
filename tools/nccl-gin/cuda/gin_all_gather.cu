// torchtitan/components/gin/cuda/gin_all_gather.cu
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

// Forward declaration from gin_lsa.cu
void gin_lsa_all_gather_cuda(
    torch::Tensor out_bytes,
    torch::Tensor local_in_bytes,
    torch::Tensor peer_ptrs_u64,
    int64_t world_size,
    int64_t rank);

// Optional convenience CUDA entry that assumes byte views + LSA peer ptrs.
void gin_all_gather_into_tensor_cuda(
    torch::Tensor out_bytes,
    torch::Tensor local_in_bytes,
    torch::Tensor peer_ptrs_u64,
    int64_t world_size,
    int64_t rank) {
  gin_lsa_all_gather_cuda(out_bytes, local_in_bytes, peer_ptrs_u64,
                          world_size, rank);
}
