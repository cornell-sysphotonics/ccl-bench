// torchtitan/components/gin/cuda/gin_lsa.cu
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <c10/cuda/CUDAStream.h>

__global__ void gin_memcpy_peer_kernel(
    const uint8_t* __restrict__ src,
    uint8_t* __restrict__ dst,
    size_t nbytes) {
  size_t idx    = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = (size_t)blockDim.x * gridDim.x;
  for (size_t i = idx; i < nbytes; i += stride) {
    dst[i] = src[i];
  }
}

static inline int ceil_div(size_t a, int b) {
  return (int)((a + b - 1) / b);
}

// LSA-based all_gather: each rank's OUT window is registered; peer_ptrs_u64
// holds base LSA pointers to every rank's OUT window.
void gin_lsa_all_gather_cuda(
    torch::Tensor out_bytes,       // uint8, [world_size * chunk_bytes]
    torch::Tensor local_in_bytes,  // uint8, [chunk_bytes]
    torch::Tensor peer_ptrs_u64,   // uint64, [world_size], base ptrs to peers' OUT windows
    int64_t world_size,
    int64_t rank) {
  TORCH_CHECK(out_bytes.is_cuda(), "out_bytes must be CUDA");
  TORCH_CHECK(local_in_bytes.is_cuda(), "local_in_bytes must be CUDA");
  TORCH_CHECK(peer_ptrs_u64.is_cuda(), "peer_ptrs_u64 must be CUDA");
  TORCH_CHECK(out_bytes.scalar_type() == torch::kUInt8, "out_bytes must be uint8");
  TORCH_CHECK(local_in_bytes.scalar_type() == torch::kUInt8, "local_in_bytes must be uint8");
  TORCH_CHECK(peer_ptrs_u64.scalar_type() == torch::kUInt64, "peer_ptrs_u64 must be uint64");
  TORCH_CHECK(out_bytes.is_contiguous(), "out_bytes must be contiguous");
  TORCH_CHECK(local_in_bytes.is_contiguous(), "local_in_bytes must be contiguous");
  TORCH_CHECK(peer_ptrs_u64.is_contiguous(), "peer_ptrs_u64 must be contiguous");
  TORCH_CHECK(rank >= 0 && rank < world_size, "bad rank");

  const size_t chunk_bytes = (size_t)local_in_bytes.numel();
  TORCH_CHECK(out_bytes.numel() == (int64_t)chunk_bytes * world_size,
              "out_bytes.numel must equal world_size * local_in_bytes.numel");

  cudaStream_t stream = c10::cuda::getDefaultCUDAStream().stream();

  // 1) Copy my local chunk into my slice of OUT (local memory)
  {
    uint8_t* out_base = out_bytes.data_ptr<uint8_t>();
    uint8_t* dst      = out_base + (size_t)rank * chunk_bytes;
    const uint8_t* src = local_in_bytes.data_ptr<uint8_t>();

    int threads = 256;
    int blocks  = ceil_div(chunk_bytes, threads);
    if (blocks > 4096) blocks = 4096;

    gin_memcpy_peer_kernel<<<blocks, threads, 0, stream>>>(
        src, dst, chunk_bytes);
    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "gin_memcpy_peer_kernel (self) failed: ",
                cudaGetErrorString(err));
  }

  // 2) For each peer p != rank: read peer’s OUT slice [p] via LSA pointer,
  //    copy into my out slice [p].
  const uint64_t* peer_ptrs = peer_ptrs_u64.data_ptr<uint64_t>();
  uint8_t* my_out_base      = out_bytes.data_ptr<uint8_t>();

  int threads = 256;
  int blocks  = ceil_div(chunk_bytes, threads);
  if (blocks > 4096) blocks = 4096;

  for (int p = 0; p < (int)world_size; ++p) {
    if (p == (int)rank) continue;

    const uint8_t* peer_out_base =
        reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(peer_ptrs[p]));

    const uint8_t* src = peer_out_base + (size_t)p * chunk_bytes;  // peer's slice p
    uint8_t* dst       = my_out_base + (size_t)p * chunk_bytes;    // my slice p

    gin_memcpy_peer_kernel<<<blocks, threads, 0, stream>>>(
        src, dst, chunk_bytes);
    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "gin_memcpy_peer_kernel (peer copy) failed: ",
                cudaGetErrorString(err));
  }
}
