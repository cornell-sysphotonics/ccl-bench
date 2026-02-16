// torchtitan/components/gin/cuda/gin_reduce_scatter.cu
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <c10/cuda/CUDAStream.h>

__global__ void gin_elementwise_add_kernel(
    const float* __restrict__ in,
    float* __restrict__ out,
    size_t n) {
  size_t idx    = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = (size_t)blockDim.x * gridDim.x;
  for (size_t i = idx; i < n; i += stride) {
    out[i] += in[i];
  }
}

__global__ void gin_memcpy_peer_kernel_rs(
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

// LSA-based reduce_scatter (sum):
// Input:  in_bytes is this rank's full input buffer [world_size * chunk_bytes].
// Output: out_bytes is this rank's shard [chunk_bytes].
// Algorithm (simple):
// 1) Copy my local segment (rank slice) into a temp buffer.
// 2) For each peer p != my_rank, pull that peer's segment for my rank via LSA
//    and add into temp.
// 3) Write temp into out_bytes.
void gin_lsa_reduce_scatter_cuda(
    torch::Tensor out_bytes,      // uint8, [chunk_bytes]
    torch::Tensor in_bytes,       // uint8, [world_size * chunk_bytes]
    torch::Tensor peer_ptrs_u64,  // uint64, [world_size], base ptrs to peers' IN windows
    int64_t world_size,
    int64_t rank) {
  TORCH_CHECK(out_bytes.is_cuda() && in_bytes.is_cuda(),
              "out_bytes and in_bytes must be CUDA");
  TORCH_CHECK(peer_ptrs_u64.is_cuda(), "peer_ptrs_u64 must be CUDA");
  TORCH_CHECK(out_bytes.scalar_type() == torch::kUInt8, "out_bytes must be uint8");
  TORCH_CHECK(in_bytes.scalar_type() == torch::kUInt8, "in_bytes must be uint8");
  TORCH_CHECK(peer_ptrs_u64.scalar_type() == torch::kUInt64,
              "peer_ptrs_u64 must be uint64");
  TORCH_CHECK(out_bytes.is_contiguous() && in_bytes.is_contiguous() &&
                  peer_ptrs_u64.is_contiguous(),
              "tensors must be contiguous");
  TORCH_CHECK(rank >= 0 && rank < world_size, "bad rank");

  const size_t chunk_bytes = (size_t)out_bytes.numel();
  TORCH_CHECK(in_bytes.numel() == (int64_t)chunk_bytes * world_size,
              "in_bytes.numel must equal world_size * out_bytes.numel");

  cudaStream_t stream = c10::cuda::getDefaultCUDAStream().stream();

  // We'll treat data as float for reduction; you can generalize later.
  TORCH_CHECK(chunk_bytes % sizeof(float) == 0,
              "reduce_scatter prototype assumes float-aligned data");
  size_t chunk_elems = chunk_bytes / sizeof(float);

  // Temp buffer for accumulation
  auto temp = torch::empty_like(out_bytes);
  uint8_t* temp_u8 = temp.data_ptr<uint8_t>();
  float* temp_f    = reinterpret_cast<float*>(temp_u8);

  // 1) Copy my own segment (rank slice) into temp
  const uint8_t* in_base = in_bytes.data_ptr<uint8_t>();
  const uint8_t* my_src  = in_base + (size_t)rank * chunk_bytes;

  int threads = 256;
  int blocks  = ceil_div(chunk_bytes, threads);
  if (blocks > 4096) blocks = 4096;

  gin_memcpy_peer_kernel_rs<<<blocks, threads, 0, stream>>>(
      my_src, temp_u8, chunk_bytes);
  auto err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess,
              "gin_memcpy_peer_kernel_rs (self) failed: ",
              cudaGetErrorString(err));

  // 2) Accumulate contributions from all other ranks
  const uint64_t* peer_ptrs = peer_ptrs_u64.data_ptr<uint64_t>();

  for (int p = 0; p < (int)world_size; ++p) {
    if (p == (int)rank) continue;

    const uint8_t* peer_in_base =
        reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(peer_ptrs[p]));

    // Peer’s segment for my rank is at offset rank * chunk_bytes
    const uint8_t* peer_seg_u8 =
        peer_in_base + (size_t)rank * chunk_bytes;
    const float* peer_seg_f =
        reinterpret_cast<const float*>(peer_seg_u8);

    threads = 256;
    blocks  = ceil_div(chunk_elems, threads);
    if (blocks > 4096) blocks = 4096;

    gin_elementwise_add_kernel<<<blocks, threads, 0, stream>>>(
        peer_seg_f, temp_f, chunk_elems);
    err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "gin_elementwise_add_kernel failed: ",
                cudaGetErrorString(err));
  }

  // 3) Write temp into out_bytes
  gin_memcpy_peer_kernel_rs<<<blocks, threads, 0, stream>>>(
      temp_u8, out_bytes.data_ptr<uint8_t>(), chunk_bytes);
  auto err2 = cudaGetLastError();
  TORCH_CHECK(err2 == cudaSuccess,
              "gin_memcpy_peer_kernel_rs (final copy) failed: ",
              cudaGetErrorString(err2));
}
