// ===== Includes =====
#include <nccl.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <assert.h>

#define CHECK_CUDA(call)                                                    \
  do {                                                                       \
    cudaError_t err = (call);                                               \
    if (err != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA Error %s:%d: %s\n", __FILE__, __LINE__,          \
              cudaGetErrorString(err));                                      \
      exit(EXIT_FAILURE);                                                    \
    }                                                                        \
  } while (0)

#define CHECK_NCCL(call)                                                    \
  do {                                                                       \
    ncclResult_t res = (call);                                              \
    if (res != ncclSuccess) {                                                \
      fprintf(stderr, "NCCL Error %s:%d: %s\n", __FILE__, __LINE__,         \
              ncclGetErrorString(res));                                      \
      exit(EXIT_FAILURE);                                                    \
    }                                                                        \
  } while (0)

// ===== Device Kernel =====
// Simple kernel that uses the NCCL device communicator
__global__ void deviceAllreduce(ncclDevComm devComm,
                                ncclWindow_t win,
                                float* data,
                                size_t count) {
  // A single blocking barrier for this CTA (block)
  // Each CTA waits here before accessing shared buffers
  ncclLsaBarrierSession session;
  ncclLsaBarrierInit(&session, devComm);

  // Example: all blocks check and progress communication
  // In real use, you’d use ncclGetLsaPointer on symmetric buffers
  // and device collaborative calls for actual data exchange.
  ncclLsaBarrier(&session, devComm, 0);

  // For demo: add some trivial modification
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < count) {
    data[idx] *= 2.0f;  // just a dummy operation
  }
}

// ===== Main =====
int main(int argc, char* argv[]) {
  const int nRanks = 4;
  int rank;
  cudaStream_t stream;
  ncclComm_t comm;

  // Expect NCCL unique ID in environment (e.g., via MPI or torchrun)
  ncclUniqueId id;
  // For simplicity: assume rank env vars exist
  rank = atoi(getenv("OMPI_COMM_WORLD_RANK"));
  int worldSize = atoi(getenv("OMPI_COMM_WORLD_SIZE"));

  assert(worldSize == nRanks && "This example assumes 4 GPUs");

  CHECK_NCCL(ncclGetUniqueId(&id));
  CHECK_NCCL(ncclCommInitRank(&comm, nRanks, id, rank));

  // Allocate symmetric buffer (must be same on all ranks)
  size_t count = 1 << 10;
  float* buff;
  CHECK_CUDA(cudaMalloc(&buff, count * sizeof(float)));

  // Initialize buffer with some values
  CHECK_CUDA(cudaMemset(buff, rank, count * sizeof(float)));

  // Register a symmetric window
  ncclWindow_t window;
  CHECK_NCCL(ncclCommWindowRegister(comm, buff,
                              count * sizeof(float),
                              &window,
                              NCCL_WIN_COLL_SYMMETRIC));

  // Create a device communicator
  ncclDevComm devComm;
  ncclDevCommRequirements reqs;
  memset(&reqs, 0, sizeof(reqs));
  // we’ll use 1 barrier per CTA in this example
  reqs.lsaBarrierCount = 1;
  CHECK_NCCL(ncclDevCommCreate(comm, &reqs, &devComm));

  // Create a CUDA stream
  CHECK_CUDA(cudaStreamCreate(&stream));

  // Launch a simple kernel that uses the NCCL device communicator
  deviceAllreduce<<<1, 256, 0, stream>>>(devComm, window, buff, count);
  CHECK_CUDA(cudaStreamSynchronize(stream));

  // Clean up
  CHECK_NCCL(ncclDevCommDestroy(devComm));
  CHECK_NCCL(ncclCommWindowDeregister(window));
  CHECK_NCCL(ncclCommDestroy(comm));
  CHECK_CUDA(cudaFree(buff));

  printf("Rank %d device Allreduce completed\n", rank);
  return 0;
}