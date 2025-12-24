// torchtitan/components/gin/cuda/gin_lsa.cpp
#define NCCL_DEVICE_API
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <nccl_device/core.h>

#include <mutex>
#include <stdexcept>
#include <vector>
#include <cstring>

// ---------- helpers ----------
#define CHECK_CUDA(call)                                                      \
  do {                                                                        \
    cudaError_t e = (call);                                                   \
    if (e != cudaSuccess) {                                                   \
      throw std::runtime_error(std::string("CUDA error: ") +                  \
                               cudaGetErrorString(e));                        \
    }                                                                         \
  } while (0)

#define CHECK_NCCL(call)                                                      \
  do {                                                                        \
    ncclResult_t r = (call);                                                  \
    if (r != ncclSuccess) {                                                   \
      throw std::runtime_error(std::string("NCCL error: ") +                  \
                               ncclGetErrorString(r));                        \
    }                                                                         \
  } while (0)

static std::mutex g_mu;

struct GinLSAContext {
  bool initialized       = false;
  int world_size         = -1;
  int rank               = -1;
  ncclComm_t comm        = nullptr;

  bool   window_registered = false;
  size_t window_bytes      = 0;
  void*  window_ptr        = nullptr;
  ncclWindow_t win{};
};

static GinLSAContext g_ctx;

// Kernel launcher implemented in gin_all_gather.cu
void gin_lsa_all_gather_cuda(
    torch::Tensor out_bytes,
    torch::Tensor local_in_bytes,
    torch::Tensor peer_ptrs_u64,
    int64_t world_size,
    int64_t rank);

// Kernel launcher implemented in gin_reduce_scatter.cu
void gin_lsa_reduce_scatter_cuda(
    torch::Tensor out_bytes,
    torch::Tensor in_bytes,
    torch::Tensor peer_ptrs_u64,
    int64_t world_size,
    int64_t rank);

// Utility: require CUDA + contiguous + uint8
static void check_cuda_contig_uint8(torch::Tensor t, const char* name) {
  if (!t.is_cuda())
    throw std::runtime_error(std::string(name) + " must be CUDA tensor");
  if (!t.is_contiguous())
    throw std::runtime_error(std::string(name) + " must be contiguous");
  if (t.dtype() != torch::kUInt8)
    throw std::runtime_error(std::string(name) + " must be uint8 view tensor");
}

static void ensure_initialized() {
  if (!g_ctx.initialized)
    throw std::runtime_error("GIN LSA not initialized. Call init_lsa(world, rank, unique_id_bytes) first.");
}

// ---- NCCL init / finalize ----

void init_lsa(int64_t world_size, int64_t rank, torch::Tensor unique_id_bytes) {
  std::lock_guard<std::mutex> lk(g_mu);
  if (g_ctx.initialized) return;

  if (!unique_id_bytes.defined() ||
      unique_id_bytes.numel() != (int64_t)sizeof(ncclUniqueId)) {
    throw std::runtime_error("unique_id_bytes must be a CPU ByteTensor of size sizeof(ncclUniqueId)");
  }
  if (unique_id_bytes.device().is_cuda()) {
    throw std::runtime_error("unique_id_bytes must be on CPU");
  }
  if (unique_id_bytes.dtype() != torch::kUInt8) {
    throw std::runtime_error("unique_id_bytes must be uint8");
  }

  ncclUniqueId id{};
  std::memcpy(&id, unique_id_bytes.data_ptr(), sizeof(ncclUniqueId));

  g_ctx.world_size = (int)world_size;
  g_ctx.rank       = (int)rank;
  CHECK_NCCL(ncclCommInitRank(&g_ctx.comm, g_ctx.world_size, id, g_ctx.rank));
  g_ctx.initialized = true;
}

torch::Tensor get_unique_id() {
  ncclUniqueId id{};
  CHECK_NCCL(ncclGetUniqueId(&id));
  auto out = torch::empty(
      {(int64_t)sizeof(ncclUniqueId)},
      torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
  std::memcpy(out.data_ptr(), &id, sizeof(ncclUniqueId));
  return out;
}

void finalize_lsa() {
  std::lock_guard<std::mutex> lk(g_mu);
  if (!g_ctx.initialized) return;

  if (g_ctx.window_registered) {
    (void)ncclCommWindowDeregister(g_ctx.comm, g_ctx.win);
    g_ctx.window_registered = false;
    g_ctx.window_bytes      = 0;
    g_ctx.window_ptr        = nullptr;
  }

  (void)ncclCommDestroy(g_ctx.comm);
  g_ctx.initialized = false;
}

// ---- LSA window management ----

// Register an LSA window for OUT buffer
void register_out_window(torch::Tensor out_bytes) {
  std::lock_guard<std::mutex> lk(g_mu);
  ensure_initialized();
  check_cuda_contig_uint8(out_bytes, "out_bytes");

  size_t bytes = (size_t)out_bytes.numel();
  void* ptr    = (void*)out_bytes.data_ptr();

  // Re-register if pointer or size changed
  if (g_ctx.window_registered) {
    if (g_ctx.window_ptr == ptr && g_ctx.window_bytes == bytes) {
      return;  // already registered
    }
    (void)ncclCommWindowDeregister(g_ctx.comm, g_ctx.win);
    g_ctx.window_registered = false;
  }

  CHECK_NCCL(ncclCommWindowRegister(
      g_ctx.comm,
      ptr,
      bytes,
      &g_ctx.win,
      NCCL_WIN_COLL_SYMMETRIC));

  g_ctx.window_registered = true;
  g_ctx.window_bytes      = bytes;
  g_ctx.window_ptr        = ptr;
}

// Get LSA peer pointers for the registered OUT window.
// Returns a CUDA uint64 tensor of size world_size; each entry is the base
// pointer for that peer’s OUT window.
torch::Tensor get_peer_ptrs_u64() {
  std::lock_guard<std::mutex> lk(g_mu);
  ensure_initialized();
  if (!g_ctx.window_registered)
    throw std::runtime_error("Window not registered. Call register_out_window(out_bytes) first.");

  auto opts      = torch::TensorOptions().dtype(torch::kUInt64).device(torch::kCUDA);
  auto peer_ptrs = torch::empty({(int64_t)g_ctx.world_size}, opts);

  std::vector<uint64_t> host(g_ctx.world_size);
  for (int p = 0; p < g_ctx.world_size; ++p) {
    void* pp = ncclGetLsaPointer(g_ctx.win, /*offset=*/0, /*lsaPeer=*/p);
    host[p]  = (uint64_t)pp;
  }

  auto host_t = torch::from_blob(
                    host.data(),
                    {(int64_t)g_ctx.world_size},
                    torch::TensorOptions().dtype(torch::kUInt64).device(torch::kCPU))
                    .clone();
  peer_ptrs.copy_(host_t, /*non_blocking=*/false);
  return peer_ptrs;
}

// ---- All-gather front-end ----

void all_gather_into_tensor(torch::Tensor out, torch::Tensor local_in) {
  ensure_initialized();

  if (!out.is_cuda() || !local_in.is_cuda())
    throw std::runtime_error("out/local_in must be CUDA tensors");
  if (!out.is_contiguous() || !local_in.is_contiguous())
    throw std::runtime_error("out/local_in must be contiguous");

  if (out.numel() != local_in.numel() * g_ctx.world_size) {
    throw std::runtime_error("out.numel must equal local_in.numel * world_size");
  }

  // Byte views for window registration + copy
  auto out_bytes = out.view(torch::kUInt8);
  auto in_bytes  = local_in.view(torch::kUInt8);

  // Window is registered on OUT buffer
  register_out_window(out_bytes);

  // Get peer pointers (LSA base pointers)
  auto peer_ptrs_u64 = get_peer_ptrs_u64();

  // Launch CUDA copy kernel
  gin_lsa_all_gather_cuda(out_bytes, in_bytes, peer_ptrs_u64,
                          g_ctx.world_size, g_ctx.rank);
}

// ---- Reduce-scatter front-end (sum) ----

void reduce_scatter_into_tensor(torch::Tensor out, torch::Tensor in) {
    ensure_initialized();
  
    if (!out.is_cuda() || !in.is_cuda())
      throw std::runtime_error("out/in must be CUDA tensors");
    if (!out.is_contiguous() || !in.is_contiguous())
      throw std::runtime_error("out/in must be contiguous");
  
    if (in.numel() != out.numel() * g_ctx.world_size) {
      throw std::runtime_error("in.numel must equal out.numel * world_size");
    }
  
    // Byte views for window registration + copy
    auto out_bytes = out.view(torch::kUInt8);
    auto in_bytes  = in.view(torch::kUInt8);
  
    // Register window on IN buffer (since all ranks will read each other's IN)
    register_out_window(in_bytes);  // you may want a separate register_in_window later
  
    // Get peer pointers (LSA base pointers to IN windows)
    auto peer_ptrs_u64 = get_peer_ptrs_u64();
  
    // Launch CUDA reduce_scatter kernel
    gin_lsa_reduce_scatter_cuda(out_bytes, in_bytes, peer_ptrs_u64,
                                g_ctx.world_size, g_ctx.rank);
}
  
// ---- pybind ----

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("get_unique_id", &get_unique_id,
            "Get NCCL unique ID (CPU ByteTensor)");
    m.def("init_lsa", &init_lsa,
            "Init NCCL comm for LSA (world, rank, unique_id_bytes)");
    m.def("finalize_lsa", &finalize_lsa, "Finalize LSA");
    m.def("register_out_window", &register_out_window,
            "Register LSA window for out buffer (uint8 CUDA tensor)");
    m.def("get_peer_ptrs_u64", &get_peer_ptrs_u64,
            "Get LSA peer ptrs for registered out window");
    m.def("all_gather_into_tensor", &all_gather_into_tensor,
            "LSA all_gather_into_tensor");
    m.def("reduce_scatter_into_tensor", &reduce_scatter_into_tensor,
        "LSA reduce_scatter_into_tensor (sum)");
}


// #include <torch/extension.h>

// // forward declaration of launcher implemented in gin_lsa.cu
// void gin_lsa_launcher(torch::Tensor x);

// TORCH_LIBRARY(gin_ext, m) {
//     m.def("lsa", &gin_lsa_launcher, "GIN LSA (device-side)");
// }
