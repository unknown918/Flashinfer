/*
 * Copyright (c) 2023 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef FLASHINFER_DECODE_CUH_
#define FLASHINFER_DECODE_CUH_
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <iostream>
#include <numeric>

#include "cp_async.cuh"
#include "math.cuh"
#include "pos_enc.cuh"
#include "utils.cuh"
#include "vec_dtypes.cuh"

namespace flashinfer {
DEFINE_HAS_MEMBER(decode_maybe_q_rope_offset)

namespace cg = cooperative_groups;
using cp_async::PrefetchMode;
using cp_async::SharedMemFillMode;

namespace {
/*!
 * \brief Load k tile from smem and compute qk
 * \tparam vec_size A template integer indicates the vector size
 * \tparam bdx A template integer indicates the block size in x dimension
 * \tparam tile_size A template integer indicates the tile size per (bdx * bdy) threads.
 * \tparam DType A template type indicates the input data type
 * \param smem A pointer to the start of shared memory
 * \param q_vec A vector of float indicates the thread-local query vector
 * \param kv_idx_base A current key-value cache indices
 * \param iter_base current iter idx
 * \param iter_bound maximum iterations
 * \param o A pointer to the output
 * \param tx Thread Id x
 * \param ty Thread Id y
 * \param tz Thread Id z
 */
template <uint32_t vec_size, uint32_t bdx, uint32_t tile_size, typename DType>
__device__ __forceinline__ void compute_qk(const DType* smem, const vec_t<float, vec_size>& q_vec,
                                           uint32_t kv_idx_base, uint32_t iter_base,
                                           uint32_t iter_bound, DType* o, const uint32_t tx,
                                           const uint32_t ty, const uint32_t tz) {
#pragma unroll
  for (uint32_t j = 0; j < tile_size; ++j) {
    vec_t<float, vec_size> k_vec;
    k_vec.cast_load(smem + (j * bdx + tx) * vec_size);
    float st = 0.f;
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      st += q_vec[i] * k_vec[i];
    }
#pragma unroll
    for (uint32_t offset = bdx / 2; offset > 0; offset /= 2) {
      st += math::shfl_xor_sync(st, offset);
    }
    // Store out
    if (iter_base + tz * tile_size + j < iter_bound) {
      if (tx == 0) {
        o[kv_idx_base + tz * tile_size + j] = static_cast<DType>(st);
      }
    }
  }
}
}  // namespace

template <typename DType, uint32_t num_stages_smem, uint32_t tile_size_per_bdx, uint32_t vec_size,
          uint32_t bdx, uint32_t bdy, uint32_t bdz>
__global__ void EstimationKernel(DType* q, DType* k, uint32_t num_qo_heads, uint32_t num_kv_heads,
                                 uint32_t seq_len, uint32_t num_pages, uint32_t kv_chunk_size,
                                 DType* out) {
  auto block = cg::this_thread_block();
  auto grid = cg::this_grid();

  constexpr uint32_t head_dim = bdx * vec_size;

  uint32_t kv_chunk_idx = blockIdx.x;
  uint32_t kv_head_idx = blockIdx.y;

  uint32_t qo_head_idx = kv_head_idx * bdy + threadIdx.y;
  uint32_t chunk_start = kv_chunk_idx * kv_chunk_size;
  uint32_t chunk_end = min(chunk_start + kv_chunk_size, seq_len);

  uint32_t kv_stride_h = head_dim;
  uint32_t kv_stride_n = num_kv_heads * head_dim;

  extern __shared__ uint8_t smem[];

  DType* k_smem = (DType*)smem;

  uint32_t tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
  vec_t<float, vec_size> q_vec;
  q_vec.cast_load(q + qo_head_idx * head_dim + tx * vec_size);
  block.sync();

  // preload k tiles
  uint32_t producer_kv_idx_base = chunk_start;
  constexpr uint32_t vec_bits = sizeof(DType) * vec_size * 8;
#pragma unroll
  for (uint32_t iter = 0; iter < num_stages_smem; ++iter) {
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kNoFill>(
          k_smem + (((iter * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) * head_dim +
              tx * vec_size,
          k + (producer_kv_idx_base + (tz * bdy + ty) * tile_size_per_bdx + j) * kv_stride_n +
              kv_head_idx * kv_stride_h + tx * vec_size,
          producer_kv_idx_base + (tz * bdy + ty) * tile_size_per_bdx + j < chunk_end);
    }
    cp_async::commit_group();
    producer_kv_idx_base += bdy * bdz * tile_size_per_bdx;
  }

  // pipelining k/v tiles loading and state updating
  uint32_t consumer_kv_idx_base = chunk_start, stage_idx = 0;

#pragma unroll 2
  for (uint32_t iter = 0; iter < ceil_div(kv_chunk_size, tile_size_per_bdx * bdy * bdz); ++iter) {
    // compute qk
    cp_async::wait_group<num_stages_smem - 1>();
    block.sync();
    compute_qk<vec_size, bdx, bdy * tile_size_per_bdx, DType>(
        k_smem + (stage_idx * bdz + tz) * bdy * tile_size_per_bdx * head_dim, q_vec,
        consumer_kv_idx_base, iter * bdy * tile_size_per_bdx * bdz, kv_chunk_size,
        out + num_pages * qo_head_idx, tx, ty, tz);
    block.sync();
    // load k
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kNoFill>(
          k_smem + (((stage_idx * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) * head_dim +
              tx * vec_size,
          k + (producer_kv_idx_base + (tz * bdy + ty) * tile_size_per_bdx + j) * kv_stride_n +
              kv_head_idx * kv_stride_h + tx * vec_size,
          producer_kv_idx_base + (tz * bdy + ty) * tile_size_per_bdx + j < chunk_end);
    }
    cp_async::commit_group();

    stage_idx = (stage_idx + 1) % num_stages_smem;
    producer_kv_idx_base += tile_size_per_bdx * bdy * bdz;
    consumer_kv_idx_base += tile_size_per_bdx * bdy * bdz;
  }
  cp_async::wait_group<0>();
  block.sync();
}

/*!
 * \brief Get the heuristic number of threads per threadblock
 * \param group_size The number of qo heads that maps to the same kv head in GQA.
 * \param sizeof_dtype The size (in terms of bytes) of the input data type
 */
constexpr uint32_t get_heuristic_num_threads(uint32_t group_size, uint32_t sizeof_dtype) {
  if (group_size == 8U) {
    if (sizeof_dtype == 1U) {
      return 256U;  // not enough registers for 512 threads
    } else {
      return 512U;
    }
  } else {
    return 128U;
  }
}

/*!
 * \brief Calculate the estimated attention scores
 * \tparam HEAD_DIM A integer indicates the head dimension
 * \tparam DType A template type indicates the data type
 * \param q The query matrix, shape: [num_qo_heads, head_dim]
 * \param k The key matrix in kv-cache, shape: [seq_len, num_kv_heads, head_dim]
 * \param num_qo_heads A integer indicates the number of heads of query and output
 * \param num_kv_heads A integer indicates the number of heads of key and value
 * \param seq_len A integer indicates the sequence length
 * \param stream The cuda stream to launch the kernel
 * \return status Indicates whether CUDA calls are successful
 */
template <uint32_t HEAD_DIM, typename DType>
cudaError_t Estimate(DType* __restrict__ q, DType* __restrict__ k, DType* __restrict__ out,
                     uint32_t num_qo_heads, uint32_t num_kv_heads, uint32_t seq_len,
                     uint32_t num_pages, cudaStream_t stream) {
  constexpr uint32_t vec_size = std::max(16UL / sizeof(DType), HEAD_DIM / 32UL);
  constexpr uint32_t bdx = HEAD_DIM / vec_size;
  auto compute_capacity = GetCudaComputeCapability();
  static_assert(bdx <= 32U);
  DISPATCH_GQA_GROUP_SIZE(num_qo_heads / num_kv_heads, GROUP_SIZE, {
    constexpr uint32_t bdy = GROUP_SIZE;
    constexpr uint32_t num_threads =
        std::max(get_heuristic_num_threads(GROUP_SIZE, sizeof(DType)), bdx * bdy);
    constexpr uint32_t bdz = num_threads / (bdx * bdy);
    constexpr uint32_t tile_size_per_bdx = GROUP_SIZE == 1 ? (sizeof(DType) == 1 ? 2U : 8U) : 1U;
    DISPATCH_COMPUTE_CAP_DECODE_NUM_STAGES_SMEM(compute_capacity, NUM_STAGES_SMEM, {
      const uint32_t smem_size =
          NUM_STAGES_SMEM * bdy * tile_size_per_bdx * bdz * HEAD_DIM * sizeof(DType);
      auto kernel =
          EstimationKernel<DType, NUM_STAGES_SMEM, tile_size_per_bdx, vec_size, bdx, bdy, bdz>;
      FLASHINFER_CUDA_CALL(
          cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

      int num_blocks_per_sm = 0;
      int num_sm = 0;
      int dev_id = 0;
      FLASHINFER_CUDA_CALL(cudaGetDevice(&dev_id));
      FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, dev_id));
      FLASHINFER_CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, kernel,
                                                                         num_threads, smem_size));

      uint32_t max_grid_size = uint32_t(num_blocks_per_sm) * uint32_t(num_sm);
      uint32_t num_chunks = max_grid_size / num_kv_heads;
      uint32_t kv_chunk_size = ceil_div(seq_len, num_chunks);

      // no need to use partition-kv
      dim3 nblks = dim3(num_chunks, num_kv_heads);
      dim3 nthrs = dim3(bdx, bdy, bdz);

      void* args[] = {
          (void*)&q,       (void*)&k,         (void*)&num_qo_heads,  (void*)&num_kv_heads,
          (void*)&seq_len, (void*)&num_pages, (void*)&kv_chunk_size, (void*)&out};
      FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
      return cudaSuccess;
    });
  });
}
}  // namespace flashinfer

#endif  // FLASHINFER_DECODE_CUH_
