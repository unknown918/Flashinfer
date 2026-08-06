/*
 * Copyright (c) 2025 by FlashInfer team.
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
#include <flashinfer/estimate.cuh>

#include "tvm_ffi_utils.h"

using namespace flashinfer;

using tvm::ffi::Tensor;

void estimate(TensorView query, TensorView pooling, TensorView kv_indptr, TensorView kv_indices,
              TensorView out) {
  CHECK_CONTIGUOUS(query)
  CHECK_CONTIGUOUS(pooling)
  CHECK_CONTIGUOUS(out)
  CHECK_DIM(3, query);    // [batch_size, num_qo_head, head_dim]
  CHECK_DIM(3, pooling);  // [max_num_pages, num_kv_head, head_dim]
  CHECK_DIM(1, kv_indptr);
  CHECK_DIM(1, kv_indices);
  CHECK_DIM(2, out);  // [batch_size * num_qo_head, max_num_pages]

  uint32_t batch_size = kv_indptr.size(0) - 1;
  uint32_t num_qo_heads = query.size(1);
  uint32_t max_num_pages = pooling.size(0) / batch_size;
  uint32_t num_kv_heads = pooling.size(1);
  uint32_t head_dim = pooling.size(2);

  TVM_FFI_ICHECK_EQ(out.size(0), batch_size * num_qo_heads);
  TVM_FFI_ICHECK_EQ(out.size(1), max_num_pages);

  cudaSetDevice(query.device().device_id);
  const cudaStream_t stream = get_stream(query.device());
  bool success = DISPATCH_DLPACK_DTYPE_TO_CTYPE(query.dtype(), c_type, [&] {
    DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
      cudaError_t status = Estimate<HEAD_DIM, c_type>(
          static_cast<c_type*>(query.data_ptr()), static_cast<c_type*>(pooling.data_ptr()),
          batch_size, num_qo_heads, num_kv_heads, max_num_pages,
          static_cast<uint32_t*>(kv_indptr.data_ptr()),
          static_cast<uint32_t*>(kv_indices.data_ptr()), static_cast<c_type*>(out.data_ptr()),
          stream);
      TVM_FFI_ICHECK(status == cudaSuccess)
          << "estimation failed with error: " << cudaGetErrorString(status);
      return true;
    });
  });

  TVM_FFI_ICHECK(success) << "estimation failed to dispatch with dtype " << query.dtype();
}