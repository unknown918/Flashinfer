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

void estimate(TensorView query, TensorView pooling, uint32_t num_valid_pages, TensorView out) {
  CHECK_CONTIGUOUS(query)
  CHECK_CONTIGUOUS(pooling)
  CHECK_CONTIGUOUS(out)
  CHECK_DIM(2, out);
  CHECK_DIM(2, query);
  CHECK_DIM(3, pooling);

  int64_t num_qo_heads = query.size(0);
  int64_t num_total_pages = pooling.size(0);
  int64_t num_kv_heads = pooling.size(1);
  int64_t head_dim = pooling.size(2);

  TVM_FFI_ICHECK_EQ(out.size(0), num_qo_heads);
  TVM_FFI_ICHECK_EQ(out.size(1), pooling.size(0));

  cudaSetDevice(query.device().device_id);
  const cudaStream_t stream = get_stream(query.device());
  bool success = DISPATCH_DLPACK_DTYPE_TO_CTYPE(query.dtype(), c_type, [&] {
    DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
      cudaError_t status = Estimate<HEAD_DIM, c_type>(
          static_cast<c_type*>(query.data_ptr()), static_cast<c_type*>(pooling.data_ptr()),
          static_cast<c_type*>(out.data_ptr()), static_cast<uint32_t>(num_qo_heads),
          static_cast<uint32_t>(num_kv_heads), static_cast<uint32_t>(num_valid_pages),
          static_cast<uint32_t>(num_total_pages), stream);
      TVM_FFI_ICHECK(status == cudaSuccess)
          << "estimation failed with error: " << cudaGetErrorString(status);
      return true;
    });
  });

  TVM_FFI_ICHECK(success) << "estimation failed to dispatch with dtype " << query.dtype();
}