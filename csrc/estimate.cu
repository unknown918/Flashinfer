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
#include <flashinfer/estimate.cuh>

#include "tvm_ffi_utils.h"

using namespace flashinfer;

using tvm::ffi::Tensor;

void estimate(TensorView q, TensorView pooling, uint32_t seq_len, TensorView out) {
  CHECK_CONTIGUOUS(pooling)
  CHECK_CONTIGUOUS(out)
  CHECK_DIM(3, pooling);
  CHECK_DIM(2, out);

  unsigned int num_qo_heads, num_kv_heads, head_dim, num_pages;

  num_qo_heads = q.size(0);
  num_pages = pooling.size(0);
  num_kv_heads = pooling.size(1);
  head_dim = pooling.size(2);

  TVM_FFI_ICHECK_EQ(out.size(0), num_qo_heads);
  TVM_FFI_ICHECK_EQ(out.size(1), pooling.size(0));

  cudaSetDevice(q.device().device_id);
  const cudaStream_t stream = get_stream(q.device());
  bool success = DISPATCH_DLPACK_DTYPE_TO_CTYPE(q.dtype(), c_type, [&] {
    DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
      cudaError_t status = Estimate<HEAD_DIM, c_type>(
          static_cast<c_type*>(q.data_ptr()), static_cast<c_type*>(pooling.data_ptr()),
          static_cast<c_type*>(out.data_ptr()), num_qo_heads, num_kv_heads, seq_len, num_pages, stream);
      TVM_FFI_ICHECK(status == cudaSuccess)
          << "Estimation failed with error: " << cudaGetErrorString(status);
      return true;
    });
  });

  TVM_FFI_ICHECK(success) << "Estimation failed to dispatch with dtype " << q.dtype();
}