// Copyright (c) 2024 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include <torch/extension.h>
#include <torch_npu/csrc/framework/utils/RandomOpAdapter.h>
#include <torch_npu/csrc/framework/utils/OpAdapter.h>
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/include/ops.h>

#include "inc/aclnn_common.h"

at::Tensor npu_ffn(const at::Tensor &x, const at::Tensor &weight1, const at::Tensor &weight2,
    std::string activation, c10::optional<at::Tensor> expert_tokens, c10::optional<at::Tensor> expert_tokens_index,
    const c10::optional<at::Tensor> &bias1, const c10::optional<at::Tensor> &bias2,
    const c10::optional<at::Tensor> &scale, const c10::optional<at::Tensor> &offset,
    const c10::optional<at::Tensor> &deq_scale1, const c10::optional<at::Tensor> &deq_scale2,
    const c10::optional<at::Tensor> &antiquant_scale1, const c10::optional<at::Tensor> &antiquant_scale2,
    const c10::optional<at::Tensor> &antiquant_offset1, const c10::optional<at::Tensor> &antiquant_offset2,
    c10::optional<int64_t> inner_precise, c10::optional<at::ScalarType> output_dtype)
{
    TORCH_CHECK(false, "Currently ffn is only supported in graph mode, not in single-operator mode!");
    at::Tensor y;
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("npu_ffn", &npu_ffn, "npu_ffn");
}