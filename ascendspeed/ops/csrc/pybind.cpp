// Copyright (c) 2023 Huawei Technologies Co., Ltd
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
#include "ops_interface.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("npu_multi_head_attention_forward", &flash_attention, "the fast realization of self-attention forward-function on ascend device");
    m.def("npu_multi_head_attention_backward", &flash_attention_grad, "the fast realization of self-attention backward-function on ascend device");
    m.def("npu_gen_mask", &gen_mask, "generate random mask on ascend device");
    m.def("npu_gen_mask_by_seed", &exist_gen_mask, "generate random mask by seed on ascend device");
}
