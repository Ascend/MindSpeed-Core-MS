// Copyright (c) 2024 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License");
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

using namespace at_npu::native;

const static int DIMS = 4;
const static int D_INDEX = 3;
const static int TWO = 2;
const static int BROADCAST_LIMIT = 1024;
const static int64_t ROTATE_HALF = 0;
const static int64_t ROTATE_INTERLEAVED = 1;

at::Tensor npu_rotary_position_embedding(const at::Tensor &x,
                                         const at::Tensor &cos,
                                         const at::Tensor &sin,
                                         c10::optional<int64_t> mode)
{
    TORCH_CHECK(x.dim() == DIMS,
                "The dims of input x should be 4 dimensional, bug got ", x.dim(), "-dimensional.");
    TORCH_CHECK(cos.dim() == DIMS,
                "The dims of input cos should be 4 dimensional, bug got ", cos.dim(), "-dimensional.");
    TORCH_CHECK(sin.dim() == DIMS,
                "The dims of input sin should be 4 dimensional, bug got ", sin.dim(), "-dimensional.");
    TORCH_CHECK(x.sizes()[D_INDEX] % TWO == 0,
                "The head_dim length of input must be an even number, but got ", x.sizes()[D_INDEX], ".");
    int64_t mode_value = mode.value_or(ROTATE_HALF);
    TORCH_CHECK(mode_value == ROTATE_HALF || mode_value == ROTATE_INTERLEAVED,
                "The mode of rotate shoule be 0(rotate_half) or 1(rotate_interleaved), but got ", mode_value, ".");

    at::Tensor y = at::empty(x.sizes(), x.options());

    ACLNN_CMD(aclnnRotaryPositionEmbedding, x, cos, sin, mode_value, y);

    return y;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_rotary_position_embedding_backward(const at::Tensor &dy,
                                                                                      const at::Tensor &cos,
                                                                                      const at::Tensor &sin,
                                                                                      const c10::optional<at::Tensor> &x_opt,
                                                                                      c10::optional<int64_t> mode)
{
    TORCH_CHECK(dy.dim() == DIMS,
                "The dims of input dy should be 4 dimensional, bug got ", dy.dim(), "-dimensional.");
    TORCH_CHECK(cos.dim() == DIMS,
                "The dims of input cos should be 4 dimensional, bug got ", cos.dim(), "-dimensional.");
    TORCH_CHECK(sin.dim() == DIMS,
                "The dims of input sin should be 4 dimensional, bug got ", sin.dim(), "-dimensional.");
    TORCH_CHECK(dy.sizes()[D_INDEX] % TWO == 0,
                "The head_dim length of input must be an even number, but got ", dy.sizes()[D_INDEX], ".");
    int64_t mode_value = mode.value_or(ROTATE_HALF);
    TORCH_CHECK(mode_value == ROTATE_HALF || mode_value == ROTATE_INTERLEAVED,
                "The mode of rotate shoule be 0(rotate_half) or 1(rotate_interleaved), but got ", mode_value, ".");
    bool check_support = true;
    int64_t broadcast_dim_num = 1;
    for (int64_t i = 0; i < dy.dim(); i++) {
        if (dy.sizes()[i] != cos.sizes()[i]) {
            broadcast_dim_num = broadcast_dim_num * dy.sizes()[i];
        }
        if (broadcast_dim_num > BROADCAST_LIMIT) {
            check_support = false;
            break;
        }
    }
    TORCH_CHECK(check_support == true,
                "The broadcast shape: [", broadcast_dim_num, "] > 1024 is too large, do not support.");

    const at::Tensor &x = c10::value_or_else(x_opt, [] { return at::Tensor(); });
    at::Tensor dx = at::empty(dy.sizes(), dy.options());
    at::Tensor dcos = at::empty(cos.sizes(), cos.options());
    at::Tensor dsin = at::empty(sin.sizes(), sin.options());

    ACLNN_CMD(aclnnRotaryPositionEmbeddingGrad, dy, cos, sin, x, mode_value, dx, dcos, dsin);

    return std::make_tuple(dx, dcos, dsin);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("npu_rotary_position_embedding", &npu_rotary_position_embedding, "rotary position embedding forward");
    m.def("npu_rotary_position_embedding_backward", &npu_rotary_position_embedding_backward, "rotary position embedding backward");
}
