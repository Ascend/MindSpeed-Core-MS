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

#include <string>
#include <vector>
#include <ATen/Tensor.h>
#include <torch/extension.h>

#include "../ops_interface.h"
#include "inc/adapter.h"

#ifdef ENABLE_ATB
#include "inc/atb_adapter.h"
#include "atb/operation.h"
#include "atb/train_op_params.h"
#endif 

using namespace std;

void inferShapeGenAttentionMask(c10::SmallVector<int64_t, N_32> &size, const std::vector<int> seq_len, int head_num)
{
    int output_shape = 0;
    for (auto item : seq_len) {
        output_shape += head_num * item * item;
    }
    size = {output_shape};
}

at::Tensor unpad_gen_attention_mask(const at::Tensor &input_tensor, const std::vector<int> seq_len, int head_num)
{
#ifndef ENABLE_ATB
    TORCH_CHECK(false, "unpad_gen_attention_mask not implemented");
#else
    atb::train::GenAttentionMaskParam param;
    param.headNum = head_num;
    for (auto item : seq_len) {
        param.seqLen.push_back(item);
    }
    c10::SmallVector<int64_t, N_32> output_shape;
    inferShapeGenAttentionMask(output_shape, seq_len, head_num);
    at::Tensor output_tensor = CreateAtTensor(output_shape, input_tensor.scalar_type());

    ParamSetter paramsetter;
    paramsetter.Input(input_tensor)
               .Output(output_tensor);

    atb::Operation* op = nullptr;
    atb::CreateOperation(param, &op);
    TORCH_CHECK(op != nullptr, "GenAttentionMaskOperation get op failed!");
    RunAtbCmd(op, paramsetter, "GenAttentionMaskOperation");
    return output_tensor;
#endif
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("npu_unpad_gen_attention_mask", &unpad_gen_attention_mask, "unpad gen attentionmask on ascend device");
}