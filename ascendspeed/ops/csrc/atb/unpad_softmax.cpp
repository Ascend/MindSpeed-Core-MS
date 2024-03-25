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

#include "inc/adapter.h"

#ifdef ENABLE_ATB
#include "inc/atb_adapter.h"
#include "atb/operation.h"
#include "atb/train_op_params.h"
#endif 

using namespace std;

at::Tensor unpad_softmax(const at::Tensor &data_input, const std::vector<int32_t> &seq_len, int32_t head_num)
{
#ifndef ENABLE_ATB
    TORCH_CHECK(false, "unpad_softmax not implemented");
#else
    atb::train::FastSoftMaxParam param;
    param.headNum = head_num;
    for (auto item : seq_len) {
        param.qSeqLen.push_back(item);
    }
    at::Tensor output_tensor = CreateAtTensor(data_input.sizes(), data_input.scalar_type());

    ParamSetter paramsetter;
    paramsetter.Input(data_input)
               .Output(output_tensor);

    atb::Operation* op = nullptr;
    atb::CreateOperation(param, &op);
    TORCH_CHECK(op != nullptr, "FastSoftMaxOperation get op failed!");
    RunAtbCmd(op, paramsetter, "FastSoftMaxOperation");
    return output_tensor;
#endif
}

at::Tensor unpad_softmax_grad(const at::Tensor &y_input, const at::Tensor &y_grad,const std::vector<int32_t> &seq_len,
    int32_t head_num)
{
#ifndef ENABLE_ATB
    TORCH_CHECK(false, "unpad_softmax_grad not implemented");
#else
    atb::train::FastSoftMaxGradParam param;
    param.headNum = head_num;
    for (auto item : seq_len) {
        param.qSeqLen.push_back(item);
    }
    at::Tensor output_tensor = CreateAtTensor(y_input.sizes(), y_input.scalar_type());

    ParamSetter paramsetter;
    paramsetter.Input(y_input)
               .Input(y_grad)
               .Output(output_tensor);

    atb::Operation* op = nullptr;
    atb::CreateOperation(param, &op);
    TORCH_CHECK(op != nullptr, "FastSoftMaxGradOperation get op failed!");
    RunAtbCmd(op, paramsetter, "FastSoftMaxGradOperation");
    return output_tensor;
#endif
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("npu_unpad_softmax", &unpad_softmax, "unpad softmax forward");
    m.def("npu_unpad_softmax_grad", &unpad_softmax_grad, "unpad softmax backward");
}