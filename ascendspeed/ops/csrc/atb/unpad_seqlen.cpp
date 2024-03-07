// Copyright (c) 2023 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
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

const static int HIDDEN_DIM = 2;

void inferShapeUnPad(c10::SmallVector<int64_t, N_32> &size, const at::Tensor input_tensor, const std::vector<int> seq_len)
{
    int sum_seq_len = 0;
    for (auto item : seq_len) {
        sum_seq_len += item;
    }
    size = {sum_seq_len, input_tensor.size(HIDDEN_DIM)};
}

at::Tensor unpad_seqlen(const at::Tensor &input_tensor, const std::vector<int> seq_len, int max_seq_len)
{
#ifndef ENABLE_ATB
    TORCH_CHECK(false, "unpad_seqlen not implemented");
#else
    atb::train::UnpadWithHiddenStateParam param;
    param.maxSeqLen = max_seq_len;
    for (auto item : seq_len) {
        param.qSeqLen.push_back(item);
    }
    c10::SmallVector<int64_t, N_32> output_shape;
    inferShapeUnPad(output_shape, input_tensor, seq_len);
    at::Tensor output_tensor = CreateAtTensor(output_shape, input_tensor.scalar_type());

    ParamSetter paramsetter;
    paramsetter.Input(input_tensor)
               .Output(output_tensor);

    atb::Operation* op = nullptr;
    atb::CreateOperation(param, &op);
    TORCH_CHECK(op != nullptr, "UnpadWithHiddenStateOperation get op failed!");
    RunAtbCmd(op, paramsetter, "UnpadWithHiddenStateOperation");
    return output_tensor;
#endif
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("npu_unpad_seqlen", &unpad_seqlen, "unpad seqlen input");
}