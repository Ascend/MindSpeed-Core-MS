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

#include "inc/adapter.h"

#ifdef ENABLE_ATB
#include "inc/atb_adapter.h"
#include "atb/operation.h"
#include "atb/train_op_params.h"
#endif 

using namespace std;

void inferShapeStridedBatchMatmul(c10::SmallVector<int64_t, N_32> &size, int head_num,
    const std::vector<int32_t> m, const std::vector<int32_t> n)
{
    int output_shape = 0;
    for (int i = 0; i < m.size(); i++) {
        output_shape += head_num * m[i] * n[i];
    }
    size = {output_shape};
}

at::Tensor unpad_strided_batch_matmul(const at::Tensor &input1, const at::Tensor &input2, int32_t trans_a, int32_t trans_b,
    const std::vector<int32_t> m, const std::vector<int32_t> k, const std::vector<int32_t> n,
    const std::vector<int32_t> lda, const std::vector<int32_t> ldb, const std::vector<int32_t> ldc,
    const std::vector<int32_t> stride_a, const std::vector<int32_t> stride_b, const std::vector<int32_t> stride_c,
    int32_t batch, int32_t head_num) 
{
#ifndef ENABLE_ATB
    TORCH_CHECK(false, "strided_batch_matmul not implemented");
#else
    atb::train::StridedBatchMatmulParam param;
    param.transposeA = trans_a;
    param.transposeB = trans_b;
    for (auto item : m) {
        param.m.push_back(item);
    }
    for (auto item : k) {
        param.k.push_back(item);
    }
    for (auto item : n) {
        param.n.push_back(item);
    }
    for (auto item : lda) {
        param.lda.push_back(item);
    }
    for (auto item : ldb) {
        param.ldb.push_back(item);
    }
    for (auto item : ldc) {
        param.ldc.push_back(item);
    }
    for (auto item : stride_a) {
        param.strideA.push_back(item);
    }
    for (auto item : stride_b) {
        param.strideB.push_back(item);
    }
    for (auto item : stride_c) {
        param.strideC.push_back(item);
    }
    param.batch = batch;
    param.headNum = head_num;
    c10::SmallVector<int64_t, N_32> output_shape;
    inferShapeStridedBatchMatmul(output_shape, head_num, m, n);
    at::Tensor output_tensor = CreateAtTensor(output_shape, input1.scalar_type());

    ParamSetter paramsetter;
    paramsetter.Input(input1)
               .Input(input2)
               .Output(output_tensor);

    atb::Operation* op = nullptr;
    atb::CreateOperation(param, &op);
    TORCH_CHECK(op != nullptr, "StridedBatchMatmulOperation get op failed!");
    RunAtbCmd(op, paramsetter, "StridedBatchMatmulOperation");
    return output_tensor;
#endif
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("npu_unpad_strided_batch_matmul", &unpad_strided_batch_matmul, "unpad stridedbatchmatmul forward and backward");
}