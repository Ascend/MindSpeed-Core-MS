
/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "../inc/adapter.h"
#include <torch_npu/csrc/core/npu/DeviceUtils.h>

at::Tensor CreateAtTensor(c10::SmallVector<int64_t, N> shape, at::ScalarType inputDtype)
{
    return CreateAtTensor(at::IntArrayRef(shape), inputDtype);
}

at::Tensor CreateAtTensor(at::IntArrayRef shape, at::ScalarType inputDtype)
{
    at::TensorOptions options = at::TensorOptions(torch_npu::utils::get_npu_device_type());
    options = options.dtype(inputDtype);
    options = options.layout(torch::kStrided).requires_grad(false);
    at::Tensor newTensor = at::zeros(shape, options);
    if (!newTensor.is_contiguous()) {
        newTensor = newTensor.contiguous();
    }
    return newTensor;
}
