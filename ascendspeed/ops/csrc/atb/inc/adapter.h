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
#ifndef ASCENDSPEED_OPS_CSRC_ATB_INC_ADAPTER_H
#define ASCENDSPEED_OPS_CSRC_ATB_INC_ADAPTER_H
#include <vector>
#include <string>
#include <torch/script.h>
#include <torch/custom_class.h>

const static int N_32 = 32;

at::Tensor CreateAtTensor(c10::IntArrayRef shape, at::ScalarType inputDtype);
at::Tensor CreateAtTensor(c10::SmallVector<int64_t, N_32> shape, at::ScalarType inputDtype);

#endif
