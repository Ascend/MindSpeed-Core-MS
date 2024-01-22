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
    m.def("flash_attention", &npu_flash_attention, "the fast realization of self-attention on ascend device",
    pybind11::arg("query"), pybind11::arg("key"), pybind11::arg("value"),
    pybind11::arg("atten_mask"), pybind11::arg("alibi_mask"),
    pybind11::arg("scale_value")=1.0, pybind11::arg("q_scale")=1.0, pybind11::arg("head_num")=1,
    pybind11::arg("io_layout")="BNSD", pybind11::arg("keep_prob")=1.0, pybind11::arg("pre_tokens")=2147483647,
    pybind11::arg("next_tokens")=1, pybind11::arg("precise_mode")=0, pybind11::arg("groups")=-1,
    pybind11::arg("sync")=false, pybind11::arg("parallel")=true);
}
