// Copyright (c) 2024 Huawei Technologies Co., Ltd
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
#include <torch/extension.h>
#include "torch_npu/csrc/framework/utils/RandomOpAdapter.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/core/npu/NPUFormat.h"
#include <torch_npu/csrc/include/ops.h>

#include "inc/aclnn_common.h"
#include "inc/mc2_utils.h"

at::Tensor format_trans(const at::Tensor &at_tensor)
{
    if (at_tensor.defined()) {
        TORCH_CHECK(torch_npu::utils::is_npu(at_tensor), "only npu tensor is supported");
        return at_npu::native::npu_format_cast(at_tensor, ACL_FORMAT_ND);
    }
    return at_tensor;
}

std::tuple<at::Tensor, at::Tensor> npu_mm_all_reduce_add_rms_norm(
    const at::Tensor &x1,
    const at::Tensor &x2,
    const at::Tensor &residual,
    const at::Tensor &gamma,
    std::string hcom,
    std::string reduce_op,
    double epsilon,
    const c10::optional<at::Tensor> &bias,
    const c10::optional<at::Tensor> &antiquant_scale,
    const c10::optional<at::Tensor> &antiquant_offset,
    const c10::optional<at::Tensor> &dequant_scale,
    int64_t antiquant_group_size,
    int64_t comm_turn)
{
    check_npu_mm_all_reduce_add_rms_norm_params(x1, x2, residual, gamma, antiquant_scale, antiquant_offset,
                                                dequant_scale);
    at::Tensor format_x1 = format_trans(x1);
    at::Tensor format_x2 = format_trans(x2);
    at::Tensor format_residual = format_trans(residual);
    at::Tensor format_gamma = format_trans(gamma);
    char *hcom_ptr = const_cast<char *>(hcom.c_str());
    char *reduce_op_ptr = const_cast<char *>(reduce_op.c_str());
    const at::Tensor &bias_const = bias.value_or(at::Tensor());
    at::Tensor y = at::empty(format_residual.sizes(), format_residual.options());
    at::Tensor norm_out = at::empty(format_residual.sizes(), format_residual.options());
    int64_t stream_mode = ACL_STOP_ON_FAILURE;
    // a8w8: x1\x2 kChar; a16w8: x2 kChar;
    if (!isIntegralType(x1.scalar_type()) && !isIntegralType(x2.scalar_type())) {
        ACLNN_CMD(aclnnMatmulAllReduceAddRmsNorm, x1, x2, bias_const, residual, gamma, epsilon, hcom_ptr,
                  reduce_op_ptr, comm_turn, stream_mode, y, norm_out);
    }
    if (isIntegralType(x1.scalar_type()) && isIntegralType(x2.scalar_type())) {
        const at::Tensor &dequant_scale_real = dequant_scale.value_or(at::Tensor());
        ACLNN_CMD(aclnnQuantMatmulAllReduceAddRmsNorm, x1, x2, bias_const, dequant_scale_real, residual, gamma,
                  epsilon, hcom_ptr, reduce_op_ptr, comm_turn, stream_mode, y, norm_out);
    }
    if (!isIntegralType(x1.scalar_type()) && isIntegralType(x2.scalar_type())) {
        const at::Tensor &antiquant_scale_real = antiquant_scale.value_or(at::Tensor());
        const at::Tensor &antiquant_offset_real = antiquant_offset.value_or(at::Tensor());
        ACLNN_CMD(aclnnWeightQuantMatmulAllReduceAddRmsNorm, x1, x2, bias_const, antiquant_scale_real,
                  antiquant_offset_real, residual, gamma, epsilon, hcom_ptr, reduce_op_ptr, comm_turn,
                  stream_mode, antiquant_group_size, y, norm_out);
    }

    return std::make_tuple(y, norm_out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("npu_mm_all_reduce_add_rms_norm", &npu_mm_all_reduce_add_rms_norm, "npu_mm_all_reduce_add_rms_norm");
}