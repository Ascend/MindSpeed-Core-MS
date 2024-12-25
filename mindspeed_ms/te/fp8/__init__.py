# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""fp8 init"""

import math

import mindspore
from mindspore import ops, numpy, nn, mint

E4M3_MAXVAL = 448.5
E5M2_MAXVAL = 57344.5

def set_amax(tensor: mindspore.Tensor, block_dim=None):
    """set fp8_amax attribute for tensor"""
    if block_dim is None:
        amax = mint.amax(mint.abs(tensor))
    else:
        # TODO torch.empty API 无对应 MS API
        amax = tensor.new_empty((math.ceil(tensor.shape[0] / block_dim[0]), math.ceil(tensor.shape[1] / block_dim[1])))
        for i in range(amax.shape[0]):
            for j in range(amax.shape[1]):
                i_start, i_end = i * block_dim[0], (i + 1) * block_dim[0]
                j_start, j_end = j * block_dim[1], (j + 1) * block_dim[1]
                amax[i, j].copy_(mint.amax(mint.abs(tensor[i_start:i_end, j_start:j_end])))

    if not hasattr(tensor, 'fp8_amax'):
        setattr(tensor, 'fp8_amax', amax)
    else:
        tensor.fp8_amax.copy_(amax)


def block_scaling_matmul(inputs, weight, inputs_scale, weight_scale, block_dim):
    """matmul for block scaling"""
    if inputs.shape[1] is not weight.shape[0]:
        raise AssertionError('shape error.')
    # TODO torch.empty API 无对应 MS API
    output = inputs.new_empty((inputs.shape[0], weight.shape[1]))

    for i in range(inputs.shape[0]):
        for j in range(weight.shape[1]):
            line = inputs[i] * weight[:, j]
            ii = i // block_dim[0]
            for jj in range(math.ceil(line.shape[-1] / block_dim[1])):
                start = jj * block_dim[1]
                end = (jj + 1) * block_dim[1]
                # TODO torch tensor.mul_ 待对标实现
                line[start:end].copy_(line[start:end].mul(inputs_scale[ii, jj] * weight_scale[jj, ii]))
            output[i, j].copy_(line.sum())

    return output


def fp8_matmul(inputs, weight, fp8_meta, key, transpose=(False, False)):
    """matmul in fp8"""
    inputs, weight = fp8_meta.pre_compute(key, inputs, weight)
    recipe = getattr(fp8_meta, key)
    if recipe.block_dim is None:
        inputs = inputs.t() if transpose[0] else inputs
        weight = weight.t() if transpose[1] else weight
        output = recipe.dequantization(mint.matmul(inputs, weight))
    else:
        output = recipe.fp8_block_scaling_matmul(inputs, weight, transpose)

    set_amax(output, recipe.block_dim)
    return output


class Cast2FP8(nn.Cell):
    """cast fp16/32 to fp8"""
    def construct(self, num, ebits, mbits):
        """forward func of Cast2FP8"""
        bias = int(2 ** (ebits - 1) - 1)
        x_tmp = ops.stop_gradient(num).clone()
        if ebits == 4:
            maxval = E4M3_MAXVAL
        elif ebits == 5:
            maxval = E5M2_MAXVAL
        else:
            maxval = (2 - 2 ** (-mbits)) * 2 ** (2 ** ebits - bias - 2)

        minval = -maxval
        num_clamp = mint.clamp(num, minval, maxval)

        # 计算non normalized的exp
        scales = mint.clamp((mint.floor(ops.stop_gradient(mint.log2(mint.abs(num_clamp)) + bias))), 1.0)
        scales = 2.0 ** (scales - mbits - bias)

        num = mint.round(num / scales) * scales
        if ebits == 4:
            num = mint.where((x_tmp > maxval) | (x_tmp < minval), numpy.nan, num)
        else:
            num = mint.where((x_tmp > maxval) | (x_tmp < minval), mint.sign(x_tmp) * numpy.inf, num)

        return num

    def bprop(self, *args):
        return args[-1], None, None


def cast_to_fp8(num, ebits=4, mbits=3):
    cast2fp8 = Cast2FP8()
    return cast2fp8(num, ebits, mbits)
