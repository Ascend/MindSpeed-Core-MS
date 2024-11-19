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
"""FusedScaleMaskSoftmax"""
from mindspore import mint, ops
from mindspore.common import dtype as mstype
from .module import Module


class FusedScaleMaskSoftmax(Module):
    r"""
    fused operation: scaling + mask + softmax

    Args:
        input_in_fp16: flag to indicate if input in fp16 data format.
        input_in_bf16: flag to indicate if input in bf16 data format.
        attn_mask_type: attention mask type (pad or causal)
        scaled_masked_softmax_fusion: flag to indicate user want to use softmax fusion
        mask_func: mask function to be applied.
        softmax_in_fp32: if true, softmax in performed at fp32 precision.
        scale: scaling factor used in input tensor scaling.

    Inputs:
        - **x** (Tensor) - The input tensor
        - **mask** (Tensor) - The mask tensor

    Outputs:
        - The output tensor.
    """

    def __init__(
            self,
            input_in_fp16,
            input_in_bf16,
            attn_mask_type,
            scaled_masked_softmax_fusion,
            mask_func,
            softmax_in_fp32,
            scale,
        ):
        super(FusedScaleMaskSoftmax, self).__init__()
        self.input_in_fp16 = input_in_fp16
        self.input_in_bf16 = input_in_bf16
        if input_in_fp16 and self.input_in_bf16:
            raise ValueError("both fp16 and bf16 flags cannot be active at the same time.")
        self.input_in_float16 = self.input_in_fp16 or self.input_in_bf16
        self.attn_mask_type = attn_mask_type
        if scaled_masked_softmax_fusion:
            raise NotImplementedError(
                "`scaled_masked_softmax_fusion` is not supported for now."
            )
        self.mask_func = mask_func
        self.softmax_in_fp32 = softmax_in_fp32
        self.scale = scale

        if self.scale is not None and not self.softmax_in_fp32:
            raise ValueError("softmax should be in fp32 when scaled")

    def construct(self, x, mask):
        """construct method"""
        return self.forward_softmax(x, mask)

    def forward_softmax(self, x, mask):
        """scale mask softmax"""
        if self.input_in_float16 and self.softmax_in_fp32:
            x = ops.cast(x, mstype.float32)

        if self.scale is not None:
            x = x * self.scale
        masked_input = self.mask_func(x, mask) if mask is not None else x

        probs = mint.nn.functional.softmax(masked_input, dim=-1)

        if self.input_in_float16 and self.softmax_in_fp32:
            if self.input_in_fp16:
                probs = ops.cast(probs, mstype.float16)
            else:
                probs = ops.cast(probs, mstype.bfloat16)

        return probs
