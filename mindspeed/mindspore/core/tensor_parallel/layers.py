#  Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
from mindspeed.mindspore.mindspore_adaptor import bprop_commn


def linear_with_frozen_weight_bprop(self, input, weight, bias, output, grad_output):
    return bprop_commn(self, grad_output)


def linear_with_grad_accumulation_and_async_communication_bprop(self, input, weight, bias, gradient_accumulation_fusion,
                                                                async_grad_allreduce, sequence_parallel, output,
                                                                grad_output_buffer, grad_output):
    return bprop_commn(self, grad_output)
