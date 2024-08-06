# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.

import torch


def pipe_register_grad_ready(self, param: torch.nn.Parameter):
    assert (self.ddp_config.overlap_grad_reduce), 'register_grad_ready() should only be called when overlap_grad_reduce is True'
    from mindspeed.moe.pipe_experts import FLAG_GRAD_REDUCE
    if self.is_last_microbatch and FLAG_GRAD_REDUCE:
        bucket = self.param_to_bucket[param]
        bucket.register_grad_ready(param)
