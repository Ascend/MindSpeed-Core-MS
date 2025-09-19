# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.

"""Patch mindspeed_mm.utils.utils"""
import torch

def quick_gelu(x: torch.Tensor) -> torch.Tensor:
    # ms adaptation: fix ms scalar mul problem
    return x * torch.sigmoid((1.702 * x.to(torch.float32)).to(x.dtype))
