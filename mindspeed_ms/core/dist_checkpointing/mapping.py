# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

""" Core library classes for representing sharding of tensors and objects.

The main expected usage is wrapping torch.Tensors in state dicts with
ShardedTensor class (mostly with the ShardedTensor.from_rank_offsets classmethod).
"""

from typing import Any, Dict

ShardedStateDict = Dict[str, Any]
