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
# =========================================================================
"""Utilities for transformer layers."""

from functools import lru_cache

from mindspore.common import dtype as mstype

import mindspore as ms
from mindspore import ops, Tensor, mint


@lru_cache(maxsize=32)
def get_default_causal_mask(sq: int) -> Tensor:
    """Return the causal upper triangular mask for softmax input."""
    return ops.triu(mint.ones((sq, sq)), diagonal=1).astype(ms.bool_)


def attention_mask_func(attention_scores: Tensor, attention_mask):
    """Note: The performance of ops.masked_fill is better than that of Tensor.masked_fill.
       Although the official documentation API shows they are the same."""
    attention_scores = ops.masked_fill(
        attention_scores,
        attention_mask.astype(mstype.bool_),
        Tensor(-10000.0, attention_scores.dtype),
    )
    return attention_scores


def gelu_impl(x):
    """OpenAI's gelu implementation."""
    return 0.5 * x * (1.0 + mint.tanh(0.7978845608028654 * x * (1.0 + 0.044715 * x * x)))


def openai_gelu(x):
    return gelu_impl(x)


# This is actually Python equivalent of torch.nn.functional.gelu(), also with type hints for ONNX exporter
def erf_gelu(x):
    return (
        x * 0.5 * (mint.erf(x / 1.41421).to(dtype=x.dtype) + mint.ones_like(x).to(dtype=x.dtype))
    )
