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
"""utils"""

__all__ = ["get_attn_mask_func", "get_num_layer_list"]

import math
import mindspore.ops as ops

from mindspore import mint
from mindspore import Tensor
from mindspore.common import dtype as mstype
from mindspeed_ms.core.parallel_state import (
    get_pipeline_model_parallel_world_size,
    get_virtual_pipeline_model_parallel_world_size
)
from mindspeed_ms.training.global_vars import get_args


def attn_mask_fill(attention_scores: Tensor, attention_mask, fill_value=-10000.0):
    """mask attention scores with the mask value"""
    attention_scores = ops.masked_fill(
        attention_scores,
        attention_mask.astype(mstype.bool_),
        Tensor(fill_value, attention_scores.dtype),
    )
    return attention_scores


def attn_mask_add(attention_scores: Tensor, attention_mask):
    """Llama attention mask function"""
    score_dtype = attention_scores.dtype
    attention_scores = ops.add(
        attention_scores, ops.Cast()(attention_mask, score_dtype)
    )
    return attention_scores


ATTNMASK_FUNC_MAP = {
    "attn_mask_fill": attn_mask_fill,
    "attn_mask_add": attn_mask_add,
}


def get_attn_mask_func(mask_func_type):
    r"""
    Get attention mask function.

    Args:
        mask_func_type (str): The attention mask function type.

    Returns:
        Function, the attention mask function.
    """
    if mask_func_type not in ATTNMASK_FUNC_MAP:
        raise KeyError("Invalid attention mask function. Supported attention "
                       "mask function are ['attn_mask_fill', 'attn_mask_add'] "
                       ", but got {}.".format(mask_func_type))
    return ATTNMASK_FUNC_MAP[mask_func_type]


def get_num_layer_list(config):
    """Get num_layer_list for pp/vpp scenario"""
    num_layer_list = []
    if config.num_layer_list:
        num_layer_list = config.num_layer_list
    else:
        args = get_args()
        standalone_embedding_stage = args.standalone_embedding_stage
        pp_stage = get_pipeline_model_parallel_world_size()
        if standalone_embedding_stage:
            pp_stage = pp_stage - 1
        vpp_stage = (get_virtual_pipeline_model_parallel_world_size()
                     if get_virtual_pipeline_model_parallel_world_size() is not None else 1)
        if pp_stage == 0:
            raise ValueError(f"pp_stage need larger than zero!")
        pp_split_num = pp_stage * vpp_stage
        num_layers = config.num_layers
        borrow_layer_num = math.ceil(num_layers / pp_split_num) * pp_split_num - num_layers
        num_layer_list = [(num_layers + borrow_layer_num) // pp_stage] * pp_stage
        if get_virtual_pipeline_model_parallel_world_size() is None:
            num_layer_list[-1] -= borrow_layer_num
            if num_layer_list[-1] <= 0:
                raise NotImplementedError(
                    f"num_layers {config.num_layers} will range to {num_layer_list} "
                    f"for pp {pp_stage}, and last stage has no layer"
                )
        else:
            # split num_layer_list from (pp_stage) to (pp_stage, vpp_stage)
            num_layer_list = [[(num_layer_list[i] // vpp_stage)] * vpp_stage for i in range(pp_stage)]
            num_layer_list[-1][-1] -= borrow_layer_num
            if num_layer_list[-1][-1] <= 0:
                raise NotImplementedError(
                    f"num_layers {config.num_layers} will range to {num_layer_list}  "
                    f"for pp {pp_stage} vpp {vpp_stage}, and last model chunk has no layer."
                )
    return num_layer_list


def get_layers_and_offset(num_layer_array, pp_stage, pp_rank, vpp_stage=None, vpp_rank=0):
    """get transformer layers nums for current rank according to num layer list"""
    pp_layout = (1,)
    if vpp_stage is not None:
        pp_layout = (pp_stage, vpp_stage)
    elif pp_stage is not None:
        pp_layout = (pp_stage,)
    if num_layer_array.shape != pp_layout:
        raise ValueError("The shape of num_layer_list {} must equal to "
                         "pp_layout {}".format(num_layer_array.shape, pp_layout))
    if vpp_stage is None:
        num_layers = num_layer_array[pp_rank]
        offset = num_layer_array[:pp_rank].sum()
        return num_layers, offset

    offset = 0
    for i in range(vpp_rank):
        offset += num_layer_array[:, i].sum()
    offset += num_layer_array[:pp_rank, vpp_rank].sum()
    num_layers = num_layer_array[pp_rank][vpp_rank]
    return num_layers, offset


def gelu_impl(x):
    """OpenAI's gelu implementation."""
    return 0.5 * x * (1.0 + mint.nn.functional.tanh(0.7978845608028654 * x *
                                                    (1.0 + 0.044715 * x * x)))

def openai_gelu(x):
    return gelu_impl(x)

def erf_gelu(x):
    return x * 0.5 * (mint.erf(x / 1.41421).to(dtype=x.dtype)+mint.ones_like(x).to(dtype=x.dtype))
