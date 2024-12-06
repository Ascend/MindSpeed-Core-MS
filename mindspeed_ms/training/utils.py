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

import mindspore as ms
from mindspore import mint
from mindspore.communication.management import init

from mindspeed_ms.tools import logger
from mindspeed_ms.core.parallel_state import get_data_parallel_world_size, initialize_model_parallel, \
    get_context_parallel_for_hybrid_ring_rank, \
    get_context_parallel_for_hybrid_ulysses_rank, \
    get_context_parallel_for_hybrid_ring_world_size, \
    get_context_parallel_for_hybrid_ulysses_world_size


def set_weight_decay(params, optimizer_config):
    """
    Set weight decay coefficient, zero for bias and layernorm, 1e-1 for rest

    Args:
        params (list[Parameter]): List of parameters to apply weight decay to.

    Returns:
        list: A list of dictionaries specifying the parameter groups and their respective weight decay coefficients.
    """
    params_map = {}
    for param in params:
        if not param.requires_grad:
            continue

        no_wd = no_weight_decay_cond_func(optimizer_config.no_weight_decay_params, param.name, param)

        scale_lr = False

        if not no_wd and not scale_lr:
            wd_mult, lr_mult = 1.0, 1.0
        elif not no_wd and scale_lr:
            wd_mult, lr_mult = 1.0, lr_mult
        elif no_wd and not scale_lr:
            wd_mult, lr_mult = 0.0, 1.0
        else:
            wd_mult, lr_mult = 0.0, lr_mult

        key = (wd_mult, lr_mult)
        if key not in params_map:
            params_map[key] = []
        params_map[key].append(param)

    param_groups = []
    for (wd_mult, lr_mult), param_list in params_map.items():
        if not param_list:
            raise ValueError("After setting weight decay, param groups should not be empty.")
        param_groups.append(
            {
                "params": param_list,
                "weight_decay": optimizer_config.weight_decay,
                "wd_mult": wd_mult,
                "lr_mult": lr_mult,
            }
        )
    return param_groups


def no_weight_decay_cond_func(no_weight_decay_params, name, param):
    if 'Default_Setup' in no_weight_decay_params:
        return name.endswith(".bias") or len(param.shape) == 1
    # Check if the parameter name ends with any of the specified suffixes
    for suffix in no_weight_decay_params:
        if name.endswith(suffix):
            return True
    return False


def set_parallel_context(parallel_config):
    """
    Sets the parallel context based on the provided parallel configuration.

    Args:
        parallel_config: The parallel configuration object containing the parallel settings.

    Returns:
        ParallelConfig: The updated parallel configuration object.

    """
    init()
    initialize_model_parallel(
        tensor_model_parallel_size=parallel_config.tensor_model_parallel_size,
        pipeline_model_parallel_size=parallel_config.pipeline_model_parallel_size,
        virtual_pipeline_model_parallel_size=parallel_config.virtual_pipeline_model_parallel_size,
        context_parallel_size=parallel_config.context_parallel_size,
    )
    logger.info(
        f"dp {get_data_parallel_world_size()} | "
        f"pp {parallel_config.pipeline_model_parallel_size} | "
        f"tp {parallel_config.tensor_model_parallel_size} | "
        f"cp {parallel_config.context_parallel_size} | "
        f"sp {parallel_config.sequence_parallel} | "
        f"vpp {parallel_config.virtual_pipeline_model_parallel_size}"
    )


def set_seed(seed):
    """
    Set the seed for random number generation.

    Parameters:
    - seed (int): The seed value to set.

    Returns:
    None
    """
    # set global seed, np seed, and dataset seed
    ms.set_seed(seed)
    # set rng seed
    ms.manual_seed(seed)

def _get_batch_on_this_cp_rank_in_hybrid_cp(batch):
    """
    Transformed batch data to support hybrid context parallelism.
    """
    u_size = get_context_parallel_for_hybrid_ulysses_world_size()
    r_size = get_context_parallel_for_hybrid_ring_world_size()

    u_rank = get_context_parallel_for_hybrid_ulysses_rank()
    r_rank = get_context_parallel_for_hybrid_ring_rank()

    for key, val in batch.items():
        if key == 'attention_mask':
            continue
        if val is not None:
            seq_dim = 1 if key != 'attention_mask' else 2
            val = val.view(
                *val.shape[0:seq_dim],
                2 * r_size,
                val.shape[seq_dim] // (2 * r_size),
                *val.shape[(seq_dim + 1):],
            )
            index = ms.tensor([r_rank, (2 * r_size - r_rank - 1)])
            val = mint.index_select(val, seq_dim, index)
            val = val.view(*val.shape[0:seq_dim], -1, *val.shape[(seq_dim + 2):])
            val = val.chunk(u_size, seq_dim)[u_rank].contiguous()
            batch[key] = val
    return batch
