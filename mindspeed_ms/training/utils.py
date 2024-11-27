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


def decay_filter(x):
    return "norm" not in x.name.lower() and "bias" not in x.name.lower()


def set_weight_decay(params, weight_decay=1e-1):
    """
    Set weight decay coefficient, zero for bias and layernorm, 1e-1 for rest

    Args:
        params (list[Parameter]): List of parameters to apply weight decay to.

    Returns:
        list: A list of dictionaries specifying the parameter groups and their respective weight decay coefficients.
    """
    decay_params = list(filter(decay_filter, params))
    other_params = list(filter(lambda x: not decay_filter(x), params))
    group_params = []
    if decay_params:
        group_params.append({"params": decay_params, "weight_decay": weight_decay, "wd_mult": 1.0})
    if other_params:
        group_params.append({"params": other_params, "weight_decay": 0.0, "wd_mult": 0.0})
    return group_params


def set_parallel_context(config):
    """
    Sets the parallel context based on the provided parallel configuration.

    Args:
        config: The parallel configuration object containing the parallel settings.

    Returns:
        ParallelConfig: The updated parallel configuration object.

    """
    init()
    initialize_model_parallel(
        tensor_model_parallel_size=config.tensor_model_parallel_size,
        pipeline_model_parallel_size=config.pipeline_model_parallel_size,
        virtual_pipeline_model_parallel_size=config.virtual_pipeline_model_parallel_size,
        zero_shard_size=parallel_config.zero_shard_size,
    )
    logger.info(
        f"dp {get_data_parallel_world_size()} | "
        f"pp {config.pipeline_model_parallel_size} | "
        f"tp {config.tensor_model_parallel_size} | "
        f"sp {config.sequence_parallel} | "
        f"vpp {config.virtual_pipeline_model_parallel_size}"
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
