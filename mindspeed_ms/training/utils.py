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

import numpy as np
import mindspore as ms
from mindspore import mint, Tensor
import mindspore.communication.comm_func as comm_func
from mindspore.communication.management import init
from mindspore.communication.comm_func import broadcast
from mindspore.communication import get_rank
from mindspeed_ms.training import get_args
from mindspeed_ms.tools import logger
from mindspeed_ms.core.parallel_state import (
    get_data_parallel_world_size,
    initialize_model_parallel,
    get_context_parallel_for_hybrid_ring_rank,
    get_context_parallel_for_hybrid_ulysses_rank,
    get_context_parallel_for_hybrid_ring_world_size,
    get_context_parallel_for_hybrid_ulysses_world_size,
    get_data_parallel_group,
    get_tensor_model_parallel_world_size,
    get_context_parallel_rank,
    get_context_parallel_world_size
)
import mindspeed_ms.core.parallel_state as mpu

# pylint: disable=W0212
def print_rank_0(message):
    """If distributed is initialized, print only on rank 0."""
    if ms.communication._comm_helper._is_initialized():
        if ms.communication.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)


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
        zero_shard_size=config.zero_shard_size,
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


def get_batch_on_this_tp_rank(data_iterator):
    """
       get batch data on tensor parallelism.
    """
    args = get_args()

    def _broadcast(item):
        if item is not None:
            broadcast(item, get_tensor_model_parallel_src_rank_ms(), group=mpu.get_tensor_model_parallel_group())

    if mpu.get_tensor_model_parallel_rank() == 0:

        if data_iterator is not None:
            data = next(data_iterator)
            data = data['dataset']
        else:
            data = None

        batch = {
            'tokens': data["tokens"].astype(ms.int32),
            'labels': data["labels"].astype(ms.int32),
            'loss_mask': data["loss_mask"].astype(ms.float32),
            'attention_mask': None if "attention_mask" not in data else data["attention_mask"].astype(ms.int8),
            'position_ids': data["position_ids"].astype(ms.int32)
        }

        if args.pipeline_model_parallel_size == 1:
            _broadcast(batch['tokens'])
            _broadcast(batch['labels'])
            _broadcast(batch['loss_mask'])
            _broadcast(batch['attention_mask'])
            _broadcast(batch['position_ids'])

        elif mpu.is_pipeline_first_stage():
            _broadcast(batch['tokens'])
            _broadcast(batch['attention_mask'])
            _broadcast(batch['position_ids'])

        elif mpu.is_pipeline_last_stage():
            _broadcast(batch['labels'])
            _broadcast(batch['loss_mask'])
            _broadcast(batch['attention_mask'])
    else:
        tokens = Tensor(
            np.empty((args.global_batch_size // get_data_parallel_world_size(), args.seq_length), dtype=np.int32))
        labels = Tensor(
            np.empty((args.global_batch_size // get_data_parallel_world_size(), args.seq_length), dtype=np.int32))
        loss_mask = Tensor(
            np.empty((args.global_batch_size // get_data_parallel_world_size(), args.seq_length), dtype=np.float32))
        if args.create_attention_mask_in_dataloader:
            attention_mask = Tensor(np.empty(
                (args.global_batch_size // get_data_parallel_world_size(), 1, args.seq_length, args.seq_length),
                dtype=np.int8))
        else:
            attention_mask = None
        position_ids = Tensor(
            np.empty((args.global_batch_size // get_data_parallel_world_size(), args.seq_length), dtype=np.int32))

        if args.pipeline_model_parallel_size == 1:
            _broadcast(tokens)
            _broadcast(labels)
            _broadcast(loss_mask)
            _broadcast(attention_mask)
            _broadcast(position_ids)

        elif mpu.is_pipeline_first_stage():
            labels = None
            loss_mask = None

            _broadcast(tokens)
            _broadcast(attention_mask)
            _broadcast(position_ids)

        elif mpu.is_pipeline_last_stage():
            tokens = None
            position_ids = None

            _broadcast(labels)
            _broadcast(loss_mask)
            _broadcast(attention_mask)

        batch = {
            'tokens': tokens,
            'labels': labels,
            'loss_mask': loss_mask,
            'attention_mask': attention_mask,
            'position_ids': position_ids
        }

    return batch


def get_batch_on_this_cp_rank(batch):
    """ Slice batch input along sequence dimension into multiple chunks,
        which are parallelized across GPUs in a context parallel group.
    """

    # With causal masking, each token only attends to its prior tokens. Simply split
    # sequence into CP chunks can result in severe load imbalance. That's to say, chunks
    # at the end of sequence have bigger workload than others. To address this issue,
    # we split sequence into 2*CP ranks. Assuming CP=2, we then get 4 chunks, chunk_0
    # and chunk_3 are assigned to GPU0, chunk_1 and chunk_2 are assigned to GPU1, so
    # that we can get balanced workload among GPUs in a context parallel group.
    cp_size = get_context_parallel_world_size()
    if cp_size > 1:
        cp_rank = get_context_parallel_rank()
        for key, val in batch.items():
            if val is not None:
                seq_dim = 1 if key != 'attention_mask' else 2
                val = val.view(
                    *val.shape[0:seq_dim],
                    2 * cp_size,
                    val.shape[seq_dim] // (2 * cp_size),
                    *val.shape[(seq_dim + 1):],
                )
                index = Tensor([cp_rank, (2 * cp_size - cp_rank - 1)])
                val = val.index_select(seq_dim, index)
                val = val.view(*val.shape[0:seq_dim], -1, *val.shape[(seq_dim + 2):])
                batch[key] = val

    return batch


def get_tensor_model_parallel_src_rank_ms():
    global_rank = get_rank()
    local_world_size = get_tensor_model_parallel_world_size()
    return (global_rank // local_world_size) * local_world_size


def average_losses_across_data_parallel_group(losses):
    """Reduce a tensor of losses across all GPUs."""
    averaged_losses = mint.cat(
        [loss.copy().view(1) for loss in losses])
    averaged_losses = comm_func.all_reduce(averaged_losses, group=get_data_parallel_group())[0]
    averaged_losses = averaged_losses / get_data_parallel_world_size()

    return averaged_losses
