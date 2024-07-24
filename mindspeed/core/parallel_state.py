# coding=utf-8
# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Expert parallel groups."""

from functools import wraps
from typing import Optional
from datetime import timedelta

import torch
import megatron

_CONTEXT_PARALLEL_GROUP_FOR_SEND_RECV_OVERLAP = None
_CONTEXT_PARALLEL_GROUP_FOR_HYBRID_ULYSSES = None
_CONTEXT_PARALLEL_GROUP_FOR_HYBRID_RING = None
_PIPELINE_MODEL_PARALLEL_GROUP_FOR_NEW_STREAM = None

_CONTEXT_PARALLEL_RANKS_FOR_HYBRID_ULYSSES = None
_CONTEXT_PARALLEL_RANKS_FOR_HYBRID_RING = None

_TENSOR_MODEL_PARALLEL_GROUP_FOR_ND1_DIM1 = None
_TENSOR_MODEL_PARALLEL_GROUP_FOR_ND1_DIM2 = None
_TENSOR_MODEL_PARALLEL_GROUP_FOR_ND2_DIM1 = None
_TENSOR_MODEL_PARALLEL_GROUP_FOR_ND2_DIM2 = None
_TENSOR_MODEL_PARALLEL_WORLD_SIZE_FOR_ND1_DIM1 = None
_TENSOR_MODEL_PARALLEL_WORLD_SIZE_FOR_ND1_DIM2 = None
_TENSOR_MODEL_PARALLEL_WORLD_SIZE_FOR_ND2_DIM1 = None
_TENSOR_MODEL_PARALLEL_WORLD_SIZE_FOR_ND2_DIM2 = None


def initialize_model_parallel_wrapper(initialize_model_parallel):
    @wraps(initialize_model_parallel)
    def wrapper(
            tensor_model_parallel_size: int = 1,
            pipeline_model_parallel_size: int = 1,
            virtual_pipeline_model_parallel_size: Optional[int] = None,
            pipeline_model_parallel_split_rank: Optional[int] = None,
            use_sharp: bool = False,
            context_parallel_size: int = 1,
            expert_model_parallel_size: int = 1,
            nccl_communicator_config_path: Optional[str] = None,
            distributed_timeout_minutes: int = 30,
            order: str = "tp-cp-ep-dp-pp",
    ):
        from megatron.training.utils import print_rank_0

        # Megatron doesn't allow ep & cp combination, set ep to 1 to bypass that, ep related groups will be regenerated
        initialize_model_parallel(
            tensor_model_parallel_size,
            pipeline_model_parallel_size,
            virtual_pipeline_model_parallel_size,
            pipeline_model_parallel_split_rank,
            use_sharp,
            context_parallel_size,
            1,
            nccl_communicator_config_path,
            distributed_timeout_minutes,
            order
        )

        rank = torch.distributed.get_rank()
        world_size: int = torch.distributed.get_world_size()
        num_tensor_model_parallel_groups: int = world_size // tensor_model_parallel_size
        num_pipeline_model_parallel_groups: int = world_size // pipeline_model_parallel_size
        data_parallel_size: int = world_size // (
                tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size
        )

        if data_parallel_size * context_parallel_size % expert_model_parallel_size != 0:
            raise RuntimeError(
                f"data_parallel_size * context_parallel_size ({data_parallel_size * context_parallel_size}) is not divisible by expert_model_parallel_size "
            )

        nccl_comm_cfgs = {}
        if nccl_communicator_config_path is not None:
            import yaml

            with open(nccl_communicator_config_path, "r") as stream:
                nccl_comm_cfgs = yaml.safe_load(stream)

        all_data_parallel_group_ranks = []
        all_data_parallel_group_ranks_with_cp = []
        for i in range(pipeline_model_parallel_size):
            start_rank = i * num_pipeline_model_parallel_groups
            end_rank = (i + 1) * num_pipeline_model_parallel_groups
            for j in range(context_parallel_size * tensor_model_parallel_size):
                ranks = range(
                    start_rank + j, end_rank, context_parallel_size * tensor_model_parallel_size
                )
                all_data_parallel_group_ranks.append(list(ranks))
            for j in range(tensor_model_parallel_size):
                ranks_with_cp = range(
                    start_rank + j, end_rank, tensor_model_parallel_size
                )
                all_data_parallel_group_ranks_with_cp.append(list(ranks_with_cp))

        timeout = timedelta(minutes=distributed_timeout_minutes)

        # Regenerate ep related groups because ep is set to 1 in initialize_model_parallel func
        tensor_and_data_group_size_with_cp: int = tensor_model_parallel_size * data_parallel_size * context_parallel_size
        num_tensor_and_data_groups_with_cp: int = world_size // tensor_and_data_group_size_with_cp
        tensor_and_expert_group_size: int = tensor_model_parallel_size * expert_model_parallel_size
        num_expert_groups: int = data_parallel_size * context_parallel_size // expert_model_parallel_size
        all_tensor_and_expert_group_ranks = []
        for i in range(num_tensor_and_data_groups_with_cp):
            for j in range(num_expert_groups):
                start_rank = i * tensor_and_data_group_size_with_cp + j * tensor_and_expert_group_size
                end_rank = i * tensor_and_data_group_size_with_cp + (j + 1) * tensor_and_expert_group_size
                ranks = range(start_rank, end_rank)
                all_tensor_and_expert_group_ranks.append(list(ranks))
                group = torch.distributed.new_group(
                    ranks, timeout=timeout,
                    pg_options=megatron.core.parallel_state.get_nccl_options('tp_exp', nccl_comm_cfgs)
                )
                if rank in ranks:
                    megatron.core.parallel_state._TENSOR_AND_EXPERT_PARALLEL_GROUP = group

        all_dp_modulo_exp_group_ranks = []
        for i in range(num_tensor_and_data_groups_with_cp):
            start_rank = i * tensor_and_data_group_size_with_cp
            end_rank = (i + 1) * tensor_and_data_group_size_with_cp
            for j in range(tensor_and_expert_group_size):
                ranks = range(start_rank + j, end_rank, tensor_and_expert_group_size)
                all_dp_modulo_exp_group_ranks.append(list(ranks))
                group = torch.distributed.new_group(
                    ranks, timeout=timeout,
                    pg_options=megatron.core.parallel_state.get_nccl_options('dp_modulo_exp', nccl_comm_cfgs)
                )
                group_gloo = torch.distributed.new_group(ranks, backend="gloo")
                if rank in ranks:
                    megatron.core.parallel_state._DATA_MODULO_EXPERT_PARALLEL_GROUP = group
                    megatron.core.parallel_state._DATA_MODULO_EXPERT_PARALLEL_GROUP_GLOO = group_gloo

        # Build expert parallel groups

        all_ep_groups = []
        for dp_cp_ranks in all_data_parallel_group_ranks_with_cp:
            for i in range(0, len(dp_cp_ranks), expert_model_parallel_size):
                ranks = dp_cp_ranks[i:i + expert_model_parallel_size]
                all_ep_groups.append(list(ranks))
                group = torch.distributed.new_group(
                    ranks, pg_options=megatron.core.parallel_state.get_nccl_options('exp', nccl_comm_cfgs)
                )
                if rank in ranks:
                    megatron.core.parallel_state._EXPERT_MODEL_PARALLEL_GROUP = group

        all_tp_groups = []
        for i in range(num_tensor_model_parallel_groups):
            ranks = range(i * tensor_model_parallel_size, (i + 1) * tensor_model_parallel_size)
            all_tp_groups.append(list(ranks))

        initialize_context_parallel_group_for_send_recv_overlap(
            tensor_model_parallel_size,
            pipeline_model_parallel_size,
            context_parallel_size,
            nccl_comm_cfgs
        )

        initialize_context_parallel_group_for_hybrid_cp(
            tensor_model_parallel_size,
            pipeline_model_parallel_size,
            context_parallel_size,
            nccl_comm_cfgs
        )

        print_rank_0(f"all tp gourps {all_tp_groups}")
        print_rank_0(f"all ep groups {all_ep_groups}")
        print_rank_0(f"all dp groups {all_data_parallel_group_ranks}")
        print_rank_0(f"all_dp_modulo_exp_group_ranks {all_dp_modulo_exp_group_ranks}")
        print_rank_0(f"all_tensor_and_expert_group_ranks {all_tensor_and_expert_group_ranks}")
        print_rank_0(f"all_data_parallel_group_ranks_with_cp {all_data_parallel_group_ranks_with_cp}")

        global _PIPELINE_MODEL_PARALLEL_GROUP_FOR_NEW_STREAM
        if _PIPELINE_MODEL_PARALLEL_GROUP_FOR_NEW_STREAM is not None:
            raise AttributeError('Pipeline parallel group for new stream is already initialized')
        num_pipeline_model_parallel_groups: int = world_size // pipeline_model_parallel_size
        for i in range(num_pipeline_model_parallel_groups):
            ranks = range(i, world_size, num_pipeline_model_parallel_groups)
            group = torch.distributed.new_group(
                ranks, pg_options=megatron.core.parallel_state.get_nccl_options('pp', nccl_comm_cfgs)
            )
            if rank in ranks:
                _PIPELINE_MODEL_PARALLEL_GROUP_FOR_NEW_STREAM = group

        from megatron.training import get_args
        args = get_args()
        initialize_ndmm_parallel_group(
            nccl_comm_cfgs,
            tensor_model_parallel_size=tensor_model_parallel_size,
            nd1_dim1_size=args.nd1_dim1_size,
            nd2_dim1_size=args.nd2_dim1_size,
        )

    return wrapper


def initialize_context_parallel_group_for_send_recv_overlap(
        tensor_model_parallel_size,
        pipeline_model_parallel_size,
        context_parallel_size,
        nccl_comm_cfgs
):
    from megatron.training import get_args
    if not get_args().use_cp_send_recv_overlap:
        return

    rank = torch.distributed.get_rank()
    world_size: int = torch.distributed.get_world_size()
    num_pipeline_model_parallel_groups: int = world_size // pipeline_model_parallel_size
    data_parallel_size: int = world_size // (
            tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size
    )
    global _CONTEXT_PARALLEL_GROUP_FOR_SEND_RECV_OVERLAP
    for i in range(pipeline_model_parallel_size):
        for j in range(data_parallel_size):
            start_rank = (
                    i * num_pipeline_model_parallel_groups
                    + j * tensor_model_parallel_size * context_parallel_size
            )
            end_rank = (
                    i * num_pipeline_model_parallel_groups
                    + (j + 1) * tensor_model_parallel_size * context_parallel_size
            )
            for k in range(tensor_model_parallel_size):
                ranks = range(start_rank + k, end_rank, tensor_model_parallel_size)
                group_send_recv_overlap = torch.distributed.new_group(
                    ranks, pg_options=megatron.core.parallel_state.get_nccl_options('cp2', nccl_comm_cfgs)
                )
                if rank in ranks:
                    _CONTEXT_PARALLEL_GROUP_FOR_SEND_RECV_OVERLAP = group_send_recv_overlap


def get_context_parallel_group_for_send_recv_overlap(check_initialized=True):
    """Get the context parallel group for send-recv overlap the caller rank belongs to."""
    if check_initialized:
        assert (
                _CONTEXT_PARALLEL_GROUP_FOR_SEND_RECV_OVERLAP is not None
        ), 'context parallel group for send-recv overlap is not initialized'
    return _CONTEXT_PARALLEL_GROUP_FOR_SEND_RECV_OVERLAP


def get_pipeline_parallel_group_for_new_stream():
    if _PIPELINE_MODEL_PARALLEL_GROUP_FOR_NEW_STREAM is None:
        raise AttributeError('Pipeline parallel group of backward is not initialized')
    return _PIPELINE_MODEL_PARALLEL_GROUP_FOR_NEW_STREAM


def initialize_context_parallel_group_for_hybrid_cp(
        tensor_model_parallel_size,
        pipeline_model_parallel_size,
        context_parallel_size,
        nccl_comm_cfgs
):
    from megatron.training import get_args
    if (not hasattr(get_args(), 'context_parallel_algo') or
            get_args().context_parallel_algo != 'hybrid_cp_algo'):
        return

    rank = torch.distributed.get_rank()
    world_size: int = torch.distributed.get_world_size()
    num_pipeline_model_parallel_groups: int = world_size // pipeline_model_parallel_size
    data_parallel_size: int = world_size // (
            tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size
    )

    ulysses_degree = get_args().ulysses_degree_in_cp
    assert (context_parallel_size > ulysses_degree and context_parallel_size % ulysses_degree == 0)
    ring_degree = context_parallel_size // ulysses_degree

    global _CONTEXT_PARALLEL_GROUP_FOR_HYBRID_ULYSSES
    global _CONTEXT_PARALLEL_RANKS_FOR_HYBRID_ULYSSES
    global _CONTEXT_PARALLEL_GROUP_FOR_HYBRID_RING
    global _CONTEXT_PARALLEL_RANKS_FOR_HYBRID_RING
    for i in range(pipeline_model_parallel_size):
        for j in range(data_parallel_size):
            start_rank = (
                    i * num_pipeline_model_parallel_groups
                    + j * tensor_model_parallel_size * context_parallel_size
            )
            end_rank = (
                    i * num_pipeline_model_parallel_groups
                    + (j + 1) * tensor_model_parallel_size * context_parallel_size
            )
            for k in range(tensor_model_parallel_size):
                # cp ranks
                ranks = list(range(start_rank + k, end_rank, tensor_model_parallel_size))
                # ulysses cp ranks. 
                # Ulysses need higher communication bandwidth than Ring.
                # Try to put Ulysses ranks in the same node.
                for m in range(ring_degree):
                    ulysses_ranks = [ranks[idx] for idx in range(m * ulysses_degree, (m + 1) * ulysses_degree)]
                    ulysses_group = torch.distributed.new_group(
                        ulysses_ranks,
                        pg_options=megatron.core.parallel_state.get_nccl_options('cp_ulysses', nccl_comm_cfgs)
                    )
                    if rank in ulysses_ranks:
                        _CONTEXT_PARALLEL_GROUP_FOR_HYBRID_ULYSSES = ulysses_group
                        _CONTEXT_PARALLEL_RANKS_FOR_HYBRID_ULYSSES = ulysses_ranks

                # ring cp ranks
                for m in range(ulysses_degree):
                    ring_ranks = [ranks[idx] for idx in range(m, len(ranks), ulysses_degree)]
                    ring_group = torch.distributed.new_group(
                        ring_ranks, pg_options=megatron.core.parallel_state.get_nccl_options('cp_ring', nccl_comm_cfgs)
                    )
                    if rank in ring_ranks:
                        _CONTEXT_PARALLEL_GROUP_FOR_HYBRID_RING = ring_group
                        _CONTEXT_PARALLEL_RANKS_FOR_HYBRID_RING = ring_ranks


def get_context_parallel_group_for_hybrid_ulysses(check_initialized=True):
    """Get the context parallel group for hybrid ulysses the caller rank belongs to."""
    if check_initialized:
        assert (
                _CONTEXT_PARALLEL_GROUP_FOR_HYBRID_ULYSSES is not None
        ), 'context parallel group for hybrid ulysses is not initialized'
    return _CONTEXT_PARALLEL_GROUP_FOR_HYBRID_ULYSSES


def get_context_parallel_for_hybrid_ulysses_world_size():
    return torch.distributed.get_world_size(group=get_context_parallel_group_for_hybrid_ulysses())


def get_context_parallel_for_hybrid_ulysses_rank():
    return torch.distributed.get_rank(group=get_context_parallel_group_for_hybrid_ulysses())


def get_context_parallel_group_for_hybrid_ring(check_initialized=True):
    """Get the context parallel group for hybrid ring the caller rank belongs to."""
    if check_initialized:
        assert (
                _CONTEXT_PARALLEL_GROUP_FOR_HYBRID_RING is not None
        ), 'context parallel group for hybrid ring is not initialized'
    return _CONTEXT_PARALLEL_GROUP_FOR_HYBRID_RING


def get_context_parallel_for_hybrid_ring_world_size():
    return torch.distributed.get_world_size(group=get_context_parallel_group_for_hybrid_ring())


def get_context_parallel_for_hybrid_ring_rank():
    return torch.distributed.get_rank(group=get_context_parallel_group_for_hybrid_ring())


def get_context_parallel_for_hybrid_ring_global_ranks():
    assert (_CONTEXT_PARALLEL_GROUP_FOR_HYBRID_RING is not None
            ), 'context parallel group for hybrid ring is not initialized'
    global _CONTEXT_PARALLEL_RANKS_FOR_HYBRID_RING
    return _CONTEXT_PARALLEL_RANKS_FOR_HYBRID_RING


def destroy_model_parallel_wrapper(destroy_model_parallel):
    @wraps(destroy_model_parallel)
    def wrapper():
        destroy_model_parallel()

        global _CONTEXT_PARALLEL_GROUP_FOR_SEND_RECV_OVERLAP
        global _PIPELINE_MODEL_PARALLEL_GROUP_FOR_NEW_STREAM
        global _CONTEXT_PARALLEL_GROUP_FOR_HYBRID_RING
        global _CONTEXT_PARALLEL_GROUP_FOR_HYBRID_ULYSSES
        global _CONTEXT_PARALLEL_RANKS_FOR_HYBRID_RING
        global _CONTEXT_PARALLEL_RANKS_FOR_HYBRID_ULYSSES
        _CONTEXT_PARALLEL_GROUP_FOR_SEND_RECV_OVERLAP = None
        _PIPELINE_MODEL_PARALLEL_GROUP_FOR_NEW_STREAM = None
        _CONTEXT_PARALLEL_GROUP_FOR_HYBRID_RING = None
        _CONTEXT_PARALLEL_GROUP_FOR_HYBRID_ULYSSES = None
        _CONTEXT_PARALLEL_RANKS_FOR_HYBRID_RING = None
        _CONTEXT_PARALLEL_RANKS_FOR_HYBRID_ULYSSES = None

    return wrapper


def get_tensor_model_parallel_group_for_nd1_dim1(check_initialized=True):
    if check_initialized and _TENSOR_MODEL_PARALLEL_GROUP_FOR_ND1_DIM1 is None:
        raise AssertionError('tensor model parallel group for nd1 dim1 is not initialized')
    return _TENSOR_MODEL_PARALLEL_GROUP_FOR_ND1_DIM1


def get_tensor_model_parallel_group_for_nd1_dim2(check_initialized=True):
    if check_initialized and _TENSOR_MODEL_PARALLEL_GROUP_FOR_ND1_DIM2 is None:
        raise AssertionError('tensor model parallel group for nd1 dim2 is not initialized')
    return _TENSOR_MODEL_PARALLEL_GROUP_FOR_ND1_DIM2


def get_tensor_model_parallel_group_for_nd2_dim1(check_initialized=True):
    if check_initialized and _TENSOR_MODEL_PARALLEL_GROUP_FOR_ND2_DIM1 is None:
        raise AssertionError('tensor model parallel group for nd2 dim1 is not initialized')
    return _TENSOR_MODEL_PARALLEL_GROUP_FOR_ND2_DIM1


def get_tensor_model_parallel_group_for_nd2_dim2(check_initialized=True):
    if check_initialized and _TENSOR_MODEL_PARALLEL_GROUP_FOR_ND2_DIM2 is None:
        raise AssertionError('tensor model parallel group for nd2 dim2 is not initialized')
    return _TENSOR_MODEL_PARALLEL_GROUP_FOR_ND2_DIM2


def get_tensor_model_parallel_world_size_for_nd1_dim1():
    global _TENSOR_MODEL_PARALLEL_WORLD_SIZE_FOR_ND1_DIM1
    if _TENSOR_MODEL_PARALLEL_WORLD_SIZE_FOR_ND1_DIM1 is None:
        _TENSOR_MODEL_PARALLEL_WORLD_SIZE_FOR_ND1_DIM1 = torch.distributed.get_world_size(
            group=get_tensor_model_parallel_group_for_nd1_dim1()
        )
    return _TENSOR_MODEL_PARALLEL_WORLD_SIZE_FOR_ND1_DIM1


def get_tensor_model_parallel_world_size_for_nd1_dim2():
    global _TENSOR_MODEL_PARALLEL_WORLD_SIZE_FOR_ND1_DIM2
    if _TENSOR_MODEL_PARALLEL_WORLD_SIZE_FOR_ND1_DIM2 is None:
        _TENSOR_MODEL_PARALLEL_WORLD_SIZE_FOR_ND1_DIM2 = torch.distributed.get_world_size(
            group=get_tensor_model_parallel_group_for_nd1_dim2()
        )
    return _TENSOR_MODEL_PARALLEL_WORLD_SIZE_FOR_ND1_DIM2


def get_tensor_model_parallel_world_size_for_nd2_dim1():
    global _TENSOR_MODEL_PARALLEL_WORLD_SIZE_FOR_ND2_DIM1
    if _TENSOR_MODEL_PARALLEL_WORLD_SIZE_FOR_ND2_DIM1 is None:
        _TENSOR_MODEL_PARALLEL_WORLD_SIZE_FOR_ND2_DIM1 = torch.distributed.get_world_size(
            group=get_tensor_model_parallel_group_for_nd2_dim1()
        )
    return _TENSOR_MODEL_PARALLEL_WORLD_SIZE_FOR_ND2_DIM1


def get_tensor_model_parallel_world_size_for_nd2_dim2():
    global _TENSOR_MODEL_PARALLEL_WORLD_SIZE_FOR_ND2_DIM2
    if _TENSOR_MODEL_PARALLEL_WORLD_SIZE_FOR_ND2_DIM2 is None:
        _TENSOR_MODEL_PARALLEL_WORLD_SIZE_FOR_ND2_DIM2 = torch.distributed.get_world_size(
            group=get_tensor_model_parallel_group_for_nd2_dim2()
        )
    return _TENSOR_MODEL_PARALLEL_WORLD_SIZE_FOR_ND2_DIM2


def initialize_ndmm_parallel_group(
    nccl_comm_cfgs: dict,
    tensor_model_parallel_size: int = 1,
    nd1_dim1_size: int = 1,
    nd2_dim1_size: int = 1,
) -> None:
    import megatron.core.parallel_state as ps
    from megatron.training import get_args
    from megatron.training.global_vars import _ensure_var_is_not_initialized

    args = get_args()
    if not args.use_nd_matmul:
        return

    global _TENSOR_MODEL_PARALLEL_GROUP_FOR_ND1_DIM1
    _ensure_var_is_not_initialized(
        _TENSOR_MODEL_PARALLEL_GROUP_FOR_ND1_DIM1, 'nd1_dim1'
    )

    global _TENSOR_MODEL_PARALLEL_GROUP_FOR_ND1_DIM2
    _ensure_var_is_not_initialized(
        _TENSOR_MODEL_PARALLEL_GROUP_FOR_ND1_DIM2, 'nd1_dim2'
    )

    global _TENSOR_MODEL_PARALLEL_GROUP_FOR_ND2_DIM1
    _ensure_var_is_not_initialized(
        _TENSOR_MODEL_PARALLEL_GROUP_FOR_ND2_DIM1, 'nd2_dim1'
    )

    global _TENSOR_MODEL_PARALLEL_GROUP_FOR_ND2_DIM2
    _ensure_var_is_not_initialized(
        _TENSOR_MODEL_PARALLEL_GROUP_FOR_ND2_DIM2, 'nd2_dim2'
    )

    if tensor_model_parallel_size % nd1_dim1_size != 0:
        raise RuntimeError(
            f"tensor_model_parallel_size can't divisible by nd1_dim1_size"
        )

    if tensor_model_parallel_size % nd2_dim1_size != 0:
        raise RuntimeError(
            f"tensor_model_parallel_size can't divisible by nd2_dim1_size"
        )

    rank = torch.distributed.get_rank()
    world_size: int = torch.distributed.get_world_size()
    num_tensor_model_parallel_group: int = world_size // tensor_model_parallel_size

    for i in range(num_tensor_model_parallel_group):
        for j in range(tensor_model_parallel_size // nd1_dim1_size):
            ranks = range(
                i * tensor_model_parallel_size + j * nd1_dim1_size,
                i * tensor_model_parallel_size + (j + 1) * nd1_dim1_size
            )
            group = torch.distributed.new_group(
                ranks, pg_options=ps.get_nccl_options('nd1_dim1', nccl_comm_cfgs)
            )
            if rank in ranks:
                _TENSOR_MODEL_PARALLEL_GROUP_FOR_ND1_DIM1 = group

        nd1_dim2_size = tensor_model_parallel_size // nd1_dim1_size
        for j in range(tensor_model_parallel_size // nd1_dim2_size):
            ranks = range(
                i * tensor_model_parallel_size + j,
                (i + 1) * tensor_model_parallel_size,
                nd1_dim1_size
            )
            group = torch.distributed.new_group(
                ranks, pg_options=ps.get_nccl_options('nd1_dim2', nccl_comm_cfgs)
            )
            if rank in ranks:
                _TENSOR_MODEL_PARALLEL_GROUP_FOR_ND1_DIM2 = group

        for j in range(tensor_model_parallel_size // nd2_dim1_size):
            ranks = range(
                i * tensor_model_parallel_size + j * nd2_dim1_size,
                i * tensor_model_parallel_size + (j + 1) * nd2_dim1_size
            )
            group = torch.distributed.new_group(
                ranks, pg_options=ps.get_nccl_options('nd2_dim1', nccl_comm_cfgs)
            )
            if rank in ranks:
                _TENSOR_MODEL_PARALLEL_GROUP_FOR_ND2_DIM1 = group

        nd2_dim2_size = tensor_model_parallel_size // nd2_dim1_size
        for j in range(tensor_model_parallel_size // nd2_dim2_size):
            ranks = range(
                i * tensor_model_parallel_size + j,
                (i + 1) * tensor_model_parallel_size,
                nd2_dim1_size
            )
            group = torch.distributed.new_group(
                ranks, pg_options=ps.get_nccl_options('nd2_dim2', nccl_comm_cfgs)
            )
            if rank in ranks:
                _TENSOR_MODEL_PARALLEL_GROUP_FOR_ND2_DIM2 = group

