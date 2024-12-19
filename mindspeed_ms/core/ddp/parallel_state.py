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
# ======================

"""Model and data parallel groups."""
import numpy as np
from mindspore import log as logger
from mindspore.communication import create_group, destroy_group, get_rank

import mindspeed_ms.core.ddp.create_comm as create_comm
from mindspeed_ms.core.ddp.create_comm import get_dp_world_size as get_data_parallel_world_size, \
    get_dp_group as get_data_parallel_group

group_info_maps = {}


valid_groups = ["dp-zero", "dp-zero-grad"]

# zero shard size, not initialized
_ZERO_SHARD_SIZE = None
_ZERO_FULL_SHARD = True
_ZERO_WITH_CP = False

_USE_ZERO3 = True

def get_zero3_flag():
    global _USE_ZERO3
    return _USE_ZERO3

def set_zero3_flag(use_zero3):
    global _USE_ZERO3
    _USE_ZERO3 = use_zero3

class GroupInfo:
    """ Comm Group Info """

    def __init__(self):
        self.group = None
        self.world_size = None
        self.rank = None
        self.global_ranks = None
        self.is_group_created = False

    def reset(self):
        if self.group is not None and self.is_group_created:
            destroy_group(self.group)
        self.group = None
        self.world_size = None
        self.rank = None
        self.global_ranks = None
        self.is_group_created = False


def get_group_info(mode):
    global group_info_maps
    if mode not in group_info_maps:
        if mode not in valid_groups:
            raise ValueError(f'the group info {mode} is not valid.')
        group_info_maps[mode] = GroupInfo()
    return group_info_maps[mode]


# pylint: disable=W0613
def initialize_zero_group(zero_shard_size=-1):
    """Initialize model data parallel groups.
    """
    # initialize zero3 shard size
    set_zero_shard_size(zero_shard_size)
    if not get_zero_full_shard_flag():
        get_zero_shard_group()
        get_zero_shard_grad_group()


# pylint: disable=W0212
def get_dp_global_ranks(with_context_parallel=False):
    return create_comm._DATA_PARALLEL_GLOBAL_RANKS_WITH_CP if with_context_parallel \
        else create_comm._DATA_PARALLEL_GLOBAL_RANKS


### get group
# pylint: disable=C0330
def _get_group_helper(mode):
    comm_group = get_group_info(mode)
    if comm_group.group is None:
        raise RuntimeError(f"{mode} parallel group is not initialized. Please check whether communication "
                           f"is initialized and {mode} in order.")
    if not comm_group.is_group_created:
        create_group(comm_group.group, comm_group.global_ranks)
        comm_group.is_group_created = True
    return comm_group.group


def get_zero_parallel_group():
    """Get the tensor model parallel group the caller rank belongs to."""
    return _get_group_helper('dp-zero')


### get world size
def _get_world_size_helper(mode):
    comm_group = get_group_info(mode)
    return comm_group.world_size


def get_zero_shard_world_size():
    """Return world size for the data parallel group."""
    return _get_world_size_helper('dp-zero')


### get rank
def _get_rank_helper(mode):
    comm_group = get_group_info(mode)
    if comm_group.rank is not None:
        return comm_group.rank
    comm_group.rank = 0 if _get_world_size_helper(mode) == 1 else get_rank(group=_get_group_helper(mode))
    return comm_group.rank


def get_zero_shard_rank():
    """Return my rank for the data parallel group."""
    return _get_rank_helper('dp-zero')



def set_zero_shard_size(zero_shard_size):
    """initialize zero3 shard size"""
    try:
        dp_size = get_data_parallel_world_size()
    except AssertionError as e:
        raise RuntimeError("When using zero3 optimizer parallel. Data parallel communication "
                           "need be initialized. Please check 'dp' in order when calling "
                           "initialize_model_parallel.") from e
    if zero_shard_size == 1:
        raise ValueError("zero_shard_size should be greater than 1")
    if zero_shard_size != -1:
        if zero_shard_size > dp_size or dp_size % zero_shard_size != 0:
            logger.warning("zero_shard_size should be less than or equal to data parallel size or "
                           "zero_shard_size should be a factor of data parallel size, but got"
                           f"{zero_shard_size}, zero_shard_size will not take effect.")
        else:
            if zero_shard_size < dp_size:
                global _ZERO_FULL_SHARD
                _ZERO_FULL_SHARD = False
            dp_size = zero_shard_size

    global _ZERO_SHARD_SIZE
    _ZERO_SHARD_SIZE = dp_size


def get_zero_shard_size():
    """get zero3 shard size"""
    global _ZERO_SHARD_SIZE
    if _ZERO_SHARD_SIZE is None:
        raise RuntimeError("Zero shard size is not initialized")
    return _ZERO_SHARD_SIZE


def get_zero_full_shard_flag():
    """get whether zero3 shard size is unsaturated or not"""
    global _ZERO_FULL_SHARD
    return _ZERO_FULL_SHARD


def _local_rank_in_zero_shard_group():
    """get loacl rank in zero shard group"""
    dp_rank = get_dp_global_ranks(_ZERO_WITH_CP)
    zero_shard_size = get_zero_shard_size()
    # [0, 1, 2, 3, 4, 5, 6, 7] -> [0, 1, 4, 5], [2, 3, 6, 7]
    # [0, 2, 4, 6] -> [0, 4], [2, 6]
    new_dp_rank_order = []
    for i in range(get_data_parallel_world_size()):
        if i == get_data_parallel_world_size() - 1:
            new_dp_rank_order.append(get_data_parallel_world_size() - 1)
        else:
            new_dp_rank_order.append((i * zero_shard_size) % (get_data_parallel_world_size() - 1))
    all_rank_list_index = np.split(np.array(new_dp_rank_order), zero_shard_size)
    current_rank_list_in_zero = None
    group_rank_id = -1
    for group_rank_id, rank_list_index in enumerate(all_rank_list_index):
        current_rank = get_rank()
        rank_list = [dp_rank[i] for i in rank_list_index]
        if current_rank in rank_list:
            current_rank_list_in_zero = rank_list
            break
    if current_rank_list_in_zero is None or group_rank_id == -1:
        raise ValueError("Can not find current rank in zero shard group or group rank id is -1")
    current_rank_list_in_zero.sort()
    return current_rank_list_in_zero, group_rank_id


def get_zero_shard_group(with_context_parallel=False):
    """Get the data parallel group the caller rank belongs to."""
    global _ZERO_WITH_CP
    _ZERO_WITH_CP = with_context_parallel
    group = get_data_parallel_group(with_context_parallel)
    if get_zero_full_shard_flag():
        return group
    dp_rank = get_dp_global_ranks(with_context_parallel)
    zero_shard_size = get_zero_shard_size()
    # [0, 1, 2, 3, 4, 5, 6, 7] -> [0, 1, 2, 3], [4, 5, 6, 7]
    # [0, 2, 4, 6] -> [0, 2], [4, 6]
    current_idx = dp_rank.index(get_rank())
    idx, _ = divmod(current_idx, zero_shard_size)
    current_rank_list_in_zero = dp_rank[idx * zero_shard_size: (idx + 1) * zero_shard_size]
    group_name = "dp-zero-" + "-".join([str(i) for i in current_rank_list_in_zero])
    comm_group = get_group_info("dp-zero")
    comm_group.group = group_name
    comm_group.global_ranks = current_rank_list_in_zero
    comm_group.world_size = zero_shard_size
    _, group_rank_id = _local_rank_in_zero_shard_group()
    comm_group.rank = group_rank_id
    return _get_group_helper("dp-zero")


def get_zero_shard_grad_group():
    """Get the data parallel group the caller rank belongs to."""
    zero_shard_size = get_zero_shard_size()
    current_rank_list_in_zero, group_rank_id = _local_rank_in_zero_shard_group()
    group_name = "dp-zero-grad-" + "-".join([str(i) for i in current_rank_list_in_zero])
    comm_group = get_group_info("dp-zero-grad")
    comm_group.group = group_name
    comm_group.global_ranks = current_rank_list_in_zero
    comm_group.world_size = zero_shard_size
    comm_group.rank = group_rank_id
    return _get_group_helper("dp-zero-grad")
