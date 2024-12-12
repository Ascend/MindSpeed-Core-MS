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
import abc
import warnings
import numpy as np

import mindspore
from mindspore.communication import create_group, destroy_group, get_group_size, get_rank
from mindspore import hal
from mindspeed_ms.tools import logger
from mindspeed_ms.training.global_vars import get_args

_GLOBAL_STREAM = None
_SP_SEND_STREAM = None
_SP_RECV_STREAM = None
_SP_SEND_OML_STREAM = None
_SP_RECV_OML_STREAM = None
_GLOBAL_MP_RANK = None

group_info_maps = {}

# special_groups has a different initialization process compared to normal_groups
normal_groups = ['tp', 'dp', 'pp', 'cp', 'dp-cp', 'tp-pp', 'tp-dp-cp', 'tp-dp', 'tp-cp']
special_groups = ['ep', 'tp-ep', 'tp-ep-pp', 'dp-independent_ep', 'vpp', 'embedding', 'position_embedding',
                  "dp-zero", "dp-zero-grad", "dp-zero-tp", "cp_ulysses", "cp_ring", 'tp_x', 'tp_y']
valid_groups = normal_groups + special_groups

# A list of global ranks for pipeline group
_PIPELINE_GLOBAL_RANKS = None
# zero shard size, not initialized
_ZERO_SHARD_SIZE = None
_ZERO_FULL_SHARD = True
_ZERO_WITH_CP = False


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


class CreateCommGroups():
    '''Generate ranks for each parallel type.'''

    def __init__(self, tp, ep, dp, pp, cp, order):
        self.tp = tp
        self.ep = ep
        self.dp = dp
        self.pp = pp
        self.cp = cp
        self.world_size = tp * dp * pp * cp

        self.name_to_size = {
            "tp": self.tp,
            "pp": self.pp,
            "dp": self.dp,
            "ep": self.ep,
            "cp": self.cp,
        }
        self.rank = get_rank()
        self.order = order

        for name, size in self.name_to_size.items():
            if name not in order:
                if size == 1:
                    order = order + '-' + name
                else:
                    raise RuntimeError(
                        f"The size of ({name}) is ({size}), \
                        but you haven't specified the order ({self.order})."
                    )

        self.order_w_ep = order
        self.order_wo_ep = '-'.join([token for token in order.split('-') if token != 'ep'])
        self.ordered_size_wo_ep = []
        self.ordered_size_w_ep = []

        for token in order.split('-'):
            if token == 'dp':
                self.ordered_size_w_ep.append(self.dp // self.ep)
                self.ordered_size_wo_ep.append(self.dp)
            elif token == 'ep':
                self.ordered_size_w_ep.append(self.ep)
            else:
                self.ordered_size_w_ep.append(self.name_to_size.get(token))
                self.ordered_size_wo_ep.append(self.name_to_size.get(token))

    def get_mask(self, order, token):
        ordered_token = order.split('-')
        token = token.split('-')
        mask = [False] * len(ordered_token)
        for t in token:
            mask[ordered_token.index(t)] = True
        return mask

    def get_ranks(self, token, independent_ep=False):
        '''Get rank group by input token.

        Arguments:
            token (str): Specify the ranks type that want to get. Use a hyphen '-' to separate multiple parallel types.
            independent_ep (bool): Whether to treat EP and DP independently. Default: False.
        '''
        if independent_ep:
            parallel_size = self.ordered_size_w_ep
            order = self.order_w_ep
        else:
            parallel_size = self.ordered_size_wo_ep
            order = self.order_wo_ep
        mask = self.get_mask(order, token)
        ranks = self._dispatch_comm_ranks(self.world_size, parallel_size, mask)
        return ranks

    def init_group(self, input_mode, independent_ep=False):
        '''Create data parallel group.'''
        mode = input_mode + '-independent_ep' if input_mode == 'dp' and independent_ep else input_mode
        comm_group = get_group_info(mode)

        if comm_group.group is not None:
            raise RuntimeError(f'{mode} parallel group is already initialized.')

        for ranks in self.get_ranks(input_mode, independent_ep=independent_ep):
            if self.rank in ranks:
                group = mode + '-' + '-'.join([str(i) for i in ranks])
                comm_group.group = group
                comm_group.global_ranks = ranks
                comm_group.world_size = len(ranks)

    def init_embedding_group(self, pipeline_model_parallel_split_rank):
        '''Init pipeline parallel group.'''
        embedding_group = get_group_info('embedding')
        position_embedding_group = get_group_info('position_embedding')
        if embedding_group.group is not None:
            raise RuntimeError('embedding group is already initialized.')
        if position_embedding_group.group is not None:
            raise RuntimeError('position embedding group is already initialized.')
        for ranks in self.get_ranks('pp'):
            # Setup embedding group (to exchange gradients between
            # first and last stages).
            if len(ranks) > 1:
                embedding_ranks = [ranks[0], ranks[-1]]
                position_embedding_ranks = [ranks[0]]
                if pipeline_model_parallel_split_rank is not None:
                    if ranks[pipeline_model_parallel_split_rank] not in embedding_ranks:
                        embedding_ranks = [
                            ranks[0],
                            ranks[pipeline_model_parallel_split_rank],
                            ranks[-1],
                        ]
                    if ranks[pipeline_model_parallel_split_rank] not in position_embedding_ranks:
                        position_embedding_ranks = [ranks[0], ranks[pipeline_model_parallel_split_rank]]
            else:
                embedding_ranks = ranks
                position_embedding_ranks = ranks

            if self.rank in embedding_ranks:
                group = 'embedding-' + '-'.join([str(i) for i in embedding_ranks])
                embedding_group.group = group
                embedding_group.global_ranks = embedding_ranks
            if self.rank in position_embedding_ranks:
                group = 'position_embedding-' + '-'.join([str(i) for i in position_embedding_ranks])
                position_embedding_group.group = group
                position_embedding_group.global_ranks = position_embedding_ranks

    def init_hybrid_cp_group(self, ulysses_degree, ring_degree):
        """"Init hybrid context parallel group."""
        cp_ring_mode = "cp_ring"
        cp_ulysses_mode = "cp_ulysses"
        org_comm_group = get_group_info("cp")
        global_ranks = org_comm_group.global_ranks
        for m in range(ring_degree):
            ulysses_ranks = [global_ranks[idx] for idx in range(m * ulysses_degree, (m + 1) * ulysses_degree)]
            if self.rank in ulysses_ranks:
                common_group = get_group_info(cp_ulysses_mode)
                group_name = cp_ulysses_mode + "-" + '-'.join([str(i) for i in ulysses_ranks])
                common_group.group = group_name
                common_group.global_ranks = ulysses_ranks
                common_group.world_size = len(ulysses_ranks)
        for m in range(ulysses_degree):
            ring_ranks = [global_ranks[idx] for idx in range(m, len(global_ranks), ulysses_degree)]
            if self.rank in ring_ranks:
                common_group = get_group_info(cp_ring_mode)
                group_name = cp_ring_mode + "-" + '-'.join([str(i) for i in ring_ranks])
                common_group.group = group_name
                common_group.global_ranks = ring_ranks
                common_group.world_size = len(ring_ranks)

    def _dispatch_comm_ranks(self, world_size, parallel_size, mask):
        """dispatch comm ranks"""
        def prefix_product(a, init=1):
            r = [init]
            for v in a:
                init = init * v
                r.append(init)
            return r

        def modulo(index, shape, stride=None):
            if stride is None:
                stride = prefix_product(shape)
            idx = [(index // d) % s for s, d in zip(shape, stride)]
            if (
                    sum([x * y for x, y in zip(idx, stride[:-1])]) != index
            ):
                raise ValueError("idx {} with shape {} mismatch the return idx {}".format(index, shape, idx))
            return idx

        masked_shape = [s for s, m in zip(parallel_size, mask) if m]
        unmasked_shape = [s for s, m in zip(parallel_size, mask) if not m]

        global_stride = prefix_product(parallel_size)
        masked_stride = [d for d, m in zip(global_stride, mask) if m]
        unmasked_stride = [d for d, m in zip(global_stride, mask) if not m]

        group_size = prefix_product(masked_shape)[-1]
        num_of_group = world_size // group_size

        ranks = []
        for group_index in range(num_of_group):
            # get indices from unmaksed for group_index.
            decomposed_group_idx = modulo(group_index, unmasked_shape)
            rank = []
            for rank_in_group in range(group_size):
                # get indices from masked for rank_in_group.
                decomposed_rank_idx = modulo(rank_in_group, masked_shape)
                masked_inner_product = sum([x * y for x, y in zip(decomposed_rank_idx, masked_stride)])
                unmasked_inner_product = sum([x * y for x, y in zip(decomposed_group_idx, unmasked_stride)])
                rank.append(masked_inner_product + unmasked_inner_product)
            ranks.append(rank)
        return ranks

    def init_tp2d_groups(self, tp_x, tp_y):
        """init 2D TP groups"""
        for mode in ('tp_x', 'tp_y'):
            comm_group = get_group_info(mode)

            if comm_group.group is not None:
                raise RuntimeError(f'{mode} parallel group is already initialized.')

            for ranks in self.get_tp2d_ranks(mode, tp_x, tp_y):
                if self.rank in ranks:
                    group = mode + '-' + '-'.join([str(i) for i in ranks])
                    comm_group.group = group
                    comm_group.global_ranks = ranks
                    comm_group.world_size = len(ranks)

    def get_tp2d_ranks(self, mode, tp_x, tp_y):
        """get 2D TP ranks"""
        num_tensor_model_parallel_group = self.world_size // self.tp

        ranks = []
        if mode == 'tp_x':
            for i in range(num_tensor_model_parallel_group):
                for j in range(self.tp // tp_x):
                    rank = range(
                        i * self.tp + j * tp_x,
                        i * self.tp + (j + 1) * tp_x
                    )
                    ranks.append(list(rank))
        if mode == 'tp_y':
            for i in range(num_tensor_model_parallel_group):
                for j in range(self.tp // tp_y):
                    rank = range(
                        i * self.tp + j,
                        (i + 1) * self.tp,
                        tp_x
                    )
                    ranks.append(list(rank))
        return ranks


# pylint: disable=W0613
def initialize_model_parallel(tensor_model_parallel_size=1,
                              pipeline_model_parallel_size=1,
                              virtual_pipeline_model_parallel_size=None,
                              pipeline_model_parallel_split_rank=None,
                              context_parallel_size=1,
                              expert_model_parallel_size=1,
                              order="tp-cp-ep-dp-pp",
                              communicator_config_path=None,
                              zero_shard_size=-1,
                              tp_2d=False,
                              tp_x=1,
                              tp_y=1,
                              **kwargs):
    """Initialize model data parallel groups.
    """

    # pylint: disable=W0212
    if not mindspore.communication._comm_helper._is_initialized():
        raise RuntimeError('mindspore.communication._comm_helper is not initialized.')
    world_size = get_group_size()

    minimum_world_size = (tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size)

    if world_size % minimum_world_size != 0:
        raise RuntimeError(
            f"world_size ({world_size}) is not divisible by tensor_model_parallel_size "
            f"({tensor_model_parallel_size}) x pipeline_model_parallel_size ({pipeline_model_parallel_size}) "
            f"x context_parallel_size ({context_parallel_size})"
        )

    data_parallel_size = world_size // minimum_world_size

    if data_parallel_size % expert_model_parallel_size != 0:
        raise RuntimeError(
            f"data_parallel_size ({data_parallel_size}) is not divisible by expert_model_parallel_size "
        )

    if expert_model_parallel_size > 1 and context_parallel_size > 1:
        raise RuntimeError(
            f"combination of expert model prallellism and context parallelism is not supported"
        )

    if virtual_pipeline_model_parallel_size is not None:
        if pipeline_model_parallel_size < 2:
            raise RuntimeError(
                "pipeline-model-parallel size should be greater than 1 with interleaved schedule"
            )
        vpp_group = get_group_info('vpp')
        vpp_group.rank = 0
        vpp_group.world_size = virtual_pipeline_model_parallel_size

    order = order.lower()
    order_list = order.split('-')
    if not order:
        raise RuntimeError(f"order can not be empty.")
    if len(set(order_list)) != len(order_list):
        raise RuntimeError(f"Duplicate elements in order ({order}).")
    if 'ep' in order:
        if 'ep-dp' not in order and 'dp-ep' not in order:
            raise RuntimeError(f"The ep and dp must be adjacent in order ({order}).")

    for key in kwargs:
        logger.warning(f"The parameter '{key}' is not used in initialize_model_parallel.")

    rank_generator = CreateCommGroups(tp=tensor_model_parallel_size,\
                                      ep=expert_model_parallel_size, \
                                      dp=data_parallel_size, pp=pipeline_model_parallel_size, \
                                      cp=context_parallel_size, order=order)

    if tp_2d:
        rank_generator.init_tp2d_groups(tp_x, tp_y)

    # Build the basic parallel groups.
    for mode in normal_groups:
        rank_generator.init_group(mode)

    # Build the expert-parallel groups which share ranks with DP.
    rank_generator.init_group('ep', independent_ep=True)
    rank_generator.init_group('tp-ep', independent_ep=True)
    rank_generator.init_group('tp-ep-pp', independent_ep=True)
    rank_generator.init_group('dp', independent_ep=True)
    global _GLOBAL_MP_RANK
    _GLOBAL_MP_RANK = [rank for rank in rank_generator.get_ranks("tp-pp")]

    # Build the pipeline-parallel related groups.
    rank_generator.init_embedding_group(pipeline_model_parallel_split_rank)
    global _PIPELINE_GLOBAL_RANKS
    all_pp_ranks = rank_generator.get_ranks('pp')
    for pp_ranks in all_pp_ranks:
        if rank_generator.rank in pp_ranks:
            _PIPELINE_GLOBAL_RANKS = pp_ranks
            break

    global _GLOBAL_STREAM
    if _GLOBAL_STREAM is not None:
        raise RuntimeError('Global stream is already initialized.')
    _GLOBAL_STREAM = hal.Stream()

    global _SP_SEND_STREAM
    global _SP_RECV_STREAM
    global _SP_SEND_OML_STREAM
    global _SP_RECV_OML_STREAM
    if context_parallel_size > 1:
        _SP_SEND_STREAM = hal.Stream()
        _SP_SEND_OML_STREAM = hal.Stream()
        _SP_RECV_OML_STREAM = hal.Stream()

    # a temporary workaround for dp group failure initialization in ms.dataset
    if get_data_parallel_world_size() > 1:
        get_data_parallel_group()
    if get_tensor_model_parallel_world_size() > 1:
        get_tensor_model_parallel_group()
    if get_context_parallel_world_size() > 1:
        get_context_parallel_group()
    # initialize zero3 shard size
    set_zero_shard_size(zero_shard_size)
    if not get_zero_full_shard_flag():
        get_zero_shard_group()
        get_zero_shard_grad_group()
    get_zero_shard_tp_group()
    # initialize hybrid context parallel group
    initialize_context_parallel_group_for_hybrid_cp(context_parallel_size, rank_generator)


def is_initialized():
    """Useful for code segments that may be accessed with or without mpu initialization"""
    comm_group = get_group_info('dp')
    return comm_group.group is not None


def is_uninitialized() -> bool:
    """Check if parallel state has been initialized

    Deprecated. Use is_initialized instead.
    """
    warnings.warn(
        "is_uninitialized is deprecated, use is_initialized instead", DeprecationWarning,
    )
    return not is_initialized()


def model_parallel_is_initialized():
    """Check if model and data parallel groups are initialized."""
    for name in ['tp', 'pp', 'dp']:
        comm_group = get_group_info(name)
        if comm_group.group is None:
            return False
    return True


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


def get_tensor_model_parallel_group():
    """Get the tensor model parallel group the caller rank belongs to."""
    return _get_group_helper('tp')


def get_tp_x_group():
    return _get_group_helper('tp_x')


def get_tp_y_group():
    return _get_group_helper('tp_y')


def get_zero_parallel_group():
    """Get the tensor model parallel group the caller rank belongs to."""
    return _get_group_helper('dp-zero')


def get_context_parallel_group():
    """Get the context parallel group the caller rank belongs to."""
    return _get_group_helper('cp')


def get_expert_model_parallel_group():
    return _get_group_helper('ep')


def get_data_parallel_group(with_context_parallel=False):
    """Get the data parallel group the caller rank belongs to."""
    return _get_group_helper('dp-cp') if with_context_parallel else _get_group_helper('dp')


def get_pipeline_model_parallel_group():
    """Get the pipeline model parallel group the caller rank belongs to."""
    return _get_group_helper('pp')


def get_embedding_group():
    """Get the embedding group the caller rank belongs to."""
    return _get_group_helper('embedding')


def get_position_embedding_group():
    """Get the position embedding group the caller rank belongs to."""
    return _get_group_helper('position_embedding')


def get_tensor_and_expert_parallel_group():
    return _get_group_helper('tp-ep')


def get_data_modulo_expert_parallel_group():
    return _get_group_helper('dp-independent_ep')


def get_model_parallel_group(with_expert_parallel=False):
    """Get the model parallel group the caller rank belongs to."""
    return _get_group_helper('tp-ep-pp') if with_expert_parallel else _get_group_helper('tp-pp')


def get_tensor_and_data_parallel_group(with_context_parallel=False):
    return _get_group_helper('tp-dp-cp') if with_context_parallel else _get_group_helper('tp-dp')


def get_tensor_and_context_parallel_group():
    return _get_group_helper('tp-cp')

def get_context_parallel_group_for_hybrid_ulysses():
    """Get the hybrid context parallel ulysses group the caller rank belongs to."""
    return _get_group_helper('cp_ulysses')

def get_context_parallel_group_for_hybrid_ring():
    """Get the hybrid context parallel ring group the caller rank belongs to."""
    return _get_group_helper('cp_ring')

### get global ranks
def _get_global_ranks_helper(mode, check_initialized=True):
    comm_group = get_group_info(mode)
    if check_initialized:
        if comm_group.global_ranks is None:
            raise RuntimeError(f"{mode} parallel group is not initialized. Please check whether communication "
                               f"is initialized and {mode} in order.")
    return comm_group.group


# pylint: disable=C0330
def get_cp_global_ranks(check_initialized=True):
    """Get all global ranks of the context parallel group that the caller rank belongs to."""
    return _get_global_ranks_helper('cp', check_initialized)


### get world size
def _get_world_size_helper(mode):
    comm_group = get_group_info(mode)
    return comm_group.world_size


def get_tensor_model_parallel_world_size():
    """Return world size for the tensor model parallel group."""
    return _get_world_size_helper('tp')


def get_tp_x_world_size():
    return _get_world_size_helper('tp_x')


def get_tp_y_world_size():
    return _get_world_size_helper('tp_y')


def get_context_parallel_world_size():
    """Return world size for the context parallel group."""
    return _get_world_size_helper('cp')


def get_expert_model_parallel_world_size():
    """Return world size for the expert model parallel group"""
    tensor_and_expert_parallel_world_size = _get_world_size_helper('tp-ep')
    return tensor_and_expert_parallel_world_size // get_tensor_model_parallel_world_size()


def get_data_parallel_world_size(with_context_parallel=False):
    """Return world size for the data parallel group."""
    return _get_world_size_helper('dp-cp') if with_context_parallel else _get_world_size_helper('dp')


def get_pipeline_model_parallel_world_size():
    """Return world size for the pipeline model parallel group."""
    return _get_world_size_helper('pp')


def get_virtual_pipeline_model_parallel_world_size():
    """Return world size for the virtual pipeline model parallel group."""
    return _get_world_size_helper('vpp')


def get_tensor_and_expert_parallel_world_size():
    """Return world size for the expert model parallel group times model parallel group.
       Currently, each expert will also be distributed across TP group by default.
    """
    return _get_world_size_helper('tp-ep')


def get_tensor_and_context_parallel_world_size():
    """Return world size for the tensor parallel group and context parallel group."""
    return _get_world_size_helper('tp-cp')


def get_data_modulo_expert_parallel_world_size():
    return _get_world_size_helper('dp-independent_ep')

def get_context_parallel_for_hybrid_ulysses_world_size():
    """Return world size for the hybrid context parallel ulysses group."""
    return _get_world_size_helper('cp_ulysses')

def get_context_parallel_for_hybrid_ring_world_size():
    """Return world size for the hybrid context parallel ring group."""
    return _get_world_size_helper('cp_ring')

### get rank
def _get_rank_helper(mode):
    comm_group = get_group_info(mode)
    if comm_group.rank is not None:
        return comm_group.rank
    comm_group.rank = 0 if _get_world_size_helper(mode) == 1 else get_rank(group=_get_group_helper(mode))
    return comm_group.rank


def get_tensor_model_parallel_rank():
    """Return my rank for the tensor model parallel group."""
    return _get_rank_helper('tp')


def get_tp_x_rank():
    return _get_rank_helper('tp_x')


def get_tp_y_rank():
    return _get_rank_helper('tp_y')


def get_zero_parallel_rank():
    """Return my rank for the tensor model parallel group."""
    return _get_rank_helper('dp-zero-tp')


def get_context_parallel_rank():
    """Return my rank for the context parallel group."""
    return _get_rank_helper('cp')


def get_expert_model_parallel_rank():
    """Return my rank for the expert parallel group"""
    tensor_and_expert_parallel_rank = _get_rank_helper('tp-ep')
    return tensor_and_expert_parallel_rank // get_tensor_model_parallel_world_size()


def get_data_parallel_rank(with_context_parallel=False):
    """Return my rank for the data parallel group."""
    return _get_rank_helper('dp-cp') if with_context_parallel else _get_rank_helper('dp')


def get_zero_shard_rank():
    """Return my rank for the data parallel group."""
    return _get_rank_helper('dp-zero')


def get_pipeline_model_parallel_rank():
    """Return my rank for the pipeline model parallel group."""
    return _get_rank_helper('pp')

def get_context_parallel_for_hybrid_ulysses_rank():
    """Return my rank for the hybrid context parallel ulysses group."""
    return _get_rank_helper('cp_ulysses')

def get_context_parallel_for_hybrid_ring_rank():
    """Return my rank for the hybrid context parallel ring group."""
    return _get_rank_helper('cp_ring')

def get_pipeline_model_parallel_first_rank():
    """Return the global rank of the first precess in the pipeline"""
    if _PIPELINE_GLOBAL_RANKS is None:
        raise Exception("Pipeline parallel group is not initialized")
    return _PIPELINE_GLOBAL_RANKS[0]


def get_pipeline_model_parallel_last_rank():
    """Return the global rank of the last precess in the pipeline"""
    if _PIPELINE_GLOBAL_RANKS is None:
        raise Exception("Pipeline parallel group is not initialized")
    last_rank_local = get_pipeline_model_parallel_world_size() - 1
    return _PIPELINE_GLOBAL_RANKS[last_rank_local]


def get_pipeline_model_parallel_prev_rank():
    """Return the global rank that precedes the caller in the pipeline"""
    if _PIPELINE_GLOBAL_RANKS is None:
        raise Exception("Pipeline parallel group is not initialized")
    rank_in_pipeline = get_pipeline_model_parallel_rank()
    world_size = get_pipeline_model_parallel_world_size()
    return _PIPELINE_GLOBAL_RANKS[(rank_in_pipeline - 1) % world_size]


def get_pipeline_model_parallel_next_rank():
    """Return the global rank that follows the caller in the pipeline"""
    if _PIPELINE_GLOBAL_RANKS is None:
        raise Exception("Pipeline parallel group is not initialized")
    rank_in_pipeline = get_pipeline_model_parallel_rank()
    world_size = get_pipeline_model_parallel_world_size()
    return _PIPELINE_GLOBAL_RANKS[(rank_in_pipeline + 1) % world_size]


def is_pipeline_first_stage(ignore_virtual=False):
    """Return True if in the first pipeline model-parallel stage, False otherwise."""
    if not ignore_virtual:
        if get_virtual_pipeline_model_parallel_world_size() is not None \
            and get_virtual_pipeline_model_parallel_rank() != 0:
            return False
    return get_pipeline_model_parallel_rank() == 0


def is_pipeline_last_stage(ignore_virtual=False):
    """Return True if in the last pipeline model-parallel stage, False otherwise."""
    if not ignore_virtual:
        vpp_world_size = get_virtual_pipeline_model_parallel_world_size()
        if vpp_world_size is not None and get_virtual_pipeline_model_parallel_rank() != (vpp_world_size - 1):
            return False
    return get_pipeline_model_parallel_rank() == (get_pipeline_model_parallel_world_size() - 1)


def is_rank_in_embedding_group(ignore_virtual=False):
    """Return true if current rank is in embedding group, False otherwise."""
    ret = False
    rank = get_rank()
    embedding_group = get_group_info('embedding')
    global_ranks = embedding_group.global_ranks
    if global_ranks is None:
        return False
    if ignore_virtual:
        return rank in global_ranks
    if rank in global_ranks:
        if rank == global_ranks[0]:
            ret = is_pipeline_first_stage(ignore_virtual=False)
        elif rank == global_ranks[-1]:
            ret = is_pipeline_last_stage(ignore_virtual=False)
        else:
            ret = True
    return ret


def get_virtual_pipeline_model_parallel_rank():
    """Get the virtual pipeline-parallel rank."""
    comm_group = get_group_info('vpp')
    return comm_group.rank


def set_virtual_pipeline_model_parallel_rank(rank):
    """Set the virtual pipeline-parallel rank."""
    comm_group = get_group_info('vpp')
    comm_group.rank = rank


def set_expert_model_parallel_rank(rank):
    """Set expert model parallel rank."""
    comm_group = get_group_info('ep')
    comm_group.rank = rank


def get_stream():
    """Return global stream. There is only one stream for each npu."""
    if _GLOBAL_STREAM is None:
        raise RuntimeError("Global stream is not initialized")
    return _GLOBAL_STREAM


def get_sp_send_stream():
    """Return send stream for sequence parallel."""
    if _SP_SEND_STREAM is None:
        raise RuntimeError("Sp send stream is not initialized")
    return _SP_SEND_STREAM


def get_sp_recv_stream():
    """Return recv stream for sequence parallel."""
    if _SP_RECV_STREAM is None:
        raise RuntimeError("Sp receive stream is not initialized")
    return _SP_RECV_STREAM


def get_sp_send_oml_stream():
    """Return send stream for sequence parallel."""
    if _SP_SEND_OML_STREAM is None:
        raise RuntimeError("Sp send oml stream is not initialized")
    return _SP_SEND_OML_STREAM


def get_sp_recv_oml_stream():
    """Return recv stream for sequence parallel."""
    if _SP_RECV_OML_STREAM is None:
        raise RuntimeError("Sp receive oml stream is not initialized")
    return _SP_RECV_OML_STREAM


def destroy_model_parallel():
    """Set the groups to none."""
    global group_info_maps
    for _, comm_group in group_info_maps.items():
        comm_group.reset()
    global _GLOBAL_STREAM
    _GLOBAL_STREAM = None
    global _SP_SEND_STREAM
    _SP_SEND_STREAM = None
    global _SP_RECV_STREAM
    _SP_RECV_STREAM = None
    global _SP_SEND_OML_STREAM
    _SP_SEND_OML_STREAM = None
    global _SP_RECV_OML_STREAM
    _SP_RECV_OML_STREAM = None


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
            if dp_size <= 8:
                logger.warning("When using zero unsaturated shard, data parallel size is recommended to be greater "
                               f"then 8, but got {dp_size}. Unless the performance may be worse.")
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


def _local_rank_in_zero_shard_group(dp_rank):
    """inner func, calculate the same param group"""
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
    group = get_group_info("dp-cp") if with_context_parallel else get_group_info("dp")
    if get_zero_full_shard_flag():
        return group.group
    dp_rank = group.global_ranks
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
    comm_group.rank = idx
    return _get_group_helper("dp-zero")


def get_zero_shard_grad_group():
    """Get the data parallel group the caller rank belongs to."""
    zero_shard_size = get_zero_shard_size()
    group = get_group_info("dp-cp") if _ZERO_WITH_CP else get_group_info("dp")
    if get_zero_full_shard_flag():
        return group.group
    dp_rank = group.global_ranks
    current_rank_list_in_zero, group_rank_id = _local_rank_in_zero_shard_group(dp_rank)
    group_name = "dp-zero-grad-" + "-".join([str(i) for i in current_rank_list_in_zero])
    comm_group = get_group_info("dp-zero-grad")
    comm_group.group = group_name
    comm_group.global_ranks = current_rank_list_in_zero
    comm_group.world_size = zero_shard_size
    comm_group.rank = group_rank_id
    return _get_group_helper("dp-zero-grad")


def get_zero_shard_tp_group():
    """Get the data parallel group the caller rank belongs to."""
    z3_group = get_group_info("dp")
    if not get_zero_full_shard_flag():
        z3_group = get_group_info("dp-zero")
    # DP [0, 2, 4, 6] --> z3_group [0, 4] [2, 6]
    # DP [1, 3, 5, 7] --> tp_group [1, 3] [5, 7]
    # TP [0, 1] [2, 3] [4, 5] [6, 7]
    z3_rank = z3_group.global_ranks
    mp_rank = _GLOBAL_MP_RANK
    new_rank_list = []
    for rank in z3_rank:
        for mp in mp_rank:
            if rank in mp:
                new_rank_list.extend(mp)
                break
    new_rank_list = list(set(new_rank_list))
    new_rank_list.sort()
    group_name = "dp-zero-tp-" + "-".join([str(i) for i in new_rank_list])
    comm_group = get_group_info("dp-zero-tp")
    comm_group.group = group_name
    comm_group.global_ranks = new_rank_list
    comm_group.world_size = len(new_rank_list)
    comm_group.rank = z3_group.rank
    return _get_group_helper("dp-zero-tp")

def initialize_context_parallel_group_for_hybrid_cp(context_parallel_size, rank_generator):
    if not is_hybrid_cp():
        return
    ulysses_degree = get_args().ulysses_degree_in_cp
    assert (context_parallel_size > ulysses_degree and context_parallel_size % ulysses_degree == 0)
    ring_degree = context_parallel_size // ulysses_degree
    rank_generator.init_hybrid_cp_group(ulysses_degree, ring_degree)

def is_hybrid_cp() -> bool:
    return hasattr(get_args(), 'context_parallel_algo') and \
        get_args().context_parallel_algo == "hybrid_cp_algo"

class CollectiveCommIntf:
    """Collective Comm Intf Class"""
    @classmethod
    @abc.abstractmethod
    def get_comm_group_world_size(cls):
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def get_comm_group(cls):
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def get_comm_rank(cls):
        raise NotImplementedError

    def get_comm_group_name(self):
        return self.comm_group_name


class TPXCollectiveComm(CollectiveCommIntf):
    """TPX Collective Comm Class"""
    @classmethod
    def get_comm_rank(cls):
        return get_tp_x_rank()

    @classmethod
    def get_comm_group_world_size(cls):
        return get_tp_x_world_size()

    @classmethod
    def get_comm_group(cls):
        return get_tp_x_group()


class TPYCollectiveComm(CollectiveCommIntf):
    """TPY Collective Comm Class"""
    @classmethod
    def get_comm_rank(cls):
        return get_tp_y_rank()

    @classmethod
    def get_comm_group_world_size(cls):
        return get_tp_y_world_size()

    @classmethod
    def get_comm_group(cls):
        return get_tp_y_group()
