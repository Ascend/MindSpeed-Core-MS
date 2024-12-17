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
"""Ring Attention utils."""

import mindspore as ms
from mindspore import Tensor, ops
import mindspore.communication as commu
from mindspore.communication import get_group_size
from mindspore.mint.distributed import get_rank

from mindspeed_ms.training.global_vars import get_args
from mindspeed_ms.core.parallel_state import get_context_parallel_rank, \
    get_data_parallel_world_size, get_context_parallel_world_size


def get_sp_chuncks(batch, input_layout, enable_dp_shard=True,
                   enable_flash_sp=False):
    """
    Slice batch input along sequence dimension into multiple chunks,
    which are parallelized across NPUs in a sequence parallel group.
    """
    sp_rank = get_context_parallel_rank()
    world_size = get_group_size()
    dp = get_data_parallel_world_size(with_context_parallel=False)
    sp = get_context_parallel_world_size()

    if not isinstance(enable_flash_sp, bool):
        raise TypeError(
            f"The type of enable_flash_sp must be bool, but got the {type(enable_flash_sp)}")
    if not enable_flash_sp:
        if input_layout == "BSH":
            seq_dim = 1
            batch_dim = 0
        elif input_layout == "BNSD":
            seq_dim = 2
            batch_dim = 0
        elif input_layout == "SBH":
            seq_dim = 0
            batch_dim = 1
        else:
            raise ValueError(
                f"Only input_layout = 'BSH' or 'BNSD' or 'SBH' is supported")
    else:
        if input_layout == "BSH":
            seq_dim = 1
            batch_dim = 0
        else:
            raise ValueError(
                f"For FlashSP, only input_layout = 'BSH' is supported")

    if not isinstance(enable_dp_shard, bool):
        raise TypeError(
            f"The type of enable_dp_shard must be bool, but got the {type(enable_dp_shard)}")

    if dp * sp != world_size:
        raise ValueError(f"The product of dp and sp should be equal to total device number,"
                         f"but got dp = {dp}, sp = {sp} and total device number = {world_size}")

    seq_len = batch.shape[seq_dim]
    if seq_len < 2 * sp:
        raise ValueError(f"The sequence length of input batch should be larger or equal to 2*sp,"
                         f"but got sequence length {seq_len} and sp is {sp}")
    if seq_len % (2 * sp) != 0:
        raise ValueError(f"The sequence length of input batch is not divisible by 2*sp,"
                         f"but got sequence length {seq_len} and sp is {sp}")

    if enable_dp_shard:
        batch_sz = batch.shape[batch_dim]
        if batch_sz % dp != 0:
            raise ValueError(f"The batch size of input batch is not divisible by dp,"
                             f"but got batch_size {batch_sz} and dp is {dp}")
        if dp > 1:
            if batch_dim == 0:
                batch = batch.view(
                    dp,
                    batch.shape[batch_dim] // dp,
                    *batch.shape[(batch_dim + 1):],
                )
            else:
                batch = batch.view(
                    *batch.shape[0:batch_dim],
                    dp,
                    batch.shape[batch_dim] // dp,
                    *batch.shape[(batch_dim + 1):],
                )
            sp_group_index = get_rank() // sp
            sp_group_index = Tensor([sp_group_index])
            batch = batch.index_select(
                batch_dim, sp_group_index).squeeze(batch_dim)

    if sp > 1:
        if seq_dim == 0:
            batch = batch.view(
                2 * sp,
                batch.shape[seq_dim] // (2 * sp),
                *batch.shape[(seq_dim + 1):],
            )
        else:
            batch = batch.view(
                *batch.shape[0:seq_dim],
                2 * sp,
                batch.shape[seq_dim] // (2 * sp),
                *batch.shape[(seq_dim + 1):],
            )

        if enable_flash_sp:
            index = Tensor([2 * sp_rank, 2 * sp_rank + 1])
        else:
            index = Tensor([sp_rank, (2 * sp - sp_rank - 1)])
        batch = batch.index_select(seq_dim, index)

        if seq_dim == 0:
            batch = batch.view(-1, *batch.shape[(seq_dim + 2):])
        else:
            batch = batch.view(
                *batch.shape[0:seq_dim], -1, *batch.shape[(seq_dim + 2):])

    return batch


def get_sp_chuncks_general(batch, input_layout, enable_dp_shard=True,
                           enable_flash_sp=False):
    """
    Slice batch input along sequence dimension into multiple chunks,
    which are parallelized across NPUs in a sequence parallel group.
    No head-to-tail data rearrangement
    """
    sp_rank = get_context_parallel_rank()
    world_size = get_group_size()
    dp = get_data_parallel_world_size(with_context_parallel=False)
    sp = get_context_parallel_world_size()
    if not isinstance(enable_flash_sp, bool):
        raise TypeError(
            f"The type of enable_flash_sp must be bool, but got the {type(enable_flash_sp)}")

    if not enable_flash_sp:
        if input_layout == "BSH":
            seq_dim = 1
            batch_dim = 0
        elif input_layout == "BNSD":
            seq_dim = 2
            batch_dim = 0
        elif input_layout == "SBH":
            seq_dim = 0
            batch_dim = 1
        else:
            raise ValueError(
                f"Only input_layout = 'BSH' or 'BNSD' or 'SBH' is supported")
    else:
        if input_layout == "BSH":
            seq_dim = 1
            batch_dim = 0
        else:
            raise ValueError(
                f"For FlashSP, only input_layout = 'BSH' is supported")
    if not isinstance(enable_dp_shard, bool):
        raise TypeError(
            f"The type of enable_dp_shard must be bool, but got the {type(enable_dp_shard)}")

    if dp * sp != world_size:
        raise ValueError(f"The product of dp and sp should be equal to total device number,"
                         f"but got dp = {dp}, sp = {sp} and total device number = {world_size}")
    seq_len = batch.shape[seq_dim]
    if seq_len < 2 * sp:
        raise ValueError(f"The sequence length of input batch should be larger or equal to 2*sp,"
                         f"but got sequence length {seq_len} and sp is {sp}")
    if seq_len % (2 * sp) != 0:
        raise ValueError(f"The sequence length of input batch is not divisible by 2*sp,"
                         f"but got sequence length {seq_len} and sp is {sp}")

    if enable_dp_shard:
        batch_sz = batch.shape[batch_dim]
        if batch_sz % dp != 0:
            raise ValueError(f"The batch size of input batch is not divisible by dp,"
                             f"but got batch_size {batch_sz} and dp is {dp}")
        if dp > 1:
            if batch_dim == 0:
                batch = batch.view(
                    dp,
                    batch.shape[batch_dim] // dp,
                    *batch.shape[(batch_dim + 1):],
                )
            else:
                batch = batch.view(
                    *batch.shape[0:batch_dim],
                    dp,
                    batch.shape[batch_dim] // dp,
                    *batch.shape[(batch_dim + 1):],
                )
            sp_group_index = get_rank() // sp
            sp_group_index = Tensor([sp_group_index])
            batch = batch.index_select(
                batch_dim, sp_group_index).squeeze(batch_dim)

    val = ops.chunk(batch, sp, axis=seq_dim)[sp_rank]

    return val


def get_sp_chuncks_attn_mask_general(attn_mask):
    """
    Slice attention_mask input along sequence dimension into multiple chunks,
    which are parallelized across NPUs in a sequence parallel group.
    No head-to-tail data rearrangement
    """
    sp_rank = get_context_parallel_rank()
    sp = get_context_parallel_world_size()

    if len(attn_mask.shape) != 2:
        raise AssertionError("The fusion attention operator currently only support 2D attention mask.")
    attn_mask = ops.chunk(attn_mask, sp, axis=0)[sp_rank]

    return attn_mask


def get_batch_on_this_cp_rank(batch, enable_flash_sp=False):
    """
    Retrieve the batch data for the current compute node (CP rank).
    This function further slices the batch tensor along the sequence
    dimension to obtain the data for the current sequence parallel rank.
    The sliced batch tensor is then returned.
    """
    args = get_args()
    cp_size = get_context_parallel_world_size()
    if not cp_size > 1:
        return batch

    if args.context_parallel_algo == 'megatron_cp_algo':
        if args.cp_attention_mask_type == 'general' or enable_flash_sp:
            batch = _get_batch_on_this_cp_rank_in_megatron_cp_general(batch)
        else:
            batch = _get_batch_on_this_cp_rank_in_megatron_cp(batch, enable_flash_sp=False)
    elif args.context_parallel_algo == 'ulysses_cp_algo':
        batch = _get_batch_on_this_cp_rank_in_ulysses_cp(batch)
    return batch

def _get_batch_on_this_cp_rank_in_megatron_cp(batch, enable_flash_sp=False):
    """
    Retrieve the batch data for the current compute node (CP rank).
    To support ring attention or flash sp with causal attention mask.
    """
    cp_rank = get_context_parallel_rank()
    cp_size = get_context_parallel_world_size()
    for key, val in batch.items():
        if key == 'attention_mask':
            continue
        if val is not None:
            seq_dim = 1 if key != 'attention_mask' else 2
            val = val.view(
                *val.shape[0:seq_dim],
                2 * cp_size,
                val.shape[seq_dim] // (2 * cp_size),
                *val.shape[(seq_dim + 1):],
            )
            if enable_flash_sp:
                index = Tensor([2 * cp_rank, 2 * cp_rank + 1], ms.int32)
            else:
                index = Tensor([cp_rank, (2 * cp_size - cp_rank - 1)], ms.int32)

            val = ops.index_select(val, seq_dim, index)
            val = val.view(
                *val.shape[0:seq_dim], -1, *val.shape[(seq_dim + 2):])

            batch[key] = val

    return batch


def _get_batch_on_this_cp_rank_in_megatron_cp_general(batch):
    """
    Retrieve the batch data for the current compute node (CP rank).
    To support ring attention with general attention mask.
    """
    cp_rank = get_context_parallel_rank()
    cp_size = get_context_parallel_world_size()
    for key, val in batch.items():
        if val is not None:
            seq_dim = 1 if key != 'attention_mask' else 2
            val = ops.chunk(val, cp_size, seq_dim)[cp_rank]

            batch[key] = val

    return batch

def _get_batch_on_this_cp_rank_in_ulysses_cp(batch):
    """
    Retrieve the batch data for the current compute node (CP rank).
    To support ulysses context parallel.
    """
    cp_rank = get_context_parallel_rank()
    cp_size = get_context_parallel_world_size()
    for key, val in batch.items():
        if key == 'attention_mask':
            continue
        if val is not None:
            seq_dim = 1 if key != 'attention_mask' else 2
            val = ops.chunk(val, cp_size, seq_dim)[cp_rank]
            batch[key] = val

    return batch

# pylint: disable=R1705
# pylint: disable=C1801
class RingP2P:
    '''Ring P2P communication.'''

    def __init__(self, ring_global_ranks, group, group_for_send_recv_overlap=None, is_backward=False) -> None:
        self.group = group
        self.group_for_send_recv_overlap = group
        if group_for_send_recv_overlap is not None:
            self.group_for_send_recv_overlap = group_for_send_recv_overlap

        global_rank = get_rank()
        ring_rank = ring_global_ranks.index(global_rank)
        ring_size = len(ring_global_ranks)
        self.next = ring_global_ranks[(ring_rank + 1) % ring_size]
        self.prev = ring_global_ranks[(ring_rank + ring_size - 1) % ring_size]
        self.ring_rank = ring_rank
        if is_backward:
            self.next, self.prev = self.prev, self.next

        self.send_recv_ops = []

    def async_send_recv(self, send_tensor, recv_tensor):
        '''Send and receive tensors asynchronously.'''
        if self.ring_rank % 2 == 0:
            send_op = commu.comm_func.isend(send_tensor, self.next, self.group)
            recv_tensor_tmp, recv_op = commu.comm_func.irecv(recv_tensor, self.prev, self.group_for_send_recv_overlap)
            self.send_recv_ops.append(send_op)
            self.send_recv_ops.append(recv_op)
        else:
            recv_tensor_tmp, recv_op = commu.comm_func.irecv(recv_tensor, self.prev, self.group)
            send_op = commu.comm_func.isend(send_tensor, self.next, self.group_for_send_recv_overlap)
            self.send_recv_ops.append(recv_op)
            self.send_recv_ops.append(send_op)
        return recv_tensor_tmp

    def wait(self):
        '''Wait for all send and recv operations to complete.'''
        if len(self.send_recv_ops) > 0:
            for op in self.send_recv_ops:
                op.wait()
            self.send_recv_ops = []
            return 1
        else:
            return 0


# pylint: disable=W0613
def forward_update(prev_attn_out, prev_softmax_max, prev_softmax_sum,
                   cur_attn_out, cur_softmax_max, cur_softmax_sum, actual_seq_qlen=None, layout='SBH'):
    """
    Updates the attention output and softmax statistics for the ring attention mechanism,
    with added parameters for enhanced flexibility and extensibility.

    This function is designed to update the attention output and related softmax statistics
    for a given sequence length in a ring attention mechanism. It handles the merging of
    previous and current attention outputs and their corresponding softmax statistics.
    The introduction of `actual_seq_qlen` and `layout` parameters allows for greater flexibility
    in handling variable sequence lengths and different tensor layouts, respectively.

    Parameters:
    - prev_attn_out (Tensor): The attention output from the previous process.
    - prev_softmax_max (Tensor): The maximum value of the softmax distribution from the previous process.
    - prev_softmax_sum (Tensor): The sum of the softmax distribution from the previous process.
    - cur_attn_out (Tensor): The attention output from the current process.
    - cur_softmax_max (Tensor): The maximum value of the softmax distribution from the current process.
    - cur_softmax_sum (Tensor): The sum of the softmax distribution from the current process.
    - actual_seq_qlen (Tensor, optional): The actual sequence length for the query. This parameter
                                      is crucial for handling variable-length sequences and ensuring
                                      that the attention mechanism operates correctly under such conditions.
                                      If not provided, it defaults to the length of the current attention output.
    - layout (str, optional): The layout format of the input tensors. This parameter allows for the specification
                              of different tensor layouts, enhancing the function's versatility across various
                              model architectures. Default is 'SBH', where:
        - S: Sequence length
        - B: Batch size
        - H: Hidden size (number of attention heads)

    Returns:
    - updated_attn_out (Tensor): The updated attention output after merging previous and current process.
    - updated_softmax_max (Tensor): The updated maximum value of the softmax distribution.
    - updated_softmax_sum (Tensor): The updated sum of the softmax distribution.
    """
    # TODO: Not available until the 'npu_ring_attention_update' is available
    # _args = get_args()

    # if hasattr(_args, 'use_fused_ring_attention_update') and _args.use_fused_ring_attention_update:
    #     return npu_ring_attention_update(prev_attn_out, prev_softmax_max, prev_softmax_sum, cur_attn_out,
    #                                      cur_softmax_max, cur_softmax_sum, actual_seq_qlen, layout)

    # update softmax_max
    origin_dtype = prev_attn_out.dtype
    softmax_max = ms.ops.maximum(prev_softmax_max, cur_softmax_max)
    prev_scale = ms.ops.exp(prev_softmax_max - softmax_max)
    cur_scale = ms.ops.exp(cur_softmax_max - softmax_max)

    # update softmax_sum
    prev_softmax_sum_scaled = prev_softmax_sum * prev_scale
    cur_softmax_sum_scaled = cur_softmax_sum * cur_scale
    softmax_sum = prev_softmax_sum_scaled + cur_softmax_sum_scaled

    # out updating scale
    prev_out_scale = prev_softmax_sum_scaled / softmax_sum
    cur_out_scale = cur_softmax_sum_scaled / softmax_sum

    # [b, n, s, 8] -> [s, b, h]
    # SBH layout
    n = prev_out_scale.shape[1]
    h = prev_attn_out.shape[-1]
    d = h // n
    prev_out_scale = prev_out_scale[..., 0].unsqueeze(3).tile((1, 1, 1, d))
    prev_out_scale = prev_out_scale.permute(2, 0, 1, 3).reshape(prev_out_scale.shape[2],
                                                                prev_out_scale.shape[0], -1).contiguous()
    cur_out_scale = cur_out_scale[..., 0].unsqueeze(3).tile((1, 1, 1, d))
    cur_out_scale = cur_out_scale.permute(2, 0, 1, 3).reshape(cur_out_scale.shape[2],
                                                              cur_out_scale.shape[0], -1).contiguous()

    # update output
    attn_out = prev_attn_out * prev_out_scale + cur_attn_out * cur_out_scale
    attn_out = attn_out.to(origin_dtype)
    return attn_out, softmax_max, softmax_sum

# pylint: disable=W0613
def causal_out_update(q_block_id, kv_block_id, cur_attn_outs, global_attn_outs,
                      q_index=None, cur_sub_out_seq_len=None):
    '''Update the global attention output with the current attention output.'''
    cur_attn_out, cur_softmax_max, cur_softmax_sum = cur_attn_outs[0], cur_attn_outs[1], cur_attn_outs[2]
    attn_out, softmax_max, softmax_sum, rng_states = global_attn_outs
    layout = 'SBH'
    # (seed, offset, numels)
    if len(cur_attn_outs) > 3:
        rng_states[kv_block_id] = (cur_attn_outs[4], cur_attn_outs[5], cur_attn_outs[6])

    if q_block_id == kv_block_id:
        attn_out = cur_attn_out
        softmax_max = cur_softmax_max
        softmax_sum = cur_softmax_sum
    elif kv_block_id <= q_block_id:
        attn_out_updated, softmax_max_updated, softmax_sum_updated = forward_update(
            attn_out, softmax_max, softmax_sum,
            cur_attn_out, cur_softmax_max, cur_softmax_sum, layout=layout
        )
        attn_out, softmax_max, softmax_sum = attn_out_updated, softmax_max_updated, softmax_sum_updated
    else:
        # [2s, b, h] -> [2, s, b, h]
        attn_out = attn_out.view(2, attn_out.shape[0] // 2, *attn_out.shape[1:])
        # [b, n, 2s, 8] -> [b, n, 2, s, 8]
        softmax_max = softmax_max.view(softmax_max.shape[0], softmax_max.shape[1],
                                       2, softmax_max.shape[2] // 2, softmax_max.shape[-1])
        softmax_sum = softmax_sum.view(softmax_sum.shape[0], softmax_sum.shape[1],
                                       2, softmax_sum.shape[2] // 2, softmax_sum.shape[-1])
        attn_out_updated, softmax_max_updated, softmax_sum_updated = forward_update(
            attn_out[1], softmax_max[:, :, 1, :, :], softmax_sum[:, :, 1, :, :],
            cur_attn_out, cur_softmax_max, cur_softmax_sum, layout=layout
        )
        attn_out[1].copy_(attn_out_updated)
        softmax_max[:, :, 1, :, :].copy_(softmax_max_updated)
        softmax_sum[:, :, 1, :, :].copy_(softmax_sum_updated)
        # [2, s, b, h] -> [2s, b, h]
        attn_out = attn_out.view(-1, *attn_out.shape[2:])
        # [b, n, 2, s, 8] -> [b, n, 2s, 8]
        softmax_max = softmax_max.view(softmax_max.shape[0], softmax_max.shape[1], -1,
                                       softmax_max.shape[-1])
        softmax_sum = softmax_sum.view(softmax_sum.shape[0], softmax_sum.shape[1], -1,
                                       softmax_sum.shape[-1])

    return [attn_out, softmax_max, softmax_sum, rng_states]

# pylint: disable=W0613
def general_out_update(q_block_id, kv_block_id, cur_attn_outs, global_attn_outs, cur_sub_out_seq_len=None):
    '''Update the global attention output with the current attention output.'''

    cur_attn_out, cur_softmax_max, cur_softmax_sum = cur_attn_outs[0], cur_attn_outs[1], cur_attn_outs[2]
    attn_out, softmax_max, softmax_sum, rng_states = global_attn_outs
    layout = 'SBH'
    # (seed, offset, numels)
    rng_states[kv_block_id] = (cur_attn_outs[4], cur_attn_outs[5], cur_attn_outs[6])
    if q_block_id == kv_block_id:
        attn_out = cur_attn_out
        softmax_max = cur_softmax_max
        softmax_sum = cur_softmax_sum
    else:
        attn_out_updated, softmax_max_updated, softmax_sum_updated = forward_update(
            attn_out, softmax_max, softmax_sum,
            cur_attn_out, cur_softmax_max, cur_softmax_sum, layout=layout
        )
        attn_out, softmax_max, softmax_sum = attn_out_updated, softmax_max_updated, softmax_sum_updated

    return [attn_out, softmax_max, softmax_sum, rng_states]
