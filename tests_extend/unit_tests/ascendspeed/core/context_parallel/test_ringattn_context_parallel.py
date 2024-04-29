import math

import pytest
import torch
import torch_npu
import torch.distributed as dist
from ascendspeed import megatron_adaptor
from megatron.training.global_vars import set_args
from megatron.training.arguments import parse_args
from ascendspeed.core.context_parallel.ring_context_parallel import ringattn_context_parallel

from commons import set_random_seed, initialize_model_parallel
from unit_tests.common import DistributedTest


DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]


def get_data_on_this_cp_rank(data, cp_size, cp_rank, dim=0):
    """ Slice data along sequence dimension into multiple chunks,
        which are parallelized across GPUs in a context parallel group.
        Dispatch data in a striped way for load-balance.
    """
    data = data.view(*data.shape[0:dim], 2 * cp_size, data.shape[dim] // (2 * cp_size), *data.shape[dim + 1:])
    index = torch.tensor([cp_rank, (2 * cp_size - cp_rank - 1)], device=data.device)
    data = data.index_select(dim, index)
    data = data.view(*data.shape[0:dim], -1, *data.shape[dim + 2:])
    return data


def get_data_on_all_cp_ranks(data, cp_size, dim=0):
    """ Combine data along sequence dimension from multiple chunks.
    """
    data = data.view(*data.shape[0:dim], 2 * cp_size, -1, *data.shape[dim + 1:])
    index = [[i, 2 * cp_size - i - 1] for i in range(cp_size)]
    index = torch.tensor(index).flatten().to(data.device)
    index = index[:, None, None, None].repeat(1, *data.shape[1:])
    out = torch.empty_like(data)
    out = out.scatter(dim=0, index=index, src=data)
    out = out.view(-1, *out.shape[2:])
    return out


def run_ringattn_cp(cp_size, bs, seq_len, dtype, cp_args):
    causal, send_recv_overlap = cp_args
    args = parse_args(None, True)
    args.use_cp_send_recv_overlap = send_recv_overlap
    set_args(args)
    initialize_model_parallel(context_parallel_size=cp_size)
    set_random_seed(1234)

    rank = dist.get_rank()
    b, n, s, d = bs, 32, seq_len, 128
    scale = 1.0 / math.sqrt(d)

    q = torch.randn(s, b, n * d, dtype=dtype, device='npu', requires_grad=True)
    k = torch.randn(s, b, n * d, dtype=dtype, device='npu', requires_grad=True)
    v = torch.randn(s, b, n * d, dtype=dtype, device='npu', requires_grad=True)
    dout = torch.randn(s, b, n * d, dtype=dtype, device='npu', requires_grad=True)

    if causal:
        attn_mask = ~torch.tril(torch.ones((2048, 2048), dtype=torch.bool, device=q.device))
    else:
        attn_mask = None
    out = torch_npu.npu_fusion_attention( \
        q, k, v, n, 'SBH', \
        pse=None, \
        padding_mask=None, \
        atten_mask=attn_mask, \
        scale=scale, \
        pre_tockens=seq_len, \
        next_tockens=0, \
        keep_prob=1., \
        inner_precise=0, \
        sparse_mode=3 if attn_mask is not None else 0
    )[0]
    out.backward(dout)

    q_ = get_data_on_this_cp_rank(q.clone().detach(), cp_size, rank)
    k_ = get_data_on_this_cp_rank(k.clone().detach(), cp_size, rank)
    v_ = get_data_on_this_cp_rank(v.clone().detach(), cp_size, rank)
    dout_ = get_data_on_this_cp_rank(dout.clone().detach(), cp_size, rank)

    for x in [q_, k_, v_]:
        x.requires_grad = True

    cp_para = dict()
    cp_para['causal'] = causal
    out_ = ringattn_context_parallel(q_, k_, v_, n, cp_para, softmax_scale=scale, attn_mask=None)
    out_.backward(dout_)

    output_list = [torch.empty_like(out_) for i in range(cp_size)]
    dist.all_gather(output_list, out_)
    out_ring = torch.cat(output_list, dim=0)
    out_ring = get_data_on_all_cp_ranks(out_ring, cp_size)

    k_grad_list = [torch.empty_like(k_) for i in range(cp_size)]
    dist.all_gather(k_grad_list, k_.grad)
    k_grad = torch.cat(k_grad_list, dim=0)
    k_grad = get_data_on_all_cp_ranks(k_grad, cp_size)

    v_grad_list = [torch.empty_like(v_) for i in range(cp_size)]
    dist.all_gather(v_grad_list, v_.grad)
    v_grad = torch.cat(v_grad_list, dim=0)
    v_grad = get_data_on_all_cp_ranks(v_grad, cp_size)

    # same as transformer_engine
    tols = dict(atol=5e-3, rtol=5e-3)
    if dtype == torch.bfloat16:
        tols = dict(atol=2.5e-2, rtol=2.5e-2)

    # compare results with and without CP
    assert torch.allclose(out, out_ring, **tols)
    assert torch.allclose(k.grad, k_grad, **tols)
    assert torch.allclose(v.grad, v_grad, **tols)


class TestRingAttnCP(DistributedTest):
    world_size = 8

    # @pytest.mark.skipif(DEVICE_NAME != 'Ascend910B', reason='device type is not supported, skip this UT!')
    @pytest.mark.skip(reason='this UT need update for new megatron version')
    @pytest.mark.parametrize("cp_args", [(True, False), (False, True)])
    def test_ringattn_context_parallel_seq8192_bs1_fp16(self, cp_args):
        run_ringattn_cp(self.world_size, 1, 8192, torch.half, cp_args)

    # @pytest.mark.skipif(DEVICE_NAME != 'Ascend910B', reason='device type is not supported, skip this UT!')
    @pytest.mark.skip(reason='this UT need update for new megatron version')
    @pytest.mark.parametrize("cp_args", [(True, True), (False, False)])
    def test_ringattn_context_parallel_seq8192_bs2_bf16(self, cp_args):
        run_ringattn_cp(self.world_size, 2, 8192, torch.bfloat16, cp_args)
