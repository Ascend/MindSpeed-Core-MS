import math

import pytest
import torch
import torch_npu
import torch.distributed as dist
from mindspeed import megatron_adaptor
from megatron.training.global_vars import set_args
from megatron.training.arguments import parse_args
from megatron.legacy.model.transformer import FlashSelfAttention
from commons import set_random_seed, initialize_model_parallel
from unit_tests.common import DistributedTest
from mindspeed.core.context_parallel.ulysses_context_parallel import UlyssesContextAttention
from mindspeed.core.parallel_state import get_context_parallel_group_for_hybrid_ulysses


DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]


def get_data_on_this_cp_rank(data, r_size, u_size, cp_rank, dim=0):
    """ Slice data along sequence dimension into multiple chunks,
        which are parallelized across GPUs in a context parallel group.
        Dispatch data in a striped way for load-balance.
    """
    r_rank = cp_rank // u_size
    u_rank = cp_rank % u_size
    data = data.view(*data.shape[0:dim], 2 * r_size, data.shape[dim] // (2 * r_size), *data.shape[dim + 1:])
    index = torch.tensor([r_rank, (2 * r_size - r_rank - 1)], device=data.device)
    data = data.index_select(dim, index)
    data = data.view(*data.shape[0:dim], -1, *data.shape[dim + 2:])
    data = data.chunk(u_size, dim=dim)[u_rank]
    return data


def run_hybridattn_cp(cp_size, u_size, bs, seq_len, dtype, cp_args):
    causal, send_recv_overlap = cp_args
    r_size = cp_size // u_size
    args = parse_args(None, True)
    args.use_cp_send_recv_overlap = send_recv_overlap
    args.cp_attention_mask_type = 'causal' if causal else 'full'
    args.context_parallel_algo = 'hybrid_cp_algo'
    args.context_parallel_size = cp_size
    args.ulysses_degree_in_cp = u_size
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

    out_ref = get_data_on_this_cp_rank(out.clone().detach(), r_size, u_size, rank)
    k_grad_ref = get_data_on_this_cp_rank(k.grad.clone().detach(), r_size, u_size, rank)
    v_grad_ref = get_data_on_this_cp_rank(v.grad.clone().detach(), r_size, u_size, rank)

    q_ = get_data_on_this_cp_rank(q.clone().detach(), r_size, u_size, rank)
    k_ = get_data_on_this_cp_rank(k.clone().detach(), r_size, u_size, rank)
    v_ = get_data_on_this_cp_rank(v.clone().detach(), r_size, u_size, rank)
    dout_ = get_data_on_this_cp_rank(dout.clone().detach(), r_size, u_size, rank)

    for x in [q_, k_, v_]:
        x.requires_grad = True


    local_attn = FlashSelfAttention(causal=causal, softmax_scale=scale)
    hybrid_attn = UlyssesContextAttention(local_attn, get_context_parallel_group_for_hybrid_ulysses())
    out_ = hybrid_attn(q_.reshape(-1, b, n, d), k_.reshape(-1, b, n, d), v_.reshape(-1, b, n, d), attn_mask)
    out_.backward(dout_)

    # same as transformer_engine
    tols = dict(atol=5e-3, rtol=5e-3)
    if dtype == torch.bfloat16:
        tols = dict(atol=2.5e-2, rtol=2.5e-2)

    # compare results with and without CP
    assert torch.allclose(out_ref, out_, **tols)
    assert torch.allclose(k_grad_ref, k_.grad, **tols)
    assert torch.allclose(v_grad_ref, v_.grad, **tols)


class TestHybridAttnCP(DistributedTest):
    world_size = 8

    @pytest.mark.skipif(DEVICE_NAME != 'Ascend910B', reason='device type is not supported, skip this UT!')
    @pytest.mark.parametrize("cp_args", [(True, True), (False, False)])
    def test_hybrid_context_parallel_seq8192_bs2_bf16(self, cp_args):
        run_hybridattn_cp(self.world_size, 2, 2, 8192, torch.bfloat16, cp_args)
