import pytest
import torch
import torch_npu
import torch.distributed as dist
import ascendspeed.megatron_adaptor
from ascendspeed.moe.moe import MoE
from ascendspeed.moe.mixtral_parallel_mlpbm import MixtralParallelMLPBM
from commons import set_random_seed
from unit_tests.common import DistributedTest
import megatron.core.parallel_state as ps
from megatron.core.transformer import TransformerConfig
from megatron.training.global_vars import set_args
from megatron.training.arguments import parse_args


def get_data_on_group_rank(data, group_size, rank, dim=0):
    data = data.view(*data.shape[0:dim], group_size, data.shape[dim] // group_size, *data.shape[dim+1:])
    index = torch.tensor([rank]).npu()
    data = data.index_select(dim, index)
    data = data.view(*data.shape[0:dim], -1, *data.shape[dim+2:])
    return data


def run_moe_cp(input_data, cp_size, ep_size, num_experts):
    args = parse_args(None, True)
    args.moe_router_topk = 2
    args.context_parallel_size = cp_size
    args.expert_model_parallel_size = ep_size
    set_args(args)

    ps.destroy_model_parallel()
    ps.initialize_model_parallel(context_parallel_size=cp_size, expert_model_parallel_size=ep_size)

    s = input_data.shape[0]
    b = input_data.shape[1]
    h = input_data.shape[2]

    cp_size = ps.get_context_parallel_world_size()
    cp_rank = ps.get_context_parallel_rank()

    config = TransformerConfig(
        num_layers = 1,
        hidden_size=h,
        num_attention_heads=1,
        use_cpu_initialization=True,
        fp16=True,
    )

    set_random_seed(1234)
    moe = MoE(
        h,
        MixtralParallelMLPBM(config, ),
        num_experts=num_experts,
        k=2,
        capacity_factor=num_experts,
        aux_loss_coef=0.0,
        ep_group=ps.get_expert_model_parallel_group(),
        noisy_gate_policy='RSample'
    ).npu()

    moe_input = get_data_on_group_rank(input_data, cp_size, cp_rank)
    return moe(moe_input)[0]


class TestCPMoE(DistributedTest):
    world_size = 8

    @pytest.mark.skip(reason='skip because of megatron ep group bug')
    @pytest.mark.parametrize("config", [(4, 8), (2, 8)])
    def test_cp_ep_loss(self, config):
        cp_size, ep_size = config
        set_random_seed(1234)
        s, b, h = 4, 4, 8
        input_data = torch.randn(s, b, h, dtype = torch.float16).npu()
        output_data = run_moe_cp(input_data, 1, ep_size, 8)
        cp_output = run_moe_cp(input_data, cp_size, ep_size, 8)
        cp_output_allgather = torch.empty_like(output_data, dtype=cp_output.dtype).npu()
        dist._all_gather_base(cp_output_allgather, cp_output, ps.get_context_parallel_group())
        assert torch.allclose(output_data, cp_output_allgather, rtol=1e-05, atol=1e-05)