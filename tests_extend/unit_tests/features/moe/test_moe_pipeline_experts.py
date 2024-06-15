import pytest
import torch

from unit_tests.common import DistributedTest
from commons import set_random_seed

from mindspeed import megatron_adaptor
from mindspeed.moe.config import Config
from mindspeed.moe.gate import TopKGate
from mindspeed.moe.experts import Experts
from mindspeed.moe.moe_layer import MOELayer
from mindspeed.moe.mixtral_parallel_mlpbm import MixtralParallelMLPBM
from megatron.core.transformer import TransformerConfig
from megatron.training.global_vars import set_args
from megatron.training.arguments import parse_args
from megatron.core.parallel_state import get_expert_model_parallel_group, destroy_model_parallel, \
    initialize_model_parallel


class TestMOELayer(DistributedTest):
    world_size = 8
    num_layers = 2
    seq_len = 16
    hidden_size = 2
    num_attention_heads = 4
    use_cpu_initialization = True
    use_fp16 = True
    num_experts = 8
    tp_size = None
    pp_size = None
    ep_size = None
    topk_config = None
    topk_gate = None

    def get_moe_layer_output(self, input_data, use_pipe_experts, use_multi_stream, num_multi_data):
        expert = Experts(self.parallel_mlp, self.num_experts).npu()
        num_local_experts = self.num_experts // self.ep_size
        moe_layer_module = MOELayer(
            self.topk_gate,
            expert,
            self.ep_size,
            num_local_experts=num_local_experts,
            sequence_parallel=True,
            pipe_experts=use_pipe_experts,
            pipe_experts_multi_stream=use_multi_stream,
            pipe_experts_multi_data=num_multi_data
        ).npu()
        expert_parallel_group = get_expert_model_parallel_group()
        moe_layer_module.set_ep_group(expert_parallel_group)
        output_data = moe_layer_module(input_data)
        output_data.sum().backward()
        return output_data, input_data.grad

    @pytest.mark.parametrize("tp_pp_ep", [(1, 1, 4)])
    @pytest.mark.parametrize("base_tuple", [(False, False, 1)])
    @pytest.mark.parametrize("pipe_tuple", [(True, False, 1), (True, True, 1), (True, False, 4), (True, True, 4)])
    def test_moe_layer(self, tp_pp_ep, base_tuple, pipe_tuple):
        set_random_seed(1)
        (self.tp_size, self.pp_size, self.ep_size) = tp_pp_ep
        args = parse_args(None, True)
        set_args(args)
        destroy_model_parallel()
        initialize_model_parallel(tensor_model_parallel_size=self.tp_size,
                                  pipeline_model_parallel_size=self.pp_size,
                                  virtual_pipeline_model_parallel_size=None,
                                  pipeline_model_parallel_split_rank=None,
                                  expert_model_parallel_size=self.ep_size)

        parallel_config = TransformerConfig(
            num_layers=self.num_layers,
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_attention_heads,
            use_cpu_initialization=self.use_cpu_initialization,
            fp16=self.use_fp16)
        self.parallel_mlp = MixtralParallelMLPBM(parallel_config)

        self.topk_config = Config(
            hidden_size=self.hidden_size,
            num_experts=self.num_experts,
            ep_size=self.ep_size
        )
        self.topk_gate = TopKGate(self.topk_config)

        input_data_base = torch.randn(self.seq_len, self.hidden_size, device=torch.npu.current_device())
        input_data_pipe = input_data_base.clone().detach()
        input_data_base.requires_grad = True
        input_data_pipe.requires_grad = True

        # base without pipeline
        args_base = parse_args(None, True)
        (args_base.use_pipe_experts, args_base.use_multi_stream, args_base.num_multi_data) = base_tuple
        set_args(args_base)
        output_base, grad_input_base = self.get_moe_layer_output(input_data_base,
                                                                 args_base.use_pipe_experts,
                                                                 args_base.use_multi_stream,
                                                                 args_base.num_multi_data)

        # with pipeline feature(s)
        args_pipe = parse_args(None, True)
        (args_pipe.use_pipe_experts, args_pipe.use_multi_stream, args_pipe.num_multi_data) = pipe_tuple
        set_args(args_pipe)
        output_pipe, grad_input_pipe = self.get_moe_layer_output(input_data_pipe,
                                                                 args_pipe.use_pipe_experts,
                                                                 args_pipe.use_multi_stream,
                                                                 args_pipe.num_multi_data)

        assert torch.allclose(output_base, output_pipe)
        assert torch.allclose(grad_input_base, grad_input_pipe)
