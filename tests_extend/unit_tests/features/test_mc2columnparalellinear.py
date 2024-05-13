import pytest
import torch
import torch_npu
from ascendspeed import megatron_adaptor
from megatron.training.global_vars import set_args
from megatron.core.tensor_parallel import ColumnParallelLinear
from megatron.core.transformer.transformer_config import TransformerConfig
from ascendspeed.core.tensor_parallel.ascend_turbo.initialize import initialize_cfg_from_args
from unit_tests.common import DistributedTest
from commons import initialize_model_parallel
from commons import set_random_seed


class MC2Args:
    tensor_model_parallel_size = 8
    use_unpad = False
    seed = 2024
    seq_len = 128
    input_size_coeff = 128
    output_size_coeff = 128
    batch_size = 8
    optimize_recomp_communication_level = True
    sequence_parallel = 1
    use_cp_send_recv_overlap = False


class TestMC2(DistributedTest):
    world_size = MC2Args.tensor_model_parallel_size

    def test_MC2ColumnParallelLinear(self):
        args = MC2Args()
        set_args(args)
        initialize_cfg_from_args(args)
        transformer_config = TransformerConfig(num_layers=1,
                                               hidden_size=12,
                                               num_attention_heads=4,
                                               use_cpu_initialization=True)
        initialize_model_parallel(args.tensor_model_parallel_size, 1)
        set_random_seed(args.seed)
        input_size = args.input_size_coeff * args.tensor_model_parallel_size
        output_size = args.output_size_coeff * args.tensor_model_parallel_size
        linear_layer = ColumnParallelLinear(input_size, 
                                            output_size, 
                                            keep_master_weight_for_test=True,
                                            init_method=transformer_config.init_method,
                                            config=transformer_config).half().npu()
        loss_weight = torch.rand([args.seq_len, args.output_size_coeff]).half().npu()
        input_ = torch.rand(args.batch_size, args.seq_len, input_size).half().npu()
        output = linear_layer(input_)
        gather_list = [torch.zeros(input_.shape).half().npu() for _ in range(self.world_size)]
        torch.distributed.all_gather(gather_list, input_)
        gather_res = torch.concat(gather_list, dim=0)
        output_naive = torch.matmul(gather_res, linear_layer.weight.t())
        assert torch.allclose(output_naive, output[0], rtol=0.005, atol=0.005)
        loss = torch.mul(output[0], loss_weight).sum()
        # Backward
        loss.backward()
        dLdY = loss_weight
        dLdA = torch.matmul(dLdY.t(), gather_res).sum(dim=0)
        ones = torch.ones(args.seq_len, args.batch_size * self.world_size).half().npu()
        dLdb = torch.matmul(ones.t(), dLdY).sum(dim=0).view(-1)
        assert torch.allclose(dLdA, linear_layer.weight.grad, rtol=0.005, atol=0.005)
        assert torch.allclose(dLdb, linear_layer.bias.grad, rtol=0.005, atol=0.005)