from unittest import mock
import queue
import pytest
import torch
from unit_tests.common import DistributedTest
from mindspeed import megatron_adaptor
from megatron.training.global_vars import set_args
from megatron.training.arguments import parse_args
from megatron.core.parallel_state import destroy_model_parallel, initialize_model_parallel
from mindspeed.core.weight_grad_store import WeightGradStore


class TestDistributedOptimizer(DistributedTest):
    world_size = 4

    @staticmethod
    def setup_weight_grad_store():
        WeightGradStore.cache = []
        WeightGradStore.weight_grad_queue = queue.Queue()
        WeightGradStore.stored_grads = None
        WeightGradStore.store_grad_cache = None

    def test_put(self):
        self.setup_weight_grad_store()
        WeightGradStore.put('input1', 'grad1', 'weight1', True, False, False)
        assert len(WeightGradStore.cache) == 1
        assert WeightGradStore.cache[0] == ('input1', 'grad1', 'weight1', True, False, False)

    def test_flush(self):
        self.setup_weight_grad_store()
        with mock.patch('mindspeed.core.weight_grad_store.is_pipeline_first_stage', return_value=False):
            WeightGradStore.put('input1', 'grad1', 'weight1', True, False, False)
            WeightGradStore.flush()
            assert WeightGradStore.cache == []
            assert not WeightGradStore.weight_grad_queue.empty()
            assert WeightGradStore.weight_grad_queue.get() == [('input1', 'grad1', 'weight1', True, False, False)]

    @pytest.mark.parametrize("tp_pp", [(2, 2)])
    @pytest.mark.parametrize("row", [True, False])
    @pytest.mark.parametrize("sequence_parallel", [True, False])
    def test_overlap_all_gather(self, tp_pp, row, sequence_parallel):
        self.setup_weight_grad_store()
        input_data = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]).npu()
        grad_output = torch.tensor([[0.01, 0.02, 0.01], [0.02, 0.01, 0.03]]).npu()
        weight = torch.tensor([[0.3, 0.2, 0.1], [0.5, 0.4, 0.3]]).npu()
        (tp, pp) = tp_pp
        in_row = row
        sequence_parallel = sequence_parallel
        args = parse_args(None, True)
        set_args(args)
        destroy_model_parallel()
        initialize_model_parallel(tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
        virtual_pipeline_model_parallel_size=None,
        pipeline_model_parallel_split_rank=None)
        WeightGradStore.stored_grads = [(input_data, grad_output, weight, sequence_parallel, in_row, False)]
        result, handle = WeightGradStore.overlap_all_gather()
        diff_value = 0.00001
        if handle is not None:
            handle.wait()
        if not sequence_parallel:
            assert result == (input_data, grad_output, weight, sequence_parallel, in_row, False)
        elif in_row:
            assert torch.allclose(result[0], input_data, rtol=diff_value, atol=diff_value)
            assert torch.allclose(result[1], torch.cat((grad_output, grad_output), dim=0), rtol=diff_value, atol=diff_value)
        else:
            assert torch.allclose(result[0], torch.cat((input_data, input_data), dim=0), rtol=diff_value, atol=diff_value)
            assert torch.allclose(result[1], grad_output, rtol=diff_value, atol=diff_value)

    @pytest.mark.parametrize("tp_pp", [(2, 2)])
    def test_pop(self, tp_pp):
        self.setup_weight_grad_store()
        input_data = torch.randn(1, 2, 3).npu()
        grad_output = torch.randn(1, 2, 3).npu()
        weight = torch.randn(3, 3).npu()
        weight.grad = torch.randn(3, 3).npu()
        setattr(weight, 'main_grad', weight.clone())
        (tp, pp) = tp_pp
        args = parse_args(None, True)
        set_args(args)
        destroy_model_parallel()
        initialize_model_parallel(tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
        virtual_pipeline_model_parallel_size=None,
        pipeline_model_parallel_split_rank=None)
        WeightGradStore.put(input_data, grad_output, weight, sequence_parallel=False, in_row=True, pipe_experts=False)
        WeightGradStore.put(input_data, grad_output, weight, sequence_parallel=False, in_row=True, pipe_experts=False)
        with mock.patch('mindspeed.core.weight_grad_store.is_pipeline_first_stage', return_value=False):
            WeightGradStore.flush()
            WeightGradStore.pop()
            assert WeightGradStore.store_grad_cache is None
            assert WeightGradStore.stored_grads is None
