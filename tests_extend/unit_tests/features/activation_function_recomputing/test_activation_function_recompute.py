import pytest
import torch
import torch_npu

from torch.autograd import Variable
from mindspeed import megatron_adaptor
from mindspeed.core.tensor_parallel.random import CheckpointWithoutOutput


class TestActivationFunctionRecomputing:

    def test_checkpoint_function(self):
        def run_function(a, b):
            return a * a + b * b

        checkpoint_without_output = CheckpointWithoutOutput()

        a = Variable(torch.randn(5, 5), requires_grad=True)
        b = Variable(torch.randn(5, 5), requires_grad=True)

        outputs = checkpoint_without_output.checkpoint(run_function, False, a, b)

        expected_outputs = run_function(a, b)
        assert torch.allclose(outputs, expected_outputs)

        checkpoint_without_output.recompute(None)

        outputs.sum().backward()

        a.grad.zero_()
        b.grad.zero_()

        a.retain_grad()
        b.retain_grad()
        recomputed_outputs = run_function(a, b)
        recomputed_outputs.sum().backward()

        assert torch.allclose(a.grad, torch.autograd.grad(expected_outputs.sum(), a)[0])
        assert torch.allclose(b.grad, torch.autograd.grad(expected_outputs.sum(), b)[0])