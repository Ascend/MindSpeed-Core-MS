import pytest
import torch
from torch import nn
from mindspeed import megatron_adaptor
from mindspeed.core.tensor_parallel.checkpoint_manager import get_pipeline_checkpoint_manager

from megatron.core.tensor_parallel import checkpoint
from unit_tests.common import DistributedTest


def checkpointed_forward(function, *args):
    return checkpoint(function, False, *args)


class TestCheckpointManager(DistributedTest):
    world_size = 1

    def test_checkpoint_manager(self):
        layer1 = nn.Sequential(nn.Linear(10, 20), nn.ReLU())
        layer2 = nn.Sequential(nn.Linear(20, 30), nn.ReLU())
        layer3 = nn.Sequential(nn.Linear(30, 20), nn.ReLU())
        layer4 = nn.Sequential(nn.Linear(20, 10), nn.ReLU())

        input_data = torch.randn(20, 10, requires_grad=True)
        expected_output = layer4(layer3(layer2(layer1(input_data))))

        manager = get_pipeline_checkpoint_manager(num_of_chunks=1)
        manager.open_ri_pipe = True
        manager.do_pre_recompute = True

        output = checkpointed_forward(layer1, input_data)
        manager.disable_recompute()
        output = checkpointed_forward(layer2, output)
        manager.enable_recompute()
        output = checkpointed_forward(layer3, output)
        manager.disable_recompute()
        output = checkpointed_forward(layer4, output)

        with pytest.raises(RuntimeError):
            assert manager.iter_fin()

        manager.batch_fin(0)

        assert torch.allclose(expected_output, output)

        expected_output.sum().backward()
        expected_grad = input_data.grad.clone().detach()
        input_data.grad.zero_()

        manager.recompute_next(0)
        output.sum().backward()
        manager.iter_fin()
        assert torch.allclose(expected_grad, input_data.grad)

