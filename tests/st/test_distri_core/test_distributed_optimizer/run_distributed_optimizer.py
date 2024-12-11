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
""" Test ParamAndGradBuffer """
import argparse
import sys

import numpy as np

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.common.dtype as mstype
import mindspore.dataset as ds
from mindspore.communication.management import init
from mindspore.nn import SoftmaxCrossEntropyWithLogits
from mindspore.mint.optim import AdamW
from mindspeed_ms.training import parse_args, core_transformer_config_from_yaml
from mindspeed_ms.training.yaml_arguments import validate_yaml
from mindspeed_ms.training.global_vars import set_global_variables
from mindspeed_ms.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from mindspeed_ms.core.parallel_state import (initialize_model_parallel, get_data_parallel_world_size,
                                              get_data_parallel_rank, get_data_parallel_group)
from mindspeed_ms.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from mindspeed_ms.core.optimizer.distrib_optimizer import DistributedOptimizer
from mindspeed_ms.core.optimizer import optimizer_config_from_args
from tests.st.test_distri_core.utils import TestData, train


def clean_args():
    """ clear args. """
    option_to_remove = '--golden'
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] in option_to_remove:
            if i + 1 < len(sys.argv):
                del sys.argv[i:i+2]
            else:
                del sys.argv[i]
        else:
            i += 1


class TestNet2(nn.Cell):
    """ test class. """
    def __init__(self, config):
        super(TestNet2, self).__init__()
        hidden_size = config.hidden_size
        self.columnlinear = ColumnParallelLinear(input_size=hidden_size,
                                                 output_size=hidden_size,
                                                 config=config,
                                                 init_method=config.init_method,
                                                 bias=config.add_bias_linear,
                                                 gather_output=False,
                                                 skip_bias_add=False,
                                                 bias_init=config.bias_init)
        self.rowlinear = RowParallelLinear(input_size=hidden_size,
                                           output_size=hidden_size,
                                           config=config,
                                           init_method=config.init_method,
                                           bias=config.add_bias_linear,
                                           input_is_parallel=True,
                                           skip_bias_add=False,
                                           bias_init=config.bias_init)
        self.loss = SoftmaxCrossEntropyWithLogits()

    def construct(self, input_, label_):
        output, _ = self.columnlinear(input_)
        output, _ = self.rowlinear(output)
        output = ops.sum(output, dim=-1, keepdim=False)
        output = ops.cast(output, mstype.float32)
        loss = self.loss(output, label_)

        return loss


def run_golden_optimizer():
    """
    Feature: test DDP with DistributedOptimizer
    Description: test DDP with DistributedOptimizer
    Expectation: test success
    """
    config = core_transformer_config_from_yaml(args)
    dataset_size = 6

    ms.set_context(device_target='Ascend', mode=ms.PYNATIVE_MODE)
    ms.set_seed(2024)
    init()
    initialize_model_parallel(tensor_model_parallel_size=1, order='tp-dp-pp')
    input_data = np.random.random((dataset_size, args.seq_length, config.hidden_size)).astype(np.float32)
    label_data = np.zeros((dataset_size, args.seq_length)).astype(np.float32)
    dataset = TestData(input_data=input_data, label_data=label_data)
    dataset = ds.GeneratorDataset(dataset, column_names=['input_ids', 'labels'],
                                  num_shards=get_data_parallel_world_size(), shard_id=get_data_parallel_rank())
    dataset = dataset.batch(args.global_batch_size)
    network_golden = TestNet2(config=config)
    optimizer_golden = AdamW(params=network_golden.get_parameters(), lr=1.0)

    train(epoch_num=1, dataset=dataset, network=network_golden, optimizer=optimizer_golden)


def run_distributed_optimizer():
    """
    Feature: test DDP with DistributedOptimizer
    Description: test DDP with DistributedOptimizer
    Expectation: test success
    """
    config = core_transformer_config_from_yaml(args)
    dataset_size = 6
    bucket_size = 10

    ms.set_context(device_target='Ascend', mode=ms.PYNATIVE_MODE)
    ms.set_seed(2024)
    init()
    initialize_model_parallel(tensor_model_parallel_size=1)
    input_data = np.random.random((dataset_size, args.seq_length, config.hidden_size)).astype(np.float32)
    label_data = np.zeros((dataset_size, args.seq_length)).astype(np.float32)
    dataset = TestData(input_data=input_data, label_data=label_data)
    dataset = ds.GeneratorDataset(dataset, column_names=['input_ids', 'labels'],
                                  num_shards=get_data_parallel_world_size(), shard_id=get_data_parallel_rank())
    dataset = dataset.batch(args.global_batch_size)
    ddp_config = DistributedDataParallelConfig(
        overlap_grad_reduce=True,
        use_distributed_optimizer=True,
        bucket_size=bucket_size,
        average_in_collective=True,
        enable_mem_align=True,
    )
    network = TestNet2(config=config)
    network_with_ddp = DistributedDataParallel(config=config,
                                               ddp_config=ddp_config,
                                               module=network)

    optimizer = AdamW(params=network_with_ddp.get_parameters(), lr=1.0)
    optimizer_config = optimizer_config_from_args(args)
    optimizer = DistributedOptimizer(optimizer=optimizer,
                                     config=optimizer_config,
                                     grad_scaler=None,
                                     init_state_fn=None,
                                     per_model_buffers=network_with_ddp.buffers,
                                     data_parallel_group=get_data_parallel_group(with_context_parallel=True),
                                     data_parallel_group_mccl=None)

    losses = train(epoch_num=1,
                   dataset=dataset,
                   network=network_with_ddp,
                   optimizer=optimizer)

    losses = list(map(lambda x: x[0], losses))
    print(losses, "get_data_parallel_rank():", get_data_parallel_rank())
    if get_data_parallel_rank() == 0:
        golden_losses = [2.0796428, 4.841238, 0.6431562]
    elif get_data_parallel_rank() == 1:
        golden_losses = [2.079449, 2.7259526, 12.919599]

    assert np.allclose(losses, golden_losses, atol=1.e-3, rtol=1.e-3)


def extra_args_provider(inner_parser):
    inner_parser.add_argument('--golden', default=False, type=bool)
    return inner_parser


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--golden', default=False, type=bool)
    parser.add_argument('--yaml-cfg', default=None, type=str)
    extra_args, _ = parser.parse_known_args()
    args, defaults = parse_args(extra_args_provider=extra_args_provider)
    args = validate_yaml(args, defaults, {})
    set_global_variables(args, False)

    if extra_args.golden:
        run_golden_optimizer()
    else:
        run_distributed_optimizer()
