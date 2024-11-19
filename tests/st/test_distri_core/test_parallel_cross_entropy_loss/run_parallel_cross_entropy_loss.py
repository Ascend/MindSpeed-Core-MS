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
"""run parallel cross attention"""

import argparse
import numpy as np

import mindspore as ms
import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.nn as nn
from mindspore import Tensor, ops, mint
from mindspore.communication.management import init
from mindspore.nn import AdamWeightDecay

from mindspeed_ms.training.yaml_arguments import validate_yaml
from mindspeed_ms.training.global_vars import set_global_variables
from mindspeed_ms.training import parse_args, core_transformer_config_from_yaml
from mindspeed_ms.core.optimizer import optimizer_config_from_args
from mindspeed_ms.core.parallel_state import (initialize_model_parallel, get_tensor_model_parallel_world_size,
                                              get_data_parallel_world_size)
from mindspeed_ms.core.tensor_parallel.layers import ColumnParallelLinear
from mindspeed_ms.training import get_loss_func
from tests.st.test_distri_core.utils import TestData, train


class ColumnParallelLinearNet(nn.Cell):
    """ColumnParallelLinear network."""
    def __init__(self,
                 in_channels,
                 out_channels,
                 loss_fn,
                 config=None,
                 weight_init=0.5,
                 bias_init='zeros',
                 gather_output=True):
        super(ColumnParallelLinearNet, self).__init__()
        self.linear = ColumnParallelLinear(input_size=in_channels,
                                           output_size=out_channels,
                                           config=config,
                                           init_method=weight_init,
                                           bias=True,
                                           gather_output=gather_output,
                                           skip_bias_add=False,
                                           bias_init=bias_init)
        self.loss = loss_fn

    def construct(self, x, labels):
        input_mask = ops.full((4,), 1, dtype=mstype.float32)
        output = self.linear(x)[0]
        labels = labels.reshape(-1)
        output = output.reshape(output.shape[1:])
        losses = self.loss(output, labels)
        loss = self.loss_reduce(input_mask, losses)
        return loss

    def loss_reduce(self, loss_mask, output_tensor):
        if output_tensor.ndim == 2:
            output_tensor = output_tensor.swapaxes(0, 1).contiguous()
        losses = output_tensor.float()
        loss_mask = loss_mask.view(-1).float()
        total_tokens = loss_mask.sum()
        loss = mint.cat([mint.sum(losses.view(-1) * loss_mask).view(1), total_tokens.view(1)])
        return loss[0] / loss[1]


def run_parallel_cross_entropy_loss(config, args):
    """test cross entropy loss."""
    dataset_size = 3
    ms.set_context(device_target="Ascend", mode=ms.PYNATIVE_MODE, deterministic='ON')
    init()
    initialize_model_parallel(tensor_model_parallel_size=config.tensor_model_parallel_size)
    print(f"dp: {get_data_parallel_world_size()}, tp: {get_tensor_model_parallel_world_size()}")

    ms.set_seed(2024)
    input_data = np.random.random((dataset_size, args.seq_length, args.seq_length)).astype(np.float32)
    label_data = np.zeros((dataset_size, args.seq_length)).astype(np.int32)
    dataset = TestData(input_data=input_data, label_data=label_data)
    dataset = ds.GeneratorDataset(dataset, column_names=['input_ids', 'labels'])
    dataset = dataset.batch(args.batch_size)

    optimizer_config = optimizer_config_from_args(args)
    loss = get_loss_func(optimizer_config)
    gather_output = False
    if args.loss_func_kwargs.loss_func_type == "CrossEntropyLoss":
        gather_output = True
    network = ColumnParallelLinearNet(in_channels=args.seq_length,
                                      out_channels=args.seq_length,
                                      loss_fn=loss,
                                      config=config,
                                      gather_output=gather_output)

    input_ids = Tensor(shape=(None, None, None), dtype=mstype.float32)
    labels = Tensor(shape=(None, None), dtype=mstype.int32)
    network.set_inputs(input_ids, labels)
    optimizer = AdamWeightDecay(params=network.get_parameters())

    train(1, dataset, network, optimizer, None)


def extra_args_provider(inner_parser):
    inner_parser.add_argument('--run_mode', type=str, default='cross_entropy_loss',
                              help="cross_entropy_loss model")
    return inner_parser


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_mode', type=str, default='cross_entropy_loss',
                        help="cross_entropy_loss model")
    parser.add_argument('--yaml-cfg', type=str, default=None,
                        help="yaml file path")
    extra_args = parser.parse_args()
    main_args, defaults = parse_args(extra_args_provider=extra_args_provider)
    main_args = validate_yaml(main_args, defaults, {})
    set_global_variables(main_args, False)
    # init config
    used_config = core_transformer_config_from_yaml(main_args)

    if extra_args.run_mode == 'cross_entropy_loss':
        used_config.tensor_model_parallel_size = 2
        main_args.tensor_model_parallel_size = 2
        run_parallel_cross_entropy_loss(used_config, main_args)
    elif extra_args.run_mode == 'vocab_parallel_cross_entropy_loss':
        main_args.loss_func_kwargs.loss_func_type = "VocabParallelCrossEntropy"
        used_config.tensor_model_parallel_size = 2
        main_args.tensor_model_parallel_size = 2
        run_parallel_cross_entropy_loss(used_config, main_args)
    elif extra_args.run_mode == 'cross_entropy_loss_single':
        run_parallel_cross_entropy_loss(used_config, main_args)
    elif extra_args.run_mode == 'vocab_parallel_cross_entropy_loss_single':
        main_args.loss_func_kwargs.loss_func_type = "VocabParallelCrossEntropy"
        run_parallel_cross_entropy_loss(used_config, main_args)
    elif extra_args.run_mode == 'cross_entropy_loss_dp2':
        run_parallel_cross_entropy_loss(used_config, main_args)
    elif extra_args.run_mode == 'vocab_parallel_cross_entropy_loss_dp2':
        main_args.loss_func_kwargs.loss_func_type = "VocabParallelCrossEntropy"
        run_parallel_cross_entropy_loss(used_config, main_args)
