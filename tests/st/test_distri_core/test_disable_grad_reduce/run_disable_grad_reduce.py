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
"""disable grad reduce test"""
import os
import argparse
from functools import partial
import numpy as np
import mindspore as ms
import mindspore.dataset as ds
from mindspore import nn, mint
from mindspore.communication import init
from mindspore.nn import AdamWeightDecay, SoftmaxCrossEntropyWithLogits

from mindspeed_ms.training import train, parse_args, core_transformer_config_from_yaml, get_args, get_model
from mindspeed_ms.training.utils import average_losses_across_data_parallel_group
from mindspeed_ms.training.global_vars import set_global_variables
from mindspeed_ms.training.yaml_arguments import validate_yaml
from mindspeed_ms.core.tensor_parallel.layers import ColumnParallelLinear
from mindspeed_ms.core.tensor_parallel.mappings import ReduceFromContextParallelRegion
from mindspeed_ms.core.parallel_state import initialize_model_parallel
from tests.st.test_distri_core.utils import TestData


seed = 2024
np.random.seed(seed)
ms.set_seed(seed)


def get_batch(data_iterator):
    """ get micro batch data """
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    return data.values()


def loss_func(loss_mask, output_tensor):
    """ loss reduction function. """
    # pylint: disable=W0621
    args = get_args()

    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()

    if args.context_parallel_size > 1:
        loss = mint.cat([mint.sum(losses.view(-1) * loss_mask).view(1), loss_mask.sum().view(1)])
        loss = ReduceFromContextParallelRegion()(loss)
        loss = loss[0] / loss[1]
    else:
        loss = mint.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    print(f"final micro loss: {loss}")

    if args.check_for_nan_in_loss_and_grad:
        global_rank = ms.communication.get_rank()
        if loss.isnan():
            raise RuntimeError(f"Rank {global_rank}: found NaN in local forward loss calculation. "
                               f"Device: {ms.hal.get_device_name()}, node: {os.uname()[1]}")

    # Reduce loss for logging
    average_loss = average_losses_across_data_parallel_group([loss])
    return loss * args.context_parallel_size, {'lm loss': average_loss[0]}


def forward_step(data_iterator, model):
    """Forward training step

    Args:
        data_iterator: Input data iterator.
        model: The model.
    """
    # get batch data
    input_data, labels = get_batch(data_iterator)
    loss_mask = mint.ones_like(labels)
    input_tensor = (input_data, labels)

    # pylint: disable=W0621
    def core_forward_func(*args):
        input_data, labels = args
        output_tensor = model(input_data, labels)
        return output_tensor

    return input_tensor, core_forward_func, partial(loss_func, loss_mask)


class ColumnNet(nn.Cell):
    """ColumnParallelLinear Net"""
    # pylint: disable=W0621
    def __init__(self, config, disable_grad_reduce):
        super().__init__()
        self.linear = ColumnParallelLinear(input_size=config.hidden_size,
                                           output_size=config.hidden_size,
                                           config=config,
                                           init_method=config.init_method,
                                           bias=config.add_bias_linear,
                                           gather_output=True,
                                           skip_bias_add=False,
                                           params_dtype=config.params_dtype,
                                           compute_dtype=config.compute_dtype,
                                           disable_grad_reduce=disable_grad_reduce)
        self.loss = SoftmaxCrossEntropyWithLogits()
        self.config = config

    def construct(self, input_, labels):
        output, _ = self.linear(input_)
        loss = self.loss(output, labels)
        return loss

# pylint: disable=W0621
def run_disable_grad_reduce(config, args):
    """run clumnparallellinear net"""
    ms.set_context(device_target="Ascend", mode=ms.PYNATIVE_MODE, pynative_synchronize=True, deterministic='ON')
    tp = config.tensor_model_parallel_size
    init()
    initialize_model_parallel(tensor_model_parallel_size=tp)

    # pylint: disable=W0613
    def model_provider_func(pre_process=True, post_process=True):
        network = ColumnNet(config, args.disable_grad_reduce)
        return network

    network = get_model(model_provider_func, config)

    dataset_size = 3
    input_data = np.random.random((dataset_size, args.seq_length // tp)).astype(np.float32)
    label_data = np.zeros((dataset_size, args.seq_length)).astype(np.float32)
    dataset = TestData(input_data=input_data, label_data=label_data)
    dataset = ds.GeneratorDataset(dataset, column_names=['input_', 'labels'], shuffle=False)
    dataset = dataset.batch(args.micro_batch_size)

    optimizer = AdamWeightDecay(params=network.get_parameters())
    train(forward_step, network, optimizer, None, dataset, None, None, config)

def extra_args_provider(inner_parser):
    inner_parser.add_argument('--run_mode', type=str, default='True',
                              help="True: run disable_grad_reduce = True")
    return inner_parser

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_mode', type=str, default='True',
                        help="True: run disable_grad_reduce = True")
    parser.add_argument('--yaml-cfg', type=str, default=None,
                        help="yaml file path")
    extra_args = parser.parse_args()
    args, defaults = parse_args(extra_args_provider=extra_args_provider)
    args = validate_yaml(args, defaults, {})
    set_global_variables(args, False)

    args.data_layout = "BSH"
    if extra_args.run_mode == 'False':
        args.disable_grad_reduce = False
    else:
        args.disable_grad_reduce = True
    config = core_transformer_config_from_yaml(args)
    run_disable_grad_reduce(config, args)
