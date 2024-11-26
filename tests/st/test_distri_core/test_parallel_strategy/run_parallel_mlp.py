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
"""run parallel mlp"""

import argparse
import os

import numpy as np

import mindspore as ms
import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.communication.management import init, get_rank
from mindspore.nn import AdamWeightDecay, SoftmaxCrossEntropyWithLogits

from mindspeed_ms.training import parse_args, core_transformer_config_from_yaml
from mindspeed_ms.core.parallel_state import initialize_model_parallel
from mindspeed_ms.legacy.model import ParallelMLP
from mindspeed_ms.legacy.model.module import Module
from mindspeed_ms.core.utils import generate_state_dict, save_strategy_file

from tests.st.test_distri_core.utils import TestData, train


class ParallelMLPNet(Module):
    """
    define a pynative MLP net
    """
    def __init__(self, config):
        super(ParallelMLPNet, self).__init__()
        self.mlp = ParallelMLP(config=config)
        self.loss = SoftmaxCrossEntropyWithLogits()
        self.cast = ops.Cast()
        self.dtype = config.compute_dtype

    def construct(self, x, labels):
        x = self.cast(x, self.dtype)
        output, _ = self.mlp(x)
        output = ops.sum(output, dim=-1, keepdim=False)
        output = self.cast(output, mstype.float32)
        loss = self.loss(output, labels)
        return loss


src_strategy_path = "mlp_strategy_ckpt_src"
dst_strategy_path = "mlp_strategy_ckpt_dst"
src_network_ckpt = "mlp_network_ckpt_src"
dst_network_ckpt = "mlp_network_ckpt_dst"


def run_parallel_mlp_src(config, args):
    """
    run pynative mode mlp and load golden ckpt to generate pynative loss
    """
    dataset_size = 10
    ms.set_context(device_target="Ascend", mode=ms.PYNATIVE_MODE, deterministic='ON')

    init()
    initialize_model_parallel(tensor_model_parallel_size=config.tensor_model_parallel_size)

    ms.set_seed(2024)
    input_data = np.random.random((dataset_size, args.seq_length, config.hidden_size)).astype(np.float32)
    label_data = np.zeros((dataset_size, args.seq_length)).astype(np.float32)
    dataset = TestData(input_data=input_data, label_data=label_data)
    dataset = ds.GeneratorDataset(dataset, column_names=['input_ids', 'labels'])
    dataset = dataset.batch(args.micro_batch_size)

    network = ParallelMLPNet(config=config)

    input_ids = Tensor(shape=(None, None, None), dtype=mstype.float32)
    labels = Tensor(shape=(None, None), dtype=mstype.float32)
    network.set_inputs(input_ids, labels)
    optimizer = AdamWeightDecay(params=network.get_parameters())

    shard_info = generate_state_dict(network, optimizer)

    strategy_file = f"./{src_strategy_path}/rank_{get_rank()}/strategy.ckpt"
    save_strategy_file(shard_info, strategy_file)

    init_values = ms.numpy.arange(0, network.mlp.mapping.weight.size)
    init_values = init_values.reshape(network.mlp.mapping.weight.shape)
    network.mlp.mapping.weight.set_data(init_values)
    print("saved raw value:", network.mlp.mapping.weight.value())

    ckpt_path = f"./{src_network_ckpt}/rank_{get_rank()}"
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    ms.save_checkpoint(optimizer, ckpt_path + "/network.ckpt")

    train(1, dataset, network, optimizer, None)


def run_parallel_mlp_dst(config, args):
    """
    run pynative mode mlp and load golden ckpt to generate pynative loss
    """
    dataset_size = 10
    config.tensor_model_parallel_size = 4
    ms.set_context(device_target="Ascend", mode=ms.PYNATIVE_MODE, deterministic='ON')

    init()
    initialize_model_parallel(tensor_model_parallel_size=config.tensor_model_parallel_size)

    ms.set_seed(2024)
    input_data = np.random.random((dataset_size, args.seq_length, config.hidden_size)).astype(np.float32)
    label_data = np.zeros((dataset_size, args.seq_length)).astype(np.float32)
    dataset = TestData(input_data=input_data, label_data=label_data)
    dataset = ds.GeneratorDataset(dataset, column_names=['input_ids', 'labels'])
    dataset = dataset.batch(args.micro_batch_size)
    network = ParallelMLPNet(config=config)

    input_ids = Tensor(shape=(None, None, None), dtype=mstype.float32)
    labels = Tensor(shape=(None, None), dtype=mstype.float32)
    network.set_inputs(input_ids, labels)
    optimizer = AdamWeightDecay(params=network.get_parameters())

    shard_info = generate_state_dict(network, optimizer)

    strategy_file = f"./{dst_strategy_path}/rank_{get_rank()}/strategy.ckpt"
    save_strategy_file(shard_info, strategy_file)
    print("before loading", network.mlp.mapping.weight.value())

    if get_rank() == 0:
        ms.transform_checkpoints(src_network_ckpt, dst_network_ckpt, "dst_checkpoint",
                                 f"./{src_strategy_path}/rank_{get_rank()}/strategy.ckpt",
                                 f"./{dst_strategy_path}/rank_{get_rank()}/strategy.ckpt")
    else:
        import time
        time.sleep(5)
    dst_params = ms.load_checkpoint(f"./{dst_network_ckpt}/rank_{get_rank()}/dst_checkpoint{get_rank()}.ckpt")
    param_not_load, _ = ms.load_param_into_net(optimizer, dst_params)
    print("param_not_load:", param_not_load)
    print("after loading", network.mlp.mapping.weight.value())

    train(1, dataset, network, optimizer, None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate_src_strategy', action='store_true', help="Generate src strategy.")
    parser.add_argument('--yaml-cfg', default=None, type=str)
    extra_args = parser.parse_args()

    def extra_parser_provider(inner_parser):
        inner_parser.add_argument('--generate_src_strategy', action='store_true', help="Generate src strategy.")
        return inner_parser

    main_args = parse_args(extra_args_provider=extra_parser_provider)
    used_config = core_transformer_config_from_yaml(main_args)
    if extra_args.generate_src_strategy:
        run_parallel_mlp_src(used_config, main_args)
    else:
        run_parallel_mlp_dst(used_config, main_args)
