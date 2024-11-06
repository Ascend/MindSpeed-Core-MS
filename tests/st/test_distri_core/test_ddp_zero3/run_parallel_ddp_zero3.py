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
"""run parallel ddp zero3"""

import argparse
import numpy as np

import mindspore as ms
import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.communication.management import init
from mindspore.nn import CrossEntropyLoss

from mindspeed_ms.training import TrainOneStepCell, train, get_model
from mindspeed_ms.core.parallel_state import initialize_model_parallel
from mindspeed_ms.core.optimizer import optimizer_config_from_args
from mindspeed_ms.legacy.model import ParallelMLP
from mindspeed_ms.legacy.model.module import Module
from mindspeed_ms.core.dist_checkpointing import save_checkpoint, load_checkpoint
from mindspeed_ms.training.yaml_arguments import core_transformer_config_from_yaml
from mindspeed_ms.training.arguments import parse_args

from tests.st.test_distri_core.utils import _transform_ckpt_helper
from tests.st.test_distri_core.utils import TestData


class ParallelMLPNet(Module):
    """
    define a pynative MLP net
    """

    def __init__(self, config):
        super(ParallelMLPNet, self).__init__()
        self.mlp0 = ParallelMLP(config=config)
        self.mlp1 = ParallelMLP(config=config)
        self.mlp2 = ParallelMLP(config=config)
        self.mlp3 = ParallelMLP(config=config)
        self.mlp4 = ParallelMLP(config=config)
        self.mlp5 = ParallelMLP(config=config)
        self.mlp6 = ParallelMLP(config=config)
        self.mlp7 = ParallelMLP(config=config)

        self.loss = CrossEntropyLoss()
        self.cast = ops.Cast()
        self.dtype = config.compute_dtype

    def construct(self, input_ids, labels):
        """ do construct and calc mean loss """
        input_id = ops.cast(input_ids, mstype.bfloat16)
        output, _ = self.mlp0(input_id)
        output, _ = self.mlp1(output)
        output, _ = self.mlp2(output)
        output, _ = self.mlp3(output)
        output, _ = self.mlp4(output)
        output, _ = self.mlp5(output)
        output, _ = self.mlp6(output)
        output, _ = self.mlp7(output)

        labels = labels
        loss = output.abs().mean()
        return loss


def run_parallel_ddp_zero3(golden, first, args):
    """
    run pynative mode in ddp zero3
    """
    model_config = core_transformer_config_from_yaml(args)

    dataset_size = 20
    zero_shard_size = -1 if (first or golden) else 2
    ms.set_context(device_target="Ascend", mode=ms.PYNATIVE_MODE, pynative_synchronize=True, deterministic='ON')

    init()
    initialize_model_parallel(tensor_model_parallel_size=model_config.tensor_model_parallel_size,
                              zero_shard_size=zero_shard_size)

    ms.set_seed(2024)

    input_data = np.random.random((dataset_size, args.seq_length, model_config.hidden_size)).astype(np.float32)
    label_data = np.zeros((dataset_size, args.seq_length)).astype(np.float32)
    dataset = TestData(input_data=input_data, label_data=label_data)
    dataset = ds.GeneratorDataset(dataset, column_names=['input_ids', 'labels'])
    dataset = dataset.batch(args.batch_size)

    # pylint: disable=W0613
    def model_provider_func(pre_process=True, post_process=True):
        network = ParallelMLPNet(config=model_config)
        return network

    network = get_model(model_provider_func, model_config)

    input_ids = Tensor(shape=(None, None, None), dtype=mstype.float32)
    labels = Tensor(shape=(None, None), dtype=mstype.float32)
    network.set_inputs(input_ids, labels)

    from mindspeed_ms.core.optimizer import get_optimizer
    optimizer_config = optimizer_config_from_args(args)
    optimizer = get_optimizer(optimizer_config, model_config, network.trainable_params(), network)

    # init train one step cell
    train_one_step_cell = TrainOneStepCell(network, optimizer, None, model_config)
    print(f"network trainable params: {network.trainable_params()}", flush=True)

    if not first:
        if not golden:
            _transform_ckpt_helper(model_config, train_one_step_cell.network_with_loss,
                                   optimizer=train_one_step_cell.optimizer, src_ckpt_path=args.save,
                                   dst_ckpt_path="./dst_ckpt", timeout=60, output_format="ckpt")
        load_ckpt_path = args.save if golden else "./dst_ckpt"
        load_checkpoint(model_config, train_one_step_cell.network_with_loss, optimizer=train_one_step_cell.optimizer,
                        opt_param_scheduler=train_one_step_cell.opt_param_scheduler, ckpt_path=load_ckpt_path)
    train(train_one_step_cell, dataset, model_config)
    if first:
        save_checkpoint(model_config,
                        train_one_step_cell.network_with_loss,
                        train_one_step_cell.optimizer,
                        train_one_step_cell.opt_param_scheduler,
                        args.save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--golden', default=False, type=bool)
    parser.add_argument('--first', default=False, type=bool)
    parser.add_argument('--yaml-cfg', default=None, type=str)
    extra_args = parser.parse_args()

    def extra_parser_provider(inner_parser):
        inner_parser.add_argument('--golden', default=False, type=bool)
        inner_parser.add_argument('--first', default=False, type=bool)
        return inner_parser

    main_args = parse_args(extra_args_provider=extra_parser_provider)
    run_parallel_ddp_zero3(extra_args.golden, extra_args.first, main_args)
