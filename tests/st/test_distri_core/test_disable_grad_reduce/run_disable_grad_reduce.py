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
import argparse
import numpy as np
import mindspore as ms
import mindspore.dataset as ds
from mindspore import nn
from mindspore.communication import init
from mindspore.nn import AdamWeightDecay, SoftmaxCrossEntropyWithLogits

from mindspeed_ms.training import train, parse_args, core_transformer_config_from_yaml
from mindspeed_ms.training.training import TrainOneStepCell
from mindspeed_ms.core.tensor_parallel.layers import ColumnParallelLinear
from mindspeed_ms.core.parallel_state import initialize_model_parallel
from tests.st.test_distri_core.utils import TestData

seed = 2024
np.random.seed(seed)
ms.set_seed(seed)

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
                                           param_init_dtype=config.params_dtype,
                                           compute_dtype=config.compute_dtype,
                                           disable_grad_reduce=disable_grad_reduce)
        self.loss = SoftmaxCrossEntropyWithLogits()

    def construct(self, input_, labels):
        output, _ = self.linear(input_)
        loss = self.loss(output, labels)
        return loss

# pylint: disable=W0621
def run_disable_grad_reduce(config, args):
    """run clumnparallellinear net"""
    ms.set_context(device_target="Ascend")
    tp = config.tensor_model_parallel_size
    init()
    initialize_model_parallel(tensor_model_parallel_size=tp)
    network = ColumnNet(config, args.disable_grad_reduce)

    dataset_size = 3
    input_data = np.random.random((dataset_size, args.seq_length // tp)).astype(np.float32)
    label_data = np.zeros((dataset_size, args.seq_length)).astype(np.float32)
    dataset = TestData(input_data=input_data, label_data=label_data)
    dataset = ds.GeneratorDataset(dataset, column_names=['input_', 'labels'], shuffle=False)
    dataset = dataset.batch(args.global_batch_size)

    optimizer = AdamWeightDecay(params=network.get_parameters())
    train_one_step_cell = TrainOneStepCell(network, optimizer, None, config)
    train(train_one_step_cell, dataset)

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
    args = parse_args(extra_args_provider=extra_args_provider)

    if extra_args.run_mode == 'False':
        args.disable_grad_reduce = False
    else:
        args.disable_grad_reduce = True
    config = core_transformer_config_from_yaml(args)
    run_disable_grad_reduce(config, args)
