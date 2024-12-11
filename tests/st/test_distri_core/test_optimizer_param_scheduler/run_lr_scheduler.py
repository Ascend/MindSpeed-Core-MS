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
"""Pipeline parallel test"""

import os
import argparse
from functools import partial
import mindspore as ms
import mindspore.dataset as ds
from mindspore import mint
from mindspore.mint.optim import AdamW
from mindspore.communication import init

from mindspeed_ms.core.tensor_parallel.mappings import ReduceFromContextParallelRegion
from mindspeed_ms.training.utils import average_losses_across_data_parallel_group
from mindspeed_ms.training.yaml_arguments import validate_yaml
from mindspeed_ms.training.global_vars import set_global_variables
from mindspeed_ms.core.dist_checkpointing import load_checkpoint
from mindspeed_ms.core.optimizer import get_optimizer_param_scheduler
from mindspeed_ms.core.parallel_state import initialize_model_parallel
from mindspeed_ms.training import TrainOneStepCell, train, get_args
from mindspeed_ms.training import parse_args, core_transformer_config_from_yaml
from tests.st.test_distri_core.test_pipeline_parallel.test_pipeline_net import PipelineTestNet, FakeData

ms.set_seed(2024)


def get_batch(data_iterator):
    """ get micro batch data """
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    return data.values()


def loss_func(loss_mask, output_tensor):
    """ loss reduction function. """
    args = get_args()

    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()

    if args.context_parallel_size > 1:
        loss = mint.cat([mint.sum(losses.view(-1) * loss_mask).view(1), loss_mask.sum().view(1)])
        loss = ReduceFromContextParallelRegion()(loss)
        loss = loss[0] / loss[1]
    else:
        loss = mint.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

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

    def core_forward_func(*args):
        input_data, labels = args
        output_tensor = model(input_data, labels)
        return output_tensor

    return input_tensor, core_forward_func, partial(loss_func, loss_mask)


def run_lr_scheduler():
    """main function."""
    args, defaults = parse_args()
    args = validate_yaml(args, defaults, {})
    set_global_variables(args, False)
    args.wrap_with_ddp = False
    config = core_transformer_config_from_yaml(args)
    ms.set_context(device_target="Ascend", mode=ms.PYNATIVE_MODE)
    init()

    initialize_model_parallel()

    # generate dataset
    dataset = FakeData(data_num=32, seq_length=args.seq_length)
    dataset_parallel = ds.GeneratorDataset(dataset, column_names=['input_ids', 'labels'], shuffle=False)
    # calculate global batch size
    dataset_parallel = dataset_parallel.batch(args.micro_batch_size)

    # init model
    network = PipelineTestNet(config)

    optimizer = AdamW(params=network.trainable_params(), lr=0.001)

    opt_param_scheduler = get_optimizer_param_scheduler(optimizer)

    # init train one step cell
    train_one_step_cell = TrainOneStepCell(network, optimizer, opt_param_scheduler, config)

    train(train_one_step_cell, dataset_parallel, forward_step)

    if args.yaml_cfg == 'test_iteration_tarining.yaml':
        assert optimizer.param_groups[0]['lr'] == 2.9999999999999997e-06
        load_checkpoint(config, network, optimizer, opt_param_scheduler, f"./output")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml-cfg', type=str, default='test_iteration_tarining.yaml',
                        help="test_iteration_tarining.yaml: test_lr_base_iteration_tarining")
    run_lr_scheduler()
