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
"""Pretrain test"""
import os
from functools import partial
import numpy as np

import mindspore as ms
import mindspore.mint as mint
import mindspore.communication.comm_func as comm_func
import mindspore.dataset as ds

from mindspeed_ms.training import (
    get_args,
    core_transformer_config_from_args,
    core_transformer_config_from_yaml,
    pretrain
)
from mindspeed_ms.training.utils import average_losses_across_data_parallel_group
import mindspeed_ms.core.parallel_state as parallel_state
from mindspeed_ms.core.tensor_parallel.mappings import ReduceFromContextParallelRegion

from test_model import TestModel
from test_data import TestData


def get_batch(data_iterator):
    """ get batch func """
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None

    return data.values()


def model_provider_func(pre_process=True, post_process=True):
    """ model provider """
    args = get_args()

    if args.yaml_cfg is not None:
        config = core_transformer_config_from_yaml(args, "language_model")
    else:
        config = core_transformer_config_from_args(args)

    network = TestModel(config, pre_process=pre_process, post_process=post_process)

    return network

# pylint: disable=W0613
def train_valid_test_dataset_provider(unuse_args):
    """ dataset provider """
    args = get_args()
    dataset = TestData(data_num=40, seq_length=args.seq_length)
    test_dataset = ds.GeneratorDataset(dataset,
                                       column_names=['tokens', 'labels', 'attention_mask'],
                                       shuffle=False)
    test_dataset = test_dataset.batch(args.micro_batch_size)
    valid_dataset = test_dataset

    return test_dataset, valid_dataset, None


def loss_func(loss_mask, output_tensor):
    """ loss function for gpt model """
    # core_r0.6.0/core_r0.8.0
    args = get_args()
    args.version = '0.6'

    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    total_tokens = loss_mask.sum()

    # core_r0.6.0 version
    if args.version == '0.6':
        if args.context_parallel_size > 1:
            loss = mint.cat([mint.sum(losses.view(-1) * loss_mask).view(1), loss_mask.sum().view(1)])
            loss = ReduceFromContextParallelRegion()(loss)
            loss = loss[0] / loss[1]
        else:
            loss = mint.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    # core_r0.8.0 version
    elif args.version == '0.8':
        loss = mint.cat([mint.sum(losses.view(-1) * loss_mask).view(1), total_tokens.view(1)])
        if args.context_parallel_size > 1:
            loss = ReduceFromContextParallelRegion()(loss)

    # Check individual rank losses are not NaN prior to DP all-reduce
    if args.check_for_nan_in_loss_and_grad:
        global_rank = ms.communication.get_rank()
        if loss[0].isnan():
            raise RuntimeError(f"Rank {global_rank}: found NaN in local forward loss calculation. "
                               f"Device: {ms.hal.get_device_name()}, node: {os.uname()[1]}")

    # Reduce loss for logging
    if args.version == '0.6':
        average_loss = average_losses_across_data_parallel_group([loss])
        return loss * args.context_parallel_size, {'lm loss': average_loss[0]}
    if args.version == '0.8':
        reporting_loss = loss.copy()
        reporting_loss = comm_func.all_reduce(reporting_loss, group=parallel_state.get_data_parallel_group())[0]
        local_num_tokens = loss[1].copy().to(ms.int32)
        return (
            loss[0] * args.context_parallel_size,
            local_num_tokens,
            {'lm loss': (reporting_loss[0], reporting_loss[1])}
        )
    raise NotImplementedError("'loss func' code version must be 0.6 or 0.8.")


def forward_step(data_iterator, model):
    """Forward training step

    Args:
        data_iterator: Input data iterator.
        model: The model.
    """
    # get batch data
    tokens, labels, attention_mask = get_batch(
        data_iterator
    )
    loss_mask = ms.Tensor(np.ones(tokens.shape), dtype=ms.int32)
    input_tensor = (tokens, labels, attention_mask)

    def core_forward_func(*args):
        tokens, labels, attention_mask = args
        output_tensor = model(tokens, attention_mask, labels=labels)
        return output_tensor

    return input_tensor, core_forward_func, partial(loss_func, loss_mask)


def run_pretrain():
    """ run pretrain process """
    train_valid_test_dataset_provider.is_distributed = True
    pretrain(
        train_valid_test_dataset_provider=train_valid_test_dataset_provider,
        model_provider=model_provider_func,
        model_type=None,
        forward_step_func=forward_step,
        process_non_loss_data_func=None,
        extra_args_provider=None,
        args_defaults={},
    )


if __name__ == '__main__':
    run_pretrain()
