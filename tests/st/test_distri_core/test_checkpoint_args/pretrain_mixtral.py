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
""" Pretrain Mixtral """
import os
from functools import partial
import numpy as np

import mindspore as ms
import mindspore.dataset as ds
from mindspore import Tensor, mint
from mindspore.communication.management import init

from mindspeed_ms.training import (
    pretrain,
    get_loss_func,
    parse_args,
    get_args,
    core_transformer_config_from_args,
    core_transformer_config_from_yaml
)
from mindspeed_ms.core.tensor_parallel.mappings import ReduceFromContextParallelRegion
from mindspeed_ms.training.utils import average_losses_across_data_parallel_group
from mindspeed_ms.training.yaml_arguments import validate_yaml
from mindspeed_ms.core.optimizer import optimizer_config_from_args
from mindspeed_ms.core.transformer.transformer_config import TransformerConfig
from tests.st.test_distri_core.utils import MixtralModel


dataset = None

def get_batch(data_iterator):
    """ get micro batch data """
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    batch = {
        'input_ids': data["input_ids"][:, :-1].astype(ms.int32),
        'labels': data["labels"][:, 1:].astype(ms.int32),
        'attention_mask': data["attention_mask"].astype(ms.int32),
    }

    return batch.values()

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
    input_data, labels, attention_mask = get_batch(data_iterator)
    loss_mask = mint.ones_like(labels)
    input_tensor = (input_data, labels, attention_mask)

    # pylint: disable=W0621
    def core_forward_func(*args):
        input_data, labels, attention_mask = args
        output_tensor = model(input_data, labels, attention_mask)
        return output_tensor

    return input_tensor, core_forward_func, partial(loss_func, loss_mask)


class TestData:
    """
    generate a test dataset
    """
    def __init__(self, dataset_size=None, input_data=None, label_data=None):
        super().__init__()
        self.dataset_size = dataset_size
        self.input_data = input_data
        self.data_shape = self.input_data.shape
        self.label_data = label_data
        seq_length = self.data_shape[1]
        self.attention_mask = np.tril(np.ones(shape=(1, seq_length - 1, seq_length - 1))).astype(np.int32)
        self.attention_mask = self.attention_mask < 0.5

    def __getitem__(self, index):
        return (Tensor(self.input_data[index]), Tensor(self.label_data[index]), Tensor(self.attention_mask))

    def __len__(self):
        return self.dataset_size

# pylint: disable=W0613
def train_valid_test_dataset_provider(unuse_args):
    """ get dataset """
    return dataset, None, None


# pylint: disable=W0621
def main(config: TransformerConfig, args):
    """ Test ParallelTransformer. """
    global dataset
    print(f"config is:\n{config}")

    init()

    # random.seed(args.seed)
    # ms.set_seed(args.seed)
    # np.random.seed(args.seed)

    input_data = np.random.randint(
        low=1, high=args.vocab_size,
        size=(10, args.seq_length + 2),
        dtype=np.int32)
    dataset = TestData(
        dataset_size=input_data.shape[0],
        input_data=input_data[:, :-1],
        label_data=input_data[:, 1:])
    dataset = ds.GeneratorDataset(
        dataset,
        column_names=['input_ids', 'labels', "attention_mask"],
        shuffle=False)
    dataset = dataset.batch(args.micro_batch_size)
    optimizer_config = optimizer_config_from_args(args)

    # build net
    def model_provider_func(pre_process=True, post_process=True):
        """ get mixtral model """
        loss = get_loss_func(optimizer_config)
        network = MixtralModel(
            config,
            parallel_output=False,
            loss_func=loss,
            pre_process=pre_process,
            post_process=post_process
        )
        return network

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
    args, defaults = parse_args()
    args = validate_yaml(args, defaults, {})

    args.deterministic_mode = True
    args.data_layout = "BSH"
    if args.yaml_cfg is None:
        config = core_transformer_config_from_args(args)
    else:
        config = core_transformer_config_from_yaml(args)
    main(config, args)
