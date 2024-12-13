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
"""Language model test"""
import os
from functools import partial
import numpy as np
import mindspore as ms
import mindspore.dataset as ds
from mindspore.nn import Adam, SoftmaxCrossEntropyWithLogits
from mindspore.communication import init
from mindspore import mint
from mindspeed_ms.legacy.model import TransformerLanguageModel
from mindspeed_ms.core.parallel_state import initialize_model_parallel
from mindspeed_ms.training import (
    train,
    parse_args,
    get_args,
    core_transformer_config_from_args,
    core_transformer_config_from_yaml
)
from mindspeed_ms.training.utils import average_losses_across_data_parallel_group
from mindspeed_ms.training.global_vars import set_global_variables
from mindspeed_ms.training.yaml_arguments import validate_yaml
from mindspeed_ms.core.tensor_parallel.mappings import ReduceFromContextParallelRegion
from tests.st.test_distri_core.utils import generate_ckpt, transform_transformerlayer_params


ms.set_seed(1024)
ds.set_seed(1024)


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
    loss_mask = ms.ops.sum(loss_mask, dim=-1, keepdim=False)
    input_tensor = (input_data, attention_mask, labels)

    # pylint: disable=W0621
    def core_forward_func(*args):
        input_data, attention_mask, labels = args
        output_tensor = model(input_data, attention_mask, labels)
        return output_tensor

    return input_tensor, core_forward_func, partial(loss_func, loss_mask)


class FakeData():
    """ generate fake data for language model test """
    def __init__(self, data_num, seq_length, input_data=None):
        super().__init__()
        if input_data is not None:
            self.input_data = input_data
            self.data_num = self.input_data.shape[0]
            self.seq_length = self.input_data[0].shape[0]
        else:
            input_data = np.random.randint(0, 100, (data_num, seq_length + 1))
            self.input_data = input_data[:, :-1]
            self.labels = input_data[:, 1:]
        ones = np.ones((data_num, 1, seq_length, seq_length), dtype=np.int32)
        self.attention_mask = np.tril(ones)

    def __getitem__(self, index):
        return ms.Tensor(self.input_data[index], dtype=ms.int32), \
               ms.Tensor(self.labels[index], dtype=ms.float32), \
               ms.Tensor(self.attention_mask[index], dtype=ms.int32)

    def __len__(self):
        return self.input_data.shape[0]


class ParallelLanguageModel(ms.nn.Cell):
    """ Test language model """
    # pylint: disable=W0621
    def __init__(self, config):
        super().__init__()
        self.language_model = TransformerLanguageModel(config, encoder_attn_mask_type=None)
        self.loss = SoftmaxCrossEntropyWithLogits(reduction='none')
        self.config = config

    def construct(self, input_ids, attention_mask, labels):
        """ Language model test forward """
        hidden_states = self.language_model(input_ids, None, attention_mask)
        output = ms.ops.sum(hidden_states, dim=-1, keepdim=False)
        loss = self.loss(output, labels)
        return loss


# pylint: disable=W0621
def run_parallel_language_model(config):
    """main function."""
    args = get_args()
    ms.set_context(device_target="Ascend", mode=ms.PYNATIVE_MODE, deterministic="ON")

    # init
    init()
    initialize_model_parallel()

    # generate dataset
    dataset = FakeData(data_num=16, seq_length=args.seq_length)
    fake_dataset = ds.GeneratorDataset(dataset,
                                       column_names=['input_ids', 'labels', 'attention_mask'],
                                       shuffle=False)
    # calculate global batch size
    fake_dataset = fake_dataset.batch(args.micro_batch_size)
    print("micro batch size: ", args.micro_batch_size, flush=True)

    # init ckpt
    param_dict = generate_ckpt(hidden_size=config.hidden_size,
                               module_type='transformer',
                               num_layers=config.num_layers,
                               prefix=None,
                               vocab_size=args.vocab_size,
                               use_embedding=True)
    param_dict = transform_transformerlayer_params(param_dict,
                                                   hidden_size=config.hidden_size,
                                                   kv_hidden_size=None,
                                                   prefix='language_model.encoder.layers.')
    # init model
    network = ParallelLanguageModel(config)

    # load ckpt
    ms.load_param_into_net(network, param_dict)

    optimizer = Adam(params=network.trainable_params(), learning_rate=0.001, beta1=0.9, beta2=0.95)

    train(forward_step, network, optimizer, None, fake_dataset, None, None, config)

if __name__ == '__main__':
    args, defaults = parse_args()
    args = validate_yaml(args, defaults, {})
    set_global_variables(args, False)

    args.wrap_with_ddp = False
    args.data_layout = "BSH"
    if args.yaml_cfg is not None:
        config = core_transformer_config_from_yaml(args)
    else:
        config = core_transformer_config_from_args(args)
    run_parallel_language_model(config)
