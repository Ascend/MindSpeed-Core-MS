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
""" Test Mixtral. """
import os
from functools import partial
import numpy as np

import mindspore as ms
import mindspore.dataset as ds
from mindspore import Tensor, mint
from mindspore.communication.management import init
from mindspore.nn import SGD

from mindspeed_ms.training import (
    get_model,
    train,
    get_loss_func,
    parse_args,
    get_args,
    core_transformer_config_from_args,
    core_transformer_config_from_yaml
)
from mindspeed_ms.training.utils import average_losses_across_data_parallel_group
from mindspeed_ms.training.yaml_arguments import validate_yaml
from mindspeed_ms.training.global_vars import set_global_variables
from mindspeed_ms.core.tensor_parallel import ReduceFromContextParallelRegion
from mindspeed_ms.core.optimizer import optimizer_config_from_args
from mindspeed_ms.core.transformer.transformer_config import TransformerConfig
from mindspeed_ms.core.parallel_state import (
    get_data_parallel_group,
    get_data_parallel_rank,
    get_expert_model_parallel_group,
    get_expert_model_parallel_rank,
    get_pipeline_model_parallel_group,
    get_pipeline_model_parallel_rank,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_context_parallel_world_size,
    initialize_model_parallel
)
from tests.st.test_distri_core.utils import MixtralModel


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
    """ reduce loss func """
    if output_tensor.ndim == 2:
        output_tensor = output_tensor.swapaxes(0, 1).contiguous()
    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    total_tokens = loss_mask.sum()
    loss = mint.cat([mint.sum(losses.view(-1) * loss_mask).view(1), total_tokens.view(1)])

    cp_world_size = get_context_parallel_world_size()
    if cp_world_size > 1:
        loss = ReduceFromContextParallelRegion()(loss)

    # Check individual rank losses are not NaN prior to DP all-reduce.
    if args.check_for_nan_in_loss_and_grad:
        global_rank = ms.communication.get_rank()
        if loss[0].isnan():
            raise ValueError(f"Rank {global_rank}: found NaN in local forward loss calculation")

    average_loss = average_losses_across_data_parallel_group([loss[0] * cp_world_size / loss[1]])

    return loss[0] * cp_world_size / loss[1], {'lm loss': average_loss[0]}


def forward_step(data_iterator, model):
    """Forward training step

    Args:
        data_iterator: Input data iterator.
        model: The model.
    """
    # pylint: disable=W0621
    args = get_args()

    # get batch data
    input_data, labels, attention_mask = get_batch(data_iterator)
    loss_mask = mint.ne(input_data, args.pad_token).astype(args.compute_dtype)
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


# pylint: disable=W0621
def run_mixtral(config: TransformerConfig):
    """ Test ParallelTransformer. """
    args = get_args()
    args.wrap_with_ddp = False
    print(f"config is:\n{config}")
    tp = config.tensor_model_parallel_size
    ep = config.expert_model_parallel_size
    pp = config.pipeline_model_parallel_size
    vpp = config.virtual_pipeline_model_parallel_size
    seq_length = args.seq_length
    micro_batch_num = args.micro_batch_size

    ms.set_context(
        device_target="Ascend",
        mode=ms.PYNATIVE_MODE,
        max_device_memory="58GB",
        deterministic='ON',
        pynative_synchronize=True
        )

    init()
    initialize_model_parallel(
        tensor_model_parallel_size=tp,
        expert_model_parallel_size=ep,
        pipeline_model_parallel_size=pp,
        virtual_pipeline_model_parallel_size=vpp
        )

    dp_group = get_data_parallel_group()
    ep_group = get_expert_model_parallel_group()
    tp_group = get_tensor_model_parallel_group()
    pp_group = get_pipeline_model_parallel_group()
    dp_rank = get_data_parallel_rank()
    ep_rank = get_expert_model_parallel_rank()
    tp_rank = get_tensor_model_parallel_rank()
    pp_rank = get_pipeline_model_parallel_rank()

    print(f"dp_group is {dp_group}, ep_group is {ep_group}, tp_group is {tp_group}, pp_group is {pp_group}", flush=True)
    print(f"dp_rank is {dp_rank}, ep_rank is {ep_rank}, tp_rank is {tp_rank}, pp_rank is {pp_rank}", flush=True)

    ms.set_seed(2024)

    golden_input_and_loss_path = args.data_path

    # load golden input and loss
    assert os.path.exists(golden_input_and_loss_path), f"'{golden_input_and_loss_path}' did not exits"
    input_and_loss = np.load(golden_input_and_loss_path, allow_pickle=True).tolist()
    input_data = input_and_loss['input']
    assert input_data.shape == (2, seq_length + 1), \
           f"expect input.shape == (2, {seq_length + 1}), but got {input_data.shape}"

    # making dataset
    if ep == 1:
        dataset = TestData(dataset_size=2, input_data=input_data, label_data=input_data)
        dataset = ds.GeneratorDataset(dataset, column_names=['input_ids', 'labels', "attention_mask"], shuffle=False)
        dataset = dataset.batch(2)
    else:
        input_data = np.tile(input_data[dp_rank % 2, None], (micro_batch_num, 1))
        dataset = TestData(dataset_size=micro_batch_num, input_data=input_data, label_data=input_data)
        dataset = ds.GeneratorDataset(dataset, column_names=['input_ids', 'labels', "attention_mask"], shuffle=False)
        dataset = dataset.batch(micro_batch_num)

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

    network = get_model(model_provider_func, config, wrap_with_ddp=args.wrap_with_ddp)

    print(f"network construct is:\n{network}")
    print("network parameters are:")
    for param in network.get_parameters():
        print(f"{param.name} {param.dtype} {param.shape}")

    optimizer = SGD(params=network.trainable_params(), learning_rate=1e-4)
    train(forward_step, network, optimizer, None, dataset, None, None, config)

if __name__ == '__main__':
    args, defaults = parse_args()
    args = validate_yaml(args, defaults, {})
    set_global_variables(args, False)
    args.data_layout = "BSH"
    args.wrap_with_ddp = False
    if args.yaml_cfg is None:
        config = core_transformer_config_from_args(args)
    else:
        config = core_transformer_config_from_yaml(args)

    if args.checkpoint_dir and not os.path.exists(args.checkpoint_dir):
        args.checkpoint_dir = None
    run_mixtral(config)
