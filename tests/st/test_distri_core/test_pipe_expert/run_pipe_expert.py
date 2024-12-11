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

import numpy as np

import mindspore as ms
import mindspore.dataset as ds
from mindspore import Tensor
from mindspore.communication.management import init
from mindspore.nn import SGD

from mindspeed_ms.training import (
    get_model,
    TrainOneStepCell,
    train,
    get_loss_func,
    parse_args,
    get_args,
    core_transformer_config_from_args,
    core_transformer_config_from_yaml
)
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
    initialize_model_parallel
)
from tests.st.test_distri_core.utils import MixtralModel


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

    network = get_model(model_provider_func, config)

    print(f"network construct is:\n{network}")
    print("network parameters are:")
    for param in network.get_parameters():
        print(f"{param.name} {param.dtype} {param.shape}")

    optimizer = SGD(params=network.trainable_params(), learning_rate=1e-4)
    train_one_step_cell = TrainOneStepCell(network, optimizer, None, config)
    # train
    train(train_one_step_cell, dataset)


if __name__ == '__main__':
    args = parse_args()
    args.data_layout = "BSH"
    if args.yaml_cfg is None:
        config = core_transformer_config_from_args(args)
    else:
        config = core_transformer_config_from_yaml(args)

    if args.checkpoint_dir and not os.path.exists(args.checkpoint_dir):
        args.checkpoint_dir = None
    run_mixtral(config)
