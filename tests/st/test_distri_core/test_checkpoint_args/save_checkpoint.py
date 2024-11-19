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
""" Save Checkpoint """
import numpy as np

import mindspore.dataset as ds
from mindspore import Tensor
from mindspeed_ms.training.yaml_arguments import validate_yaml
from mindspeed_ms.training.initialize import initialize_mindspeed_ms

from mindspeed_ms.training import (
    get_model,
    get_loss_func,
    parse_args,
    TrainOneStepCell,
    core_transformer_config_from_args,
    core_transformer_config_from_yaml
)
from mindspeed_ms.core.optimizer import (
    optimizer_config_from_args,
    get_optimizer,
    get_optimizer_param_scheduler
)
from mindspeed_ms.core.transformer.transformer_config import TransformerConfig
from mindspeed_ms.core.dist_checkpointing import save_checkpoint
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
def main(config: TransformerConfig, args):
    """ Test ParallelTransformer. """
    print(f"config is:\n{config}")
    initialize_mindspeed_ms()

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

    network = get_model(model_provider_func, config, wrap_with_ddp=args.wrap_with_ddp)

    optimizer = get_optimizer(
        optimizer_config,
        config,
        network.trainable_params(),
        network
    )

    opt_param_scheduler = get_optimizer_param_scheduler(optimizer)

    train_one_step_cell = TrainOneStepCell(
        network, optimizer, opt_param_scheduler, config
    )

    save_checkpoint(
        config,
        train_one_step_cell.network_with_loss,
        train_one_step_cell.optimizer,
        train_one_step_cell.opt_param_scheduler,
        args.save
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
