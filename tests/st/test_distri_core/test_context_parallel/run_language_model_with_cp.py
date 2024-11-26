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
import argparse
import numpy as np
import mindspore as ms
import mindspore.dataset as ds
import mindspore.ops as ops
from mindspore.nn import Adam, SoftmaxCrossEntropyWithLogits
from mindspore.communication import init
from mindspeed_ms.training import TrainOneStepCell, train
from mindspeed_ms.legacy.model import TransformerLanguageModel
from mindspeed_ms.legacy.model.enums import AttnMaskType
from mindspeed_ms.core.parallel_state import (
    initialize_model_parallel,
    get_context_parallel_world_size,
    get_context_parallel_group
)
from mindspeed_ms.core.context_parallel.utils import get_batch_on_this_cp_rank
from mindspeed_ms.core.config import (
    set_global_args,
    init_configs_from_yaml,
    TrainingConfig,
    ModelParallelConfig,
    TransformerConfig,
    DatasetConfig,
)
from tests.st.test_distri_core.utils import generate_ckpt, transform_transformerlayer_params

ms.set_seed(1024)
ds.set_seed(1024)


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
    def __init__(self, config):
        super().__init__()
        self.language_model = TransformerLanguageModel(config, encoder_attn_mask_type=AttnMaskType.causal)
        self.loss = SoftmaxCrossEntropyWithLogits(reduction='mean')
        if get_context_parallel_world_size() > 1:
            self.allgather = ops.AllGather(get_context_parallel_group())

    def construct(self, input_ids, attention_mask, labels):
        """ Language model test forward """
        hidden_states = self.language_model(input_ids, None, attention_mask)
        output = ms.ops.sum(hidden_states, dim=-1, keepdim=False)
        if get_context_parallel_world_size() > 1:
            seq_dim = 1
            output = self.allgather(output)
            split_outputs = ops.split(
                output, output.shape[0] // get_context_parallel_world_size(), axis=0
            )
            output = ops.cat(split_outputs, axis=seq_dim)

            labels = self.allgather(labels)
            split_labels = ops.split(
                labels, labels.shape[0] // get_context_parallel_world_size(), axis=0
            )
            labels = ops.cat(split_labels, axis=seq_dim)

        labels = ms.Tensor(labels, dtype=ms.float32)
        loss = self.loss(output, labels)
        cp_world_size = get_context_parallel_world_size()
        loss = loss * cp_world_size
        return loss

def get_batch(data_iterator):
    """
    Retrieve the next batch of data from the data iterator.
    """

    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None

    batch = {
        'input_ids': data["input_ids"].astype(ms.int32),
        'labels': data["labels"].astype(ms.int32),
        'attention_mask': data["attention_mask"].astype(ms.int32),
    }
    batch = get_batch_on_this_cp_rank(batch)
    # for key, value in batch.items():
    #     print(f"{key}: {value.shape}")
    return batch

def run_parallel_language_model(training_config, parallel_config, model_config, dataset_config):
    """main function."""
    ms.set_context(device_target="Ascend", mode=ms.PYNATIVE_MODE, deterministic="ON")

    # init
    init()
    initialize_model_parallel(context_parallel_size=parallel_config.context_parallel_size)

    # generate dataset
    dataset = FakeData(data_num=16, seq_length=model_config.seq_length)
    fake_dataset = ds.GeneratorDataset(dataset,
                                       column_names=['input_ids', 'labels', 'attention_mask'],
                                       shuffle=False)
    # calculate global batch size
    global_batch_size = dataset_config.batch_size * dataset_config.micro_batch_num
    fake_dataset = fake_dataset.batch(global_batch_size)
    print("global batch size: ", global_batch_size, flush=True)

    # init ckpt
    param_dict = generate_ckpt(hidden_size=model_config.hidden_size,
                               module_type='transformer',
                               num_layers=model_config.num_layers,
                               prefix=None,
                               vocab_size=model_config.vocab_size,
                               use_embedding=True)
    param_dict = transform_transformerlayer_params(param_dict,
                                                   hidden_size=model_config.hidden_size,
                                                   kv_hidden_size=None,
                                                   prefix='language_model.encoder.layers.')
    # init model
    network = ParallelLanguageModel(model_config)

    # load ckpt
    ms.load_param_into_net(network, param_dict)

    optimizer = Adam(params=network.trainable_params(), learning_rate=0.001, beta1=0.9, beta2=0.95)

    # init train one step cell
    train_one_step_cell = TrainOneStepCell(network, optimizer, None, training_config, model_config)

    # train
    train(train_one_step_cell, fake_dataset, training_config, get_batch_func=get_batch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--generate_golden', action='store_true', help="Generate golden data for test."
    )
    args, _ = parser.parse_known_args()
    # set_global_args()
    CONFIG_PATH = "test_language_model_with_cp.yaml"
    assert os.path.exists(CONFIG_PATH) and CONFIG_PATH.endswith(('.yaml', '.yml'))
    training_config_main, parallel_config_main, dataset_config_main, model_config_main = init_configs_from_yaml(
        CONFIG_PATH, [TrainingConfig, ModelParallelConfig, DatasetConfig, TransformerConfig]
    )

    global_args = argparse.Namespace(**{**vars(training_config_main), **vars(parallel_config_main),
                                        **vars(dataset_config_main), **vars(model_config_main)})
    set_global_args(global_args)
    run_parallel_language_model(training_config_main, parallel_config_main, model_config_main, dataset_config_main)
