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
import numpy as np
import mindspore as ms
import mindspore.dataset as ds
from mindspore.nn import Adam, SoftmaxCrossEntropyWithLogits
from mindspore.communication import init
from mindspeed_ms.legacy.model import TransformerLanguageModel
from mindspeed_ms.training import (
    TrainOneStepCell,
    train,
    parse_args,
    get_args,
    core_transformer_config_from_args,
    core_transformer_config_from_yaml
)
from mindspeed_ms.core.parallel_state import initialize_model_parallel, get_data_parallel_world_size
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
    # pylint: disable=W0621
    def __init__(self, config):
        super().__init__()
        self.language_model = TransformerLanguageModel(config, encoder_attn_mask_type=None)
        self.loss = SoftmaxCrossEntropyWithLogits(reduction='mean')

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
    global_batch_size = args.global_batch_size // get_data_parallel_world_size()
    fake_dataset = fake_dataset.batch(global_batch_size)
    print("global batch size: ", global_batch_size, flush=True)

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

    # init train one step cell
    train_one_step_cell = TrainOneStepCell(network, optimizer, None, config)

    # train
    train(train_one_step_cell, fake_dataset)


if __name__ == '__main__':
    args = parse_args()
    if args.yaml_cfg is not None:
        config = core_transformer_config_from_yaml(args)
    else:
        config = core_transformer_config_from_args(args)
    run_parallel_language_model(config)
