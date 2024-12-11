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
"""Mcore transformerblock test"""
import glob
import os
import argparse
import sys
import numpy as np

import mindspore as ms
import mindspore.dataset as ds
from mindspore.communication import init

from mindspeed_ms.legacy.model.enums import ModelType
from mindspeed_ms.legacy.model.module import Module as LegacyModule
from mindspeed_ms.legacy.model.transformer import ParallelTransformer
from mindspeed_ms.core.optimizer import get_optimizer, optimizer_config_from_args
from mindspeed_ms.training import TrainOneStepCell

from mindspeed_ms.core.transformer.transformer_block import TransformerBlock
from mindspeed_ms.core.transformer.module import Module as McoreModule
from mindspeed_ms.core.parallel_state import initialize_model_parallel, \
                                             get_tensor_model_parallel_world_size, \
                                             get_tensor_model_parallel_rank, \
                                             get_data_parallel_rank, \
                                             get_data_parallel_world_size
from mindspeed_ms.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from mindspeed_ms.training import parse_args, get_args, core_transformer_config_from_yaml, get_model
from mindspeed_ms.training.yaml_arguments import validate_yaml
from mindspeed_ms.training.global_vars import set_global_variables
from mindspeed_ms.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from mindspeed_ms.tools.utils import barrier_world

seed = 2024
np.random.seed(seed)

class TestData:
    """
    generate a test dataset
    """
    def __init__(self, input_data):
        super().__init__()
        self.input_data = input_data['hidden_states']
        self.attention_mask = input_data['attention_mask']
        self.dataset_size = self.input_data.shape[0]

    def __getitem__(self, index):
        index = int(index)
        return (ms.Tensor(self.input_data[index]), ms.Tensor(self.attention_mask[index], dtype=ms.bool_))

    def __len__(self):
        return self.dataset_size


def get_batch(data_iterator):
    """ get micro batch data """
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    return data.values()


def loss_func(output_tensor):
    """ loss func """
    return output_tensor, {'lm loss': output_tensor}


def forward_step(data_iterator, model):
    """Forward training step

    Args:
        data_iterator: Input data iterator.
        model: The model.
    """
    # get batch data
    input_data, attention_mask = get_batch(data_iterator)
    input_tensor = (input_data, attention_mask)

    def core_forward_func(*args):
        input_data, attention_mask = args
        output_tensor = model(input_data, attention_mask)
        return output_tensor

    return input_tensor, core_forward_func, loss_func


def param_name_legacy2mcore(legacy_name):
    """transform legacy param name to mcore param name"""
    mcore_name = legacy_name.replace("parallel_transformer", "decoder")
    mcore_name = mcore_name.replace("input_norm", "input_layernorm")
    mcore_name = mcore_name.replace("post_attention_norm", "pre_mlp_layernorm")
    mcore_name = mcore_name.replace("attention", "self_attention")
    mcore_name = mcore_name.replace("out_proj", "linear_proj")
    mcore_name = mcore_name.replace("qkv_proj", "linear_qkv")
    mcore_name = mcore_name.replace("mapping", "linear_fc1")
    mcore_name = mcore_name.replace("projection", "linear_fc2")
    mcore_name = mcore_name.replace("final_norm", "final_layernorm")
    return mcore_name


def clean_args():
    """ clean args for megatron """
    option_to_remove = ('--run_mode',
                        '--data_dir',
                        '--ckpt_dir')
    # Process and remove the option from sys.argv
    # Start at 1 to skip the script name
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] in option_to_remove:
            # Remove the option and its value
            if i + 1 < len(sys.argv):
                del sys.argv[i:i+2]
            else:
                del sys.argv[i]
        else:
            i += 1


class ParallelTransformerNet(LegacyModule):
    """ mindspore net """
    def __init__(
            self,
            config,
            pre_process=True,
            post_process=True
        ):
        super().__init__(config)
        args = get_args()
        self.pre_process = pre_process
        self.post_process = post_process
        self.use_rope = args.position_embedding_type == 'rope'
        self.get_rotary_embedding = RotaryEmbedding(config.hidden_size
                                                    // config.num_attention_heads, rotary_percent=1.0)
        self.parallel_transformer = ParallelTransformer(config=config, model_type=ModelType.encoder_or_decoder,
                                                        pre_process=pre_process, post_process=post_process)

    def construct(self, hidden_states, attention_mask):
        """ mindspore net forward """
        args = get_args()
        if self.pre_process:
            hidden_states = hidden_states.astype(ms.bfloat16)
            hidden_states = hidden_states.swapaxes(0, 1)
        else:
            hidden_states = None

        if not self.use_rope:
            rotary_pos_emb = None
        else:
            rotary_pos_emb = self.get_rotary_embedding(args.seq_length)

        output = self.parallel_transformer(hidden_states=hidden_states,
                                           attention_mask=attention_mask,
                                           rotary_pos_emb=rotary_pos_emb)
        if self.post_process:
            output = output.abs().mean()
        return output

    def set_input_tensor(self, input_tensor) -> None:
        """Sets input tensor to the model."""
        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]

        assert len(input_tensor) == 1, 'input_tensor should only be length 1 for gpt/bert'
        self.parallel_transformer.set_input_tensor(input_tensor[0])

class TransformerBlockNet(McoreModule):
    """ mindspore net """
    def __init__(
            self,
            config,
            pre_process=True,
            post_process=True
        ):
        super().__init__(config)
        args = get_args()
        self.pre_process = pre_process
        self.post_process = post_process
        self.share_embeddings_and_output_weights = False
        self.use_rope = args.position_embedding_type == 'rope'
        self.get_rotary_embedding = RotaryEmbedding(config.hidden_size
                                                    // config.num_attention_heads, rotary_percent=1.0)
        transformer_layer_spec = get_gpt_layer_local_spec()
        self.decoder = TransformerBlock(
            config=config,
            spec=transformer_layer_spec,
            pre_process=pre_process,
            post_process=post_process
        )

    def construct(self, hidden_states, attention_mask):
        """ mindspore net forward """
        args = get_args()
        if self.pre_process:
            hidden_states = hidden_states.astype(ms.bfloat16)
            hidden_states = hidden_states.swapaxes(0, 1)
        else:
            hidden_states = None

        if not self.use_rope:
            rotary_pos_emb = None
        else:
            rotary_pos_emb = self.get_rotary_embedding(args.seq_length)

        output = self.decoder(hidden_states=hidden_states,
                              attention_mask=attention_mask,
                              rotary_pos_emb=rotary_pos_emb)
        if self.post_process:
            output = output.abs().mean()
        return output

    def set_input_tensor(self, input_tensor) -> None:
        """Sets input tensor to the model."""
        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]

        assert len(input_tensor) == 1, 'input_tensor should only be length 1 for gpt/bert'
        self.decoder.set_input_tensor(input_tensor[0])

def run_legacy(data_dir, ckpt_dir):
    """ test mindspore """
    # init config
    args, defaults = parse_args()
    args = validate_yaml(args, defaults, {})
    set_global_variables(args, False)
    # we use ms.load_param_into_net, ddp is not supported
    args.wrap_with_ddp = False
    args.use_distributed_optimizer = False
    config = core_transformer_config_from_yaml(args)
    optimizer_config = optimizer_config_from_args(args)

    ms.set_seed(seed)

    # set env
    ms.set_context(device_target="Ascend", mode=ms.PYNATIVE_MODE, deterministic='ON')
    init()
    initialize_model_parallel(tensor_model_parallel_size=config.tensor_model_parallel_size,
                              pipeline_model_parallel_size=config.pipeline_model_parallel_size,
                              virtual_pipeline_model_parallel_size=config.virtual_pipeline_model_parallel_size)

    # pylint: disable=W0613
    def model_provider_func(pre_process=True, post_process=True):
        """ model provider func """
        network = ParallelTransformerNet(config=config,
                                         pre_process=pre_process,
                                         post_process=post_process)
        return network
    network = get_model(model_provider_func, config, wrap_with_ddp=args.wrap_with_ddp)

    # get ckpt and dataset
    data_dict, ckpt_dict = load_ckpt_and_data(args.global_batch_size,
                                              args.seq_length,
                                              config.hidden_size,
                                              config.compute_dtype,
                                              config.compute_dtype,
                                              data_dir,
                                              ckpt_dir,
                                              network)
    dataset = TestData(data_dict)
    dataset = ds.GeneratorDataset(dataset, column_names=['hidden_states', "attention_mask"], shuffle=False)
    dataset = dataset.batch(args.micro_batch_size)

    # load ckpt
    ms.load_param_into_net(network, ckpt_dict)

    optimizer = get_optimizer(
        optimizer_config,
        config,
        network.trainable_params(),
        network,
        grad_allreduce_op=args.loss_reduction
    )

    train_one_step_cell = TrainOneStepCell(network, optimizer, None, config)
    train_one_step_cell.set_train()
    config = train_one_step_cell.config
    for _ in range(args.train_iters):
        dataset_iterator = [dataset.create_dict_iterator(), dataset.create_dict_iterator()]
        loss, _, _, _, _ = train_one_step_cell(forward_step, dataset_iterator, False)
        print(f"Output not lm loss: {loss['lm loss'].abs()}")

# 启动/入参
def run_mcore(data_dir, ckpt_dir):
    """ test mindspore """
    # init config
    args, defaults = parse_args()
    args = validate_yaml(args, defaults, {})
    set_global_variables(args, False)
    # we use ms.load_param_into_net, ddp is not supported
    args.wrap_with_ddp = False
    args.use_distributed_optimizer = False
    config = core_transformer_config_from_yaml(args)
    optimizer_config = optimizer_config_from_args(args)

    ms.set_seed(seed)

    # set env
    ms.set_context(device_target="Ascend", mode=ms.PYNATIVE_MODE, deterministic='ON')
    init()
    initialize_model_parallel(tensor_model_parallel_size=config.tensor_model_parallel_size,
                              pipeline_model_parallel_size=config.pipeline_model_parallel_size,
                              virtual_pipeline_model_parallel_size=config.virtual_pipeline_model_parallel_size)

    # pylint: disable=W0613
    def model_provider_func(pre_process=True, post_process=True):
        """ model provider func """
        network = TransformerBlockNet(config=config,
                                      pre_process=pre_process,
                                      post_process=post_process)
        return network
    network = get_model(model_provider_func, config, wrap_with_ddp=args.wrap_with_ddp)

    # get ckpt and dataset
    data_dict, ckpt_dict = load_ckpt_and_data(args.global_batch_size,
                                              args.seq_length,
                                              config.hidden_size,
                                              config.compute_dtype,
                                              config.compute_dtype,
                                              data_dir,
                                              ckpt_dir,
                                              network)
    dataset = TestData(data_dict)
    dataset = ds.GeneratorDataset(dataset, column_names=['hidden_states', "attention_mask"], shuffle=False)
    dataset = dataset.batch(args.micro_batch_size)

    # load ckpt
    mcore_ckpt = {}
    ckpt_dict_keys = ckpt_dict.copy()
    for key in list(ckpt_dict_keys):
        mcore_key = param_name_legacy2mcore(key)
        mcore_ckpt[mcore_key] = ckpt_dict[key]
    ms.load_param_into_net(network, mcore_ckpt)

    optimizer = get_optimizer(
        optimizer_config,
        config,
        network.trainable_params(),
        network,
        grad_allreduce_op=args.loss_reduction
    )

    train_one_step_cell = TrainOneStepCell(network, optimizer, None, config)
    train_one_step_cell.set_train()
    config = train_one_step_cell.config
    for _ in range(args.train_iters):
        dataset_iterator = [dataset.create_dict_iterator(), dataset.create_dict_iterator()]
        loss, _, _, _, _ = train_one_step_cell(forward_step, dataset_iterator, False)
        print(f"Output not lm loss: {loss['lm loss'].abs()}")


def save_random_ckpt(model, save_path="data/alone/random_ckpt/"):
    """ save random ckpt """

    state_dict = model.parameters_dict()
    tp_rank = get_tensor_model_parallel_rank() if get_tensor_model_parallel_world_size() > 1 else 0
    dp_rank = get_data_parallel_rank() if get_data_parallel_world_size() > 1 else 0

    if dp_rank == 0:
        for name, value in state_dict.items():
            if value is None:
                continue

            np_value = np.random.randn(*value.shape).astype(np.float32)
            save_name = save_path + name + f"_tp{tp_rank}.npy"
            np.save(save_name, np_value)
    barrier_world()


def save_random_data(batch_size,
                     seq_length,
                     hidden_size,
                     save_path="data/alone/random_data/"):
    """ save random data """
    dp_rank = get_data_parallel_rank() if get_data_parallel_world_size() > 1 else 0
    tp_rank = get_tensor_model_parallel_rank() if get_tensor_model_parallel_world_size() > 1 else 0

    shape = (batch_size, seq_length, hidden_size)
    mask_shape = (batch_size, 1, seq_length, seq_length)

    if tp_rank == 0:
        hidden_states = np.random.randn(*shape).astype(np.float32) * (dp_rank + 1)
        attention_mask = 1 - np.tril(np.ones(mask_shape)).astype(np.float32)
        hidden_states_save_name = save_path + f'hidden_states_dp{dp_rank}.npy'
        attention_mask_save_name = save_path + f'attention_mask_dp{dp_rank}.npy'
        np.save(hidden_states_save_name, hidden_states)
        np.save(attention_mask_save_name, attention_mask)
    barrier_world()


# pylint: disable=W0613
def load_ckpt_and_data(batch_size,
                       seq_length,
                       hidden_size,
                       params_dtype,
                       compute_dtype,
                       data_dir="data/alone/random_data/",
                       ckpt_dir="data/alone/random_ckpt/",
                       model=None):
    """ load ckpt and data. """
    tensor_method = ms.Tensor
    tp_size = get_tensor_model_parallel_world_size()
    dp_size = get_data_parallel_world_size()
    tp_rank = get_tensor_model_parallel_rank() if tp_size > 1 else 0
    dp_rank = get_data_parallel_rank() if dp_size > 1 else 0

    # save random ckpt
    if not os.path.exists(ckpt_dir):
        raise FileNotFoundError(f"{ckpt_dir} is not exists !")
    num_files = len(glob.glob(os.path.join(ckpt_dir, "*.npy")))
    if num_files == 0:
        barrier_world()
        save_random_ckpt(model, ckpt_dir)

    # save random data
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"{data_dir} is not exists !")
    num_files = len(glob.glob(os.path.join(data_dir, "*.npy")))
    if num_files == 0:
        barrier_world()
        save_random_data(batch_size, seq_length, hidden_size, data_dir)

    # loading data
    data_dict = {}
    data_npy_files = glob.glob(os.path.join(data_dir, f"*_dp{dp_rank}.npy"))
    for cur_npy_file in data_npy_files:
        cur_name = cur_npy_file.split('/')[-1].replace(f"_dp{dp_rank}.npy", "")
        data_dict[cur_name] = np.load(cur_npy_file)

    # loading ckpt
    ckpt_dict = {}
    ckpt_npy_files = glob.glob(os.path.join(ckpt_dir, f"*_tp{tp_rank}.npy"))
    for cur_ckpt_npy_file in ckpt_npy_files:
        cur_name = cur_ckpt_npy_file.split('/')[-1].replace(f"_tp{tp_rank}.npy", "")
        cur_ckpt = ms.Parameter(tensor_method(np.load(cur_ckpt_npy_file), dtype=params_dtype), name=cur_name)
        ckpt_dict[cur_name] = cur_ckpt
    return data_dict, ckpt_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_mode', type=str, default='test_legacy',
                        help="legacy / mcore")
    parser.add_argument('--data_dir', type=str, default="data/alone/random_data/",
                        help="load data path")
    parser.add_argument('--ckpt_dir', type=str, default="data/alone/random_ckpt/",
                        help="load ckpt path")
    cmd_args, _ = parser.parse_known_args()
    clean_args()

    if cmd_args.run_mode == 'test_legacy':
        run_legacy(cmd_args.data_dir, cmd_args.ckpt_dir)
    elif cmd_args.run_mode == 'test_mcore':
        run_mcore(cmd_args.data_dir, cmd_args.ckpt_dir)
    else:
        raise TypeError("Wrong run mode !")
