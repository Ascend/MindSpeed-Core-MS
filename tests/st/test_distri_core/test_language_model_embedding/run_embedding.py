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
"""Mcore languagemodelembedding test case"""
import glob
import os
import argparse
import sys
import numpy as np

import mindspore as ms
from mindspore import nn
from mindspore.communication import init

from tests.st.test_distri_core.utils import save_output_data

from mindspeed_ms.legacy.model.module import Module as LegacyModule
from mindspeed_ms.legacy.model.language_model import Embedding as LegacyEmbedding

from mindspeed_ms.core.models.common.embeddings.language_model_embedding import \
    LanguageModelEmbedding as McoreEmbedding
from mindspeed_ms.core.transformer.module import Module as McoreModule
from mindspeed_ms.core.parallel_state import initialize_model_parallel, \
                                             get_pipeline_model_parallel_world_size, \
                                             get_tensor_model_parallel_world_size, get_tensor_model_parallel_rank, \
                                             get_data_parallel_rank, get_data_parallel_group, \
                                             get_data_parallel_world_size
from mindspeed_ms.core.transformer import ModuleSpec, build_module
from mindspeed_ms.training import parse_args, get_args, core_transformer_config_from_yaml, get_model
from mindspeed_ms.tools.utils import barrier_world

seed = 2024
np.random.seed(seed)

def clean_args():
    """ clean args for megatron """
    option_to_remove = ('--run_mode',
                        '--data_dir',
                        '--ckpt_dir',
                        '--output_dir')
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


class McoreEmbeddingNet(McoreModule):
    """ mindspore net """
    def __init__(self, config):
        super().__init__(config)
        model_args = get_args()
        embedding_spec = ModuleSpec(module=McoreEmbedding)
        self.embedding = build_module(embedding_spec,
                                      config=config,
                                      vocab_size=model_args.vocab_size,
                                      max_sequence_length=model_args.seq_length,
                                      position_embedding_type=model_args.position_embedding_type)

    def construct(self, input_ids, position_ids, tokentype_ids=None):
        """ mindspore net forward """
        output = self.embedding(input_ids, position_ids, tokentype_ids)
        loss = output.astype(ms.float32).abs().mean()
        return loss, output


class LegacyEmbeddingNet(LegacyModule):
    """ mindspore net """
    def __init__(self, config):
        super().__init__(config)
        model_args = get_args()
        self.embedding = LegacyEmbedding(
            hidden_size=config.hidden_size,
            vocab_size=model_args.vocab_size,
            max_sequence_length=model_args.seq_length,
            embedding_dropout_prob=0.0,
            config=config)

    def construct(self, input_ids, position_ids, tokentype_ids=None):
        """ mindspore net forward """
        output = self.embedding(input_ids, position_ids, tokentype_ids)
        loss = output.astype(ms.float32).abs().mean()
        return loss, output


def run_legacy(data_dir, ckpt_dir, save_output_dir):
    """ test mindspore """
    # init config
    args = parse_args()
    config = core_transformer_config_from_yaml(args)

    ms.set_seed(seed)

    # set env
    ms.set_context(device_target="Ascend", mode=ms.PYNATIVE_MODE, deterministic='ON')
    init()
    initialize_model_parallel(tensor_model_parallel_size=config.tensor_model_parallel_size)

    global_batch_size = args.global_batch_size

    # pylint: disable=W0613
    def model_provider_func(pre_process=True, post_process=True):
        """ model provider func """
        network = LegacyEmbeddingNet(config)
        return network
    network = get_model(model_provider_func, config)
    if isinstance(network, nn.CellList) and len(network) == 1:
        network = network[0]
    network.set_train(True)

    # get ckpt and dataset
    data_dict, ckpt_dict = load_ckpt_and_data(global_batch_size,
                                              args.seq_length,
                                              config.hidden_size,
                                              config.compute_dtype,
                                              ms.int32,
                                              data_dir,
                                              ckpt_dir,
                                              network)
    print(f"all of inputs for model: {data_dict.keys()}")
    grad_position = list(data_dict.keys()).index("input_ids")

    ms.load_param_into_net(network, ckpt_dict)

    # run model
    trainable_params = network.trainable_params()
    if get_pipeline_model_parallel_world_size() == 1:
        forward_backward_func = ms.value_and_grad(network,
                                                  grad_position=grad_position,
                                                  weights=trainable_params,
                                                  has_aux=True)
        (loss, output), (dout, dw) = forward_backward_func(**data_dict)
        if get_data_parallel_world_size() > 1:
            # reduce loss
            loss = (ms.communication.comm_func.all_reduce(loss, group=get_data_parallel_group())[0]
                    / get_data_parallel_world_size())
            # reduce dout
            dout = (ms.communication.comm_func.all_reduce(dout, group=get_data_parallel_group())[0]
                    / get_data_parallel_world_size())
            # reduce dw
            dw = list(dw)
            for idx, cur_dw in enumerate(dw):
                cur_dw = (ms.communication.comm_func.all_reduce(cur_dw, group=get_data_parallel_group())[0]
                          / get_data_parallel_world_size())
                dw[idx] = cur_dw
    else:
        raise NotImplementedError("Need to be Implemented when pp_size > 1")

    dp_rank = get_data_parallel_rank() if get_data_parallel_world_size() > 1 else 0
    tp_rank = get_tensor_model_parallel_rank() if get_tensor_model_parallel_world_size() > 1 else 0

    print(f"dp rank: {dp_rank}, tp rank: {tp_rank}, loss: {loss.asnumpy()}")

    # save forward
    save_output_data(output.astype(ms.float32).asnumpy(),
                     os.path.join(save_output_dir, 'legacy_forward'),
                     'output',
                     f'dp{dp_rank}_tp{tp_rank}')
    # save dout
    save_output_data(dout.astype(ms.float32).asnumpy(),
                     os.path.join(save_output_dir, 'legacy_backward'),
                     'dout',
                     f'tp{tp_rank}')
    # save dw
    weight_names = [param.name for param in trainable_params]
    for idx, cur_dw in enumerate(dw):
        save_output_data(cur_dw.astype(ms.float32).asnumpy(),
                         os.path.join(save_output_dir, 'legacy_backward'),
                         weight_names[idx] + '_dw',
                         f'tp{tp_rank}')


# 启动/入参
def run_mcore(data_dir, ckpt_dir, save_output_dir):
    """ test mindspore """
    # init config
    args = parse_args()
    config = core_transformer_config_from_yaml(args)

    ms.set_seed(seed)

    # set env
    ms.set_context(device_target="Ascend", mode=ms.PYNATIVE_MODE, deterministic='ON')
    init()
    initialize_model_parallel(tensor_model_parallel_size=config.tensor_model_parallel_size)

    global_batch_size = args.global_batch_size

    # pylint: disable=W0613
    def model_provider_func(pre_process=True, post_process=True):
        """ model provider func """
        network = McoreEmbeddingNet(config)
        return network
    network = get_model(model_provider_func, config)
    if isinstance(network, nn.CellList) and len(network) == 1:
        network = network[0]
    network.set_train(True)

    # get ckpt and dataset
    data_dict, ckpt_dict = load_ckpt_and_data(global_batch_size,
                                              args.seq_length,
                                              config.hidden_size,
                                              config.compute_dtype,
                                              ms.int32,
                                              data_dir,
                                              ckpt_dir,
                                              network)
    print(f"all of inputs for model: {data_dict.keys()}")
    grad_position = list(data_dict.keys()).index("input_ids")

    ms.load_param_into_net(network, ckpt_dict)

    # run model
    trainable_params = network.trainable_params()
    if get_pipeline_model_parallel_world_size() == 1:
        forward_backward_func = ms.value_and_grad(network,
                                                  grad_position=grad_position,
                                                  weights=trainable_params,
                                                  has_aux=True)
        (loss, output), (dout, dw) = forward_backward_func(**data_dict)
        if get_data_parallel_world_size() > 1:
            # reduce loss
            loss = (ms.communication.comm_func.all_reduce(loss, group=get_data_parallel_group())[0]
                    / get_data_parallel_world_size())
            # reduce dout
            dout = (ms.communication.comm_func.all_reduce(dout, group=get_data_parallel_group())[0]
                    / get_data_parallel_world_size())
            # reduce dw
            dw = list(dw)
            for idx, cur_dw in enumerate(dw):
                cur_dw = (ms.communication.comm_func.all_reduce(cur_dw, group=get_data_parallel_group())[0]
                          / get_data_parallel_world_size())
                dw[idx] = cur_dw
    else:
        raise NotImplementedError("Need to be Implemented when pp_size > 1")

    dp_rank = get_data_parallel_rank() if get_data_parallel_world_size() > 1 else 0
    tp_rank = get_tensor_model_parallel_rank() if get_tensor_model_parallel_world_size() > 1 else 0

    print(f"dp rank: {dp_rank}, tp rank: {tp_rank}, loss: {loss.asnumpy()}")

    # save forward
    save_output_data(output.astype(ms.float32).asnumpy(),
                     os.path.join(save_output_dir, 'mcore_forward'),
                     'output',
                     f'dp{dp_rank}_tp{tp_rank}')
    # save dout
    save_output_data(dout.astype(ms.float32).asnumpy(),
                     os.path.join(save_output_dir, 'mcore_backward'),
                     'dout',
                     f'tp{tp_rank}')
    # save dw
    weight_names = [param.name for param in trainable_params]
    for idx, cur_dw in enumerate(dw):
        save_output_data(cur_dw.astype(ms.float32).asnumpy(),
                         os.path.join(save_output_dir, 'mcore_backward'),
                         weight_names[idx] + '_dw',
                         f'tp{tp_rank}')


def save_random_ckpt(model, save_path="data/alone/random_ckpt/"):
    """ save random ckpt """

    state_dict = model.parameters_dict()
    tp_rank = get_tensor_model_parallel_rank() if get_tensor_model_parallel_world_size() > 1 else 0
    dp_rank = get_data_parallel_rank() if get_data_parallel_world_size() > 1 else 0

    if dp_rank == 0:
        for name, value in state_dict.items():
            if value is None:
                continue
            print(f"tp rank{tp_rank}, saving weight:{name}, shape:{value.shape}")
            np_value = np.random.randn(*value.shape).astype(np.float32)
            save_name = save_path + name + f"_tp{tp_rank}.npy"
            np.save(save_name, np_value)
    barrier_world()


# pylint: disable=W0613
def save_random_data(batch_size,
                     seq_length,
                     hidden_size,
                     save_path="data/alone/random_data/"):
    """ save random data """

    dp_rank = get_data_parallel_rank() if get_data_parallel_world_size() > 1 else 0
    tp_rank = get_tensor_model_parallel_rank() if get_tensor_model_parallel_world_size() > 1 else 0

    shape = (batch_size, seq_length)

    if tp_rank == 0:
        print(f"saving data dp rank {dp_rank}")
        input_ids = np.random.randint(0, 100, size=shape).astype(np.int32) + (dp_rank + 1)
        position_ids = np.random.randint(0, 100, size=shape).astype(np.int32) + (dp_rank + 1)
        input_ids_save_name = save_path + f'input_ids_dp{dp_rank}.npy'
        position_ids_save_name = save_path + f'position_ids_dp{dp_rank}.npy'
        np.save(input_ids_save_name, input_ids)
        np.save(position_ids_save_name, position_ids)
    barrier_world()


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
        print("save random ckpt !")
        barrier_world()
        save_random_ckpt(model, ckpt_dir)

    # save random data
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"{data_dir} is not exists !")
    num_files = len(glob.glob(os.path.join(data_dir, "*.npy")))
    if num_files == 0:
        print(f"saving random data dp rank {dp_rank}")
        barrier_world()
        save_random_data(batch_size, seq_length, hidden_size, data_dir)

    # loading data
    data_dict = {}
    data_npy_files = glob.glob(os.path.join(data_dir, f"*_dp{dp_rank}.npy"))
    for cur_npy_file in data_npy_files:
        cur_data = tensor_method(np.load(cur_npy_file), dtype=compute_dtype)
        cur_name = cur_npy_file.split('/')[-1].replace(f"_dp{dp_rank}.npy", "")
        data_dict[cur_name] = cur_data

    # loading ckpt
    ckpt_dict = {}
    ckpt_npy_files = glob.glob(os.path.join(ckpt_dir, f"*_tp{tp_rank}.npy"))
    ori_params_dtype = params_dtype
    for cur_ckpt_npy_file in ckpt_npy_files:
        cur_name = cur_ckpt_npy_file.split('/')[-1].replace(f"_tp{tp_rank}.npy", "")
        if 'position' in cur_ckpt_npy_file:
            params_dtype = ms.float32
        else:
            params_dtype = ori_params_dtype
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
    parser.add_argument('--output_dir', type=str, default="data/alone/output/",
                        help="load ckpt path")
    cmd_args, _ = parser.parse_known_args()
    clean_args()

    if cmd_args.run_mode == 'test_legacy':
        run_legacy(cmd_args.data_dir, cmd_args.ckpt_dir, cmd_args.output_dir)
    elif cmd_args.run_mode == 'test_mcore':
        run_mcore(cmd_args.data_dir, cmd_args.ckpt_dir, cmd_args.output_dir)
    else:
        raise TypeError("Wrong run mode!")
