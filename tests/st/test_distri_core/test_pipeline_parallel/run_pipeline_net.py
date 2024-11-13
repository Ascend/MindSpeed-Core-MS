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
"""Pipeline parallel test"""
import argparse
import numpy as np
import mindspore as ms
import mindspore.dataset as ds
from mindspore.nn import Adam
from mindspore.mint.optim import AdamW
from mindspore.communication import init

from mindspeed_ms.training.training import TrainOneStepCell, train, get_model
from mindspeed_ms.core.optimizer import DistributedOptimizer
from mindspeed_ms.training.arguments import parse_args
from mindspeed_ms.core.optimizer import optimizer_config_from_args
from mindspeed_ms.training.yaml_arguments import core_transformer_config_from_yaml
from mindspeed_ms.core.parallel_state import (
    initialize_model_parallel,
    get_data_parallel_group,
    get_pipeline_model_parallel_group,
    get_pipeline_model_parallel_rank,
    get_pipeline_model_parallel_world_size
)

from test_pipeline_net import PipelineTestNet, FakeData
from tests.st.test_distri_core.utils import get_layer_str_param

ms.set_seed(2024)

def generate_ckpt(vocab_size,
                  seq_length,
                  model_config,
                  standalone_embedding_stage,
                  share_weight):
    """ get ckpt dict """
    hidden_size = model_config.hidden_size
    num_layers = model_config.num_layers
    ckpt = {}
    layer_str_dict = get_layer_str_param(model_config, standalone_embedding_stage)
    print(f"layer_str_dict = {layer_str_dict}")
    embedding_param = ms.Parameter(ms.Tensor(np.random.random((vocab_size, hidden_size)),
                                             ms.float32),
                                   name='embedding.weight')
    ckpt['embedding.weight'] = embedding_param
    for i in range(num_layers):
        idx = i
        if model_config.noop_layers:
            if idx in model_config.noop_layers:
                continue
        layer_str = layer_str_dict[idx]
        # first
        param_name = f'fake_transformer.fake_transformer_layers.{layer_str}.first_liner.weight'
        ckpt[param_name] = ms.Parameter(ms.Tensor(np.random.random((seq_length, hidden_size)), ms.float32),
                                        name=param_name)
        # second
        param_name = f'fake_transformer.fake_transformer_layers.{layer_str}.second_liner.weight'
        ckpt[param_name] = ms.Parameter(ms.Tensor(np.random.random((hidden_size, seq_length)), ms.float32),
                                        name=param_name)
    if not share_weight:
        ckpt['fake_head.weight'] = ms.Parameter(ms.Tensor(np.random.random((vocab_size, hidden_size)),
                                                          ms.float32),
                                                name='fake_head.weight')
    elif get_pipeline_model_parallel_world_size() > 1:
        ckpt['fake_head.weight'] = embedding_param
    ckpt['final_norm.beta'] = ms.Parameter(ms.Tensor(np.zeros((hidden_size,)),
                                                     ms.float32),
                                           name='final_norm.beta')
    ckpt['final_norm.gamma'] = ms.Parameter(ms.Tensor(np.ones((hidden_size,)),
                                                      ms.float32),
                                            name='final_norm.gamma')
    return ckpt


def run_pipeline(model_config, train_args, dynamic_dataset=False):
    """main function."""
    ms.set_context(device_target="Ascend", mode=ms.PYNATIVE_MODE, deterministic='ON')
    init()

    # init context
    pp = model_config.pipeline_model_parallel_size
    if model_config.virtual_pipeline_model_parallel_size is not None and \
       model_config.virtual_pipeline_model_parallel_size > 1:
        vpp = model_config.virtual_pipeline_model_parallel_size
    else:
        vpp = None

    initialize_model_parallel(pipeline_model_parallel_size=pp,
                              virtual_pipeline_model_parallel_size=vpp)
    print("pp stage num: {}".format(pp), flush=True)
    print("vpp size: {}".format(vpp), flush=True)
    print("dp group {} | pp group {}".format(get_data_parallel_group(), \
                                             get_pipeline_model_parallel_group()), flush=True)
    print("current pp rank {}".format(get_pipeline_model_parallel_rank()), flush=True)

    # get ckpt
    ckpt_dict = generate_ckpt(train_args.vocab_size,
                              train_args.seq_length,
                              model_config,
                              train_args.standalone_embedding_stage,
                              not train_args.untie_embeddings_and_output_weights)
    # generate dataset
    dataset = FakeData(data_num=32, seq_length=train_args.seq_length, dynamic_dataset=dynamic_dataset)
    dataset_parallel = ds.GeneratorDataset(dataset, column_names=['input_ids', 'labels'], shuffle=False)
    # calculate global batch size
    global_batch_size = train_args.global_batch_size * train_args.micro_batch_size
    dataset_parallel = dataset_parallel.batch(global_batch_size)
    print("global batch size: ", global_batch_size, flush=True)

    # init model
    def model_provider_func(pre_process=True, post_process=True):
        """ model provider func """
        network = PipelineTestNet(model_config, pre_process=pre_process, post_process=post_process)
        # load ckpt
        ms.load_param_into_net(network, ckpt_dict)
        return network
    network = get_model(model_provider_func, model_config)

    if train_args.wrap_with_ddp and train_args.use_distributed_optimizer:
        optimizer_config = optimizer_config_from_args(train_args)
        optimizer = AdamW(params=network.trainable_params(), lr=0.001)
        per_model_buffers = {}
        for model_idx, model_chunk in enumerate(network):
            per_model_buffers[model_idx] = model_chunk.buffers
        optimizer = DistributedOptimizer(optimizer=optimizer,
                                         config=optimizer_config,
                                         grad_scaler=None,
                                         init_state_fn=None,
                                         per_model_buffers=per_model_buffers,
                                         data_parallel_group=get_data_parallel_group(with_context_parallel=True))
    else:
        optimizer = Adam(params=network.trainable_params(), learning_rate=0.001, beta1=0.9, beta2=0.95)

    # init train one step cell
    train_one_step_cell = TrainOneStepCell(network, optimizer, None, model_config)

    print(f"network trainable params: {network.trainable_params()}", flush=True)

    train(train_one_step_cell, dataset_parallel)


def run_standalone(model_config, train_args):
    """main function."""
    ms.set_context(device_target="Ascend", mode=ms.PYNATIVE_MODE, deterministic='ON')
    init()

    initialize_model_parallel()

    # get ckpt
    ckpt_dict = generate_ckpt(train_args.vocab_size,
                              train_args.seq_length,
                              model_config,
                              train_args.standalone_embedding_stage,
                              not train_args.untie_embeddings_and_output_weights)

    # generate dataset
    dataset = FakeData(data_num=32, seq_length=train_args.seq_length)
    dataset_parallel = ds.GeneratorDataset(dataset, column_names=['input_ids', 'labels'], shuffle=False)
    # calculate global batch size
    global_batch_size = train_args.global_batch_size * train_args.micro_batch_size
    dataset_parallel = dataset_parallel.batch(global_batch_size)
    print("global batch size: ", global_batch_size, flush=True)

    # init model
    def model_provider_func(pre_process=True, post_process=True):
        """ model provider func """
        network = PipelineTestNet(model_config, pre_process=pre_process, post_process=post_process)
        # load ckpt
        ms.load_param_into_net(network, ckpt_dict)
        return network
    network = get_model(model_provider_func, model_config)

    if train_args.wrap_with_ddp and train_args.use_distributed_optimizer:
        optimizer_config = optimizer_config_from_args(train_args)
        optimizer = AdamW(params=network.trainable_params(), lr=0.001)
        per_model_buffers = {}
        for model_idx, model_chunk in enumerate(network):
            per_model_buffers[model_idx] = model_chunk.buffers
        optimizer = DistributedOptimizer(optimizer=optimizer,
                                         config=optimizer_config,
                                         grad_scaler=None,
                                         init_state_fn=None,
                                         per_model_buffers=per_model_buffers,
                                         data_parallel_group=get_data_parallel_group(with_context_parallel=True))
    else:
        optimizer = Adam(params=network.trainable_params(), learning_rate=0.001, beta1=0.9, beta2=0.95)

    # init train one step cell
    train_one_step_cell = TrainOneStepCell(network, optimizer, None, model_config)

    print(f"network trainable params: {network.trainable_params()}", flush=True)

    train(train_one_step_cell, dataset_parallel)


def extra_args_provider(inner_parser):
    inner_parser.add_argument('--run_mode', type=str, default='pp',
                              help="pp: run pp process standalone: run standalone process")
    return inner_parser


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_mode', type=str, default='pp',
                        help="pp: run pp process standalone: run standalone process")
    parser.add_argument('--yaml-cfg', type=str, default=None,
                        help="yaml file path")
    extra_args = parser.parse_args()
    args = parse_args(extra_args_provider=extra_args_provider)

    config = core_transformer_config_from_yaml(args)

    if extra_args.run_mode == 'standalone_without_share':
        run_standalone(config, args)
    elif extra_args.run_mode == 'standalone_with_share':
        args.untie_embeddings_and_output_weights = False
        run_standalone(config, args)
    elif extra_args.run_mode == 'standalone_with_ddp':
        args.untie_embeddings_and_output_weights = False
        args.wrap_with_ddp = True
        args.use_distributed_optimizer = True
        args.bucket_size = 10
        run_standalone(config, args)
    elif extra_args.run_mode == 'pp_without_share':
        run_pipeline(config, args)
    elif extra_args.run_mode == 'pp_with_share':
        args.untie_embeddings_and_output_weights = False
        run_pipeline(config, args)
    elif extra_args.run_mode == 'custom_pp_with_share':
        args.untie_embeddings_and_output_weights = False
        config.num_layer_list = [2, 3, 1, 2]
        run_pipeline(config, args)
    elif extra_args.run_mode == 'pp_interleaved':
        config.virtual_pipeline_model_parallel_size = 2
        args.untie_embeddings_and_output_weights = False
        run_pipeline(config, args)
    elif extra_args.run_mode == 'pp_with_ddp':
        args.untie_embeddings_and_output_weights = False
        args.wrap_with_ddp = True
        args.use_distributed_optimizer = True
        args.bucket_size = 10
        run_pipeline(config, args)
    elif extra_args.run_mode == 'custom_pp_interleaved':
        config.virtual_pipeline_model_parallel_size = 2
        args.untie_embeddings_and_output_weights = False
        config.num_layer_list = [[1, 1], [1, 1], [1, 1], [1, 1]]
        run_pipeline(config, args)
    elif extra_args.run_mode == 'pp_with_standalone_embedding_stage':
        args.untie_embeddings_and_output_weights = False
        args.standalone_embedding_stage = True
        config.pipeline_model_parallel_size = 2
        config.virtual_pipeline_model_parallel_size = 2
        run_pipeline(config, args)
    elif extra_args.run_mode == 'pp_with_noop_layers':
        args.untie_embeddings_and_output_weights = False
        config.virtual_pipeline_model_parallel_size = 2
        config.pipeline_model_parallel_size = 2
        config.num_layers += 2
        config.noop_layers = [4, 5]
        print(f"config.noop_layers = {config.noop_layers}")
        run_pipeline(config, args)
    elif extra_args.run_mode == 'pp_interleaved_with_variable_seq_length':
        config.virtual_pipeline_model_parallel_size = 2
        args.untie_embeddings_and_output_weights = False
        config.variable_seq_lengths = True
        run_pipeline(config, args)
    elif extra_args.run_mode == 'pp_interleaved_with_variable_seq_length_dynamic_data':
        config.virtual_pipeline_model_parallel_size = 2
        args.untie_embeddings_and_output_weights = False
        config.variable_seq_lengths = True
        run_pipeline(config, args, dynamic_dataset=True)
