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
import os
import argparse
from functools import partial
import numpy as np
import mindspore as ms
from mindspore import mint
import mindspore.dataset as ds
from mindspore.nn import Adam
from mindspore.mint.optim import AdamW
from mindspore.communication import init

from mindspeed_ms.core.tensor_parallel.mappings import ReduceFromContextParallelRegion
from mindspeed_ms.training import train, get_model, get_args, evaluate_and_print_results
from mindspeed_ms.training.utils import average_losses_across_data_parallel_group
from mindspeed_ms.core.optimizer import DistributedOptimizer
from mindspeed_ms.training.arguments import parse_args
from mindspeed_ms.core.optimizer import optimizer_config_from_args
from mindspeed_ms.training.global_vars import set_global_variables
from mindspeed_ms.training.yaml_arguments import core_transformer_config_from_yaml, validate_yaml
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
    input_data, labels = get_batch(data_iterator)
    loss_mask = mint.ones_like(labels)
    input_tensor = (input_data, labels)

    # pylint: disable=W0621
    def core_forward_func(*args):
        input_data, labels = args
        output_tensor = model(input_data, labels)
        return output_tensor

    return input_tensor, core_forward_func, partial(loss_func, loss_mask)


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
    dataset_parallel = dataset_parallel.batch(train_args.micro_batch_size)
    if vpp is not None and vpp > 1:
        dataset_parallel = [dataset_parallel] * vpp

    # init model
    def model_provider_func(pre_process=True, post_process=True):
        """ model provider func """
        network = PipelineTestNet(model_config, pre_process=pre_process, post_process=post_process)
        # load ckpt
        ms.load_param_into_net(network, ckpt_dict)
        return network
    network = get_model(model_provider_func, model_config, wrap_with_ddp=args.wrap_with_ddp)

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

    print(f"network trainable params: {network.trainable_params()}", flush=True)

    prefix = f'pipeline before train'
    evaluate_and_print_results(prefix, forward_step, dataset_parallel, network, 0,
                               None, model_config, verbose=True, write_to_tensorboard=False)

    train(forward_step, network, optimizer, None, dataset_parallel, None, None, model_config)

    prefix = f'pipeline after train'
    evaluate_and_print_results(prefix, forward_step, dataset_parallel, network, args.train_iters,
                               None, model_config, verbose=True, write_to_tensorboard=False)


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
    dataset_parallel = dataset_parallel.batch(train_args.micro_batch_size)

    # init model
    def model_provider_func(pre_process=True, post_process=True):
        """ model provider func """
        network = PipelineTestNet(model_config, pre_process=pre_process, post_process=post_process)
        # load ckpt
        ms.load_param_into_net(network, ckpt_dict)
        return network
    network = get_model(model_provider_func, model_config, wrap_with_ddp=args.wrap_with_ddp)

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

    print(f"network trainable params: {network.trainable_params()}", flush=True)

    prefix = f'standalone before train'
    evaluate_and_print_results(prefix, forward_step, dataset_parallel, network, 0,
                               None, model_config, verbose=True, write_to_tensorboard=False)

    train(forward_step, network, optimizer, None, dataset_parallel, None, None, model_config)

    prefix = f'standalone after train'
    evaluate_and_print_results(prefix, forward_step, dataset_parallel, network, args.train_iters,
                               None, model_config, verbose=True, write_to_tensorboard=False)


def extra_args_provider(inner_parser):
    """ extra args provider """
    inner_parser.add_argument('--run_mode', type=str, default='pp',
                              help="pp: run pp process standalone: run standalone process")
    return inner_parser


# pylint: disable=W0621
def set_vars(var_name, var_value, args, config):
    """ set var for args and config """
    setattr(args, var_name, var_value)
    setattr(config, var_name, var_value)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_mode', type=str, default='pp',
                        help="pp: run pp process standalone: run standalone process")
    parser.add_argument('--yaml-cfg', type=str, default=None,
                        help="yaml file path")
    extra_args = parser.parse_args()
    args, defaults = parse_args(extra_args_provider=extra_args_provider)
    args = validate_yaml(args, defaults, {})
    set_global_variables(args, False)
    args.data_layout = "BSH"
    args.wrap_with_ddp = False

    config = core_transformer_config_from_yaml(args)

    if extra_args.run_mode == 'standalone_without_share':
        run_standalone(config, args)
    elif extra_args.run_mode == 'standalone_with_share':
        set_vars("untie_embeddings_and_output_weights", False, args, config)
        run_standalone(config, args)
    elif extra_args.run_mode == 'standalone_with_ddp':
        set_vars("untie_embeddings_and_output_weights", False, args, config)
        set_vars("wrap_with_ddp", True, args, config)
        set_vars("use_distributed_optimizer", True, args, config)
        set_vars("ddp_bucket_size", 10, args, config)
        run_standalone(config, args)
    elif extra_args.run_mode == 'pp_without_share':
        run_pipeline(config, args)
    elif extra_args.run_mode == 'pp_with_share':
        set_vars("untie_embeddings_and_output_weights", False, args, config)
        run_pipeline(config, args)
    elif extra_args.run_mode == 'custom_pp_with_share':
        set_vars("untie_embeddings_and_output_weights", False, args, config)
        set_vars("num_layer_list", [2, 3, 1, 2], args, config)
        run_pipeline(config, args)
    elif extra_args.run_mode == 'pp_interleaved':
        set_vars("virtual_pipeline_model_parallel_size", 2, args, config)
        set_vars("untie_embeddings_and_output_weights", False, args, config)
        run_pipeline(config, args)
    elif extra_args.run_mode == 'pp_with_ddp':
        set_vars("untie_embeddings_and_output_weights", False, args, config)
        set_vars("wrap_with_ddp", True, args, config)
        set_vars("use_distributed_optimizer", True, args, config)
        set_vars("ddp_bucket_size", 10, args, config)
        run_pipeline(config, args)
    elif extra_args.run_mode == 'custom_pp_interleaved':
        set_vars("virtual_pipeline_model_parallel_size", 2, args, config)
        set_vars("untie_embeddings_and_output_weights", False, args, config)
        set_vars("num_layer_list", [[1, 1], [1, 1], [1, 1], [1, 1]], args, config)
        run_pipeline(config, args)
    elif extra_args.run_mode == 'pp_with_standalone_embedding_stage':
        set_vars("virtual_pipeline_model_parallel_size", 2, args, config)
        set_vars("pipeline_model_parallel_size", 2, args, config)
        set_vars("untie_embeddings_and_output_weights", False, args, config)
        set_vars("standalone_embedding_stage", True, args, config)
        run_pipeline(config, args)
    elif extra_args.run_mode == 'pp_with_noop_layers':
        set_vars("virtual_pipeline_model_parallel_size", 2, args, config)
        set_vars("pipeline_model_parallel_size", 2, args, config)
        set_vars("untie_embeddings_and_output_weights", False, args, config)
        set_vars("num_layers", config.num_layers + 2, args, config)
        set_vars("noop_layers", [4, 5], args, config)
        run_pipeline(config, args)
    elif extra_args.run_mode == 'pp_interleaved_with_variable_seq_length':
        set_vars("virtual_pipeline_model_parallel_size", 2, args, config)
        set_vars("untie_embeddings_and_output_weights", False, args, config)
        set_vars("variable_seq_lengths", True, args, config)
        run_pipeline(config, args)
    elif extra_args.run_mode == 'pp_interleaved_with_variable_seq_length_dynamic_data':
        set_vars("virtual_pipeline_model_parallel_size", 2, args, config)
        set_vars("untie_embeddings_and_output_weights", False, args, config)
        set_vars("variable_seq_lengths", True, args, config)
        run_pipeline(config, args, dynamic_dataset=True)
