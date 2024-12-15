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
"""run parallel dp0 gather"""

import os
import argparse
from functools import partial
import numpy as np

import mindspore as ms
import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.ops as ops
from mindspore import mint
from mindspore import Tensor
from mindspore.communication.management import init
from mindspore.nn import CrossEntropyLoss

from mindspeed_ms.training import train, get_model, get_loss_func, get_args
from mindspeed_ms.training.utils import average_losses_across_data_parallel_group
from mindspeed_ms.training.global_vars import set_global_variables
from mindspeed_ms.training.yaml_arguments import core_transformer_config_from_yaml, validate_yaml
from mindspeed_ms.training.arguments import parse_args
from mindspeed_ms.core import parallel_state
from mindspeed_ms.core.dist_checkpointing import load_checkpoint
from mindspeed_ms.core.optimizer import optimizer_config_from_args, get_optimizer
from mindspeed_ms.core.tensor_parallel.mappings import ReduceFromContextParallelRegion
from mindspeed_ms.legacy.model import ParallelMLP
from mindspeed_ms.legacy.model.module import Module

from tests.st.test_distri_core.utils import MixtralModel

def get_batch(data_iterator):
    """ get micro batch data """
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    return data.values()


def loss_func(loss_mask, output_tensor):
    """ loss reduction function. """
    args = get_args()

    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()

    if args.context_parallel_size > 1:
        loss = mint.cat([mint.sum(losses.view(-1) * loss_mask).view(1), loss_mask.sum().view(1)])
        loss = ReduceFromContextParallelRegion()(loss)
        loss = loss[0] / loss[1]
    else:
        loss = mint.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    print(f"final micro loss: {loss}")

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
    args = get_args()
    input_data, labels, attention_mask = get_batch(data_iterator)
    loss_mask = mint.ne(input_data[:, :-1], args.pad_token).astype(args.compute_dtype)
    input_tensor = (input_data, labels, attention_mask)

    def core_forward_func(*input_args):
        local_input, local_labels, local_attention_mask = input_args
        output_tensor = model(local_input, local_labels, local_attention_mask)
        return output_tensor

    return input_tensor, core_forward_func, partial(loss_func, loss_mask)


class ParallelMLPNet(Module):
    """
    define a pynative MLP net
    """

    def __init__(self, config):
        super(ParallelMLPNet, self).__init__()
        self.mlp0 = ParallelMLP(config=config)
        self.mlp1 = ParallelMLP(config=config)
        self.mlp2 = ParallelMLP(config=config)

        self.loss = CrossEntropyLoss()
        self.cast = ops.Cast()
        self.dtype = config.compute_dtype
        self.config = config

    def construct(self, input_ids, labels):
        """ do construct and calc mean loss """
        input_id = ops.cast(input_ids, mstype.bfloat16)

        output, bias = self.mlp0(input_id)
        if bias is not None:
            output = output + bias
        output, bias = self.mlp1(output)
        if bias is not None:
            output = output + bias
        output, bias = self.mlp2(output)
        if bias is not None:
            output = output + bias

        labels = labels
        loss = output.abs().mean()
        return loss

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
        self.attention_mask = np.tril(np.ones(shape=(1, seq_length-1, seq_length-1))).astype(np.int32)
        self.attention_mask = self.attention_mask < 0.5

    def __getitem__(self, index):
        return (Tensor(self.input_data[index]), Tensor(self.label_data[index]), Tensor(self.attention_mask))

    def __len__(self):
        return self.dataset_size

def run_parallel_dp0_gather(base, config, args):
    """
    run pynative mode in dp0 gather
    """
    tp = args.tensor_model_parallel_size
    ep = args.expert_model_parallel_size
    pp = args.pipeline_model_parallel_size
    vocab_size = args.vocab_size
    seq_length = args.seq_length

    ms.set_context(device_target="Ascend", mode=ms.PYNATIVE_MODE, deterministic='ON', pynative_synchronize=True)
    init()
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=tp, expert_model_parallel_size=ep,
                                             context_parallel_size=args.context_parallel_size,
                                             pipeline_model_parallel_size=pp)
    dp_group = parallel_state.get_data_parallel_group()
    ep_group = parallel_state.get_expert_model_parallel_group()
    tp_group = parallel_state.get_tensor_model_parallel_group()
    pp_group = parallel_state.get_pipeline_model_parallel_group()
    dp_rank = parallel_state.get_data_parallel_rank()
    ep_rank = parallel_state.get_expert_model_parallel_rank()
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    pp_size = parallel_state.get_pipeline_model_parallel_world_size()
    tp_size = parallel_state.get_tensor_model_parallel_world_size()
    dp_size = parallel_state.get_data_parallel_world_size()
    print(f"dp_group is {dp_group}, ep_group is {ep_group}, tp_group is {tp_group}, pp_group is {pp_group}", flush=True)
    print(f"dp_rank is {dp_rank}, ep_rank is {ep_rank}, tp_rank is {tp_rank}, pp_rank is {pp_rank}", flush=True)
    print(f"pp_size is {pp_size}, tp_size is {tp_size}, dp_size is {dp_size}", flush=True)

    ms.set_seed(2024)
    # making dataset
    if args.epochs == 1:
        dataset_size = args.global_batch_size * args.train_iters
    elif args.epochs > 1:
        dataset_size = args.global_batch_size * 5
    input_data = np.random.randint(low=1, high=vocab_size, size=(dataset_size, seq_length+1), dtype=np.int32)
    dataset = TestData(dataset_size=dataset_size, input_data=input_data, label_data=input_data)
    print(f"dataset size is {len(dataset)}, global_batch_size is {args.global_batch_size}, "
          f"micro_batch_size is {args.micro_batch_size}")
    dataset = ds.GeneratorDataset(dataset, column_names=['input_ids', 'labels', "attention_mask"], shuffle=False,
                                  num_shards=dp_size, shard_id=dp_rank)
    dataset = dataset.batch(args.micro_batch_size)
    optimizer_config = optimizer_config_from_args(main_args)

    # build net
    def model_provider_func(pre_process=True, post_process=True):
        """ get mixtral model """
        loss = get_loss_func(optimizer_config)
        network = MixtralModel(config, parallel_output=False, loss_func=loss, pre_process=pre_process,
                               post_process=post_process)
        return network

    network = get_model(model_provider_func, config, args.wrap_with_ddp)

    print(f"network construct is:\n{network}")
    print("network parameters are:")
    for param in network.get_parameters():
        print(f"{param.name} {param.dtype} {param.shape}")

    optimizer = get_optimizer(optimizer_config, config, network.trainable_params(), network)

    # init train one step cell
    print(f"network trainable params: {network.trainable_params()}", flush=True)

    resume_dict = None
    if not base:
        target_dp_rank = 0
        global_rank = pp_rank * dp_size * tp_size + target_dp_rank * tp_size + tp_rank
        load_ckpt_path = args.save + f"/rank_{global_rank}/network_rank_{global_rank}-0_3.ckpt"
        args.save_interval = 100
        args.resume_training = True

        resume_dict = load_checkpoint(config,
                                      network[0],
                                      optimizer=optimizer,
                                      opt_param_scheduler=None,
                                      ckpt_path=load_ckpt_path)

    train(forward_step, network, optimizer, None, dataset, None, None, config, resume_dict=resume_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', default=False, type=bool)
    parser.add_argument('--yaml-cfg', default=None, type=str)
    extra_args = parser.parse_args()

    def extra_parser_provider(inner_parser):
        inner_parser.add_argument('--base', default=False, type=bool)
        return inner_parser

    main_args, defaults = parse_args(extra_args_provider=extra_parser_provider)
    main_args = validate_yaml(main_args, defaults, {})
    set_global_variables(main_args, False)
    all_config = core_transformer_config_from_yaml(main_args)
    run_parallel_dp0_gather(extra_args.base, all_config, main_args)
