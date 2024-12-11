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
""" Test Resume_Training. """
import argparse
import os
from functools import partial
import numpy as np

import mindspore as ms
import mindspore.dataset as ds
from mindspore import Tensor
from mindspore.communication.management import init
from mindspore.communication import get_rank
from mindspore.mint.optim import AdamW
from mindspore import mint

from mindspeed_ms.training import parse_args, core_transformer_config_from_yaml
from mindspeed_ms.training.arguments import _print_args
from mindspeed_ms.training.yaml_arguments import validate_yaml
from mindspeed_ms.training.global_vars import set_global_variables
from mindspeed_ms.training.utils import average_losses_across_data_parallel_group
from mindspeed_ms.core.tensor_parallel import ReduceFromContextParallelRegion
from mindspeed_ms.core import parallel_state
from mindspeed_ms.training import get_model, TrainOneStepCell, train, get_loss_func, get_args
from mindspeed_ms.core.optimizer import get_optimizer_param_scheduler, optimizer_config_from_args
from mindspeed_ms.core.dist_checkpointing import load_checkpoint
from mindspeed_ms.tools.resume_ckpt import get_resume_checkpoint
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


def run_resume_training(config, args):
    """ Test ParallelTransformer. """
    print(f"config is:\n{config}")
    _print_args("final args", args)
    tp = args.tensor_model_parallel_size
    ep = args.expert_model_parallel_size
    pp = args.pipeline_model_parallel_size
    vocab_size = args.vocab_size
    seq_length = args.seq_length
    if args.enable_compile_cache:
        print(f"compile_cache will be save to: {args.compile_cache_path}")
    ms.set_context(device_target="Ascend", mode=ms.PYNATIVE_MODE, deterministic='ON', pynative_synchronize=True,
                   enable_compile_cache=args.enable_compile_cache, compile_cache_path=args.compile_cache_path)
    init()
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=tp, expert_model_parallel_size=ep,
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
    optimizer = AdamW(params=network.trainable_params(), lr=1e-4)
    opt_param_scheduler = get_optimizer_param_scheduler(optimizer)
    # load golden ckpt
    resume_dict = None
    if args.resume_training is True and os.path.exists(args.load):
        rank_path = os.path.join(args.load, f"rank_{get_rank()}")
        meta_path = os.path.join(rank_path, "meta.json")
        if not os.path.exists(meta_path):
            print(f"Could not find meta.json in directory {rank_path}, using latest ckpt in {rank_path}")
        resume_ckpt_name = get_resume_checkpoint(
            checkpoint_dir=args.load,
            resume_training=args.resume_training,
            resume_by_meta=True)
        print(f"resume_ckpt_name is {resume_ckpt_name}")
        assert resume_ckpt_name == f"network_rank_{get_rank()}-{args.latest_epoch_step}.ckpt", \
            f"expect newest ckpt is network_rank_{get_rank()}-{args.latest_epoch_step}.ckpt, but got {resume_ckpt_name}"
        if resume_ckpt_name is True:
            ckpt_path = args.load
        elif isinstance(resume_ckpt_name, str):
            # hard code to load step 5 ckpt, resume train another 5 steps
            resume_ckpt_name = f"network_rank_{get_rank()}-{args.load_epoch_step}.ckpt"
            ckpt_path = os.path.join(rank_path, resume_ckpt_name)
        print(f"ckpt_path is {ckpt_path}")
        resume_dict = load_checkpoint(
            config=config,
            model=network,
            optimizer=optimizer if not args.no_load_optim else None,
            opt_param_scheduler=opt_param_scheduler,
            ckpt_path=ckpt_path,
            format=args.dist_ckpt_format
            )

    train_one_step_cell = TrainOneStepCell(network, optimizer, opt_param_scheduler, config)
    # train
    train(train_one_step_cell, dataset, forward_step, resume_dict=resume_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml-cfg', default=None, type=str)
    parser.add_argument('--resume_training', action='store_true', help="resume training")
    parser.add_argument('--enable_compile_cache', action='store_true', help="enable compile cache")
    parser.add_argument('--compile_cache_path', type=str, default=None, help="where to save/load compile_cache")
    parser.add_argument('--crc_check', action='store_true', help="crc check")
    parser.add_argument('--output_dir', type=str, default="./output", help="dir to put log、ckpt and complie cache")
    parser.add_argument('--load_checkpoint', type=str, default="", help="where to load ckpt")
    parser.add_argument('--training_iters', type=int, default=10, help="training_iters")
    parser.add_argument('--save_interval', type=int, default=None, help="training_iters")
    parser.add_argument('--no_load_optim', action='store_true', help="load optim or not")
    parser.add_argument('--no_load_rng', action='store_true', help="load rng or not")
    parser.add_argument('--new_dataset', action='store_true', help="use new dataset or not")
    parser.add_argument('--learning_rate', type=float, default=0.0009, help="learning_rate")
    parser.add_argument('--override_opt_param_scheduler', action='store_true', help="use config scheduler")
    parser.add_argument('--tp', type=int, default=2, help="tp size")
    parser.add_argument('--pp', type=int, default=2, help="pp size")
    parser.add_argument('--gbs', type=int, default=1, help="global batch size")
    parser.add_argument('--epochs', type=int, default=1, help="epochs")
    parser.add_argument('--latest_epoch_step', type=str, default='0_10', help="saved latest epoch and step")
    parser.add_argument('--load_epoch_step', type=str, default='0_5', help="specified a epoch and step to load")
    cli_args = parser.parse_args()

    def extra_parser_provider(inner_parser):
        """ Get extra args"""
        inner_parser.add_argument('--resume_training', action='store_true', help="resume training")
        inner_parser.add_argument('--enable_compile_cache', action='store_true', help="enable compile cache")
        inner_parser.add_argument('--compile_cache_path', type=str, default=None,
                                  help="where to save/load compile_cache")
        inner_parser.add_argument('--crc_check', action='store_true', help="crc check")
        inner_parser.add_argument('--output_dir', type=str, default="./output",
                                  help="dir to put log、ckpt and complie cache")
        inner_parser.add_argument('--load_checkpoint', type=str, default="", help="where to load ckpt")
        inner_parser.add_argument('--training_iters', type=int, default=10, help="training_iters")
        inner_parser.add_argument('--save_interval', type=int, default=None, help="training_iters")
        inner_parser.add_argument('--no_load_optim', action='store_true', help="load optim or not")
        inner_parser.add_argument('--no_load_rng', action='store_true', help="load rng or not")
        inner_parser.add_argument('--new_dataset', action='store_true', help="use new dataset or not")
        inner_parser.add_argument('--learning_rate', type=float, default=0.0009, help="learning_rate")
        inner_parser.add_argument('--override_opt_param_scheduler', action='store_true', help="use config scheduler")
        inner_parser.add_argument('--tp', type=int, default=2, help="tp size")
        inner_parser.add_argument('--pp', type=int, default=2, help="pp size")
        inner_parser.add_argument('--gbs', type=int, default=1, help="global batch size")
        inner_parser.add_argument('--latest_epoch_step', type=str, default='0_10', help="saved latest epoch and step")
        inner_parser.add_argument('--load_epoch_step', type=str, default='0_5',
                                  help="specified a epoch and step to load")
        return inner_parser

    main_args, defaults = parse_args(extra_args_provider=extra_parser_provider)
    main_args = validate_yaml(main_args, defaults, {})
    set_global_variables(main_args, False)
    all_config = core_transformer_config_from_yaml(main_args)
    all_config.tensor_model_parallel_size = cli_args.tp
    all_config.pipeline_model_parallel_size = cli_args.pp

    main_args.latest_epoch_step = cli_args.latest_epoch_step
    main_args.load_epoch_step = cli_args.load_epoch_step
    main_args.tensor_model_parallel_size = cli_args.tp
    main_args.pipeline_model_parallel_size = cli_args.pp
    main_args.global_batch_size = cli_args.gbs
    main_args.epochs = cli_args.epochs
    main_args.resume_training = cli_args.resume_training
    main_args.enable_compile_cache = cli_args.enable_compile_cache
    if cli_args.compile_cache_path is None:
        main_args.compile_cache_path = os.path.join(cli_args.output_dir, "compile_cache")
    else:
        main_args.compile_cache_path = cli_args.compile_cache_path
    main_args.crc_check = cli_args.crc_check
    main_args.save = cli_args.output_dir
    main_args.profile_save_path = os.path.join(cli_args.output_dir, "profile")
    main_args.load = cli_args.load_checkpoint
    main_args.train_iters = cli_args.training_iters
    main_args.save_interval = cli_args.save_interval
    main_args.no_load_optim = cli_args.no_load_optim
    main_args.no_load_rng = cli_args.no_load_rng
    main_args.new_dataset = cli_args.new_dataset
    main_args.lr = cli_args.learning_rate
    main_args.override_opt_param_scheduler = cli_args.override_opt_param_scheduler
    if cli_args.override_opt_param_scheduler:
        main_args.use_checkpoint_opt_param_scheduler = False
    run_resume_training(all_config, main_args)
