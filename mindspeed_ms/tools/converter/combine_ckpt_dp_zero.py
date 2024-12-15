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
'''combine ms bucket data tool'''
import os
import json
import shutil
import datetime
import argparse
from collections import OrderedDict
from multiprocessing import Process

import numpy as np
import mindspore as ms
from mindspore import _checkparam as validator


def get_last_checkpoint(ckpt_path, format_="ckpt"):
    """Get last timestamp checkpoint under ckpt_path."""
    ckpt_list = [
        checkpoint for checkpoint in os.listdir(ckpt_path)
        if checkpoint.endswith(f'.{format_}')
    ]
    if not ckpt_list:
        return None
    ckpt_list = sorted(ckpt_list, key=lambda x: os.path.getmtime(os.path.join(ckpt_path, x)))
    return os.path.join(ckpt_path, ckpt_list[-1])


def combine_bucket_data(param_total_dict, prefix, buffer_id, bucket_id, no_save_optim):
    '''combine bucket data'''
    shard_name = "buffer_{}_bucket_{}".format(buffer_id, bucket_id)
    param_name = prefix + '.' + shard_name if prefix else shard_name
    bucket_data_list = []
    # collect this bucket data from all dp rank
    for rank, state_dict in param_total_dict.items():
        if not param_name in state_dict:
            raise KeyError("There is no data found for buffer {} bucket {} in rank_{}'s state_dict."
                           .format(buffer_id, bucket_id, rank))
        bucket_data_list.append(state_dict[param_name].asnumpy())
        state_dict.pop(param_name)
        if no_save_optim:
            # filter optimizer parameters
            state_dict.pop('exp_avg.' + shard_name, None)
            state_dict.pop('exp_avg_sq.' + shard_name, None)
    # concatenate sharded bucket data
    bucket_data = np.concatenate(bucket_data_list)
    return bucket_data


def combine_zero3_data(param_total_dict, param_name, no_save_optim):
    '''combine bucket data'''
    # collect this bucket data from all dp rank
    param_data_list = []
    for rank, state_dict in param_total_dict.items():
        if not param_name in state_dict:
            raise KeyError("There is no data found for param {} in rank_{}'s state_dict."
                           .format(param_name, rank))
        param_data_list.append(state_dict[param_name].asnumpy())
        state_dict.pop(param_name)
        if no_save_optim:
            state_dict.pop('exp_avg.' + param_name, None)
            state_dict.pop('exp_avg_sq.' + param_name, None)
    # concatenate sharded data
    param_data = np.concatenate(param_data_list)
    return param_data

def check_key(key):
    key_list = ['learning_rate', 'weight_decay', 'epoch', 'step', 'default_generator']
    if any(x in key for x in key_list):
        return True
    return False

def get_parameter_state_dp_zero(strategy, param_total_dict, no_save_optim):
    '''get parameter state on dp_zero'''
    # extract buffer info
    buffer_info = strategy['buffer_info']
    buffer_num = len(buffer_info)
    if no_save_optim:
        # only save model parameters
        param_buffer_data = {"": {}}
    else:
        param_buffer_data = {"": {}, "exp_avg": {}, "exp_avg_sq": {}}
    # assemble parameter and states data from all dp rank
    for buffer_id in range(buffer_num):
        buffer_id = str(buffer_id)
        buffer_size = buffer_info[buffer_id]['buffer_size']
        bucket_num = buffer_info[buffer_id]['bucket_num']
        # create buffer for params and states
        for key in param_buffer_data:
            param_buffer_data[key][buffer_id] = np.zeros(shape=(buffer_size), dtype=np.float32)
        offset = 0
        for bucket_id in range(bucket_num):
            bucket_size = 0
            for key in param_buffer_data:
                bucket_data = combine_bucket_data(
                    param_total_dict,
                    prefix=key,
                    buffer_id=buffer_id,
                    bucket_id=bucket_id,
                    no_save_optim=no_save_optim
                )
                bucket_size = bucket_data.shape[0]
                # copy bucket data into buffer
                param_buffer_data[key][buffer_id][offset:offset+bucket_size] = bucket_data
            offset += bucket_size
    # reassemble parameters
    new_state_dict = {}
    param_info = strategy['param_info']
    for param_name in param_info:
        buffer_idx, _, start_idx, end_idx = param_info[param_name]['range_map']
        buffer_idx = str(buffer_idx)
        param_shape = param_info[param_name]['shape']
        for key, data in param_buffer_data.items():
            param = ms.Parameter(
                data[buffer_idx][start_idx:end_idx].reshape(param_shape),
                name=key+'.'+param_name if key else param_name,
                requires_grad=True,
            )
            new_state_dict[param.name] = param

    # assemble zero3 params
    if 'zero3_params' in strategy:
        zero3_params = strategy['zero3_params']
        for param_name in zero3_params:
            if no_save_optim and (param_name.startswith('exp_avg.') or param_name.startswith('exp_avg_sq.')):
                continue
            param_data = combine_zero3_data(param_total_dict, param_name, no_save_optim)
            param = ms.Parameter(
                param_data,
                name=param_name,
                requires_grad=True,
            )
            new_state_dict[param.name] = param

    # add remain non-parameter-related parameters such as rng, epoch_num, step_num.
    for _, state_dict in param_total_dict.items():
        for key, value in state_dict.items():
            if key in new_state_dict.keys():
                continue
            if no_save_optim and check_key(key):
                continue
            new_state_dict[key] = value
    return new_state_dict


def transform_ckpt_dp_zero_by_rank(
        shard_info_file,
        rank_id,
        checkpoints_dir,
        output_dir,
        src_format="ckpt",
        dst_format="ckpt",
        copy_to_all_dp_ranks=True,
        no_save_optim=False,
    ):
    '''transform dp0 ckpt'''
    with open(shard_info_file, 'r') as f:
        strategy = json.load(f)
    # extract dp rank list
    dp_rank_list = strategy['dp_rank_list']
    dp_rank_list.sort()
    if not rank_id in dp_rank_list:
        raise ValueError("rank_id {} not in the dp_rank_list from the strategy file under its directory. "
                         "Please check dist_opt_shard_info file.")

    print("[{}][Rank {}] Start collecting checkpoints from all dp rank.".format(
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), rank_id))
    param_total_dict = OrderedDict()
    for rank in dp_rank_list:
        checkpoint_file = get_last_checkpoint(os.path.join(checkpoints_dir, "rank_{}".format(rank)), format_=src_format)
        state_dict = ms.load_checkpoint(checkpoint_file, format=src_format)
        param_total_dict[rank] = state_dict
    print("[{}][Rank {}] End collecting checkpoints from all dp rank.".format(
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), rank_id))

    # combine data from all dp rank
    param_state_dict = get_parameter_state_dp_zero(strategy, param_total_dict, no_save_optim)

    # save combined checkpoint
    this_rank_ckpt_file = get_last_checkpoint(
        os.path.join(checkpoints_dir, "rank_{}".format(rank_id)),
        format_=src_format,
    ).split('/')[-1]
    this_rank_ckpt_file = this_rank_ckpt_file.replace(
        '.{}'.format(src_format),
        '_dp_merged.{}'.format(dst_format),
    )
    save_dir = os.path.join(output_dir, "rank_{}".format(rank_id))
    os.makedirs(save_dir, exist_ok=True)
    save_file_path = os.path.join(
        save_dir,
        this_rank_ckpt_file
    )
    ms.save_checkpoint(param_state_dict, save_file_path, format=dst_format)
    print("[{}][Rank {}] Save dp combined checkpoint under: {}".format(
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), rank_id, save_file_path))

    if copy_to_all_dp_ranks:
        # copy generated checkpoint file to each dp rank's checkpoint directory
        for rank in dp_rank_list:
            # skip current rank
            if rank == rank_id:
                continue
            rank_ckpt_dir = os.path.join(output_dir, "rank_{}".format(rank))
            os.makedirs(rank_ckpt_dir, exist_ok=True)
            file_name = this_rank_ckpt_file.replace('rank_{}'.format(rank_id), 'rank_{}'.format(rank))
            print("[{}][Rank {}] Copy merged checkpoint to dir {}".format(
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), rank_id, rank_ckpt_dir))
            shutil.copy(save_file_path, os.path.join(rank_ckpt_dir, file_name))


def transform_ckpt_dp_zero(
        opt_shard_info_dir,
        checkpoints_dir,
        output_dir,
        src_format='ckpt',
        dst_format='ckpt',
        copy_to_all_dp_ranks=True,
        max_proccess_limit=8,
        rank_section=None,
        no_save_optim=False,
    ):
    '''transform dp0 ckpt'''
    if rank_section is None:
        rank_section = [None, None]
    validator.check_string(src_format, ["ckpt", "safetensors"], "src_format")
    validator.check_string(dst_format, ["ckpt", "safetensors"], "dst_format")
    if not isinstance(rank_section, (list, tuple)) or len(rank_section) != 2:
        raise ValueError("Invalid rank_section, should be list or tuple with size 2.")
    start_rank, end_rank = rank_section[0], rank_section[1]
    start_passed_ = not start_rank is None
    end_passed_ = not end_rank is None
    if not all([start_passed_, end_passed_]) and any([start_passed_, end_passed_]):
        raise ValueError("`start_rank` and `end_rank` should be set at the same time, but got {}".format(rank_section))
    validator.check_value_type("start_rank", start_rank, [int, type(None)])
    validator.check_value_type("end_rank", end_rank, [int, type(None)])
    if isinstance(start_rank, int) and isinstance(end_rank, int) and start_rank >= end_rank:
        raise ValueError("`start_rank` should less than `end_rank`, but got {} and {}".format(start_rank, end_rank))

    # get all dirs under opt_shard_info
    shard_info_dirs = [
        os.path.join(opt_shard_info_dir, f) for f in os.listdir(opt_shard_info_dir) \
        if os.path.isdir(os.path.join(opt_shard_info_dir, f))
    ]

    # filter dirs, keep dirs with dist_opt_shard_info json file exist
    # and rank_id is in section [start_rank:end_rank]
    # these files will work as unit for multiprocessing transformation
    def _shard_info_dir_filter(dir_path):
        nonlocal start_passed_, end_passed_
        rank_id = int(dir_path.split('/')[-1].split('_')[-1])
        shard_info_file_exist_ = os.path.exists(
            os.path.join(dir_path, "dist_opt_shard_info_rank_{}-0_0.json".format(rank_id)))
        in_section_ = True
        if start_passed_ and end_passed_:
            in_section_ = start_rank <= rank_id < end_rank

        return shard_info_file_exist_ and in_section_

    shard_info_dirs = filter(_shard_info_dir_filter, shard_info_dirs)

    processes = []
    activate_processes = 0
    # multiprocessing transform checkpoints
    for dir_path in shard_info_dirs:
        rank_id = int(dir_path.split('/')[-1].split('_')[-1])
        shard_info_file = os.path.join(dir_path, "dist_opt_shard_info_rank_{}-0_0.json".format(rank_id))
        p = Process(
            target=transform_ckpt_dp_zero_by_rank,
            args=(
                shard_info_file,
                rank_id,
                checkpoints_dir,
                output_dir,
                src_format,
                dst_format,
                copy_to_all_dp_ranks,
                no_save_optim,
            )
        )
        p.start()
        processes.append(p)
        activate_processes += 1
        # if activated processes num exceeds max_proccess_limit
        # wait the earliest process in queue finish
        if activate_processes >= max_proccess_limit:
            p = processes.pop(0)
            p.join()
            activate_processes -= 1
    # wait all process finish
    for p in processes:
        p.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--strategy-dir',
        default='./output/opt_shard_info',
        help='Optimizer shard strategy save directory path.'
    )
    parser.add_argument(
        '--checkpoint-dir',
        default='./output',
        help='Checkpoint save directory path.'
    )
    parser.add_argument(
        '--output-dir',
        default='./output',
        help='Save path for dp merged checkpoints.'
    )
    parser.add_argument(
        '--src-format',
        type=str,
        default='ckpt',
        help='Original checkpoint format, support [`ckpt`, `safetensors`]. Default: `ckpt`.'
    )
    parser.add_argument(
        '--dst-format',
        type=str,
        default='ckpt',
        help='Output checkpoint format, support [`ckpt`, `safetensors`]. Default: `ckpt`.'
    )
    parser.add_argument(
        '--copy-to-all-dp-ranks',
        action='store_true',
        help='Copy combined checkpoint to all rank directories in this dp group.'
    )
    parser.add_argument(
        '--max-proccess-limit',
        type=int,
        default=8,
        help='Maximum number of process used to transform checkpoints.'
    )
    parser.add_argument(
        '--rank-section',
        nargs='+',
        type=int,
        default=[None, None],
        help='Specify the rank section will be processed.'
    )
    parser.add_argument(
        '--no-save-optim',
        action='store_true',
        help='Filter optimizer parameters'
    )

    args = parser.parse_args()
    transform_ckpt_dp_zero(
        args.strategy_dir,
        args.checkpoint_dir,
        args.output_dir,
        args.src_format,
        args.dst_format,
        args.copy_to_all_dp_ranks,
        args.max_proccess_limit,
        args.rank_section,
        args.no_save_optim,
    )
