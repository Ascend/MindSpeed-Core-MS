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
'''transform ckpt helper'''
import argparse
import os
import datetime
import shutil
import mindspore as ms


def get_global_rank_dir(dp_rank, cp_rank, tp_rank, tp_size, pp_rank, dp_size, cp_size):
    """Get dir of global rank."""
    global_rank = (
        pp_rank * dp_size * cp_size * tp_size +
        dp_rank * cp_size * tp_size +
        cp_rank * tp_size +
        tp_rank
    )
    return f"rank_{global_rank}"


def global_rank_to_mp_rank(src_path, dp_size, tp_size, pp_size, cp_size, iteration=1, output_format='ckpt'):
    """ global_rank_to_mp_rank """
    iteration_path = os.path.join(src_path, "iter_{:07d}".format(iteration))
    try:
        shutil.rmtree(iteration_path)
    except FileNotFoundError as e:
        print(f"skip delete {iteration_path}: {e}")
    for pp in range(pp_size):
        for tp in range(tp_size):
            rank_name = get_global_rank_dir(0, 0, tp, tp_size, pp, dp_size, cp_size)
            new_name = f'mp_rank_{tp:02d}'
            if pp_size > 1:
                new_name += f'_{pp:03d}'
            mp_rank_path = os.path.join(iteration_path, new_name)
            shutil.move(os.path.join(src_path, rank_name), mp_rank_path)
            for file_name in os.listdir(mp_rank_path):
                if file_name.endswith(".{}".format(output_format)):
                    shutil.move(os.path.join(mp_rank_path, file_name),
                                os.path.join(mp_rank_path, "model_optim_rng.{}".format(output_format)))

    for dir_name in os.listdir(src_path):
        print(f"dir_name {dir_name}")
        if dir_name.startswith('rank_'):
            shutil.rmtree(os.path.join(src_path, dir_name))


# local mp to global rank
def mp_rank_to_global_rank(root_path, dp_size, cp_size, iteration, output_dir="temp_rearrange_ckpt",
                           output_format='ckpt'):
    """ mp_rank_to_global_rank """
    src_path = os.path.join(root_path, 'iter_{:07d}'.format(iteration))
    if os.path.exists(os.path.join(root_path, "latest_checkpointed_iteration.txt")):
        shutil.copy(os.path.join(root_path, "latest_checkpointed_iteration.txt"), output_dir)
    shutil.copytree(os.path.join(root_path, "strategy"), os.path.join(output_dir, "strategy"))
    mp_rank_list = [
        dir_name for dir_name in os.listdir(src_path)
        if dir_name.startswith('mp_rank')
    ]
    tp_size = 1
    for dir_name in mp_rank_list:
        tp_size = max(int(dir_name.split("_")[2]) + 1, tp_size)

    dp0_rank_list = set()
    this_rank_ckpt_file = {}
    for root, _, files in os.walk(src_path):
        for file in files:
            match = file.endswith('model_optim_rng.ckpt')
            dp_rank = 0
            cp_rank = 0
            if match:
                dir_name = os.path.basename(root)
                print(dir_name)
                if dir_name.startswith('mp_rank_'):
                    split_list = dir_name.split("_")
                    tp_rank = int(split_list[2])
                    pp_rank = 0
                    if len(split_list) == 4:
                        pp_rank = int(split_list[3])
                    global_rank = (
                        pp_rank * dp_size * cp_size * tp_size +
                        dp_rank * cp_size * tp_size +
                        cp_rank * tp_size +
                        tp_rank
                    )
                    dp0_rank_list.add(global_rank)
                    new_file_name = f'model_optim_rng.{output_format}'
                    src_file_path = os.path.join(root, file)
                    global_rank_path = os.path.join(output_dir, f"rank_{global_rank}")
                    os.makedirs(global_rank_path, exist_ok=True)
                    dst_file_path = os.path.join(global_rank_path, new_file_name)
                    this_rank_ckpt_file[global_rank] = dst_file_path
                    shutil.copy(src_file_path, dst_file_path)
                    print(f'Copied {src_file_path} to {dst_file_path}')

    dp_rank_list = []
    dp_step = tp_size
    for x in dp0_rank_list:
        dp_rank_list.append([x + dp_step * dp_rank for dp_rank in range(dp_size * cp_size)])

    for rank_list in dp_rank_list:
        rank_id = rank_list[0]
        for dp_rank in rank_list:
            # skip current rank
            if dp_rank == rank_id:
                continue
            rank_ckpt_dir = os.path.join(output_dir, "rank_{}".format(dp_rank))
            os.makedirs(rank_ckpt_dir, exist_ok=True)
            file_name = this_rank_ckpt_file[rank_id]
            print("[{}][Rank {}] Copy merged checkpoint to dir {}".format(
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), rank_id, rank_ckpt_dir))
            shutil.copy(file_name, os.path.join(rank_ckpt_dir, 'model_optim_rng.ckpt'))


def transform_ckpt_to_new_strategy(src_ckpt_path, dst_ckpt_path, ckpt_prefix="network", output_format='ckpt'):
    """ helper function for transform ckpt """

    src_merged_strategy_file = dst_ckpt_path + "/src_merged_strategy.ckpt"
    dst_merged_strategy_file = dst_ckpt_path + "/dst_merged_strategy.ckpt"
    ms.merge_pipeline_strategys(os.path.join(src_ckpt_path, "strategy"), src_merged_strategy_file)
    ms.merge_pipeline_strategys(os.path.join(dst_ckpt_path, "strategy"), dst_merged_strategy_file)
    print(f"src_ckpt_path {src_ckpt_path}")
    ms.transform_checkpoints(src_ckpt_path, dst_ckpt_path, ckpt_prefix,
                             src_merged_strategy_file,
                             dst_merged_strategy_file,
                             output_format=output_format)


def run(args):
    '''main func'''
    src_dir = args.src_dir
    dst_dir = args.dst_dir
    src_dp_size = args.src_dp_size
    src_cp_size = args.src_cp_size
    dst_dp_size = args.dst_dp_size
    dst_tp_size = args.dst_tp_size
    dst_pp_size = args.dst_pp_size
    dst_cp_size = args.dst_cp_size
    iteration = args.iteration
    output_format = args.output_format
    if iteration == -1:
        with open(os.path.join(src_dir, "latest_checkpointed_iteration.txt"), 'r') as file:
            iteration = int(file.read().strip())

    temp_dir = os.path.join(src_dir, "temp_rearrange_ckpt")
    try:
        shutil.rmtree(temp_dir)
    except FileNotFoundError as e:
        print(f"skip delete: {e}")
    os.makedirs(temp_dir)
    mp_rank_to_global_rank(src_dir, src_dp_size, src_cp_size, iteration, output_dir=temp_dir,
                           output_format=output_format)

    transform_ckpt_to_new_strategy(temp_dir, dst_dir, output_format=output_format)

    global_rank_to_mp_rank(dst_dir, dst_dp_size, dst_tp_size, dst_pp_size, dst_cp_size, iteration=iteration,
                           output_format=output_format)
    shutil.rmtree(temp_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src-dir', default='./output',
                        help='Source ckpt path.')
    parser.add_argument('--dst-dir', default='./transformed_output',
                        help='Target ckpt path to convert.')
    parser.add_argument('--output-format', type=str, default='ckpt',
                        help='Output checkpoint format, support [`ckpt`]. Default: `ckpt`.')
    parser.add_argument('--src-dp-size', type=int, default=1,
                        help='Degree of data model parallelism of source ckpt.')
    parser.add_argument('--src-cp-size', type=int, default=1,
                        help='Degree of context model parallelism of source ckpt.')
    parser.add_argument('--dst-dp-size', type=int, default=1,
                        help='Degree of data model parallelism of target ckpt.')
    parser.add_argument('--dst-pp-size', type=int, default=1,
                        help='Degree of pipeline model parallelism of target ckpt.')
    parser.add_argument('--dst-tp-size', type=int, default=1,
                        help='Degree of tensor model parallelism of target ckpt.')
    parser.add_argument('--dst-cp-size', type=int, default=1,
                        help='Degree of context model parallelism of target ckpt.')
    parser.add_argument('--iteration', type=int, default=-1,
                        help='Iteration ckpt to be converted.')

    parse_args = parser.parse_args()
    run(parse_args)
