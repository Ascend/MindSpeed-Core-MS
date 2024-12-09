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
'''convert *.pt to *.ckpt tool'''
import os
import ast
import json
import shutil
import inspect
import argparse
import datetime
import collections
import multiprocessing
from pathlib import Path

import torch
import numpy as np
import mindspore as ms
from mindspore import Tensor, Parameter

DEBUG = False
PARAM_KEY_MAP = {
    'dense_h_to_4h': 'mapping',
    'dense_4h_to_h': 'projection',
    'self_attention': 'attention',
    'query_key_value': 'qkv_proj',
    'dense': 'out_proj',
}

DTYPE_MAP = {
    torch.bfloat16: ms.bfloat16,
    torch.float16: ms.float16,
    torch.float32: ms.float32,
}

WD_INCR_STYLE = ['constant', 'linear', 'cosine']
LR_DECAY_STYLE = ['constant', 'WSD', 'linear', 'cosine', 'inverse-square-root']
OPT_SCHEDULER_MAP = {
    'state_step': 'fp32',
    'max_lr': 'fp64',
    'lr_warmup_steps': 'int64',
    'num_steps': 'int64',
    'lr_decay_style': 'int64',
    'lr_decay_steps': 'int64',
    'min_lr': 'fp64',
    'start_wd': 'fp64',
    'end_wd': 'fp64',
    'wd_incr_style': 'int64',
    'wd_incr_steps': 'int64',
}


def log_info(msg, override=False):
    """log util"""
    if DEBUG or override:
        now = datetime.datetime.now().strftime("%m-%d %H:%M:%S")
        caller_frame = inspect.stack()[1]
        caller_lineno = caller_frame[2]
        caller_name = caller_frame[3]
        pid = multiprocessing.current_process().pid
        print(f"[{now},{pid}] filenum:{caller_lineno} {caller_name}() - {msg}", flush=True)


def parse_args():
    """Parse all arguments."""
    parser = argparse.ArgumentParser(description='Pt-to-ckpt conversion Arguments',
                                     allow_abbrev=False)
    group = parser.add_argument_group(title='File Path/Location')

    group.add_argument('--megatron-path', type=str, default=None, required=True,
                       help='Path to megatron pt files.')
    group.add_argument('--ms-path', type=str, default=None, required=True,
                       help='Path for saving Mindspore Ckpt.')
    group.add_argument('--param-map-path', type=str, default=None,
                       help='Path to param_map files.')

    group = parser.add_argument_group(title='distributed')
    group.add_argument('--pp-size', type=int, default=1,
                       help='Degree of pipeline model parallelism.')
    group.add_argument('--tp-size', type=int, default=1,
                       help='Degree of tensor model parallelism.')
    group.add_argument('--dp-size', type=int, default=1,
                       help='Degree of data model parallelism.')
    group.add_argument('--cp-size', type=int, default=1,
                       help='Degree of context model parallelism.')
    group.add_argument('--parallel-init-order', type=str, default='tp-cp-ep-dp-pp',
                       help='expected parallel initialized order, only support tp-cp-ep-dp-pp now')
    group.add_argument('--vpp-size', type=int, default=1,
                       help='euqals to virtual_pipeline_model_parallel_size, '
                            'vpp_size * num_layers_per_virtual_pipeline_stage == num_layers // pp_size')
    group.add_argument('--stage-list', type=int, default=None, nargs='+',
                       help='Number of layers per pipeline stage')
    group.add_argument('--num-layers', type=int, default=1,
                       help='Number of layers in Megatron models')

    group = parser.add_argument_group(title='Advanced feature')
    group.add_argument('--generate-all-dp-cp',
                       action='store_true', default=False,
                       help='Whether generate all dp ckpt, or just dp-cp-0.')
    group.add_argument('--convert-param-only',
                       action='store_true', default=False,
                       help=('Convert `model_optim_rng.pt`, which only contains bf16 model parameter without optimizer;'
                             ' If you want to convert model with optimizer、optimizer scheduler and other args, please '
                             'do NOT use this argument.'))
    group.add_argument('--file-prefix',
                       type=str, default='network',
                       help='Mindspore checkpoint filename')
    group.add_argument('--format',
                       type=str, default='ckpt', choices=['ckpt', 'safetensors'],
                       help='Mindspore checkpoint file format')
    group.add_argument('--dist-opt-content',
                       type=str, default=["param", "exp_avg", "exp_avg_sq"], nargs='+',
                       help='Torch distributed file content. i.e exp_avg in Adam optimizer')
    group.add_argument('--debug',
                       action='store_true', default=False,
                       help='print debug info')
    group.add_argument('--multiprocess-off',
                       action='store_true', default=False,
                       help='Turn off multiprocess.')
    group.add_argument('--process-limit', type=int, default=4,
                       help='Max num of processes.')
    group.add_argument('--process_timeout', type=int, default=3600,
                       help='Timeout for each process.')
    # Parse.
    return parser.parse_args()


def print_keys(ckpt, s=1):
    """print key util"""
    if isinstance(ckpt, dict):
        for k, v in ckpt.items():
            print(f"structure{s}, Got keys: {k}")
            print_keys(v, s+1)


def get_pt2ms_args(weight_pt_path, save_args_path, param_name_list=None):
    """convert pt args to ms args"""
    log_info(f"weight_pt_path: {weight_pt_path}")
    pt = torch.load(weight_pt_path, map_location="cpu")
    save_args(weight_pt_path, save_args_path, pt, param_name_list)
    flatten_param_groups = {}

    if 'optimizer' in pt:
        pt_param_groups = pt['optimizer']['optimizer']['param_groups']
        for i, param_lr_dict in enumerate(pt_param_groups):
            for k, v in param_lr_dict.items():
                if k == 'lr':
                    k = 'learning_rate'
                elif k == 'step':
                    k = 'state_step'
                    flatten_param_groups[k] = [v]
                    continue
                flatten_param_groups[k + f'_group_{i}'] = v

    if 'opt_param_scheduler' in pt:
        pt_opt_scheduler = pt['opt_param_scheduler']
        if 'wd_incr_style' in pt_opt_scheduler:
            pt_opt_scheduler['wd_incr_style'] = WD_INCR_STYLE.index(pt_opt_scheduler['wd_incr_style'])
        if 'lr_decay_style' in pt_opt_scheduler:
            pt_opt_scheduler['lr_decay_style'] = LR_DECAY_STYLE.index(pt_opt_scheduler['lr_decay_style'])
        flatten_param_groups.update(pt_opt_scheduler)

    return_dict = {}
    pt_iteration = int(Path(weight_pt_path).parts[-3].split('_')[-1])
    return_dict['step_num'] = int(pt_iteration)
    return_dict['epoch_num'] = 0

    for k, v in flatten_param_groups.items():
        if isinstance(v, list):
            return_dict[k] = Parameter([float(ele) for ele in v])
        elif k in OPT_SCHEDULER_MAP and OPT_SCHEDULER_MAP[k] == 'fp64':
            return_dict[k] = Parameter(ms.Tensor(v, dtype=ms.float64))
        elif k in OPT_SCHEDULER_MAP and OPT_SCHEDULER_MAP[k] == 'fp32':
            return_dict[k] = Parameter(float(v))
        elif k in OPT_SCHEDULER_MAP and OPT_SCHEDULER_MAP[k] == 'int64':
            return_dict[k] = Parameter(int(v))
        elif isinstance(v, (int, float)):
            return_dict[k] = Parameter(float(v))
    return return_dict


def get_pts_path(args):
    '''get pt iterator under pt_path'''
    path = args.megatron_path
    convert_param_only = args.convert_param_only
    pts_path = []
    for dir_item in os.listdir(path):
        if dir_item.startswith('mp_rank'):
            pt_dir = os.path.join(path, dir_item)
            dist_pt_path = os.path.join(pt_dir, "distrib_optim.pt")
            model_pt_path = os.path.join(pt_dir, "model_optim_rng.pt")
            if not convert_param_only and not os.path.exists(dist_pt_path):
                raise FileExistsError(f"{dist_pt_path} not exists, maybe wrong '--megatron-path'?")
            if not os.path.exists(model_pt_path):
                raise FileExistsError(f"{model_pt_path} not exists, maybe wrong '--megatron-path'?")
            pts_path.append((dist_pt_path, model_pt_path))
    if not pts_path:
        raise FileExistsError(f"No *.pt found in {path}, maybe wrong '--megatron-path'?")
    return pts_path


def get_pt2ms_info(para_map_path: str, pt_path):
    '''Map that contains location and data info to convert parameter to ms format'''
    files = [f for f in os.listdir(para_map_path) if f.startswith(f"param_map_buffer")]
    if not files:
        raise ValueError("param_map not found, may be wrong '--param-map-path'?")
    log_info(f"Param_map files: {files}")
    pt_dir_parts = Path(pt_path).parts[-2].split("_")
    if len(pt_dir_parts) == 4:
        pp = int(pt_dir_parts[-1])
    elif len(pt_dir_parts) == 3:
        pp = 0
    param_keymap = {}
    for f in files:
        f_pp = int(f.split('vpp')[0].split("pp")[-1])
        if f_pp != pp:
            continue
        vpp = f.split('pp')[-1].split('vpp')[-1].split('.')[0]
        file_path = os.path.join(para_map_path, f)
        with open(file_path) as json_file:
            param_keymap[int(vpp)] = json.load(json_file)
    log_info(f"param_keymap: {param_keymap}")
    return param_keymap


def save_ms_ckpt(retrived_data, path, file_prefix, format_="safetensors", dir_format="", extra_args=None):
    """save ms ckpt func"""
    path_obj = Path(path)
    ms_path = path_obj/dir_format/f"{file_prefix}.{format_}"
    log_info(f"start generating ms ckpt: {str(ms_path)}")
    ms_path.parent.mkdir(parents=True, exist_ok=True)
    ms.save_checkpoint(retrived_data, str(ms_path), format=format_, append_dict=extra_args)
    return ms_path


def copy_ckpt(ms_path, dir_format):
    """copy ms dp-0 ckpt to other dp"""
    target_path = ms_path.parents[1]/dir_format/ms_path.name
    log_info(f"start copying ckpt: {ms_path} to {target_path}")
    try:
        if target_path.parent.exists():
            shutil.rmtree(target_path.parent)
        target_path.parent.mkdir()
        shutil.copy(ms_path, target_path)
        log_info(f"copy {ms_path} to {target_path} ended.")
    except:
        raise ValueError("fail to copy ckpt, maybe wrong '--ms-path' or insufficient disk space "
                         "or insufficient permissions?")


def get_vpp_layers_map(total_layers, vpp_size, pp):
    """generate vpp local layer_id to global layer_id """
    pp_list = None
    if isinstance(vpp_size, int) and vpp_size > 1:
        vpp_np = np.arange(total_layers).reshape(vpp_size, total_layers // vpp_size)
        pp_np = np.split(vpp_np, pp, -1)
        pp_list = [item.tolist() for item in pp_np]

    return pp_list


def _convert_param_name(pt_param_name, data_key, pt_path, stage_list, vpp_layer_mapping, vpp_stage):
    """convert param name"""
    parts = [data_key] if data_key != "param" else []
    parts.extend(pt_param_name.split("."))
    extra_key = ["module"]
    pt_dir_parts = Path(pt_path).parts[-2].split("_")
    pipeline_on = len(pt_dir_parts) == 4
    if pipeline_on:
        pp_stage = int(pt_dir_parts[-1])
        cur_layer_offset = sum(stage_list[:pp_stage])
    log_info(f"pt parameter name: {parts}")
    for i, p in enumerate(parts):
        if p in extra_key:
            parts.pop(i)
        elif p in PARAM_KEY_MAP:
            parts[i] = PARAM_KEY_MAP[p]
        elif vpp_layer_mapping and p == "layers":
            pt_layer_num = int(parts[i + 1])
            try:
                ms_vpp_layer = vpp_layer_mapping[pp_stage][vpp_stage][pt_layer_num]
            except IndexError:
                raise IndexError("get 'ms_vpp_layer' failed, may be wrong '--param-map-path' "
                                 "or wrong tp/pp/vpp/num_layers config? Try to keep "
                                 "'--save' path clean when generate *.pt.")
            parts[i+1] = str(ms_vpp_layer)
        elif pipeline_on and p == "layers":
            parts[i+1] = str(int(parts[i + 1]) + cur_layer_offset)
    log_info(f"ms parameter name: {parts}")
    ms_para_name = ".".join(parts)
    return ms_para_name


def save_args(weight_pt_path, save_args_path, pt, param_name_list):
    """save args"""
    needed_args = {}
    needed_args['args'] = pt.get('args', None)
    needed_args['checkpoint_version'] = pt.get('checkpoint_version', None)
    needed_args['iteration'] = pt.get('iteration', None)
    needed_args['optimizer'] = pt.get('optimizer', None)
    needed_args['opt_param_scheduler'] = pt.get('opt_param_scheduler', None)
    needed_args['rng_state'] = pt.get('rng_state', None)
    needed_args['num_floating_point_operations_so_far'] = pt.get('num_floating_point_operations_so_far', None)

    if param_name_list is not None:
        needed_args['param_name_list'] = param_name_list

    dir_path = os.path.dirname(weight_pt_path)
    dir_name = os.path.basename(dir_path)
    dir_name_in_parts = dir_name.split("_")
    tp = int(dir_name_in_parts[2])
    pp = int(dir_name_in_parts[3]) if len(dir_name_in_parts) == 4 else 0

    args_pt_path = save_args_path / f"args_tp{tp:02}_pp{pp:03}.pt"
    torch.save(needed_args, args_pt_path)


def convert_pt2ms_param_and_optim(param_keymap: dict, param_dict: dict, pt_path, stage_list, data_keys=None,
                                  pt_param_dtype=torch.bfloat16, ms_dtype=ms.float32, vpp_layer_mapping=None):
    """convert param and optimizer"""
    if data_keys is None:
        data_keys = ["param", "exp_avg", "exp_avg_sq"]
    cur_bucket = 0
    end_buffer_index = 0
    retrived_data = {}
    vpp_stage_list = sorted(param_keymap.keys())
    for vpp_stage in vpp_stage_list:
        try:
            model0_buffer0 = param_dict[vpp_stage][(pt_param_dtype, torch.float32)]
        except IndexError:
            raise IndexError("get 'ms_vpp_layer' failed, may be wrong '--param-map-path' "
                             "or wrong tp/pp/vpp/num_layers config? Try to keep "
                             "'--save' path clean when generate *.pt")
        buffer2bucket_offset = 0
        for para_name, attrs in param_keymap[vpp_stage].items():
            log_info(f"for param map vpp: {vpp_stage}, param_keymap: {param_keymap[vpp_stage]}")
            prev_end_buffer_index = end_buffer_index
            param_dtype, shape, start_buffer_index, end_buffer_index, bucket_id = attrs
            converted_type = ms_dtype if ms_dtype else DTYPE_MAP[ast.literal_eval(param_dtype)] #
            if bucket_id > cur_bucket:
                cur_bucket = bucket_id
                buffer2bucket_offset = prev_end_buffer_index
            start_bucket_index = start_buffer_index - buffer2bucket_offset
            end_bucket_index = end_buffer_index - buffer2bucket_offset
            for k in data_keys:
                log_info(f"Pt content type:{k}, bucket_id: {bucket_id}, "
                         f"start_bucket_index:{start_bucket_index}, end_bucket_index: {end_bucket_index}")
                extrated_value = model0_buffer0[k][bucket_id][start_bucket_index:end_bucket_index]
                value_np = extrated_value.float().detach().cpu().numpy().reshape(shape)
                value_ms = Parameter(Tensor(input_data=value_np, dtype=converted_type))
                ms_para_name = _convert_param_name(para_name, k, pt_path, stage_list, vpp_layer_mapping, vpp_stage)
                retrived_data[ms_para_name] = value_ms
    return retrived_data


def flatten_param_dict(prefix, param_dict, flattened_param_dict):
    """flatten folded pt param dict"""
    if isinstance(param_dict, dict):
        for k, v in param_dict.items():
            prefix_copy = prefix.copy()
            prefix_copy.append(k)
            flatten_param_dict(prefix_copy, v, flattened_param_dict)
    elif isinstance(param_dict, torch.Tensor):
        new_name = ".".join(prefix)
        flattened_param_dict[new_name] = param_dict


def convert_pt2ms_param_only(param_dict: dict, pt_path, stage_list, ms_dtype=ms.float32, vpp_layer_mapping=None):
    """convert param only"""
    retrived_data = collections.OrderedDict()
    vpp_size = len(vpp_layer_mapping[0]) if vpp_layer_mapping else 1
    param_name_list = []
    for vpp_stage in range(vpp_size):
        flattened_dict = collections.OrderedDict()
        model_key = f'model{vpp_stage}' if vpp_size > 1 else 'model'
        flatten_param_dict([], param_dict[model_key], flattened_dict)
        param_names = [f"{model_key}.{key}" for key in flattened_dict.keys()]
        param_names.reverse()
        param_name_list.append(param_names)

        for k, v in flattened_dict.items():
            new_key = _convert_param_name(k, "param", pt_path, stage_list, vpp_layer_mapping, vpp_stage)
            v = v.float().detach().cpu().numpy()
            new_param = Parameter(Tensor(v, dtype=ms_dtype))
            retrived_data[new_key] = new_param

    print(f"param_name_list {str(param_name_list)}")
    return retrived_data, param_name_list


def get_ms_dir_format(tp_rank, tp_size, cp_rank, cp_size, dp_rank, dp_size, pp_rank, order='tp-cp-ep-dp-pp'):
    """map tp-cp-dp-pp rank to global rank util. Assume init order is tp-cp-ep-dp-pp"""
    if order != 'tp-cp-ep-dp-pp':
        raise ValueError(f"args.parallel_init_order only support 'tp-cp-ep-dp-pp' for now, but got {order} ")
    global_rank = (
        pp_rank * dp_size * cp_size * tp_size +
        dp_rank * cp_size * tp_size +
        cp_rank * tp_size +
        tp_rank
    )
    return f"rank_{global_rank}"


def run(args, dist_path, model_path):
    """main func"""
    log_info(f"start converting: {dist_path}")
    para_map_path = args.param_map_path
    ms_path = args.ms_path
    total_layers = args.num_layers
    vpp_size = args.vpp_size
    pp_size = args.pp_size
    if total_layers % pp_size != 0:
        raise ValueError(f"num_layers {total_layers} is not divisible by pp_size {pp_size}, "
                         "maybe wrong num_layers or pp_size arguments?")
    stage_list = args.stage_list if args.stage_list else [total_layers // pp_size] * pp_size
    log_info(f"stage_list: {stage_list}")
    dp_size = args.dp_size
    tp_size = args.tp_size
    cp_size = args.cp_size
    generate_all_dp_cp = args.generate_all_dp_cp
    file_prefix = args.file_prefix
    format_ = args.format
    pt_content = args.dist_opt_content
    convert_param_only = args.convert_param_only

    save_args_path = Path(all_args.ms_path) / "pt_meta"
    # save pt args
    vpp_layer_mapping = get_vpp_layers_map(total_layers, vpp_size, pp_size)
    log_info(f"vpp_layer_mapping: {vpp_layer_mapping}")
    log_info(f"Parameter conversion begins.")
    if not convert_param_only:
        param_dict = torch.load(dist_path, map_location="cpu")
        param_keymap = get_pt2ms_info(para_map_path, model_path)
        retrived_data = convert_pt2ms_param_and_optim(
            param_keymap=param_keymap,
            param_dict=param_dict,
            pt_path=dist_path,
            stage_list=stage_list,
            data_keys=pt_content,
            pt_param_dtype=torch.bfloat16,
            ms_dtype=ms.float32,
            vpp_layer_mapping=vpp_layer_mapping
        )
        args2save = get_pt2ms_args(model_path, save_args_path)

    else:
        param_dict = torch.load(model_path, map_location="cpu")
        retrived_data, param_name_list = convert_pt2ms_param_only(
            param_dict=param_dict,
            pt_path=model_path,
            stage_list=stage_list,
            ms_dtype=ms.float32,
            vpp_layer_mapping=vpp_layer_mapping
        )
        args2save = get_pt2ms_args(model_path, save_args_path, param_name_list)

    print(f"converted param is:")
    for k, v in retrived_data.items():
        print(f"{k} {v.shape} {v.dtype}")

    pt_dir_format = Path(model_path).parts[-2]
    log_info(f"convert parameter conversion ended.")

    pt_dir_parts = pt_dir_format.split('_')
    tp_rank = int(pt_dir_parts[2])
    pp_rank = int(pt_dir_parts[3]) if len(pt_dir_parts) == 4 else 0
    dp_rank_0 = 0
    cp_rank_0 = 0
    ms_dir_format = get_ms_dir_format(tp_rank, tp_size, cp_rank_0, cp_size,
                                      dp_rank_0, dp_size, pp_rank, order=args.parallel_init_order)
    ms_path = save_ms_ckpt(
        retrived_data,
        ms_path,
        file_prefix=file_prefix,
        format_=format_,
        dir_format=ms_dir_format,
        extra_args=args2save
    )
    if generate_all_dp_cp:
        for dp_rank in range(1, dp_size):
            for cp_rank in range(1, cp_size):
                ms_dir_format = get_ms_dir_format(tp_rank, tp_size, cp_rank, cp_size,
                                                  dp_rank, dp_size, pp_rank, order=args.parallel_init_order)
                copy_ckpt(ms_path, ms_dir_format)
    pt_path = Path(model_path).parent
    log_info(f"convert '{pt_path}' to '{ms_path}' successfully.")
    return dist_path

def throw_error(e):
    print(
        "multiprocess caught ERROR:\n"
        f"    {e}\n"
        "please add '--multiprocess-off' to obtain more error imformation.\n"
    )

if __name__ == '__main__':
    all_args = parse_args()
    log_info(f'Got args: {all_args}', override=True)
    if not all_args.convert_param_only and all_args.param_map_path is None:
        raise ValueError(f"When '--convert-param-only' is NOT enable, means converting both model and optimizer, "
                         "please specify '--param-map-path'.")
    DEBUG = all_args.debug

    pt_meta_path = Path(all_args.ms_path) / "pt_meta"
    # make sure 'pt_meta_path' exists
    pt_meta_path.mkdir(parents=True, exist_ok=True)
    # copy param_map to ms_path
    if not all_args.convert_param_only:
        param_map_list = Path(all_args.param_map_path).glob("*.json")
        if not param_map_list:
            log_info(f"no param_map*.json was found under directory {all_args.param_map_path}, "
                     f"so no param_map*.json will be copy to {pt_meta_path}")
        for param_map_file in param_map_list:
            dest_path = pt_meta_path / param_map_file.name
            log_info(f"copy {param_map_file} to {dest_path}")
            shutil.copy(param_map_file, dest_path)

    pt_file_list = get_pts_path(all_args)

    if all_args.multiprocess_off:
        for dist_file_path, model_file_path in pt_file_list:
            run(all_args, dist_file_path, model_file_path)
    else:
        with multiprocessing.Pool(processes=all_args.process_limit) as pool:
            results = []
            for dist_file_path, model_file_path in pt_file_list:
                result = pool.apply_async(run, (all_args, dist_file_path, model_file_path), error_callback=throw_error)
                results.append(result)
            output = [result.wait(all_args.process_timeout) for result in results]
        log_info("-------------end convert pt to ckpt-------------")
