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
'''convert mindspore ckpt to torch pt'''
import os
import sys
import json
import argparse
import multiprocessing
from datetime import datetime
from collections import OrderedDict

import torch
import numpy as np
import mindspore as ms


sys.path.insert(
    0, '/path/to/Megatron-LM')


# 1. specify args：ms_ckpt path、param_map path、megatron_pt path、vpp
ms_path = ''
param_map_path = ''
megatron_path = ''
src_model_format = ''
num_layers = 0
dp_size = 0
cp_size = 0
tp_size = 0
pp_size = 0
vpp_size = 0
noop = []
convert_param_only = False


param_key_mapping = {
    'dense_h_to_4h': 'mapping',
    'dense_4h_to_h': 'projection',
    'self_attention': 'attention',
    'query_key_value': 'qkv_proj',
    'dense': 'out_proj',
}

add_key = [
    "query_key_value.weight",
    "dense.weight",
    "dense_h_to_4h.weight",
    "dense_4h_to_h.weight",
    "output_layer.weight",
]

curr_iterations = None
file_flag = None


def log_with_time(log_str):
    '''log tool'''
    now = datetime.now()
    print(f">{now.strftime('%H:%M:%S')}  : {log_str}", flush=True)


def parse_args():
    """Parse all arguments."""
    parser = argparse.ArgumentParser(description='ckpt-to-pt conversion Arguments',
                                     allow_abbrev=False)

    group = parser.add_argument_group(title='distributed')
    group.add_argument('--num-layers', type=int, default=0, required=True,
                       help='Number of layers in models')
    group.add_argument('--dp-size', type=int, default=0, required=True,
                       help='Degree of data model parallelism.')
    group.add_argument('--cp-size', type=int, default=1, required=True,
                       help='Degree of context model parallelism.')
    group.add_argument('--tp-size', type=int, default=0, required=True,
                       help='Degree of tensor model parallelism.')
    group.add_argument('--pp-size', type=int, default=0, required=True,
                       help='Degree of pipeline model parallelism.')
    group.add_argument('--vpp-size', type=int, default=0, required=True,
                       help='Number of virtual pipeline per pipeline stage')
    group.add_argument('--src-model-format', type=str, default="ckpt", required=False,
                       help='Path to param_map files.')
    group.add_argument('--noop', type=int, nargs='+', required=False,
                       help='Number of virtual pipeline per pipeline stage')

    group = parser.add_argument_group(title='File Path/Location')
    group.add_argument('--ms-path', type=str, default=None, required=True,
                       help='Path to mindspore checkpoint files.')
    group.add_argument('--param-map-path', type=str, default=None, required=True,
                       help='Path to param_map files.')
    group.add_argument('--megatron-path', type=str, default=None, required=True,
                       help='Path for saving Mindspore ckpt.')
    group.add_argument('--convert-param-only',
                       action='store_true', default=False,
                       help='Convert only the model parameter without optimizer params;')
    group.add_argument('--process-limit', type=int, default=4,
                       help='Max num of processes.')

    # Parse.
    return parser.parse_args()


def set_args(args):
    '''set args tool'''
    global ms_path, param_map_path, megatron_path, num_layers, dp_size, cp_size, tp_size, pp_size, \
           convert_param_only, vpp_size, noop, src_model_format
    ms_path = args.ms_path
    param_map_path = args.param_map_path
    megatron_path = args.megatron_path
    num_layers = args.num_layers
    dp_size = args.dp_size
    cp_size = args.cp_size
    tp_size = args.tp_size
    pp_size = args.pp_size
    vpp_size = args.vpp_size
    convert_param_only = args.convert_param_only
    noop = args.noop
    src_model_format = args.src_model_format
    if noop is None:
        noop = []
    print("-------------args-------------")
    print(f"> ms_path = {ms_path}")
    print(f"> param_map_path = {param_map_path}")
    print(f"> megatron_path = {megatron_path}")
    print(f"> num_layers = {num_layers}")
    print(f"> dp_size = {dp_size}")
    print(f"> cp_size = {cp_size}")
    print(f"> tp_size = {tp_size}")
    print(f"> pp_size = {pp_size}")
    print(f"> vpp_size = {vpp_size}")
    print(f"> src_model_format = {src_model_format}")
    print(f"> noop = {noop}")


def get_ckpt():
    '''load ckpt and save tp/pp info, replace param name and layer id, no optim state included'''
    global ms_path, param_map_path, megatron_path, num_layers, dp_size, cp_size, tp_size, pp_size, \
           vpp_size, noop, src_model_format
    rst_list = []
    for pp in range(pp_size):
        for tp in range(tp_size):
            dir_name = f"rank_{pp*dp_size*cp_size*tp_size+tp}"
            save_dir_name = f"mp_rank_{tp:02d}_{pp:03d}"
            ckpt_dir = os.path.join(ms_path, dir_name)
            if src_model_format == "ckpt":
                safetensors_files = [file for file in os.listdir(
                    ckpt_dir) if file.endswith('ckpt')]
            elif src_model_format == "safetensors":
                safetensors_files = [file for file in os.listdir(
                    ckpt_dir) if file.endswith('safetensors')]
            else:
                raise ValueError("src_model_format only support ckpt or safetensors!")
            safetensors_files = [os.path.join(
                ckpt_dir, file) for file in safetensors_files]
            safetensors_files.sort(key=os.path.getmtime, reverse=True)
            latest_safetensors_file = safetensors_files[0] if safetensors_files else None
            ckpt_path = latest_safetensors_file
            rst_list.append([dir_name, ckpt_path, tp, pp, save_dir_name])
    return rst_list


def get_param_map(tp, pp, vpp):
    '''read tp/pp info from param_map'''
    global ms_path, param_map_path, megatron_path, num_layers, dp_size, tp_size, pp_size, vpp_size, noop
    param_map_name = f"param_map_buffer0_dp0tp{tp}pp{pp}vpp{vpp}.json"
    param_map_file = os.path.join(param_map_path, param_map_name)
    with open(param_map_file, "r") as f:
        param_map = json.load(f)
    return param_map


def cal_bucket_num(param_map):
    '''cal bucket number'''
    bucket_list = [int(item[-1]) for item in param_map.values()]
    bucket_num = max(bucket_list) + 1
    return bucket_num


def update_ms_key(ms_key, vpp_ms_layers, vpp_megatron_layers):
    '''update ms key'''
    ms_key_list = ms_key.split(".")
    layers_index = ms_key_list.index("layers")
    layer_num_index = layers_index + 1
    layer_num = int(ms_key_list[layer_num_index])
    layer_in_megatron_index = vpp_megatron_layers.index(layer_num)
    layer_in_ms_index = vpp_ms_layers[layer_in_megatron_index]
    ms_key_list[layer_num_index] = str(layer_in_ms_index)
    new_ms_key = ".".join(ms_key_list)
    return new_ms_key


def cal_corr_vpp_layers(vpp, pp):
    '''cal ms vpp layers'''
    global ms_path, param_map_path, megatron_path, num_layers, dp_size, tp_size, pp_size, vpp_size, noop
    vpp_layers = np.split(np.arange(num_layers), vpp_size, axis=-1)[vpp]
    vpp_ms_layers = np.split(vpp_layers, pp_size, axis=-1)[pp].tolist()
    vpp_ms_layers = [layer for layer in vpp_ms_layers if layer not in noop]
    vpp_megatron_layers = np.arange(len(vpp_ms_layers)).tolist()
    return vpp_ms_layers, vpp_megatron_layers


def save_pt(megatron_model_path, dir_name, distrib_model, model_optim_rng):
    '''save pt file'''
    global ms_path, param_map_path, megatron_path, num_layers, dp_size, tp_size, pp_size, vpp_size, noop
    save_dir = os.path.join(megatron_model_path, dir_name)
    os.makedirs(save_dir, exist_ok=True)

    # save optim
    if not convert_param_only:
        distrib_save_path = os.path.join(save_dir, "distrib_optim.pt")
        torch.save(distrib_model, distrib_save_path)
        log_with_time(f"-------successful saved {distrib_save_path}----------")

    # save param
    model_optim_save_path = os.path.join(save_dir, "model_optim_rng.pt")
    torch.save(model_optim_rng, model_optim_save_path)
    log_with_time(f"-------successful saved {model_optim_save_path}----------")


def add_args(model_optim, tp, pp, curr_iter):
    '''add args tool'''
    global ms_path, param_map_path, megatron_path, num_layers, dp_size, tp_size, pp_size, vpp_size, noop

    args_path = os.path.join(param_map_path, f"args_tp{tp:02}_pp{pp:03}.pt")
    pt_args = torch.load(args_path, map_location="cpu")

    model_optim_with_args = {}

    model_optim_with_args['args'] = pt_args['args']

    model_optim_with_args['checkpoint_version'] = pt_args['checkpoint_version']
    model_optim_with_args['iteration'] = curr_iter.value
    if len(model_optim) == 1:
        model_optim_with_args[f'model'] = model_optim[f'model0']
    else:
        for i in range(len(model_optim)):
            model_optim_with_args[f'model{i}'] = model_optim[f'model{i}']
    if not convert_param_only:
        model_optim_with_args['optimizer'] = pt_args['optimizer']
        for i in range(len(pt_args['optimizer']['optimizer']['param_groups'])):
            model_optim_with_args['optimizer']['optimizer']['param_groups'][i]['step'] = curr_iter.value
    model_optim_with_args['opt_param_scheduler'] = pt_args['opt_param_scheduler']
    model_optim_with_args['opt_param_scheduler']['num_steps'] = curr_iter.value * \
        pt_args['args'].global_batch_size
    model_optim_with_args['rng_state'] = pt_args['rng_state']
    model_optim_with_args['num_floating_point_operations_so_far'] = pt_args['num_floating_point_operations_so_far']

    return model_optim_with_args


def reconstruct_model_optim(model_optim_rng, tp, pp, curr_iter):
    '''reconstruct model optim'''
    global ms_path, param_map_path, megatron_path, num_layers, dp_size, tp_size, pp_size, vpp_size, noop
    model_optim = {}
    for vpp in range(vpp_size):
        lm_dict = {
            'embedding': {},
            'encoder': OrderedDict(),
            'output_layer': OrderedDict(),
        }
        for items in model_optim_rng[vpp]:
            key = list(items.keys())[0]
            value = list(items.values())[0]
            key = ".".join(key.split(".")[1:])
            branch = ''
            if "embedding" in key:
                middle_key = key.split(".")[2]  # "word_embedding"
                last_key = ".".join(key.split(".")[3:])  # "weight"
                lm_dict['embedding'][middle_key] = OrderedDict()
                lm_dict["embedding"][middle_key][last_key] = value
            elif "encoder" in key:
                last_key = ".".join(key.split(".")[2:])
                lm_dict['encoder'][last_key] = value
                branch = 'encoder'
            elif "output_layer" in key:
                last_key = ".".join(key.split(".")[2:])
                lm_dict['output_layer'][last_key] = value
                branch = 'output_layer'
            else:
                continue

            # add `_extra_state` layer
            extra_name = ".".join(key.split(".")[-2:])
            if extra_name in add_key:
                extra_key = last_key.replace("weight", "_extra_state")
                if branch == 'encoder':
                    lm_dict['encoder'][extra_key] = None
                elif branch == 'output_layer':
                    lm_dict['output_layer'][extra_key] = None
                else:
                    pass

        # del layer
        if not lm_dict['embedding']:
            del lm_dict['embedding']
        if not lm_dict['encoder']:
            del lm_dict['encoder']
        if not lm_dict['output_layer']:
            del lm_dict['output_layer']

        model_optim[f'model{vpp}'] = {}
        model_optim[f'model{vpp}']['language_model'] = lm_dict

        model_optim_with_args = add_args(model_optim, tp, pp, curr_iter)
    return model_optim_with_args


def process_ckpt_to_pt(ckpt_path, tp, pp, save_dir_name, dir_name, curr_iter):
    '''ckpt to pt main process'''
    log_with_time(f"> processing {dir_name}, tp = {tp}, pp = {pp}")
    global ms_path, param_map_path, megatron_path, num_layers, dp_size, tp_size, pp_size, vpp_size, noop, \
           src_model_format
    if src_model_format == "safetensors":
        ckpt = ms.load_checkpoint(ckpt_path, format="safetensors")
    else:
        ckpt = ms.load_checkpoint(ckpt_path)

    with curr_iter.get_lock():
        if curr_iter.value < 0:
            curr_iter.value = int(
                ckpt["step_num"].asnumpy().astype(np.float32))
        megatron_model_path = os.path.join(
            megatron_path, f"iter_{curr_iter.value:07}")
        os.makedirs(megatron_model_path, exist_ok=True)
        iterations_path = os.path.join(
            megatron_path, "latest_checkpointed_iteration.txt")
        if not os.path.exists(iterations_path):
            with open(iterations_path, "w") as f:
                f.write(str(curr_iter.value))

    distrib_optim_model = {}

    per_bucket_numel = []
    per_bucket_numel_unpadded = []
    if not convert_param_only:
        distrib_optim_model["per_bucket_numel"] = per_bucket_numel
        distrib_optim_model["per_bucket_numel_unpadded"] = per_bucket_numel_unpadded

    model_curr = []
    for vpp in range(vpp_size):
        log_with_time(f"> processing tp = {tp}, pp = {pp}, vpp = {vpp}")

        model_curr_vpp = []
        # 3. fetch param_map according to to/pp, multiple param_map need to be read when vpp_size > 1
        param_map = get_param_map(tp, pp, vpp)

        bucket_num = cal_bucket_num(param_map)
        param = [torch.tensor([], dtype=torch.float32)
                 for i in range(bucket_num)]

        if not convert_param_only:
            exp_avg = [torch.tensor([], dtype=torch.float32)
                       for i in range(bucket_num)]
            exp_avg_sq = [torch.tensor([], dtype=torch.float32)
                          for i in range(bucket_num)]

            # 4. build megatron model, read parameter from *.ckpt according to param_map
            distrib_optim_model[vpp] = {}
            distrib_optim_model[vpp][(torch.bfloat16, torch.float32)] = {}
            distrib_optim_model[vpp][(
                torch.bfloat16, torch.float32)]["param"] = param
            distrib_optim_model[vpp][(
                torch.bfloat16, torch.float32)]["exp_avg"] = exp_avg
            distrib_optim_model[vpp][(
                torch.bfloat16, torch.float32)]["exp_avg_sq"] = exp_avg_sq

        cur_bucket_numel = [0 for i in range(bucket_num)]
        cur_bucket_numel_unpadded = [0 for i in range(bucket_num)]

        # mindspore pp/vpp → megatron layer_num
        vpp_ms_layers, vpp_megatron_layers = cal_corr_vpp_layers(vpp, pp)

        for key in param_map:
            ms_key = ".".join(key.split(".")[1:])
            if "layers" in ms_key:
                ms_key = update_ms_key(
                    ms_key, vpp_ms_layers, vpp_megatron_layers)
            for k, v in param_key_mapping.items():
                if k in ms_key:
                    ms_key = ms_key.replace(k, v)
            shape = param_map[key][1]
            bucket_id = int(param_map[key][-1])
            newshape = [1]
            for i in shape:
                newshape[0] *= i
            weight_value = torch.tensor(ckpt[ms_key].asnumpy().astype(
                np.float32).reshape(newshape), dtype=torch.float32)
            weight_value_for_megatron = torch.tensor(ckpt[ms_key].asnumpy().astype(
                np.float32), dtype=torch.bfloat16)
            model_curr_vpp.append({key: weight_value_for_megatron})
            if not convert_param_only:
                exp_avg_value = torch.tensor(ckpt["exp_avg." + ms_key].asnumpy().astype(
                    np.float32).reshape(newshape), dtype=torch.float32)
                exp_avg_sq_value = torch.tensor(
                    ckpt["exp_avg_sq." + ms_key].asnumpy().astype(np.float32).reshape(newshape), dtype=torch.float32
                )
            # 5. build buffer
            param[bucket_id] = torch.cat((param[bucket_id], weight_value), 0)
            if not convert_param_only:
                exp_avg[bucket_id] = torch.cat(
                    (exp_avg[bucket_id], exp_avg_value), 0)
                exp_avg_sq[bucket_id] = torch.cat(
                    (exp_avg_sq[bucket_id], exp_avg_sq_value), 0)

            # bucket_id ＆ shape
            cur_bucket_numel[bucket_id] += newshape[0]
            cur_bucket_numel_unpadded[bucket_id] += newshape[0]

        # 6. merge different vpp，add head information: (per_bucket_numel、per_bucket_numel_unpadded)
        per_bucket_numel.append(
            {(torch.bfloat16, torch.float32): cur_bucket_numel})
        per_bucket_numel_unpadded.append(
            {(torch.bfloat16, torch.float32): cur_bucket_numel_unpadded})

        model_curr_vpp.reverse()
        model_curr.append(model_curr_vpp)

    # 7. build model_optim_rng.pt
    model_optim_rng_sorted = reconstruct_model_optim(
        model_curr, tp, pp, curr_iter)

    save_pt(megatron_model_path, save_dir_name,
            distrib_optim_model, model_optim_rng_sorted)


def process_ckpt_to_pt_wrapper(args):
    '''main func'''
    global curr_iterations
    ckpt_path, tp, pp, save_dir_name, dir_name = args
    return process_ckpt_to_pt(ckpt_path, tp, pp, save_dir_name, dir_name, curr_iterations)


def init(val):
    '''init func'''
    global curr_iterations, file_flag
    curr_iterations = val


if __name__ == "__main__":
    log_with_time("-------------start convert ckpt to pt-------------")
    all_args = parse_args()
    set_args(all_args)

    # 2. read mindspore ckpt file, save tp/pp info
    rst = get_ckpt()
    curr_iterations = multiprocessing.Value('i', -1)
    lock = multiprocessing.Lock()

    with multiprocessing.Pool(processes=all_args.process_limit, initializer=init, initargs=(curr_iterations,)) as pool:
        results = pool.map(process_ckpt_to_pt_wrapper, [(
            ckpt_path, tp, pp, save_dir_name, dir_name) for dir_name, ckpt_path, tp, pp, save_dir_name in rst])

    log_with_time("-------------end convert ckpt to pt-------------")
