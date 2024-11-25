# 模型转换脚本使用说明

## megatron转为ms

### 前置条件

在使用 megatron 脚本进行训练时，会保存 param_map 信息，此信息为 megatron 优化器模型中，桶具体以何种排布进行数据构造，
在 megatron 转 ms 模型以及 ms 模型转 megatron 模型时均依赖此信息；为确保 param_map 信息正确，请在每次执行 megatron 脚本前，
删除 `/cache/buffers` 及 `--save` 目录。

使用脚本：convert_pt2ckpt.py

参数说明：python convert_pt2ckpt.py -h

```log
usage: convert_pt2ckpt.py [-h] --megatron-path MEGATRON_PATH --ms-path MS_PATH [--param-map-path PARAM_MAP_PATH] [--pp-size PP_SIZE] [--tp-size TP_SIZE] [--dp-size DP_SIZE]
                          [--cp-size CP_SIZE] [--parallel-init-order PARALLEL_INIT_ORDER] [--vpp-size VPP_SIZE] [--stage-list STAGE_LIST [STAGE_LIST ...]] [--num-layers NUM_LAYERS]
                          [--generate-all-dp-cp] [--convert-param-only] [--file-prefix FILE_PREFIX] [--format {ckpt,safetensors}]
                          [--dist-opt-content DIST_OPT_CONTENT [DIST_OPT_CONTENT ...]] [--debug] [--multiprocess-off] [--process-limit PROCESS_LIMIT] [--process_timeout PROCESS_TIMEOUT]

Pt-to-ckpt conversion Arguments

optional arguments:
  -h, --help            show this help message and exit

File Path/Location:
  --megatron-path MEGATRON_PATH
                        Path to megatron pt files.
  --ms-path MS_PATH     Path for saving Mindspore Ckpt.
  --param-map-path PARAM_MAP_PATH
                        Path to param_map files.

distributed:
  --pp-size PP_SIZE     Degree of pipeline model parallelism.
  --tp-size TP_SIZE     Degree of tensor model parallelism.
  --dp-size DP_SIZE     Degree of data model parallelism.
  --cp-size CP_SIZE     Degree of context model parallelism.
  --parallel-init-order PARALLEL_INIT_ORDER
                        expected parallel initialized order, only support tp-cp-ep-dp-pp now
  --vpp-size VPP_SIZE   euqals to virtual_pipeline_model_parallel_size, vpp_size * num_layers_per_virtual_pipeline_stage == num_layers // pp_size
  --stage-list STAGE_LIST [STAGE_LIST ...]
                        Number of layers per pipeline stage
  --num-layers NUM_LAYERS
                        Number of layers in Megatron models

Advanced feature:
  --generate-all-dp-cp  Whether generate all dp ckpt, or just dp-cp-0.
  --convert-param-only  Convert `model_optim_rng.pt`, which only contains bf16 model parameter without optimizer; If you want to convert model with optimizer、optimizer scheduler and other
                        args, please do NOT use this argument.
  --file-prefix FILE_PREFIX
                        Mindspore checkpoint filename
  --format {ckpt,safetensors}
                        Mindspore checkpoint file format
  --dist-opt-content DIST_OPT_CONTENT [DIST_OPT_CONTENT ...]
                        Torch distributed file content. i.e exp_avg in Adam optimizer
  --debug               print debug info
  --multiprocess-off    Turn off multiprocess.
  --process-limit PROCESS_LIMIT
                        Max num of processes.
  --process_timeout PROCESS_TIMEOUT
                        Timeout for each process.
```

### 使用示例：

```python
export PYTHONPATH=/path/to/Megatron-LM:$PYTHONPATH
python /path/to/mindspeed_ms/tools/converter/convert_pt2ckpt.py \
       --megatron-path /data/ckpt/torch_baseline_ckpts/iter_0001000/ \
       --param-map-path /home/ma-user/work/ckpt_convertion/param_map \
       --ms-path /cache/ms_save_sandbox_2/ \
       --num-layers 64 \
       --pp-size 4 \
       --dp-size 1 \
       --vpp-per-stage 4 \
       --tp-size 8 \
       --process-limit 4 \
       --debug \
       > convert_ckpt.log 2>&1 &
```

- 可以在生成的日志中搜索 `successfully`, 源文件有几个文件夹，就应该有几个 `successfully`，可以依此判断转换是否成功。

- 上述为简化的转换场景，在vpp 打开的场景下，转换 tp*pp 份分布式 pt 文件。测试或debug问题建议添加 `--debug`、 `--multiprocess-off`

### 默认目录结构示例

1. 分布式优化器pt

```text
pt_path
├── iter_0000005
|   ├── mp_rank_00_000
|   |   ├──distrib_optim.pt
|   |   ├──model_optim_rng.pt
|   ├── mp_rank_00_001
|   |   ...
|   ├── param_map/buffers
|   |   ├── param_map_buffer0_dp0tp0pp0vpp0.json
|   |   ├── param_map_buffer0_dp0tp0pp0vpp1.json
|   |   ...
```

2. 转换文件保存后

```text
ms_path/ms
├── rank0
|   ├──network.ckpt
├── rank1
|   ├──network.ckpt
|   ...
```

3. param_map

```text
param_map/buffers
├── args_tp00_pp000.pt
├── args_tp00_pp001.pt
├── ...
├── param_map_buffer0_dp0tp0pp0vpp0.json
├── param_map_buffer0_dp0tp0pp0vpp1.json
|   ...
```
