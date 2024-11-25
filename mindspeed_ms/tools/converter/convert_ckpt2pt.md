### ms模型转为megatron模型

#### 前置条件：

1. 在使用megatron脚本进行训练时，会保存param_map信息，此信息为megatron优化器模型中，桶具体以何种排布进行数据构造，在megatron转ms模型以及ms模型转megatron模型时均依赖此信息；同时依赖megatron训练时保存的param_map、以及megatron模型转为ms模型时保存的args_xx_xxx.pt的模型参数文件，此两者需保存在同一路径下，路径下直接为json及pt文件。
2. 使用ms训练保存的模型，其参数以桶形式在dp域内进行分布式保存，需要先使用combine_ckpt_dp_zero.py脚本进行参数合并（dp=1也需要执行此项操作），之后使用合并后的模型做ms到megatron的转换，combine_ckpt_dp_zero.py脚本使用请参考同级目录下combine_ckpt_dp_zero.md文件；

#### 使用脚本：

convert_ckpt2pt.py

#### 参数说明：

```text
--num-layers:        int, 总层数
--dp-size:           int, dp数
--tp-size:           int, 模型并行数
--pp-size:           int, 流水线并行数
--vpp-per-size:      int, vpp数，每个pp stage下有几个vpp
--src-model-format:  str, ms模型格式，可选ckpt、safetensors，不传默认ckpt
--noop:              int, 空层所在位置，当前仅支持在最后，不传默认没有空层
--ms-path:           str, ms模型路径，此路径下为rank_xx的模型文件夹
--param-map-dir:     str, param_map及args.pt对应的文件夹
--megatron-path:     str, 需要保存的转换后的megatron模型路径
```

#### 示例：

```shell
python convert_ckpt2pt_v4_update_newer_v2.py --num-layers 8 \
    --dp-size 1 \
    --tp-size 2 \
    --pp-size 4 \
    --vpp-per-stage 2 \
    --noop 62 63 \
    --ms-path /home/ma-user/work/ckpt2pt/ms_ckpt \
    --param-map-path /home/ma-user/work/work/ckpt2pt/param_map/buffers \
    --megatron-path /home/ma-user/work/ckpt2pt/megatron_pt
```
