## tp/pp/dp转换脚本

### 前置条件：

1. 只支持ckpt为新命名方式mp_rank_xx_xxx下的tp/pp/dp转换
2. 非分布式优化器下训练保存的ckpt，可以直接运行该脚本进行转换；分布式优化器下训练保存的ckpt,其参数以桶形式在dp域内进行分布式保存，需要先使用combine_ckpt_dp_zero.py脚本进行参数合并（dp=1也需要执行此项操作），之后使用合并后的ckpt运行该脚本进行转换。
3. 目标ckpt路径下需要预置strategy文件夹，表示要转换到何种并行策略

### 功能概述：

transform_ckpt_helper.py主要实现了3个动作：

1. `mp_rank_to_global_rank()`: 把新命名方式的ckpt(mp_rank_xx_xxx)转换成旧命名方式的ckpt
2. `transform_ckpt_to_new_strategy()`: 使用ms框架的`mindspore.transform_checkpoints`进行策略转换
3. `global_rank_to_mp_rank()`: 把旧命名方式的ckpt恢复成新命名方式

### 转换前的源ckpt和目标ckpt文件

源ckpt文件目录结构：

```bash
src_dir/
    ├── iter_0000005
    │   ├── mp_rank_00_000
    |   |    └── model_optim_rng.ckpt
    │   └── mp_rank_00_001
    |        └── model_optim_rng.ckpt
    ├── latest_checkpointed_iteration.txt
    └── strategy
        ├── strategy0.ckpt
        └── strategy1.ckpt
```

目标ckpt文件目录结构：

```bash
dst_dir/
    └── strategy
        ├── strategy0.ckpt
        ├── strategy1.ckpt
        ├── strategy2.ckpt
        └── strategy3.ckpt
```

### 参数说明：

- --src-dir         :str,被转换的源ckpt路径
- --dst-dir         :str,转换到目标ckpt路径
- --output-format   :str,权重文件格式，支持 [`ckpt`]。默认值：`ckpt`
- --src-dp-size     :int,源ckpt的dp数。默认值：1
- --src-dp-size     :int,源ckpt的cp数。默认值：1
- --dst-dp-size     :int,目标ckpt的dp数。默认值：1
- --dst-pp-size     :int,目标ckpt的pp数。默认值：1
- --dst-tp-size     :int,目标ckpt的tp数。默认值：1
- --iteration       :int,被转换的iteration。默认值：-1

### 示例：

```shell
python transform_ckpt_helper.py --src-dir output_dp2tp1pp1 \
    --dst-dir output_dp2tp1pp2 \
    --src-dp-size 2 \
    --src-cp-size 1 \
    --dst-dp-size 2 \
    --dst-tp-size 1 \
    --dst-pp-size 2 \
    --iteration 5
```

