# 模型训练系统测试

## Mixtral-8x7B训练测试

### 硬件要求

训练的最低硬件配置:

| 硬件 |       配置       |
| :--: | :--------------: |
| NPU | 16 x Ascend NPUs |

### 准备工作

1. 克隆仓库到本地服务器

```shell
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout -f 9de386d0
# 同步megatron的bug fix，在megatron创建__init__.py文件
touch megatron/__init__.py
mkdir logs
mkdir model_from_hf
mkdir dataset
mkdir ckpt
```

2. 搭建环境

```bash
# python3.8
conda create -n test python=3.8
conda activate test

# 安装 torch 和 torch_npu
pip install torch-2.1.0-cp38-cp38m-manylinux2014_aarch64.whl
pip install torch_npu-2.1.0*-cp38-cp38m-linux_aarch64.whl
pip install apex-0.1_ascend*-cp38-cp38m-linux_aarch64.whl

# 修改 ascend-toolkit 路径
# export LD_LIBRARY_PATH=
# export ASCEND_TOOLKIT_HOME=
# export LD_LIBRARY_PATH=
# export LD_LIBRARY_PATH=
# export PYTHONPATH=
# export PATH=
# export ASCEND_AICPU_PATH=
# export ASCEND_OPP_PATH=
# export TOOLCHAIN_HOME=
# export ASCEND_HOME_PATH=
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 安装加速库
git clone https://gitee.com/ascend/AscendSpeed.git
cd AscendSpeed
git checkout master
pip install -r requirements.txt 
pip3 install -e .
cd ..

# 安装其余依赖库
pip install -r requirements.txt 
```

3. 下载 Mixtral-8x7B 的 [词表和tokenizer](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/tree/main)

```shell
#!/bin/bash
cd ./model_from_hf/
git lfs install
git clone https://huggingface.co/mistralai/Mixtral-8x7B-v0.1
mv Mixtral-8x7B-v0.1 Mixtral-8x7B
cd ..
```

### 模型训练

1. 准备数据集

下载 Mixtral-8x7B [数据集](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet)

```shell
# 下载数据
cd ./dataset
wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
cd ..
# 处理数据   
mkdir ./dataset/Mixtral-8x7B/
python ./tools/preprocess_data.py \
    --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
    --tokenizer-name-or-path ./model_from_hf/Mixtral-8x7B/ \
    --output-prefix ./dataset/Mixtral-8x7B/alpaca \
    --workers 4 \
    --log-interval 1000 \
    --tokenizer-type PretrainedFromHF
```

2. 配置 Mixtral-8x7B 预训练脚本：***pretrain_mixtral.sh***

```
# 拷贝mixtral训练所用脚本到Megatron-LM目录下
cp ../AscendSpeed/tests_extend/system_tests/pretrain_mixtral.sh .

```

```shell
# 按照如下内容修改pretrain_mixtral.sh测试脚本文件
# 设置 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh 

# 根据实际情况配置词表、数据集、模型参数保存路径
DATA_PATH="./dataset/Mixtral-8x7B/alpaca_text_document"
TOKENIZER_MODEL="./model_from_hf/Mixtral-8x7B/"
CKPT_SAVE_DIR="./ckpt/Mixtral-8x7B/"

# 根据分布式集群实际情况配置分布式参数
GPUS_PER_NODE=8
MASTER_ADDR="your master node IP"
MASTER_PORT=6000
NNODES=2
NODE_RANK="current node id"
WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))

# 根据实际需要设置训练并行策略
TP=2
PP=4
EP=2
```

3. 开启确定性计算

在pretrain_mixtral.sh脚本添加 `export HCCL_DETERMINISTIC=TRUE`

另外，在pretrain_gpt.py中添加代码
```
# ptdbdg_ascced 参见 https://gitee.com/ascend/tools/blob/master/ptdbg_ascend/README.md
from ptdbdg_ascced import seed_all
seed_all(mode=True)
```


4. 启动 Mixtral-8x7B 预训练脚本: ***pretrain_mixtral.sh***

```shell
bash pretrain_mixtral.sh
```

**注意**：如果使用多机训练，需要设置多机数据共享，非主节点通过数据共享读取主节点数据。或者，直接将主节点生成的数据复制到非主节点。
