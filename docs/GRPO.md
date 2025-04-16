# RL使用

## 转换脚本

test_convert_rl.sh 脚本自动执行以下设置过程：

1. 克隆所需仓库：
   - MindSpeed-LLM（commit id：421ef7bcb83fb31844a1efb688cde71705c0526e）
   - MindSpeed（commit id：0dfa0035ec54d9a74b2f6ee2867367df897299df）
   - MindSpeed-RL（分支：2.0.0）
   - Megatron-LM（分支：core_r0.8.0）
   - msadapter（分支：master）
   - vllm（分支：v0.7.3）
   - vllm-ascend（commit id：0713836e95fe993feefe334945b5b273e4add1f1）
   - transformers（分支：v4.47.0）

2. 使用 transfer.py 工具运行代码转换

## 使用说明

1. 克隆 MindSpeed-Core-MS 仓库：

   ```shell
   git clone -b feature-0.2 https://gitee.com/ascend/MindSpeed-Core-MS.git
   cd MindSpeed-Core-MS/
   ```

2. 使转换脚本可执行并运行：

   ```shell
   chmod +x test_convert_rl.sh
   ./test_convert_rl.sh
   ```

3. 数据集权重准备
   - 参考MindSpeed-RL仓库下[grpo.md](https://gitee.com/ascend/MindSpeed-RL/blob/master/docs/algorithms/grpo.md)

## 脚本启动

1. 启动任务前需要确保运行：

   ```bash
   source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=0
   source /usr/local/Ascend/ascend-toolkit/latest/env/ascend_env.sh
   ```

2. 设置环境变量

   ```bash
   MindSpeed_Core_MS_PATH=$(pwd)
   Ascend_PATH=/usr/loacl/Ascend/ascend-toolkit/latest/
   export PYTHONPATH=${MindSpeed_Core_MS_PATH}/msadapter/mindtorch:${MindSpeed_Core_MS_PATH}/MindSpeed-LLM/:${MindSpeed_Core_MS_PATH}/MindSpeed/: \
                     ${MindSpeed_Core_MS_PATH}/Megatron-LM/:${MindSpeed_Core_MS_PATH}/vllm/:${MindSpeed_Core_MS_PATH}/vllm-ascend/: \
                     ${MindSpeed_Core_MS_PATH}/transformers/src/: \
                     ${Ascend_PATH}/python/site-packages:${Ascend_PATH}/opp/built-in/op_impl/ai_core/tbe:
   ```

3. 在**MindSpeed-RL/configs/XXX.yaml**中，将tokenizer_name_or_path、data_path、load字段内容修改为刚刚准备的分词目录、数据集目录和权重目录，如下：

   ```shell
   # e.g:
   tokenizer_name_or_path: ./data/models/Qwen2.5-7B
   data_path: ./dataset/data
   load: ./ckpt
   ```

## 启动命令

1. qwen25-7b-r1-zero

   ```shell
   cd MindSpeed-RL
   python cli/train_grpo.py --config-name grpo_trainer_qwen25_7b | tee logs/grpo_trainer_qwen25_7b.log
   ```

2. deepseek v3-r1-zero

   ```shell
   cd MindSpeed-RL
   python cli/train_grpo.py --config-name r1_zero_deepseekv3_671b | tee logs/r1_zero_deepseekv3_671b.log
   ```