# 基于 MindSpore AI 框架的 GRPO-DeepSeek-R1-Qwen2.5-7B 使用指南

Group Relative Policy Optimization (GRPO) 是 Deepseek-Math 中提出的训练方法，它移除了 PPO 中对 Critic 模型的依赖，而是通过计算同一 prompt 多次重复采样输出的相对奖励来估计优势函数，这一创新大大减少了显存占用，提高了算法在强化学习任务中的效率。

在 GRPO 方法中包含了三个关键模型：Actor，Reference，Reward。其中 Actor 和 Reference 模型是通过 SFT 后得到的策略模型，而 Reward 模型则是通过规则奖励来评估。GRPO 的核心训练目标是优化 Actor 模型的策略，使其在执行强化学习任务时能够产生更优的动作序列，更符合任务目标的预期。

本篇工作基于 MindSpore AI 架使用 Qwen2.5-7B 模型复现 GRPO-DeepSeek-R1 在 Math 领域的工作。

## 依赖的三方库版本

- MindSpeed-LLM（commit id：71c5af4d72078d826fd93fec6980004f0de51132）
- MindSpeed（分支：core_r0.8.0, commit id：31aaf3d4ca86234b15f4a5d3af20bd6df06e7d45）
- MindSpeed-RL（分支：master, commit id：559db0856891e5f8504a0b21d4b26969a82241df）
- Megatron-LM（分支：core_r0.8.0）
- MSAdapter（分支：master）
- vllm（分支：v0.7.3）
- vllm-ascend（commit id：0713836e95fe993feefe334945b5b273e4add1f1）
- transformers（分支：v4.47.0）
- accelerate（分支：v1.6.0）
- safetensors（版本：0.5.1）
- huggingface_hub（分支：v0.29.2）

## 模型选择

 [Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B) 模型指令遵从度高，有一定概率能引导模型输出 `<think>...</think><answer>...$\boxed{}</answer>` 格式回复，训练曲线符合预期，在评测集上提升较大。

## 数据预处理

以 DeepScaler 为例：

数据集下载地址：[DeepScaler](https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset/tree/main)

数据集下载可以基于网页直接下载，也可以基于命令行下载，比如：

```shell
# 读取deepscaler数据集
mkdir dataset
cd dataset/
wget https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset/resolve/main/deepscaler.json --no-check
cd ..
```

数据预处理的 yaml 配置文件放置于 `MindSpeed-RL/configs/datasets` 文件夹下，通过以下命令进行数据集预处理：

```shell
# 读取configs/datasets/deepscaler.yaml文件
bash examples/data/preprocess_data.sh deepscaler
```

数据集处理配置可以根据需求自行配置，以下是数据集处理的 yaml 文件中基础参数的介绍：

- `input`：数据集的路径，需指定具体文件，例如/datasets/deepscaler.json
- `tokenizer_type`：指定分词器的类型，例如 HuggingFaceTokenizer 使用 Hugging Face 库提供的分词器来对文本进行分词处理;
- `tokenizer_name_or_path`：指定分词器的名称或路径;
- `output_prefix`：输出结果的前缀路径，例如 /datasets/data;
- `workers`：设置处理数据时使用的 worker 数;
- `prompt_type`: 用于指定对话模板，能够让 base 模型微调后能具备更好的对话能力，prompt-type 的可选项可以在 configs/model/templates.json 文件内查看;
- `log_interval`：设置日志记录的间隔，每处理多少条数据时记录一次日志，用于监控数据处理的进度和状态;
- `handler_name`：指定处理数据的处理器名称；
- `seq_length`：设置数据预处理最大序列长度，超过了会过滤掉;

## 权重转换

根据 GRPO 算法要求，Actor 和 Reference 模型应该使用 SFT 微调后的模型进行初始化，Reward 模型应该使用规则奖励。GRPO 算法模型权重均使用 Megatron-mcore 格式，其他格式的权重需要进行模型权重转换。可参考[权重转换部分](https://gitee.com/ascend/MindSpeed-RL/blob/master/docs/algorithms/grpo.md)

以 Qwen2.5-7B 模型的权重转换脚本为参考，权重转换步骤如下:

### 获取权重文件

hf 权重文件可从 Huggingface 网站获取，请根据模型的使用场景灵活选择，在此以 [Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B) 为例。

### hf 转 mcore

在训练前，需要将 Hugging Face 权重转换成Mcore格式，示例脚本启动命令和配置参数如下：

```bash
# 脚本中路径请按真实情况配置
bash examples/ckpt/ckpt_convert_qwen25_hf2mcore.sh
```

> 注：这里会调用到 MindSpeed-LLM 仓，进行权重转换前需先确认环境变量已配备完毕。

配置参数介绍

- `use-mcore-models`：启用 MCore 模型；
- `model-type`：指定模型类型，如 GPT；
- `load-model-type`：指定加载模型的类型，如 hf（Hugging Face）；
- `save-model-type`：指定保存模型的类型，如 mg；
- `target-tensor-parallel-size`：设置目标张量并行大小；
- `target-pipeline-parallel-size`：设置目标流水线并行大小；
- `add-qkv-bias`：是否进行 QKV 偏置；
- `load-dir`：加载 Hugging Face 权重的路径；
- `save-dir`：保存转换后权重的路径；
- `tokenizer-model`：分词器模型文件的路径；
- `model-type-hf`：指定 Hugging Face 模型类型，如 llama2；
- `params-dtype`：指定参数的数据类型，如 bf16。

### mcore 转 hf

训练结束后，如果需要将生成的mcore格式权重转换回 Hugging Face 格式，可以参照以下示例脚本命令及脚本参数：

```shell
# 脚本中路径请按真实情况配置
bash examples/ckpt/ckpt_convert_qwen25_mcore2hf.sh
```

配置参数与上文一致，但需注意以下事项：

- 权重转换转回 Hugging Face 格式时，tp 和 pp 配置需配置为1；
- load-model-type 参数配置为 mg，save-model-type 参数配置为 hf ;
- save-dir 路径需要填入原始 HF 模型路径，新权重会存于 HF 原始权重文件下的 mg2hg 目录下，如/qwen2.5_7b_hf/mg2hg/

## 脚本启动

1. 启动任务前需要确保运行：

   ```bash
   source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=0
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   ```

2. 在**MindSpeed-RL/configs/*_qwen25_7b.yaml**中，将`tokenizer_name_or_path`、`data_path`、`load`字段内容修改为刚刚准备的分词目录、数据集目录和权重目录，如下：

   ```shell
   # e.g:
   tokenizer_name_or_path: ./data/models/Qwen2.5-7B
   data_path: ./dataset/data
   load: ./ckpt
   ```

   > 注：yaml 中需添加参数`megatron_training.ai_framework：mindspore`

   卡数配置参数介绍

    - `rl_config.actor_resource.num_npus`：actor 需使用的卡数，如 4；
    - `rl_config.reference_resource.num_npus`：reference 需使用的卡数，如 2；
    - `rl_config.reward_resource.num_npus`：reward 需使用的卡数，如 2。

3. 任务拉起

    需保证任务拉起时可用的 npu 数量满足`2.`中yaml配置所需的卡数要求

- 若`2.`中 yaml 配置的为**单机**参数，可执行如下命令拉起任务：

     ```shell
     cd MindSpeed-RL
     bash examples/grpo/grpo_trainer_qwen25_7b.sh
     ```

     > 注：脚本中`--config-name`应修改为`2.`中设置的 yaml 文件名称, 如`grpo_trainer_qwen25_7b`

- 若`2.`中 yaml 配置的为**多机**参数，可执行如下命令拉起任务：

  **主节点执行：**

  ```shell
  bash examples/r1/qwen25/r1_zero_qwen25_7b_master.sh
  ```

  > 注：脚本中`DEFAULT_YAML`应修改为`2.`中设置的 yaml 名称，根据实际机器设置`NNODES`和`NPUS_PER_NODE`

  配置参数介绍

    - `DEFAULT_YAML`：指定参数配置的 yaml 名称，如 r1_zero_qwen25_7b.yaml；
    - `NNODES`：共使用多少节点训练，如 2；
    - `NPUS_PER_NODE`：每个节点有多少张卡，如 8。

  **从节点执行：**

  ```shell
  bash examples/r1/qwen25/r1_zero_qwen25_7b_worker.sh
  ```

  > 注：脚本中`NNODES`和`NPUS_PER_NODE`应于主节点配置一致，`MASTER_ADDR`应为主节点`ip`

  配置参数介绍

    - `NNODES`：共使用多少节点训练，如 2；
    - `NPUS_PER_NODE`：每个节点有多少张卡，如 8；
    - `MASTER_ADDR`：主节点 IP 地址。
