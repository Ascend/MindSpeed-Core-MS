## DeepSeek V3

### 模型描述

**DeepSeek-V3** 是深度求索公司（DeepSeek）开发的一款先进的人工智能模型，主要用于处理自然语言任务（如对话、文本生成等）和多模态任务（如图像、文本结合的分析）。它基于大规模数据训练，具备强大的理解和生成能力，能够像人类一样理解上下文并生成流畅的文本。DeepSeek-V3 不仅支持多种语言，还可以根据特定需求进行定制化训练，适用于智能客服、内容创作、教育辅助等多种场景。相比前代版本，它在性能、速度和资源效率上都有显著提升，同时注重安全性和合规性，确保生成的内容符合规范。简单来说，DeepSeek-V3 是一款功能强大、应用广泛且易于集成的人工智能工具。

### 仓库介绍

```shell
DeepSeek-V3
    ├── README.md
    ├── ckpt_convert_deepseek3_hf2mcore.sh          # 权重格式转换脚本(.safetensor -> .pt)
    ├── ckpt_convert_deepseek3_mcore2hf.sh          # 权重格式转换脚本(.pt -> .safetensor)
    ├── ckpt_convert_deepseek3_merge_lora2hf.sh     # lora权重格式转换脚本
    ├── convert_ckpt_deepseek3.py
    ├── convert_ckpt_deepseek3_mcore2hf.py
    ├── data_convert_deepseek3_instruction.sh       # SFT微调数据处理脚本
    ├── data_convert_deepseek3_pretrain.sh          # 预训练数据处理脚本
    ├── pretrain_deepseek3_671b_4k_ptd.sh           # 671b预训练任务启动脚本
    ├── tune_deepseek3_671b_4k_full_ptd.sh          # 671b微调任务启动脚本
    ├── tune_deepseek3_671b_4k_lora_ptd.sh          # 671b Lora微调任务启动脚本
    └── tune_deepseek3_671b_4k_qlora_ptd.sh
```

## 预训练

### 数据集准备

以Enwiki数据集为例:

- 数据集下载地址：[ENWIKI](https://gitee.com/link?target=https%3A%2F%2Fhuggingface.co%2Fdatasets%2Flsb%2Fenwiki20230101)

  数据集下载可以基于网页直接下载，也可以基于命令行下载，比如：

  ```shell
  mkdir dataset
  cd dataset/
  wget https://huggingface.co/datasets/lsb/enwiki20230101/resolve/main/data/train-00000-of-00042-d964455e17e96d5a.parquet
  cd ..
  ```

- 分词模型下载：通过HuggingFace网站下载[DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3/tree/main)的分词，或通过命令进行下载

  ```shell
  mkdir -p model_from_hf/deepseek3-hf/
  cd model_from_hf/deepseek3-hf/
  wget https://huggingface.co/deepseek-ai/DeepSeek-V3/resolve/main/tokenizers.json
  wget https://huggingface.co/deepseek-ai/DeepSeek-V3/resolve/main/tokenizers_config.json
  cd ../../
  ```

- 使用**MindSpeed-LLM/examples/mcore/deepseek3/data_convert_deepseek3_pretrain.sh**预处理脚本进行数据格式转化，脚本中的**input**、**tokenizer-name-or-path**两个[参数](./pretrain_dataset.md)需要对应修改为下载的数据集和分词路径, 。

  ```shell
  cd MindSpeed-LLM
  python ./preprocess_data.py \
      --input ./dataset/train-00000-of-00042-d964455e17e96d5a.parquet \
      --tokenizer-name-or-path ./model_from_hf/deepseek3-hf/ \
      --tokenizer-type PretrainedFromHF \
      --handler-name GeneralPretrainHandler \
      --output-prefix ./dataset/enwiki \
      --json-keys text \
      --workers 4 \
      --log-interval 1000
  ```

- 处理后的目录结构如下，后续拉起任务时可以试用/dataset/enwiki/enwiki_text_document作为数据集路径

  ```shell
  /dataset/enwiki
    ├── enwiki_text_document.bin
    └── enwiki_text_document.idx
  ```

### 脚本启动

在**examples/mcore/deepseek3/pretrain_deepseek3_671b_4k_ptd.sh**模型拉起脚本中，将DATA_PATH、TOKENIZER_PATH字段内容修改为刚刚准备的数据集目录和分词目录，如下：

```shell
# e.g:
# DATA_PATH="./dataset/enwiki/enwiki_text_document"
# TOKENIZER_PATH="./model_from_hf/deepseek3-hf/"

cd MindSpeed-LLM
bash examples/mcore/deepseek3/pretrain_deepseek3_671b_4k_ptd.sh
```

## SFT微调

### 数据集准备

以Alpaca数据集为例:

- 数据集下载地址：[ALPACA](https://gitee.com/link?target=https%3A%2F%2Fhuggingface.co%2Fdatasets%2Flsb%2Fenwiki20230101)

  数据集下载可以基于网页直接下载，也可以基于命令行下载，比如：

  ```shell
  mkdir dataset
  cd dataset/
  wget https://huggingface.co/datasets/tatsu-lab/alpaca/blob/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
  cd ..
  ```

- 分词模型下载：通过HuggingFace网站下载[DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3/tree/main)的分词，或通过命令进行下载

  ```shell
  mkdir -p model_from_hf/deepseek3-hf/
  cd model_from_hf/deepseek3-hf/
  wget https://huggingface.co/deepseek-ai/DeepSeek-V3/resolve/main/tokenizers.json
  wget https://huggingface.co/deepseek-ai/DeepSeek-V3/resolve/main/tokenizers_config.json
  cd ../../
  ```

- 使用**MindSpeed-LLM/examples/mcore/deepseek3/data_convert_deepseek3_instruction.sh**预处理脚本进行数据格式转化，脚本中的**input**、**tokenizer-name-or-path**两个[参数](./alpaca_dataset.md)需要对应修改为下载的数据集和分词路径, 。

  ```shell
  cd MindSpeed-LLM
  python ./preprocess_data.py \
      --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
      --tokenizer-name-or-path ./model_from_hf/deepseek3-bf16-hf \
      --output-prefix ./finetune_dataset/alpaca \
      --handler-name AlpacaStyleInstructionHandler \
      --tokenizer-type PretrainedFromHF \
      --workers 4 \
      --log-interval 1000 \
      --overwrite-cache \
      --prompt-type deepseek3
  ```

- 处理后的目录结构如下，后续拉起任务时可以试用/finetune_dataset/alpaca/alpaca作为数据集路径

  ```shell
  /dataset/enwiki
    ├── alpaca_packed_attention_mask_document.bin
    ├── alpaca_packed_attention_mask_document.idx
    ├── alpaca_packed_input_ids_document.bin
    ├── alpaca_packed_input_ids_document.idx
    ├── alpaca_packed_labels_document.bin
    └── alpaca_packed_labels_document.idx
  ```

###  

### 权重转换准备

- 权重下载 [DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3/tree/main)

  数据集下载可以基于网页直接下载，也可以基于命令行下载，比如：

  ```shell
  mkdir -p model_from_hf/deepseek3-bf16-hf
  cd model_from_hf/deepseek3-bf16-hf
  wget https://huggingface.co/deepseek-ai/DeepSeek-V3/resolve/main/config.json
  wget https://huggingface.co/deepseek-ai/DeepSeek-V3/resolve/main/tokenizers.json
  wget https://huggingface.co/deepseek-ai/DeepSeek-V3/resolve/main/tokenizers_config.json
  wget https://huggingface.co/deepseek-ai/DeepSeek-V3/resolve/main/model.safetensors.index.json
  for i in {000..163}; do wget https://huggingface.co/deepseek-ai/DeepSeek-V3/resolve/main/model-00${i}-of-000163.safetensors;done
  cd ../../
  ```

- 权重转换

  使用**MindSpeed-LLM/examples/mcore/deepseek3/ckpt_convert_deepseek3_hf2mcore.sh**权重转换脚本将huggingface权重转化为megatron权重，脚本中的 **load-dir** 参数需要对应修改为下载权重的路径, 。

  ```shell
  cd MindSpeed-LLM
  python examples/mcore/deepseek3/convert_ckpt_deepseek3.py \
      --moe-grouped-gemm \
      --target-tensor-parallel-size 2 \
      --target-pipeline-parallel-size 8 \
      --target-expert-parallel-size 32 \
      --load-dir ./model_from_hf/deepseek3-bf16-hf \
      --save-dir ./model_weights/deepseek3-mcore \
      --num-layers 64 \
      --num-nextn-predict-layers 1 \
      --num-layers-per-virtual-pipeline-stage 2 \
      --noop-layers 47,62,63
      # --num-layer-list, --moe-tp-extend-ep 等参数根据任务需要进行配置
  ```

  | 参数                                    | 说明                                                         |
  | --------------------------------------- | ------------------------------------------------------------ |
  | --moe-grouped-gemm                      | 当每个专家组有多个专家时，可以使用Grouped GEMM功能来提高利用率和性能。 |
  | --target-tensor-parallel-size           | 张量并行度，默认值为1。                                      |
  | --target-pipeline-parallel-size         | 流水线并行度，默认值为1。                                    |
  | --target-expert-parallel-size           | 专家并行度，默认值为1。                                      |
  | --num-layers-per-virtual-pipeline-stage | 虚拟流水线并行，默认值为None,。<br />注意参数--num-layers-per-virtual-pipeline-stage 和 --num-layer-list 不能同时使用。 |
  | --load-dir                              | 已经反量化为bf16数据格式的huggingface权重。                  |
  | --save-dir                              | 转换后的megatron格式权重的存储路径。                         |
  | --num-nextn-predict-layers              | MTP层的层数。如不需要MTP层，可设置为0。最大可设置为1。<br />默认值为0。 MTP层权重默认存储在最后一个pp stage。 |
  | --num-layers                            | 模型层数，该层数不包含MTP层。默认值为61。<br />如配置空操作层，num-layers的值应为总层数（不包含MTP层）加上空操作层层数。 |
  | --noop-layers                           | 自定义空操作层。与--num-layer-list互斥，二者选其一使用。默认值为None。 |

### 脚本启动

在**examples/mcore/deepseek3/tune_deepseek3_671b_4k_full_ptd.sh**模型拉起脚本中，将DATA_PATH、TOKENIZER_PATH、CKPT_LOAD_DIR字段内容修改为刚刚准备的数据集目录、分词目录和权重目录，如下：

```shell
# e.g:
# DATA_PATH="./dataset/enwiki/enwiki_text_document"
# TOKENIZER_PATH="./model_from_hf/deepseek3-hf/"
# CKPT_LOAD_DIR=“./model_weights/deepseek3-mcore”

cd MindSpeed-LLM
bash examples/mcore/deepseek3/tune_deepseek3_671b_4k_full_ptd.sh
```