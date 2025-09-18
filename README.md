<p align="center"> <img src="docs/logo.png" height="103px" width="700px"> </p>

<p align="center">
    <a href="https://gitee.com/ascend/MindSpeed-Core-MS/blob/dev/LICENSE">
    <a href="https://gitee.com/ascend/MindSpeed-Core-MS/blob/dev/LICENSE">
        <img alt="Badge" src="https://img.shields.io/badge/License-MIT-blue.svg">
    </a>
    <a href="https://gitee.com/ascend/MindSpeed-Core-MS">
        <img alt="Documentation" src="https://img.shields.io/website/http/huggingface.co/docs/transformers/index.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a>
        <img src="https://app.codacy.com/project/badge/Grade/1710faac5e634acaabfc26b0a778cdde">
    </a>
</p>

MindSpeed-Core-MS 是链接华为自研AI框架[MindSpore](https://www.mindspore.cn/install/)+华为[昇腾大模型加速解决方案MindSpeed](https://www.hiascend.cn/software/mindspeed)的重要组件，旨在提供华为全栈易用的端到端的自然语言模型以及多模态模型训练解决方案。MindSpeed-Core-MS 提供了代码一键适配功能，可帮助用户将模型使能加速库MindSpeed/MindSpeed-LLM/MindSpeed-MM以及三方库依赖由PyTorch无缝切换为MindSpore，以此获得更极致的性能体验。另外，MindSpeed-Core-MS 也提供了动态图调试工具，使用户在分布式训练场景下更容易地进行代码调试和debug。

---

# 配套版本与支持模型

## 配套版本

MindSpeed-Core-MS的依赖配套如下表，安装步骤参考[基础安装指导](./docs/INSTALLATION.md)。

| 依赖软件         | 版本                                                                                                                               |
| ---------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| 昇腾NPU驱动固件  | [在研版本](https://www.hiascend.com/hardware/firmware-drivers/community?product=1&model=30&cann=8.0.RC3.alpha002&driver=1.0.26.alpha) |
| 昇腾 CANN        | [在研版本](https://www.hiascend.com/zh/developer/download/community/result?module=cann)                                               |
| MindSpore        | [master](https://www.mindspore.cn/install/)                                                                                            |
| MSAdapter        | [在研版本](https://openi.pcl.ac.cn/OpenI/MSAdapter.git)                                                                               |
| Python           | >=3.9                                                                                                                              |
| Python三方库依赖 | requirements.txt                                                                                                                   |

注：Python 三方库依赖文件`requirements.txt`列举的是模型训练所需要的Python三方库。

# 使用指南

按照[基础安装指导](./docs/INSTALLATION.md)完成相关基础依赖安装后，用户可根据具体使用场景（MindSpeed-LLM/MindSpeed-MM/MindSpeed-RL）进行相应的自动适配。在自动适配前，请确保：

- 基础依赖已安装
- 所部署容器网络可用，python已安装
- git已完成配置，可以正常进行clone操作

## 仓库拉取

执行以下命令拉取MindSpeed-Core-MS代码仓，并安装Python三方依赖库

```shell
git clone https://gitee.com/ascend/MindSpeed-Core-MS.git -b master
cd MindSpeed-Core-MS
pip install -r requirements.txt
```

## 一键适配

MindSpeed-Core-MS提供了一键适配命令脚本，集成了基于MindSpeed进行模型训练的相关代码仓拉取、代码自动适配、环境变量设置等功能，用户根据使用场景（大语言模型/多模态模型/强化学习）执行相应命令即可完成一键自动适配。

**注意：各个使用场景下代码仓不能混用，建议针对各场景使用独立目录。**

### 大语言模型：MindSpeed-LLM
执行以下操作进行一键适配后，用户即可进行大语言模型训练：

```shell
source auto_convert.sh llm
cd MindSpeed-LLM
```

此处提供以下大语言模型训练拉起流程作为参考。

- [**DEEPSEEK-V3预训练**](https://gitee.com/ascend/MindSpeed-LLM/blob/master/examples/mindspore/deepseek3/README.md)

若在环境中`PYTHONPATH`等环境变量失效（例如退出容器后再进入等），可执行如下命令重新设置环境变量

```shell
# 在MindSpeed-Core-MS目录下执行
MindSpeed_Core_MS_PATH=$(pwd)
export PYTHONPATH=${MindSpeed_Core_MS_PATH}/MSAdapter:${MindSpeed_Core_MS_PATH}/MSAdapter/msa_thirdparty:${MindSpeed_Core_MS_PATH}/Megatron-LM:${MindSpeed_Core_MS_PATH}/MindSpeed:${MindSpeed_Core_MS_PATH}/MindSpeed-LLM:$PYTHONPATH
```

---

### 多模态模型：MindSpeed-MM
执行以下操作进行一键适配后，用户即可进行多模态模型训练：

```shell
source auto_convert.sh mm
cd MindSpeed-MM
```

此处提供以下多模态模型训练拉起流程作为参考。

- [**Qwen2.5VL 微调**](https://gitee.com/ascend/MindSpeed-MM/blob/master/examples/mindspore/qwen2.5vl/README.md)

若在环境中`PYTHONPATH`等环境变量失效（例如退出容器后再进入等），可执行如下命令重新设置环境变量

```shell
# 在MindSpeed-Core-MS目录下执行
MindSpeed_Core_MS_PATH=$(pwd)
export PYTHONPATH=${MindSpeed_Core_MS_PATH}/MSAdapter:${MindSpeed_Core_MS_PATH}/MSAdapter/msa_thirdparty:${MindSpeed_Core_MS_PATH}/Megatron-LM:${MindSpeed_Core_MS_PATH}/MindSpeed:${MindSpeed_Core_MS_PATH}/MindSpeed-MM/:$PYTHONPATH
```

---

### 强化学习：MindSpeed-RL
执行以下操作进行一键适配后，用户即可进行强化学习模型训练：

```shell
# deepseek v3-r1-zero、qwen25-7b-r1-zero
source auto_convert_rl.sh
cd MindSpeed-RL
```

此处提供以下强化模型训练拉起流程作为参考。

- [**Qwen2.5-7B GRPO**](./docs/GRPO_Qwen25_7B.md)

若在环境中`PYTHONPATH`等环境变量失效（例如退出容器后再进入等），可执行如下命令重新设置环境变量

```shell
# 在MindSpeed-Core-MS目录下执行
MindSpeed_Core_MS_PATH=$(pwd)
export PYTHONPATH=${MindSpeed_Core_MS_PATH}/MSAdapter:${MindSpeed_Core_MS_PATH}/MSAdapter/msa_thirdparty:${MindSpeed_Core_MS_PATH}/Megatron-LM:${MindSpeed_Core_MS_PATH}/MindSpeed:${MindSpeed_Core_MS_PATH}/MindSpeed-LLM:${MindSpeed_Core_MS_PATH}/transformers/src/:${MindSpeed_Core_MS_PATH}/vllm/:${MindSpeed_Core_MS_PATH}/vllm-ascend/:${MindSpeed_Core_MS_PATH}/accelerate/src/:${MindSpeed_Core_MS_PATH}/safetensors_dir/:${MindSpeed_Core_MS_PATH}/huggingface_hub/src/:${MindSpeed_Core_MS_PATH}/MindSpeed-RL/:$PYTHONPATH
```

# 常见问题 FAQ

相关FAQ请参考链接：[FAQ](./docs/FAQ.md)

# 文档目录

详细文档请参考：[docs](./docs)

---

# 版本维护策略

MindSpeed-Core-MS版本有以下五个维护阶段：

| **状态**      | **时间** | **说明**                                                                                                                       |
| ------------------- | -------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| 计划                | 1—3 个月      | 计划特性                                                                                                                             |
| 开发                | 3 个月         | 开发特性                                                                                                                             |
| 维护                | 6-12 个月      | 合入所有已解决的问题并发布版本，针对不同的MindSpeed-Core-MS版本采取不同的维护策略，常规版本和长期支持版本维护周期分别为6个月和12个月 |
| 无维护              | 0—3 个月      | 合入所有已解决的问题，无专职维护人员，无版本发布                                                                                     |
| 生命周期终止（EOL） | N/A            | 分支不再接受任何修改                                                                                                                 |

MindSpeed-Core-MS已发布版本维护策略：

| **MindSpeed-Core-MS版本** | **维护策略** | **当前状态** | **发布时间** | **后续状态** | **EOL日期** |
| ------------------------------- | ------------------ | ------------------ | ------------------ | ------------------ | ----------------- |
| feature-0.2                     | Demo               | 维护               |          2025.4.15          |          预计2025/09/30起无维护          |                 |
| r0.3.0                     | Demo               | 维护               |          2025.7.30          |          预计2025/12/30起无维护          |                 |
|             master                    | /               | 开发               |       预计2025.12.30             |          /          |          /         |

---

# 安全声明

[MindSpeed-Core-MS安全声明](./docs/SECURITYNOTE.md)

# 免责声明

## 致MindSpeed-Core-MS使用者

1. MindSpeed-Core-MS提供的模型仅供您用于非商业目的。
2. 对于各模型，MindSpeed-Core-MS平台仅提示性地向您建议可用于训练的数据集，华为不提供任何数据集，如您使用这些数据集进行训练，请您特别注意应遵守对应数据集的License，如您因使用数据集而产生侵权纠纷，华为不承担任何责任。
3. 如您在使用MindSpeed-Core-MS过程中，发现任何问题（包括但不限于功能问题、合规问题），请在Gitee提交issue，我们将及时审视并解决。
