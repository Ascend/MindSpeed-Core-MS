#### ！！！【重要通知】MSAdapter仓因平台网络问题暂不可用，或导致MindSpore-Core-MS仓库MindSpore后端一键启动暂不可用！！！

## 安装指导

请参考首页[依赖信息](https://gitcode.com/ascend/MindSpeed-Core-MS/blob/dev/README.md)选择下载对应依赖版本。除驱动固件安装在物理机系统外，其余软件建议使用容器镜像进行安装。

### 驱动固件安装

下载[驱动固件](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha003/softwareinst/instg/instg_0003.html?Mode=PmIns&OS=Debian&Software=cannToolKit)，请根据系统和硬件产品型号选择匹配版本的`driver`和`firmware`。参考[安装NPU驱动固件](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha003/softwareinst/instg/instg_0005.html?Mode=PmIns&OS=Debian&Software=cannToolKit)官方指导或执行以下命令安装：

```shell
chmod +x Ascend-hdk-<chip_type>-npu-driver_<version>_linux-<arch>.run
chmod +x Ascend-hdk-<chip_type>-npu-firmware_<version>.run
./Ascend-hdk-<chip_type>-npu-driver_<version>_linux-<arch>.run --full --install-for-all
./Ascend-hdk-<chip_type>-npu-firmware_<version>.run --full
```

### CANN安装

下载[CANN](https://www.hiascend.com/developer/download/community/result?module=cann)，请根据系统和硬件产品型号选择匹配版本的。参考[安装CANN软件包](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha003/softwareinst/instg/instg_0008.html?Mode=PmIns&OS=Debian&Software=cannToolKit)官方指导或执行以下命令安装：

```shell
# 安装文件名跟随版本迭代及硬件版本，根据实际修改
chmod +x Ascend-cann-toolkit_<version>_linux-<arch>.run
./Ascend-cann-toolkit_<version>_linux-<arch>.run --install
chmod +x Ascend-cann-kernels-<chip_type>_<version>_linux-<arch>.run
./Ascend-cann-kernels-<chip_type>_<version>_linux-<arch>.run --install
source /usr/local/Ascend/ascend-toolkit/set_env.sh # 安装nnal包需要source环境变量
chmod +x Ascend-cann-nnal-<chip_type>_<version>_linux-<arch>.run
./Ascend-cann-nnal-<chip_type>_<version>_linux-<arch>.run --install
# 设置环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=0
```

### MindSpore 安装

参考[MindSpore官方安装指导](https://www.mindspore.cn/install)，根据系统类型、CANN版本及Python版本选择匹配的对应的安装命令进行安装，安装前请确保网络畅通。或执行以下命令安装：

```shell
pip install mindspore==2.7.1
```

