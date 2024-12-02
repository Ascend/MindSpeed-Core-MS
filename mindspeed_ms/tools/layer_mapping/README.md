# layer_mapping配置说明

layer_mapping配置文件用于在Mindformers中使用msprobe工具，与Megatron进行跨框架的Layer层比对，具体用法请参考[跨框架的Layer层比对](https://gitee.com/ascend/mstt/blob/master/debug/accuracy_tools/msprobe/docs/11.accuracy_compare_MindSpore.md#28-%E8%B7%A8%E6%A1%86%E6%9E%B6%E7%9A%84layer%E5%B1%82%E6%AF%94%E5%AF%B9)。文件的具体格式说明，请参考[自定义映射文件(Layer)](https://gitee.com/ascend/mstt/blob/master/debug/accuracy_tools/msprobe/docs/11.accuracy_compare_MindSpore.md#35-%E8%87%AA%E5%AE%9A%E4%B9%89%E6%98%A0%E5%B0%84%E6%96%87%E4%BB%B6layer)。

> msprobe 是 MindStudio Training Tools 工具链下专用于精度调试的工具包，功能涵盖精度预检、溢出检测以及精度对比等。目前支持 PyTorch 和 MindSpore 框架。更多信息请参见[msprobe官方文档](https://gitee.com/ascend/mstt/tree/master/debug/accuracy_tools/msprobe) 。
