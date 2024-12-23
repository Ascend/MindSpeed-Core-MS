mindspeed_ms.core.parallel_state.get_context_parallel_world_size
================================================================

.. py:function:: mindspeed_ms.core.parallel_state.get_context_parallel_world_size()

    返回上下文并行组的world大小。

    返回：
        - 返回world的大小。

    样例：

    .. note::
        - 运行样例之前，需要配置好通信环境变量。
        - 针对Ascend设备，推荐使用msrun启动方式，无第三方以及配置文件依赖。详见 `msrun启动 <https://www.mindspore.cn/docs/zh-CN/master/model_train/parallel/msrun_launcher.html>`_ 。
