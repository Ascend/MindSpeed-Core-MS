mindspeed_ms.core.config.DatasetConfig
======================================

.. py:class:: mindspeed_ms.core.config.DatasetConfig(dataset_dir: str = "./dataset", shuffle: bool = False, batch_size: int = 1, micro_batch_num: int = 1, train_samples: int = 0, data_layout: str = "BSH", eos_token_id: int = 0, **kwargs)

    数据集配置类。

    参数：
        - **dataset_dir** (str，可选) - 数据集文件目录。默认值： ``"./dataset"`` 。
        - **shuffle** (bool，可选) - 是否打乱数据集。默认值： ``False`` 。
        - **batch_size** (int，可选) - 一次迭代（iteration）所处理的数据样本的数量。默认值： ``1`` 。
        - **micro_batch_num** (int，可选) - 使用流水线并行或者梯度累积时的微批数量。默认值： ``1`` 。
        - **train_samples** (int，可选) - 训练样本数量。默认值： ``0`` 。
        - **data_layout** (str，可选) - 输入布局。默认值： ``"BSH"`` 。
        - **eos_token_id** (int，可选) - 序列结束（EoS，End-of-Sequence）的token id。默认值： ``0`` 。
        - **kwargs** (dict) - 额外的关键字配置参数。
