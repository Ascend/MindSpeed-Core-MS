mindspeed_ms.core.config.init_configs_from_yaml
===============================================

.. py:function:: mindspeed_ms.core.config.init_configs_from_yaml(file_path: str, config_classes=None, **kwargs)

    从指定的YAML配置文件初始化Config类实例。

    参数：
        - **file_path** (str) - YAML配置文件路径。
        - **config_classes** (Union[list[BaseConfig], None]，可选) - 待初始化的Config类，支持[TrainingConfig，ModelParallelConfig，OptimizerConfig，DatasetConfig，LoraConfig，TransformerConfig，MoEConfig]。当没有上述Config类传入时，所有已知的配置项将被初始化为 :class:`mindspeed_ms.core.config.AllConfig` 。默认值： ``None`` 。
        - **kwargs** (dict) - 额外的关键字配置参数。

    返回：
        - 返回初始化的Config类的实例。当没有Config类传入时，返回 `AllConfig` 类的实例。

    异常：
        - **ValueError** - 入参 `file_path` 不是字符串。
        - **ValueError** - 入参 `file_path` 不是以yaml或者yml结尾。
