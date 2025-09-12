# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
"""
Patch for MindSpore adaptation.

This module provides utilities for MindSpore framework,
based on mindspeed_args settings.
"""
from mindspeed.patch_utils import MindSpeedPatchesManager as aspm
from mindspeed_mm.mindspore.data.datasets.utils import process_in_cpu_wrapper
from mindspeed_mm.mindspore.data.data_utils.func_utils.convert import preprocess_dataset
from mindspeed_mm.mindspore.models.vision.vision_encoders.qwen2vl_vit_model import get_window_index, qwen2vlvit_selfattention_forward
from mindspeed_mm.mindspore.utils.transformer_model_config import get_model_config
from mindspeed_mm.mindspore.models.common.communications import _gather


def apply_mindspore_patch():
    """
    Registers and applies all MindSpore adaptation patches.
    """
    aspm.register_patch('mindspeed_mm.data.datasets.qwen2vl_dataset.get_qwen2vl_dataset', process_in_cpu_wrapper)
    aspm.register_patch(
        'mindspeed_mm.data.data_utils.func_utils.convert.SupervisedDatasetProcessor.preprocess_dataset',
        preprocess_dataset)
    aspm.register_patch(
        'mindspeed_mm.models.vision.vision_encoders.qwen2vl_vit_model.Qwen2VLViT.get_window_index',
        get_window_index)
    aspm.register_patch(
        'mindspeed_mm.models.vision.vision_encoders.qwen2vl_vit_model.Qwen2vlVitSelfAttention.forward',
        qwen2vlvit_selfattention_forward)
    aspm.register_patch('mindspeed_mm.utils.transformer_model_config.get_model_config', get_model_config)
    aspm.register_patch('mindspeed_mm.models.common.communications._gather', _gather)
    aspm.apply_patches()

apply_mindspore_patch()
