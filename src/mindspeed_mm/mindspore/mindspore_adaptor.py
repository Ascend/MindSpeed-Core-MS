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

    # patch glm
    from mindspeed_mm.mindspore.models.vision.vision_encoders.glm4v_vl_vit_model import glm4v_vision_embeddings_forward
    aspm.register_patch('mindspeed_mm.models.vision.vision_encoders.glm4v_vl_vit_model.Glm4vVisionEmbeddings.forward',
                        glm4v_vision_embeddings_forward)

    # patch dsvl2
    from mindspeed.mindspore.core.tensor_parallel.mappings import all_to_all_forward
    aspm.register_patch('megatron.core.tensor_parallel.mappings._AllToAll.forward', all_to_all_forward)

    # patch llava
    from mindspeed.mindspore.core.transformer.module import fp32_to_float16
    from mindspeed.mindspore.legacy.model.module import float16_to_fp32
    aspm.register_patch('megatron.core.transformer.module.fp32_to_float16', fp32_to_float16)
    aspm.register_patch('megatron.core.transformer.module.float16_to_fp32', float16_to_fp32)
    from mindspeed_mm.mindspore.utils.utils import quick_gelu
    aspm.register_patch('mindspeed_mm.utils.utils.quick_gelu', quick_gelu)

    aspm.apply_patches()

apply_mindspore_patch()
