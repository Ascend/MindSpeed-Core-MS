# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Wrapped Layer Norm."""

from mindspeed_ms.core.transformer.transformer_config import TransformerConfig
from mindspeed_ms.legacy.model.norm import LayerNorm


class WrappedLayerNorm(LayerNorm):
    """WrappedLayerNorm."""
    # pylint: disable=W0613
    def __init__(self,
                 config: TransformerConfig,
                 hidden_size: int,
                 eps: float = 1e-5,
                 persist_layer_norm: bool = False,
                 zero_centered_gamma: bool = False,
                 normalization: str = "LayerNorm"):
        self.config = config
        if self.config.layernorm_zero_centered_gamma:
            raise ValueError(f"zero_centered_gamma not supported by LayerNorm")

        if not self.config.normalization == "LayerNorm":
            raise ValueError(f'({self.config.normalization}) is not supported in by Layernorm')

        if self.config.persist_layer_norm:
            raise ValueError(f"persist_layer_norm not supported by LayerNorm")

        if self.config.memory_efficient_layer_norm:
            raise ValueError(f"memory_efficient_layer_norm not supported by LayerNorm")

        super().__init__(
            normalized_shape=hidden_size,  # applied to last len(normalized_shape.size) dimensions
            eps=eps,
        )
