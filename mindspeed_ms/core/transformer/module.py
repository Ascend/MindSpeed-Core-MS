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
"""Module."""

from mindspore import nn
from mindspeed_ms.core.transformer.transformer_config import TransformerConfig
from mindspeed_ms.legacy.model.module import get_default_dict_for_module


def param_is_not_shared(param):
    return not hasattr(param, 'shared') or not param.shared


class Module(nn.Cell):
    """Base Mindformer module inhertied by all Models.

    Args:
        config (TransformerConfig): Transformer config
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

    def set_is_first_microbatch(self):
        """Sets the is_first_microbatch flag if it exists."""

        for m in self.modules():
            if hasattr(m, "is_first_microbatch"):
                m.is_first_microbatch = True

    def sharded_state_dict(self):
        """iterate over the subcells to construct the total sharded state dict"""

        sharded_state_dict = {}

        # Recurse into subcells
        def update_sharded_dict_for_single_cell(subcell):
            nonlocal sharded_state_dict
            if hasattr(subcell, 'sharded_state_dict'):
                sharded_state_dict.update(subcell.sharded_state_dict())
            else:
                if isinstance(subcell, (nn.SequentialCell, nn.CellList)):
                    for inner_layer in subcell:
                        update_sharded_dict_for_single_cell(inner_layer)
                else:
                    sharded_state_dict.update(get_default_dict_for_module(subcell, recurse=True))
        for subcell in self.cells():
            update_sharded_dict_for_single_cell(subcell)
        # Handle params in the current cell
        sharded_state_dict.update(get_default_dict_for_module(self, recurse=False))

        return sharded_state_dict
