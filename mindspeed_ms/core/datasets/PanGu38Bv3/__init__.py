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

"""PanGu38Bv3 init"""
from .data_samplers import FullBatchPretrainingSampler
from .dataset import (
    check_mismatch,
    get_batch,
    get_ltor_reset_masks_and_position_ids,
    get_dataset
)
from .decoder_packed_mtf_dataset import (
    build_train_valid_test_datasets,
    DecoderPackedMTFDataset
)
from .error_utils import (
    check_condition,
    check_divisible,
    check_divisible_by_zero,
    check_equal,
    check_exist,
    check_type
)
from .gpt_from_mr_dataset import GPTFromMRDataset
from .mtf_dataset import MTFDataset, get_packed_indexed_dataset
from .prompter import AlpacaTemplate, Prompter
from .prompter_sft import (
    PanguSftTemplate,
    PanguPrompter,
    PanguLlama3SftTemplate,
    PanguLlama3Prompter,
    PanguQwen15SftTemplate,
    PanguQwen15Prompter
)

from . import data_handler
from . import data_handler_sft


__all__ = ['FullBatchPretrainingSampler',
           'check_mismatch',
           'get_batch',
           'get_ltor_reset_masks_and_position_ids',
           'get_dataset',
           'build_train_valid_test_datasets',
           'DecoderPackedMTFDataset',
           'check_condition',
           'check_divisible',
           'check_divisible_by_zero',
           'check_equal',
           'check_exist',
           'check_type',
           'GPTFromMRDataset',
           'MTFDataset',
           'get_packed_indexed_dataset',
           'AlpacaTemplate',
           'Prompter',
           'PanguSftTemplate',
           'PanguPrompter',
           'PanguLlama3Prompter',
           'PanguLlama3SftTemplate',
           'PanguQwen15Prompter',
           'PanguQwen15SftTemplate']
__all__.extend(data_handler.__all__)
__all__.extend(data_handler_sft.__all__)
