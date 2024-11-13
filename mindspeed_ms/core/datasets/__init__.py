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

"""datasets init"""
from .blended_dataset import BlendedDataset
from .blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from .blended_megatron_dataset_config import BlendedMegatronDatasetConfig
from .gpt_dataset import GPTDataset, MockGPTDataset, GPTDatasetConfig
from .indexed_dataset import IndexedDataset, IndexedDatasetBuilder, get_bin_path, get_idx_path
from .megatron_dataset import MegatronDataset, MockDataset
from .megatron_tokenizer import MegatronTokenizer
from .utils import Split, compile_helpers, log_single_rank, normalize
from . import PanGu38Bv3

__all__ = ['BlendedDataset',
           'BlendedMegatronDatasetBuilder',
           'BlendedMegatronDatasetConfig',
           'GPTDataset',
           'MockGPTDataset',
           'GPTDatasetConfig',
           'IndexedDataset',
           'IndexedDatasetBuilder',
           'get_bin_path',
           'get_idx_path',
           'MegatronDataset',
           'MockDataset',
           'MegatronTokenizer',
           'Split',
           'compile_helpers',
           'log_single_rank',
           'normalize']
__all__.extend(PanGu38Bv3.__all__)
