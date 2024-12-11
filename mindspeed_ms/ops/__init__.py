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

"""ops init"""
from .npu_rms_norm import npu_rms_norm
from .swiglu import swiglu
from .npu_rotary_position_embedding import npu_rotary_position_embedding
from .npu_fusion_attention import npu_fusion_attention
from .sparse_attention_score import SparseAttentionScore, SparseAttentionScoreGrad
