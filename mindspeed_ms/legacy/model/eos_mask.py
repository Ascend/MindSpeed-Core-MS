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
"""EosMask"""

import numpy as np
from mindspore import Tensor, nn, ops, mint, _no_grad
import mindspore.common.dtype as mstype


class EosMask(nn.Cell):
    """
    Generate attention mask corresponding to a specific token.

    Args:
        batch_size (int): batch_size from config
        seq_len (int): seq_len from config
        eod_token_id (int): eod_token_id from config
        reset_position_ids (bool): reset_position_ids from config
    """
    def __init__(self, batch_size, seq_len, eod_token_id, reset_position_ids):
        """Cal attention mask in device."""
        super().__init__()
        self.seq_len = seq_len
        self.position_ids = Tensor(np.broadcast_to(np.expand_dims(np.arange(seq_len), 0), (batch_size, seq_len)),
                                   dtype=mstype.int32)
        self.tril = ops.Tril()
        self.cast = ops.Cast()
        self.expand_dim = ops.ExpandDims()
        self.eod_token = eod_token_id
        self.reset_position_ids = reset_position_ids

    @_no_grad()
    def construct(self, input_ids):
        """construct method"""
        # input_ids: [bs, seq_len]
        eod_idx = self.cast(mint.eq(input_ids, self.eod_token), mstype.float16)
        attention_mask = mint.cumsum(eod_idx, 1) - eod_idx
        row = self.expand_dim(attention_mask, 1)
        col = self.expand_dim(attention_mask, 2)
        row = mint.tile(row, (1, self.seq_len, 1))
        col = mint.tile(col, (1, 1, self.seq_len))
        mat = mint.eq(row, col)
        mat = self.cast(mat, mstype.uint8)
        mask = self.tril(mat)
        # [bs, seq_len, seq_len]
        if self.reset_position_ids:
            reset_position_ids = self.position_ids.copy()
            for i, eod_idx_ in enumerate(eod_idx):
                p_id_offset = mint.zeros(self.seq_len, dtype=mstype.float16)
                eod_idx_ = mint.nonzero(eod_idx_).reshape(-1)
                if eod_idx_.shape[0] == 0:
                    continue
                eod_index_offset = eod_idx_ + 1
                eod_index_offset[1:] = eod_index_offset[1:] - eod_index_offset[:-1]
                if eod_idx_[-1]+1 < self.seq_len:
                    p_id_offset[eod_idx_+1] = eod_index_offset
                else:
                    if len(eod_index_offset) == 1:
                        continue
                    p_id_offset[eod_idx_[:-1]+1] = eod_index_offset[:-1]
                p_id_offset = mint.cumsum(p_id_offset, 0)
                reset_position_ids[i] -= p_id_offset
            return reset_position_ids, mint.sub(1, mask)

        return self.position_ids, mint.sub(1, mask)
