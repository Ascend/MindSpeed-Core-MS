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
"""test data"""
import numpy as np
import mindspore as ms
import mindspeed_ms.core.parallel_state as parallel_state

class TestData:
    """ generate test data for language model test """
    def __init__(self, data_num, seq_length):
        super().__init__()
        input_data = np.random.randint(0, 100, (data_num, seq_length+1))
        self.input_data = input_data[:, :-1]
        self.labels = input_data[:, 1:]
        ones = np.ones((data_num, 1, seq_length, seq_length), dtype=np.int32)
        self.attention_mask = np.tril(ones)

        if parallel_state.get_data_parallel_world_size() > 1:
            per_rank_size = data_num // parallel_state.get_data_parallel_world_size()
            dp_rank = parallel_state.get_data_parallel_rank()
            self.input_data = self.input_data[dp_rank * per_rank_size: (dp_rank + 1) * per_rank_size]
            self.labels = self.labels[dp_rank * per_rank_size: (dp_rank + 1) * per_rank_size]
            self.attention_mask = self.attention_mask[dp_rank * per_rank_size: (dp_rank + 1) * per_rank_size]

    def __getitem__(self, index):
        return (ms.Tensor(self.input_data[index], dtype=ms.int32),
                ms.Tensor(self.labels[index], dtype=ms.int32),
                ms.Tensor(self.attention_mask[index], dtype=ms.int32))

    def __len__(self):
        return self.input_data.shape[0]
