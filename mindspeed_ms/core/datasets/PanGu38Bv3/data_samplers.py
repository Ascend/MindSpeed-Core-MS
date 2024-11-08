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

"""Dataloaders."""

class FullBatchPretrainingSampler:
    """
    FullBatch pretraining sampler
    """
    def __init__(self, total_samples, consumed_samples, micro_batch_size, micro_batch_num,
                 data_parallel_rank, data_parallel_size, drop_last=True):
        # Keep a copy of input params for later use.
        self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        self.micro_batch_size = micro_batch_size
        self.micro_batch_num = micro_batch_num
        self.data_parallel_rank = data_parallel_rank
        self.micro_batch_times_data_parallel_size = \
            self.micro_batch_size * data_parallel_size
        self.full_batch_data_parallel_size = \
            self.micro_batch_size * self.micro_batch_num * data_parallel_size
        self.drop_last = drop_last

        # Sanity checks.
        assert self.total_samples > 0, \
            'no sample to consume: {}'.format(self.total_samples)
        assert self.consumed_samples < self.total_samples, \
            'no samples left to consume: {}, {}'.format(self.consumed_samples,
                                                        self.total_samples)
        assert self.micro_batch_size > 0
        assert data_parallel_size > 0
        assert self.data_parallel_rank < data_parallel_size, \
            'data_parallel_rank should be smaller than data size: {}, ' \
            '{}'.format(self.data_parallel_rank, data_parallel_size)

    def __len__(self):
        return self.total_samples // self.full_batch_data_parallel_size

    def get_start_end_idx(self, micro_batch_idx):
        start_idx = (self.data_parallel_rank * self.micro_batch_size +
                     micro_batch_idx * self.micro_batch_times_data_parallel_size)
        end_idx = start_idx + self.micro_batch_size
        return start_idx, end_idx

    def __iter__(self):
        batch = []
        # Last batch will be dropped if drop_last is not set False
        for idx in range(self.consumed_samples, self.total_samples):
            batch.append(idx)
            if len(batch) == self.full_batch_data_parallel_size:
                batch_per_dp = []
                for micro_batch_idx in range(self.micro_batch_num):
                    start_idx, end_idx = self.get_start_end_idx(micro_batch_idx)
                    batch_per_dp += batch[start_idx:end_idx]
                yield batch_per_dp
                batch = []
                batch_per_dp = []

        # Check the last partial batch and see drop_last is set
        if batch and not self.drop_last:
            start_idx, end_idx = self.get_start_end_idx()
            yield batch[start_idx:end_idx]
