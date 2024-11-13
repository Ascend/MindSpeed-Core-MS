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

"""Simple blended dataset."""

import hashlib
import json
import logging
from collections import OrderedDict
from typing import Dict, List, Union

import numpy
import torch

from mindspeed_ms.core.datasets.blended_megatron_dataset_config import BlendedMegatronDatasetConfig
from mindspeed_ms.core.datasets.megatron_dataset import MegatronDataset
from mindspeed_ms.core.datasets.utils import log_single_rank
from mindspeed_ms.core.config import get_args

logger = logging.getLogger(__name__)

_VERBOSE = False


class SimpleBlendedDataset(torch.utils.data.Dataset):
    """Simplified BlendedDataset for data converted from MindRecord (args.is_dataset_from_mr)
    This simplified implementation removes the need to build index.
    This relies on several assumptions:
    - Each dataset has equal weight.
    - Each dataset contains the same number of sequences.
    - All sequences have been shuffled within and across datasets.
    - Its behavior is controlled by args.simple_blend,
    - which may take the following values (assuming there are n datasets (ds) and each dataset contains m instances):
    - no: The original Megatron BlendedDataset is used.
    - (When args.is_dataset_from_mr is True,
    - we usually modify the code in _get_prefixes_weights_and_sizes_for_blend
    - to int(math.ceil(target_num_samples * weight)).)
    - inter: data sequence: ds_0[0], ds_1[0], ...,
      ds_n[0], ds_0[1], ds_1[1], ..., ds_n[1], ds_0[2], ...
      This should be the same as the original Megatron BlendedDataset behavior.
    - intra: data sequence: ds_0[0], ds_0[1], ..., ds_0[m], ds_1[0], ds_1[1], ..., ds_1[m], ds_2[0], ds_2[1],
      ... This may be beneficial when there's need to resume from interrupted training.

    A note on the size:
    The size parameter is usually train_samples,
    which is usually computed from args.train_iters * args.global_batch_size.
    When args.is_dataset_from_mr is True,
    it is recommended that the size is equal to the number of data instances (n*m).
    If size > n*m, error will be raised; if size < n*m,
    some data are not used for training. In other words,
    it is recommended to set args.train_iters to n*m // args.global_batch_size.

    Args:
        datasets (List[MegatronDataset]): The MegatronDataset instances to blend

        weights (List[int]): The weights which determines the dataset blend ratios

        size (int): The number of samples to draw from the blend

        config (BlendedMegatronDatasetConfig): The config

    Raises:
        RuntimeError: When the dataset has fewer or more samples than 'size' post-initialization
    """

    def __init__(
            self,
            datasets: List[MegatronDataset],
            weights: List[int],
            size: int,
            config: BlendedMegatronDatasetConfig) -> None:
        assert len(datasets) == len(weights)
        assert all(weight == 1 for weight in weights), "SimpleBlendedDataset requires all weights to be 1"
        assert all(map(lambda _: type(_) == type(datasets[0]), datasets))

        # Alert user to unnecessary blending
        if len(datasets) == 1:
            log_single_rank(
                logger, logging.WARNING, f"Building a BlendedDataset for a single MegatronDataset"
            )

        args = get_args()
        self.simple_blend = args.simple_blend
        assert self.simple_blend == "inter" or self.simple_blend == "intra"
        self.dataset_size = len(datasets[0])
        assert all(len(dataset) == self.dataset_size for dataset in datasets), \
        "SimpleBlendedDataset expects that each dataset contains the same number of sequences."

        self.datasets = datasets
        self.weights = weights
        self.size = size
        self.config = config

        assert self.size <= len(self.datasets) * self.dataset_size, \
        f"Requesting a number of samples {self.size} \
        larger than the number of the provided data instances {len(self.datasets) * self.dataset_size}."
        if self.size < len(self.datasets) * self.dataset_size:
            log_single_rank(
                logger, logging.WARNING, \
                f"Requesting a number of samples {self.size} smaller than the number of the provided data instances \
                    {len(self.datasets) * self.dataset_size}. Some data will not be used."
            )

        unique_identifiers = OrderedDict()
        unique_identifiers["class"] = type(self).__name__
        unique_identifiers["datasets"] = [dataset.unique_identifiers for dataset in self.datasets]
        unique_identifiers["weights"] = self.weights
        unique_identifiers["size"] = self.size

        self.unique_description = json.dumps(
            unique_identifiers, indent=4, default=lambda obj: obj.unique_identifiers
        )
        self.unique_description_hash = hashlib.md5(
            self.unique_description.encode("utf-8")
        ).hexdigest()

        log_single_rank(logger, logging.INFO, f"> {type(self).__name__} length: {len(self)}")

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Dict[str, Union[int, numpy.ndarray]]:
        if self.simple_blend == "inter":
            dataset_id = idx % len(self.datasets)
            dataset_sample_id = idx // len(self.datasets)
        else:
            dataset_id = idx // self.dataset_size
            dataset_sample_id = idx % self.dataset_size
        return {
            "dataset_id": dataset_id,
            **self.datasets[dataset_id][dataset_sample_id],
        }
