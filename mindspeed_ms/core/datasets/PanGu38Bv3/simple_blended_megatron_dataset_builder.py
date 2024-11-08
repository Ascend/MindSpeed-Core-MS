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

"""Simple blended dataset builder."""

import logging
import math
from typing import List, Optional, Tuple, Union

import torch

from mindspeed_ms.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from mindspeed_ms.core.datasets.PanGu38Bv3.simple_blended_dataset import SimpleBlendedDataset
from mindspeed_ms.core.datasets.megatron_dataset import LowLevelDataset, MegatronDataset, MockDataset
from mindspeed_ms.core.datasets.utils import Split

logger = logging.getLogger(__name__)

MidLevelDataset = Union[MegatronDataset, MockDataset]

TopLevelDataset = Union[SimpleBlendedDataset, MidLevelDataset]

DistributedDataset = Union[
    TopLevelDataset, MidLevelDataset, LowLevelDataset, torch.utils.data.Dataset
]


class SimpleBlendedMegatronDatasetBuilder(BlendedMegatronDatasetBuilder):
    """Simplified version of BlendedMegatronDatasetBuilder
    See docstring of SimpleBlendedDataset for details.
    The only code differences from the superclass are:
    - BlendedDataset is changed to SimpleBlendedDataset.
    - _get_prefixes_weights_and_sizes_for_blend is slightly modified.

    Args:
        cls (Type[MegatronDataset]): The class to instantiate, must inherit from MegatronDataset

        sizes (List[int]): The minimum number of total samples to draw from each split, varies with blend

        is_built_on_rank (Callable): A callable which returns True

        if the dataset should be built on the current rank and False otherwise.

        It should be Megatron Core parallelism aware i.e. global rank, local group rank,

        and virtual rank may inform its return value.

        config (BlendedMegatronDatasetConfig): The config object which informs dataset creation
    """

    def _build_blended_dataset_splits(self,) -> List[Optional[TopLevelDataset]]:
        """Build all dataset splits according to the provided blend(s)

        See the BlendedMegatronDatasetBuilder.build alias for more information.

        Returns:
            List[Optional[TopLevelDataset]]: A list containing a dataset instance (or None) per split
        """

        # Return fake "mock" datasets
        if self.config.mock:

            return self._build_megatron_dataset_splits(None, None, self.sizes)

        # All splits come from the same distribution
        if self.config.blend:
            blend = self.config.blend
            split = self.config.split_matrix

            # Blend consists of a single prefix
            if len(blend) == 1:
                return self._build_megatron_dataset_splits(blend[0], split, self.sizes)

            # Blend consists of multiple weights and prefixes
            (
                prefix_per_dataset,
                weight_per_dataset,
                sizes_per_dataset,
            ) = _get_prefixes_weights_and_sizes_for_blend(blend, self.sizes)

            megatron_datasets = [[] for _ in range(len(Split))]

            for i in range(len(prefix_per_dataset)):
                megatron_datasets_split = self._build_megatron_dataset_splits(
                    prefix_per_dataset[i], split, sizes_per_dataset[i]
                )
                for j in range(len(megatron_datasets_split)):
                    megatron_datasets[j].append(megatron_datasets_split[j])

            # Sum over all contributing datasets, per split
            size_per_split = list(map(sum, zip(*sizes_per_dataset)))

            blended_datasets = []
            for i in range(len(megatron_datasets)):
                is_none = map(lambda _: _ is None, megatron_datasets[i])

                if split[i] is None:
                    assert all(is_none)
                    blended_datasets.append(None)
                else:
                    assert all(is_none) or not any(is_none)
                    blended_datasets.append(
                        self.build_generic_dataset(
                            SimpleBlendedDataset,
                            self.is_built_on_rank,
                            megatron_datasets[i],
                            weight_per_dataset,
                            size_per_split[i],
                            self.config,
                        )
                    )
            return blended_datasets

        # Each split comes from a separate distribution

        blended_datasets = []
        for i in range(len(Split)):
            blend = self.config.blend_per_split[i]

            # Blend is not provided
            if not blend:
                blended_datasets.append(None)
                continue

            split_spoof = [None] * len(Split)
            split_spoof[i] = (0.0, 1.0)
            sizes_spoof = [0] * len(Split)
            sizes_spoof[i] = self.sizes[i]

            # Blend consists of a single prefix
            if len(blend) == 1:
                blended_datasets.append(
                    self._build_megatron_dataset_splits(blend[0], split_spoof, sizes_spoof)[i]
                )

            # Blend consists of multiple weights and prefixes
            else:
                (
                    prefix_per_dataset,
                    weight_per_dataset,
                    sizes_per_dataset,
                ) = _get_prefixes_weights_and_sizes_for_blend(blend, sizes_spoof)

                megatron_datasets = []
                for j in range(len(prefix_per_dataset)):
                    megatron_datasets.append(
                        self._build_megatron_dataset_splits(
                            prefix_per_dataset[j], split_spoof, sizes_per_dataset[j],
                        )[i]
                    )

                size_per_split = list(map(sum, zip(*sizes_per_dataset)))

                blended_datasets.append(
                    self.build_generic_dataset(
                        SimpleBlendedDataset,
                        self.is_built_on_rank,
                        megatron_datasets,
                        weight_per_dataset,
                        size_per_split[i],
                        self.config,
                    )
                )

            return blended_datasets

def _get_prefixes_weights_and_sizes_for_blend(
        blend: List[str], target_num_samples_per_split: List[int]
) -> Tuple[List[str], List[int], List[List[int]]]:
    """Determine the contribution of the MegatronDataset splits to the SimpleBlendedDataset splits

    Args:
        blend (List[str]): e.g. ["1", "path/to/dataset_1_prefix", "1", "path/to/dataset_2_prefix"]

        target_num_samples_per_split (List[int]): The number of samples to target for each BlendedDataset split

    Returns:
        Tuple[List[str], List[int], List[List[int]]]:
        The prefix strings e.g. ["path/to/dataset_1_prefix", "path/to/dataset_2_prefix"],
        the int weights e.g. [1, 1], and the number of samples to request per MegatronDataset per split
    """
    weights, prefixes = zip(
        *[(int(blend[i]), blend[i + 1].strip()) for i in range(0, len(blend), 2)]
    )

    sizes_per_dataset = [
        [
            int(math.ceil(target_num_samples / len(weights)))
            for target_num_samples in target_num_samples_per_split
        ]
        for _ in weights
    ]

    return prefixes, weights, sizes_per_dataset
