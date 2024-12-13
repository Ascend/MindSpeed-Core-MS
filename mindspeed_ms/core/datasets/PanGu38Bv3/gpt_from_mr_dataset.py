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

"""GPT from MR dataset."""

import logging
from typing import Dict, Optional

import numpy
import torch

from mindspeed_ms.core.datasets.blended_megatron_dataset_config import BlendedMegatronDatasetConfig
from mindspeed_ms.core.datasets.gpt_dataset import GPTDatasetConfig
from mindspeed_ms.core.datasets.indexed_dataset import IndexedDataset
from mindspeed_ms.core.datasets.megatron_dataset import LowLevelDataset, MegatronDataset
from mindspeed_ms.core.datasets.utils import Split

logger = logging.getLogger(__name__)


class GPTFromMRDataset(MegatronDataset):
    """The base GPT dataset for data converted from MindRecord.

    Args:
        indexed_dataset (IndexedDataset): The IndexedDataset around which to build the
        MegatronDataset

        indexed_indices (numpy.ndarray): The set of the documents indices to expose

        num_samples (int): The number of samples to draw from the indexed dataset

        index_split (Split): The indexed_indices Split

        config (GPTDatasetConfig): The GPT-specific container for all config sourced parameters
    """

    def __init__(
            self,
            dataset: IndexedDataset,
            dataset_path: Optional[str],
            indices: numpy.ndarray,
            num_samples: int,
            index_split: Split,
            config: GPTDatasetConfig) -> None:
        super().__init__(dataset, dataset_path, indices, num_samples, index_split, config)

        self.masks_and_position_ids_are_cacheable = not any(
            [
                self.config.reset_position_ids,
                self.config.reset_attention_mask,
                self.config.eod_mask_loss,
            ]
        )
        self.masks_and_position_ids_are_cached = False
        self.cached_attention_mask = None
        self.cached_loss_mask = None
        self.cached_position_ids = None

    def _finalize(self) -> None:
        """Abstract method implementation

        Load or build/cache the document, sample, and shuffle indices
        """
        assert isinstance(self.config, GPTDatasetConfig)
        sequence_length = getattr(self.config, "sequence_length")
        assert all([self.dataset.sequence_lengths[i] == sequence_length + 1 for i in range(len(self.dataset))])
        assert self.num_samples <= len(self.dataset), (
            f"Requesting {self.num_samples} instances but the dataset only contains {len(self.dataset)}."
        )


    def __len__(self) -> int:
        """Abstract method implementation

        Returns:
            int: The length of the dataset
        """
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, numpy.ndarray]:
        """Abstract method implementation

        Args:
            idx (int): The index into the dataset

        Returns:
            Dict[str, numpy.ndarray]: The text ids and (optionally) the document ids wrapped in a
            dictionary
        """
        text = numpy.array(self.dataset[idx], dtype=numpy.int64)

        text = torch.from_numpy(text).long()
        labels = text[1:].contiguous()
        tokens = text[:-1].contiguous()

        assert not torch.any(
            tokens >= self.config.tokenizer.vocab_size
            ), "An input token is out of bounds of the tokenizer vocabulary"

        if (
                not self.masks_and_position_ids_are_cacheable
                or not self.masks_and_position_ids_are_cached
            ):
            attention_mask, loss_mask, position_ids = _get_ltor_masks_and_position_ids(
                tokens,
                self.config.tokenizer.eod,
                self.config.reset_position_ids,
                self.config.reset_attention_mask,
                self.config.eod_mask_loss,
                self.config.create_attention_mask,
            )
            if self.masks_and_position_ids_are_cacheable:
                self.cached_attention_mask = attention_mask
                self.cached_loss_mask = loss_mask
                self.cached_position_ids = position_ids
                self.masks_and_position_ids_are_cached = True
        else:
            attention_mask = self.cached_attention_mask
            loss_mask = self.cached_loss_mask
            position_ids = self.cached_position_ids

        if self.config.create_attention_mask:
            return {
                "tokens": tokens,
                "labels": labels,
                "attention_mask": attention_mask,
                "loss_mask": loss_mask,
                "position_ids": position_ids,
            }
        return {
            "tokens": tokens,
            "labels": labels,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
        }

    @staticmethod
    def is_multimodal() -> bool:
        """Abstract method implementation

        Returns:
            bool: False
        """
        return False

    @staticmethod
    def is_split_by_sequence() -> bool:
        """Abstract method implementation

        Returns:
            bool: True
        """
        return True

    @staticmethod
    def numel_low_level_dataset(low_level_dataset: LowLevelDataset) -> int:
        """Return the number of elements in the underlying low level dataset for the purpose of
        segregating the train/valid/test split indices

        It may be that the low level dataset can be split any number of ways, depending on the mid
        level dataset it supports, which is why we define the "number of elements" function
        separately from the __len__ function here in the mid level dataset class

        Args:
            low_level_dataset (LowLevelDataset): The underlying low level dataset

        Returns:
            int: The number of elements in the underlying low level dataset
        """
        return low_level_dataset.sequence_lengths.shape[0]

    @staticmethod
    def build_low_level_dataset(
            dataset_path: str, config: BlendedMegatronDatasetConfig) -> LowLevelDataset:
        """Build the low level dataset via a function to be called from within
        BlendedMegatronDatasetBuilder.build_generic_dataset

        It may be that the low level dataset spans any subset of train/valid/test splits, which is
        why we define a static "build" function separately from the constructor in the mid level
        dataset class

        Args:
            dataset_path (str): The real path on disk to the dataset

            config (BlendedMegatronDatasetConfig): The dataset config

        Returns:
            LowLevelDataset: The low level dataset
        """
        return IndexedDataset(dataset_path, multimodal=False, mmap=config.mmap_bin_files)

def _get_ltor_masks_and_position_ids(
        data: torch.Tensor,
        eod_token: int,
        reset_position_ids: bool,
        reset_attention_mask: bool,
        eod_mask_loss: bool,
        create_attention_mask: bool,
):
    """Build masks and position id for left to right model.

    Args:
        data (torch.Tensor): The data tenor that holds the tokens from the dataset

        eod_token (int): ID of the token to that is considered the EOD

        reset_position_ids (bool): Switch to reset the document position ID's

        reset_attention_mask (bool): Switch to reset the attention mask

        eod_mask_loss (bool): Switch to enable the EOD mask loss

        create_attention_mask (bool): Switch to enable the attention masks generation.

        Can be disabled if attention kernel generates masks by itself.

    Returns:
        torch.Tensor: Attention mask needed to be used for Attention

        torch.Tensor: The mask used for loss value during training

        torch.Tensor: The position ID's of the token
    """
    seq_length = data.numel()

    if create_attention_mask:
        attention_mask = torch.tril(
            torch.ones((seq_length, seq_length), device=data.device)
        ).unsqueeze(0)
    else:
        attention_mask = None

    # Loss mask.
    loss_mask = torch.ones(seq_length, dtype=torch.float, device=data.device)
    if eod_mask_loss:
        loss_mask[data == eod_token] = 0.0

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long, device=data.device)
    # We need to clone as the ids will be modified based on batch index.
    if reset_position_ids:
        position_ids = position_ids.clone()

    if reset_position_ids or reset_attention_mask:
        # Find indices where EOD token is.
        eod_index = position_ids[data == eod_token]
        # Detach indices from positions if going to modify positions.
        if reset_position_ids:
            eod_index = eod_index.clone()

        # Loop through EOD indices:
        prev_index = 0
        for j in range(eod_index.numel()):
            i = eod_index[j]
            # Mask attention loss.
            if reset_attention_mask and attention_mask is not None:
                attention_mask[0, (i + 1) :, : (i + 1)] = 0
            # Reset positions.
            if reset_position_ids:
                position_ids[(i + 1) :] -= i + 1 - prev_index
                prev_index = i + 1

    if attention_mask is not None:
        # Convert attention mask to binary:
        attention_mask = attention_mask < 0.5

    return attention_mask, loss_mask, position_ids
