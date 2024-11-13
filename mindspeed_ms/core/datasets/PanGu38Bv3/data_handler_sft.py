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

"""Data handle for sft."""

import os
import sys
import time
import glob
import json
import logging

import torch
import numpy as np
from datasets import load_dataset
from transformers import AddedToken

from mindspeed_ms.core.datasets import indexed_dataset
from .prompter_sft import PanguLlama3SftTemplate, PanguLlama3Prompter
from .prompter_sft import PanguQwen15SftTemplate, PanguQwen15Prompter
from .prompter_sft import PanguSftTemplate, PanguPrompter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__all__ = ["get_dataset_handler", "build_dataset"]

DEFAULT_CACHE_DIR = "~/tmp"


class BaseDatasetHandler():
    """
    a base handler to tokenize or/and prompt your own dataset
    """

    def __init__(self, args, raw_datasets, tokenizer, splitter):
        self.args = args
        self.tokenizer = tokenizer
        self.splitter = splitter
        self.raw_datasets = raw_datasets
        self.max_seq_len = args.seq_length
        self.tokenized_dataset = None

    @property
    def _unwrapped_tokenizer(self):
        """get huggingface tokenizer"""
        return self.tokenizer.tokenizer

    def get_tokenized_data(self):
        """get tokenized(and prompted) data"""
        columns = next(iter(self.raw_datasets)).keys()
        remove_columns = list(set(columns) - set(self.args.json_keys))
        proc_kwargs = {} if self.args.streaming else {"num_proc": self.args.workers}
        return self.raw_datasets.map(self._filter, remove_columns=remove_columns, **proc_kwargs)

    def serialize_to_disk(self):
        """save idx and bin to disk"""
        startup_start = time.time()
        # tokenize raw data generate model inputs
        if not self.tokenized_dataset:
            self.tokenized_dataset = self.get_tokenized_data()

        # output *.bin and *.idx files for megatron train
        output_bin_files = {}
        output_idx_files = {}
        builders = {}
        level = "document"
        if self.args.split_sentences:
            level = "sentence"

        logger.info("Vocab size: %s", self.tokenizer.vocab_size)
        logger.info("Output prefix: %s", self.args.output_prefix)
        for key in self.args.json_keys:
            output_bin_files[key] = f"{self.args.output_prefix}_{key}_{level}_{self.args.part}.bin"
            output_idx_files[key] = f"{self.args.output_prefix}_{key}_{level}_{self.args.part}.idx"
            # vocab_size=None : use int32 dtype for -100 will be used in labels
            builders[key] = indexed_dataset.IndexedDatasetBuilder(output_bin_files[key])
        startup_end = time.time()
        proc_start = time.time()
        total_bytes_processed = 0
        logger.info("Time to startup:%s", startup_end - startup_start)

        skip_num = 0
        pack_data = self.args.pack
        pack_lengths = {}
        pack_sentences = {}
        for key in self.args.json_keys:
            pack_lengths[key] = 0
            pack_sentences[key] = []
        for i, doc in enumerate(iter(self.tokenized_dataset), start=1):
            for key in self.args.json_keys:
                sentences = doc[key]
                if not sentences:
                    continue
                if pack_data:
                    for sentence in sentences:
                        if self.args.seq_length is not None and len(sentence) >= self.args.seq_length:
                            skip_num += 1
                            continue
                        # append
                        if (pack_lengths[key]+len(sentence)) > self.args.seq_length:
                            total_bytes_processed += len(pack_sentences[key]) * np.int32().itemsize
                            builders[key].add_item(torch.IntTensor(pack_sentences[key]))
                            pack_lengths[key] = len(sentence)
                            pack_sentences[key] = sentence
                            builders[key].end_document()
                        else:
                            if i == 1:
                                pack_sentences[key] = sentence
                                pack_lengths[key] = len(sentence)
                            else:
                                pack_sentences[key].extend(sentence)
                                pack_lengths[key] = pack_lengths[key] + len(sentence)
                else:
                    for sentence in sentences:
                        if self.args.seq_length is not None and len(sentence) >= self.args.seq_length:
                            skip_num += 1
                            continue

                        total_bytes_processed += len(sentence) * np.int32().itemsize
                        builders[key].add_item(torch.IntTensor(sentence))
                        builders[key].end_document()
            if i % self.args.log_interval == 0:
                current = time.time()
                elapsed = current - proc_start
                mbs = total_bytes_processed / elapsed / 1024 / 1024
                logger.info("Processed %s documents (%s docs/s, %s MB/s).", i, i / elapsed, mbs)

        logger.info("Skip %s sample exceeded seq-length(%s)", skip_num // 3, self.args.seq_length)
        for key in self.args.json_keys:
            builders[key].finalize(output_idx_files[key])

    def _tokenize(self, prompt):
        result = self._unwrapped_tokenizer(text=prompt)
        result["labels"] = result["input_ids"].copy()

        return result

    def _filter(self, sample):
        """prompt and tokenize"""
        return NotImplemented



class PanguInstructionHandler(BaseDatasetHandler):
    """
    a general instruction dataset handler
    """

    def __init__(self, args, raw_datasets, tokenizer, splitter):
        super().__init__(args, raw_datasets, tokenizer, splitter)
        self.prompter = PanguPrompter(PanguSftTemplate())
        self.train_on_inputs = False
        ################# add special tokens: [unused9-12] ###########################
        self.tokenizer.tokenizer.add_special_tokens({"additional_special_tokens": [AddedToken("[unused9]")]})
        self.tokenizer.tokenizer.add_special_tokens({"additional_special_tokens": [AddedToken("[unused10]")]})
        self.tokenizer.tokenizer.add_special_tokens({"additional_special_tokens": [AddedToken("[unused11]")]})
        self.tokenizer.tokenizer.add_special_tokens({"additional_special_tokens": [AddedToken("[unused12]")]})
        self.args.json_keys = ["input_ids", "attention_mask", "labels"]
        # use 'packed' string to mark that this is a packed dataset
        self.args.output_prefix = self.args.output_prefix + "_packed"
        self.ignored_label = -100
        self.is_multi_turn = self._is_muti_turn()
        print("length of tokenizer is, ", len(self.tokenizer.tokenizer))

    @property
    def _instruction_key(self) -> str:
        return "instruction"

    @property
    def _input_key(self) -> str:
        return "input"

    @property
    def _output_key(self) -> str:
        return "output"

    @property
    def _human_prefix(self) -> str:
        raise NotImplementedError

    @property
    def _assistant_prefix(self) -> str:
        raise NotImplementedError

    def _is_muti_turn(self) -> bool:
        return True

    def _format_msg(self, data):
        """format sample info"""
        messages = []
        turns = int(len(data) / 2)
        for i in range(turns):
            messages.append(data[i*2])
            messages.append(data[i*2+1])
        return messages

    def _filter(self, sample):
        messages = self._format_msg(sample['data'])
        # sys prompt
        # usr/tool/assistant prompt
        tokenized_full_prompt = {
            "input_ids": [],
            "attention_mask": [],
            "labels": []
        }
        full_prompt = ''
        # add sys token
        meta_prompts = sample["meta_prompt"] # 这是个list
        if not isinstance(meta_prompts, list):
            meta_prompts = [meta_prompts]

        prompt = ''
        for meta in meta_prompts:
            prompt += self.prompter.template.start_token + self.prompter.template.system_token + meta \
                     + self.prompter.template.end_token
        tokenized_prompt = self.tokenizer.tokenizer.encode(prompt)
        tokenized_full_prompt["input_ids"].extend(tokenized_prompt)
        tokenized_full_prompt["labels"].extend([self.ignored_label] * len(tokenized_prompt))
        full_prompt += prompt
        # add dialog token
        for message in messages:
            # usr/tool
            if message["role"] == self.prompter.user_role or message["role"] == self.prompter.tool_role:
                if message["role"] == self.prompter.user_role:
                    replaced_name = self.prompter.template.user_token
                else:
                    replaced_name = self.prompter.template.tool_token
                prompt = self.prompter.template.start_token + replaced_name + message["content"] \
                         + self.prompter.template.end_token
                tokenized_prompt = self.tokenizer.tokenizer.encode(prompt)
                tokenized_full_prompt["input_ids"].extend(tokenized_prompt)
                tokenized_full_prompt["labels"].extend([self.ignored_label] * len(tokenized_prompt))
                full_prompt += prompt
            else:
                answer = self.prompter.template.start_token + self.prompter.template.assistant_token
                tokenized_answer = self.tokenizer.tokenizer.encode(answer)
                tokenized_full_prompt["input_ids"].extend(tokenized_answer)
                tokenized_full_prompt["labels"].extend([self.ignored_label] * len(tokenized_answer))
                full_prompt += answer

                answer = message["content"] + self.prompter.template.end_token
                tokenized_answer = self.tokenizer.tokenizer.encode(answer)
                tokenized_full_prompt["input_ids"].extend(tokenized_answer)
                tokenized_full_prompt["labels"].extend(tokenized_answer)
                full_prompt += answer

        tokenized_full_prompt["attention_mask"] = [1] * len(tokenized_full_prompt["input_ids"])
        # add eod
        if self.args.append_eod:
            tokenized_full_prompt["input_ids"].append(self.tokenizer.eod)
            tokenized_full_prompt["attention_mask"].append(1)
            tokenized_full_prompt["labels"].append(self.tokenizer.eod)

        for key in self.args.json_keys:
            tokenized_full_prompt[key] = [tokenized_full_prompt[key]]
        assert len(tokenized_full_prompt["input_ids"]) == len(tokenized_full_prompt["attention_mask"])
        assert len(tokenized_full_prompt["input_ids"]) == len(tokenized_full_prompt["labels"])
        return tokenized_full_prompt


class PanguLlama3InstructionHandler(BaseDatasetHandler):
    """
    a general instruction dataset handler
    """

    def __init__(self, args, raw_datasets, tokenizer, splitter):
        super().__init__(args, raw_datasets, tokenizer, splitter)
        self.prompter = PanguLlama3Prompter(PanguLlama3SftTemplate())
        self.train_on_inputs = False
        self.args.json_keys = ["input_ids", "attention_mask", "labels"]
        # use 'packed' string to mark that this is a packed dataset
        self.args.output_prefix = self.args.output_prefix + "_packed"
        self.ignored_label = -100
        self.is_multi_turn = self._is_muti_turn()

    @property
    def _instruction_key(self) -> str:
        return "instruction"

    @property
    def _input_key(self) -> str:
        return "input"

    @property
    def _output_key(self) -> str:
        return "output"

    @property
    def _human_prefix(self) -> str:
        raise NotImplementedError

    @property
    def _assistant_prefix(self) -> str:
        raise NotImplementedError

    def _is_muti_turn(self) -> bool:
        return True

    def _format_msg(self, data):
        """format sample info"""
        messages = []
        turns = int(len(data) / 2)
        for i in range(turns):
            messages.append(data[i*2])
            messages.append(data[i*2+1])
        return messages

    # transform function
    def _filter(self, sample):
        messages = self._format_msg(sample['data'])
        # sys prompt
        sys_prompt = '\n'.join(sample['meta_prompt'])
        # usr/tool/assistant prompt
        tokenized_full_prompt = {
            "input_ids": [],
            "attention_mask": [],
            "labels": []
        }
        full_prompt = ''
        # add sys token
        prompt = (self.prompter.template.begin_token + self.prompter.template.system_token + "\n\n" + sys_prompt +
                  self.prompter.template.eot_token + "\n")
        tokenized_prompt = self.tokenizer.tokenizer.encode(prompt)
        tokenized_full_prompt["input_ids"].extend(tokenized_prompt)
        # ignore system prompt loss
        tokenized_full_prompt["labels"].extend([self.ignored_label] * len(tokenized_prompt))
        full_prompt += prompt
        # add dialog token different policy for different role
        for message in messages:
            # 处理[unused11],[unused12],替换为llama3中special token
            message["content"] = message["content"].replace("[unused11]", "<|reserved_special_token_11|>")
            message["content"] = message["content"].replace("[unused12]", "<|reserved_special_token_12|>")
            # usr/tool
            if message["role"] == self.prompter.user_role or message["role"] == self.prompter.tool_role:
                prompt = (self.prompter.template.user_token + "\n" + message["content"] +
                          self.prompter.template.eot_token + "\n")
                tokenized_prompt = self.tokenizer.tokenizer.encode(prompt)
                # check tokenizer, 有的tokenizer.json 在encode的时候会在前面加begin_token

                assert tokenized_prompt[0] != 128000
                tokenized_full_prompt["input_ids"].extend(tokenized_prompt)
                tokenized_full_prompt["labels"].extend([self.ignored_label] * len(tokenized_prompt))
                full_prompt += prompt
            else:
                answer = self.prompter.template.assistant_token + "\n" + message["content"] \
                          + self.prompter.template.eot_token + "\n"
                tokenized_answer = self.tokenizer.tokenizer.encode(answer)
                tokenized_full_prompt["input_ids"].extend(tokenized_answer)
                tokenized_full_prompt["labels"].extend(tokenized_answer)
                full_prompt += answer
        tokenized_full_prompt["attention_mask"] = [1] * len(tokenized_full_prompt["input_ids"])
        # add eod
        if self.args.append_eod:
            tokenized_full_prompt["input_ids"].append(self.tokenizer.eod)
            tokenized_full_prompt["attention_mask"].append(1)
            tokenized_full_prompt["labels"].append(self.tokenizer.eod)

        for key in self.args.json_keys:
            tokenized_full_prompt[key] = [tokenized_full_prompt[key]]
        assert len(tokenized_full_prompt["input_ids"]) == len(tokenized_full_prompt["attention_mask"])
        assert len(tokenized_full_prompt["input_ids"]) == len(tokenized_full_prompt["labels"])
        return tokenized_full_prompt


class PanguQwen15InstructionHandler(BaseDatasetHandler):
    """
    a general instruction dataset handler
    """

    def __init__(self, args, raw_datasets, tokenizer, splitter):
        super().__init__(args, raw_datasets, tokenizer, splitter)
        self.prompter = PanguQwen15Prompter(PanguQwen15SftTemplate())
        self.train_on_inputs = False
        # add special tokens: [unused11], [unused12]
        self.tokenizer.tokenizer.add_special_tokens({"additional_special_tokens": [AddedToken("[unused11]")]})
        self.tokenizer.tokenizer.add_special_tokens({"additional_special_tokens": [AddedToken("[unused12]")]})
        self.args.json_keys = ["input_ids", "attention_mask", "labels"]
        # use 'packed' string to mark that this is a packed dataset
        self.args.output_prefix = self.args.output_prefix + "_packed"
        self.ignored_label = -100
        self.is_multi_turn = self._is_muti_turn()

    @property
    def _instruction_key(self) -> str:
        return "instruction"

    @property
    def _input_key(self) -> str:
        return "input"

    @property
    def _output_key(self) -> str:
        return "output"

    @property
    def _human_prefix(self) -> str:
        raise NotImplementedError

    @property
    def _assistant_prefix(self) -> str:
        raise NotImplementedError

    def _is_muti_turn(self) -> bool:
        return True

    def _format_msg(self, data):
        """format sample info"""
        messages = []
        turns = int(len(data) / 2)
        for i in range(turns):
            messages.append(data[i*2])
            messages.append(data[i*2+1])
        return messages

    def _filter(self, sample):
        messages = self._format_msg(sample['data'])
        # sys prompt
        sys_prompt = '\n'.join(sample['meta_prompt'])
        # usr/tool/assistant prompt
        tokenized_full_prompt = {
            "input_ids": [],
            "attention_mask": [],
            "labels": []
        }
        full_prompt = ''
        # add sys token
        prompt = self.prompter.template.start_token + self.prompter.template.system_token + "\n" + sys_prompt \
                 + self.prompter.template.end_token + "\n"
        tokenized_prompt = self.tokenizer.tokenizer.encode(prompt)
        tokenized_full_prompt["input_ids"].extend(tokenized_prompt)
        tokenized_full_prompt["labels"].extend([self.ignored_label] * len(tokenized_prompt))
        full_prompt += prompt
        # add dialog token
        for message in messages:
            # usr/tool
            if message["role"] == self.prompter.user_role or message["role"] == self.prompter.tool_role:
                prompt = self.prompter.template.start_token + message["role"] + "\n" + message["content"] \
                         + self.prompter.template.end_token + "\n"
                tokenized_prompt = self.tokenizer.tokenizer.encode(prompt)
                tokenized_full_prompt["input_ids"].extend(tokenized_prompt)
                tokenized_full_prompt["labels"].extend([self.ignored_label] * len(tokenized_prompt))
                full_prompt += prompt
            else:
                # ignore assistant_token
                answer = self.prompter.template.start_token + self.prompter.template.assistant_token + "\n"
                tokenized_answer = self.tokenizer.tokenizer.encode(answer)
                tokenized_full_prompt["input_ids"].extend(tokenized_answer)
                tokenized_full_prompt["labels"].extend([self.ignored_label] * len(tokenized_answer))
                full_prompt += answer
                # real predict
                answer = message["content"] + self.prompter.template.end_token + "\n"
                tokenized_answer = self.tokenizer.tokenizer.encode(answer)
                tokenized_full_prompt["input_ids"].extend(tokenized_answer)
                tokenized_full_prompt["labels"].extend(tokenized_answer)
                full_prompt += answer
        tokenized_full_prompt["attention_mask"] = [1] * len(tokenized_full_prompt["input_ids"])
        # add eod
        if self.args.append_eod:
            tokenized_full_prompt["input_ids"].append(self.tokenizer.eod)
            tokenized_full_prompt["attention_mask"].append(1)
            tokenized_full_prompt["labels"].append(self.tokenizer.eod)

        for key in self.args.json_keys:
            tokenized_full_prompt[key] = [tokenized_full_prompt[key]]
        assert len(tokenized_full_prompt["input_ids"]) == len(tokenized_full_prompt["attention_mask"])
        assert len(tokenized_full_prompt["input_ids"]) == len(tokenized_full_prompt["labels"])
        return tokenized_full_prompt

def _get_handler_cls(handler_name=None):
    """choose dataset class by dataset_name"""
    current_module = sys.modules.get(__name__)
    if not current_module:
        raise Exception("current module not found")
    handler = getattr(current_module, handler_name, None)
    if handler is None:
        handler = GeneralPretrainHandler
    logger.info("dataset will use %s to handle dataset", handler.__name__)
    return handler


def get_dataset_handler(args, raw_dataset, tokenizer, splitter):
    """
    get a handler instance
    """
    handler = _get_handler_cls(args.handler_name)

    handler_instance = handler(args, raw_dataset, tokenizer, splitter)
    return handler_instance


def _get_data_format(files):
    """get format with largest number"""
    all_support_format = {
        'parquet': 'parquet',
        'arrow': 'arrow',
        'csv': 'csv',
        'json': 'json',
        'jsonl': 'json',
        'txt': 'text'
    }
    format_num = {}
    for file in files:
        ext = file.split('.')[-1]
        format_num[ext] = format_num.get(ext, 0) + 1
    exts_with_num = sorted(format_num.items(), key=lambda x: x[1], reverse=True)
    has_data_file = False
    for ext, _ in exts_with_num:
        if ext in all_support_format:
            has_data_file = True
            break
    return (ext, all_support_format.get(ext)) if has_data_file else (None, None)


def _has_py_script(input_name):
    """find if has python script file"""
    if os.path.isdir(input_name):
        dir_name = os.path.basename(input_name)
        has_py_script = os.path.exists(os.path.join(input_name, dir_name + '.py'))
    else:
        has_py_script = input_name.split('.')[-1] == 'py'
    return has_py_script


def build_dataset(args):
    """loading dataset by huggingface"""
    if args.handler_name == "MOSSInstructionHandler" or args.handler_name == "MOSSMultiTurnHandler":
        # for MOSS, streaming is needed.
        args.streaming = True
    if args.hf_datasets_params:
        with open(args.hf_datasets_params, 'r') as fin:
            param_dict = json.load(fin)
        return load_dataset(**param_dict)
    cache_dir = DEFAULT_CACHE_DIR
    split_flag = "train"
    load_from_local = os.path.exists(args.input)
    if load_from_local:
        if _has_py_script(args.input):
            logger.info("loading data from a local python script")
            raw_datasets = load_dataset(
                args.input,
                split=split_flag,
                num_proc=None if args.streaming else args.workers,
                cache_dir=cache_dir,
                streaming=args.streaming
            )
        else:
            data_files = [args.input] if os.path.isfile(args.input) else \
                glob.glob(os.path.join(args.input, '*'))
            ext, data_format = _get_data_format(data_files)
            filtered_data_files = list(filter(lambda x: x.split('.')[-1] == ext, data_files))
            if filtered_data_files:
                logger.info("loading data from local file, format: %s,"
                            " file num: %s", data_format, len(data_files))
                raw_datasets = load_dataset(
                    data_format,
                    split=split_flag,
                    data_files=filtered_data_files,
                    num_proc=None if args.streaming else args.workers,
                    cache_dir=cache_dir,
                    streaming=args.streaming
                )
            else:
                raise Exception("unknown local data!")
    else:
        logger.info("loading data from remote huggingface")
        raw_datasets = load_dataset(
            args.input,
            split=split_flag,
            num_proc=None if args.streaming else args.workers,
            cache_dir=cache_dir,
            streaming=args.streaming
        )
    return raw_datasets


def build_sft_dataset(args):
    """loading pangu sft dataset """
    cache_dir = DEFAULT_CACHE_DIR
    split_flag = "train"
    # load_from_local = os.path.exists(args.input)
    load_from_local = True
    if load_from_local:
        # raw_datasets = load_dataset(args.input)
        start = time.time()
        raw_datasets = load_dataset(
            'json',
            data_files=args.input
        )
        print(time.time()-start, raw_datasets)
        raw_datasets = raw_datasets['train']

    return raw_datasets
