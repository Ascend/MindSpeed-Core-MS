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

"""Prompter for SFT."""

from dataclasses import dataclass

@dataclass
class PanguSftTemplate:
    system_token = "系统："
    user_token = "用户："
    assistant_token = "助手："
    tool_token = "工具："
    start_token = "[unused9]"
    end_token = "[unused10]"


class PanguPrompter():
    """
    Pangu prompter
    """
    def __init__(self, template, verbose: bool = False):
        self._verbose = verbose
        self.template = template
        self.user_role = "user"
        self.tool_role = "tool"
        self.assistant_role = "assistant"


@dataclass
class PanguLlama3SftTemplate:
    system_token = "<|start_header_id|>system<|end_header_id|>"
    user_token = "<|start_header_id|>user<|end_header_id|>"
    assistant_token = "<|start_header_id|>assistant<|end_header_id|>"
    begin_token = "<|begin_of_text|>"
    end_token = "<|end_of_text|>"
    eot_token = "<|eot_id|>"
    system = ""


class PanguLlama3Prompter():
    """
    Pangu Llama3 prompter
    """
    def __init__(self, template, verbose: bool = False):
        self._verbose = verbose
        self.template = template
        self.user_role = "user"
        self.tool_role = "tool"
        self.assistant_role = "assistant"

    def generate_training_prompt(self, system_prompt, messages) -> str:
        """generate training prompt"""
        prompt = (self.template.begin_token + "\n" + self.template.system_token +
                  "\n\n" + system_prompt + self.template.eot_token + "\n")

        for message in messages:
            if message["role"] == self.user_role or message["role"] == self.tool_role:
                prompt += self.template.user_token + "\n" + message["content"] + self.template.eot_token + "\n"
            else:
                prompt += self.template.assistant_token + "\n" + message["content"] \
                          + self.template.eot_token + "\n"
        return prompt


@dataclass
class PanguQwen15SftTemplate:
    system_token = "system"
    user_token = "user"
    assistant_token = "assistant"
    eos_token = "<|endoftext|>"
    start_token = "<|im_start|>"
    end_token = "<|im_end|>"
    system = "system You are a helpful assistant"


class PanguQwen15Prompter():
    """
    Pangu Qwen15 prompter
    """
    def __init__(self, template, verbose: bool = False):
        self._verbose = verbose
        self.template = template
        self.user_role = "user"
        self.tool_role = "tool"
        self.assistant_role = "assistant"

    def generate_training_prompt(self, system_prompt, messages) -> str:
        """generate training prompt"""
        prompt = (self.template.begin_token + "\n" + self.template.system_token +
                  "\n\n" + system_prompt + self.template.eot_token + "\n")

        for message in messages:
            if message["role"] == self.user_role or message["role"] == self.tool_role:
                prompt += self.template.user_token + "\n" + message["content"] + self.template.eot_token + "\n"
            else:
                prompt += self.template.assistant_token + "\n" + message["content"] \
                          + self.template.eot_token + "\n"
        return prompt
