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
"""Mindspeed global variables."""

_GLOBAL_ARGS = None


def get_args():
    assert _GLOBAL_ARGS is not None, 'global arguments is not initialized.'
    return _GLOBAL_ARGS


def set_global_variables(args):
    assert args is not None
    global _GLOBAL_ARGS
    _GLOBAL_ARGS = args


def set_args(args):
    global _GLOBAL_ARGS
    _GLOBAL_ARGS = args
