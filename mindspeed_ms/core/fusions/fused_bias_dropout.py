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
# =========================================================================
"""Fused Bias Dropout."""

from mindspore import mint


# pylint: disable=R1705
def _bias_dropout_add_func(x_with_bias, residual, prob, training):
    """bias dropout add func."""

    x, bias = x_with_bias  # unpack

    residual = residual if residual.dtype == x.dtype else residual.to(x.dtype)

    # The Dropout operation, Residual Addition and the tensor returning can be
    # done generically outside the if statement, but that stops fusing of Bias
    # Addition-Dropout-Residual Addition operation. So doing it together inside
    # the conditional branch to improve performance
    if bias is not None:
        x = x + bias
        out = mint.nn.functional.dropout(x, p=prob, training=training)
        out = residual + out
        return out
    else:
        out = mint.nn.functional.dropout(x, p=prob, training=training)
        out = residual + out
        return out


def bias_dropout_add_unfused(training):
    """bias dropout unfused."""

    def _bias_dropout_add(x_with_bias, residual, prob):
        return _bias_dropout_add_func(x_with_bias, residual, prob, training)

    return _bias_dropout_add


def get_bias_dropout_add(training, fused):
    """get bias dropout add"""

    if fused:
        raise NotImplementedError("fused dropout is not supported now.")
    return bias_dropout_add_unfused(training)
