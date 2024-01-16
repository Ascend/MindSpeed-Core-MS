// Copyright (c) 2023 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <string>
#include <vector>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/core/npu/DeviceUtils.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include <torch/script.h>
#include <torch/custom_class.h>
#include <torch_npu/csrc/aten/NPUGeneratorImpl.h>
#include "../ops_interface.h"
#include "inc/adapter.h"

#ifdef ENABLE_ATB
#include "inc/atb_adapter.h"
#include "atb/operation.h"
#include "atb/train_op_params.h"
#endif 

using namespace std;

namespace {
const static int QKV_DIM_NUM = 3;
const static int QKV_DIM_NUM_BNSD = 4;
const static int BNSD_DIM_B = 0;
const static int BNSD_DIM_S = 2;
const static int BNSD_DIM_D = 3;
const static int BSH_DIM_B = 0;
const static int BSH_DIM_S = 1;
const static int BSH_DIM_H = 2;
const static int SBH_DIM_B = 1;
const static int SBH_DIM_S = 0;
const static int SBH_DIM_H = 2;

enum FlashAttentionFormat : int {
    BNSD = 0,
    BSH,
    SBH
}; 

enum class DropOutStatus {
    DROPOUT_NORMAL = 0,
    DROPOUT_NONE,
    DROPOUT_ALL
};

DropOutStatus get_status(double keep_prob)
{
    if (keep_prob == 0) {
        return DropOutStatus::DROPOUT_ALL;
    }
    if (keep_prob == 1.) {
        return DropOutStatus::DROPOUT_NONE;
    }
    return DropOutStatus::DROPOUT_NORMAL;
}

at::Tensor gen_mask_impl(const at::Tensor &self, const at::Scalar &keep_prob, const at::Scalar &seed,
    const int64_t offset, const int64_t numels)
{
    int64_t length = (numels + 256 - 1) / 256 * 256 / 8;
    c10::TensorOptions options = self.options();
    at::Tensor mask = at::empty(at::IntArrayRef{length + 32}, options.dtype(at::kByte));
    at::SmallVector<int64_t, N> offsetList = {0, offset};
    const int64_t seed1 = 0;
    at_npu::native::OpCommand cmd;
    cmd.Name("StatelessDropOutGenMask")
        .Input(at::IntArrayRef{numels})
        .Input(keep_prob, self.scalar_type(), at_npu::native::CompileType::MEMORY_HOST_COMPILE_DEPENDENT)
        .Input(seed, at::ScalarType::Int)
        .Input(at::Scalar(seed1), at::ScalarType::Int)
        .Input(offsetList, at::kLong, at_npu::native::CompileType::MEMORY_HOST_COMPILE_INDEPENDENT)
        .Output(mask)
        .Run();
    return mask;
}

at::Tensor gen_mask_dispatch(const at::Tensor &self, const at::Scalar &keep_prob, const at::Scalar &seed,
    const int64_t offset, const int64_t numels, const bool gen_mask_parallel, const bool sync)
{
    at::Tensor mask;
    mask = gen_mask_impl(self, keep_prob, seed, offset, numels);
    return mask;
}

void InferShapeFlashAttention(c10::SmallVector<int64_t, N> &size, int64_t io_layout, int64_t head_num, const at::Tensor &query)
    {
        if (io_layout == BNSD) {
            // BNSD
            size = {query.size(BNSD_DIM_B), head_num, query.size(BNSD_DIM_S)};
        } else if (io_layout == BSH) {
            // BSH
            size = {query.size(BSH_DIM_B), head_num, query.size(BSH_DIM_S)};
        } else if (io_layout == SBH) {
            // SBH
            size = {query.size(SBH_DIM_B), head_num, query.size(SBH_DIM_S)};
        }
    }

void CheckFlashAttention(const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
                         int64_t head_num, int64_t io_layout)
{
    TORCH_CHECK(query.scalar_type() == at::ScalarType::Half || query.scalar_type() == at::ScalarType::BFloat16,
                "Input Q dtype ", query.scalar_type(),
                " invalid, should be float16 or bfloat16");
    TORCH_CHECK(key.scalar_type() == at::ScalarType::Half || key.scalar_type() == at::ScalarType::BFloat16,
                "Input K dtype ", key.scalar_type(),
                " invalid, should be float16 or bfloat16");
    TORCH_CHECK(value.scalar_type() == at::ScalarType::Half || value.scalar_type() == at::ScalarType::BFloat16,
                "Input V dtype ", value.scalar_type(),
                " invalid, should be float16 or bfloat16");
    int dim_s;
    int dim_b;
    int dim_h;
    int dim_num = QKV_DIM_NUM;
    if (io_layout == BNSD) {
        // BNSD
        dim_s = BNSD_DIM_S;
        dim_b = BNSD_DIM_B;
        dim_h = BNSD_DIM_D;
        dim_num = QKV_DIM_NUM_BNSD;
    } else if (io_layout == BSH) {
        // BSH
        dim_s = BSH_DIM_S;
        dim_b = BSH_DIM_B;
        dim_h = BSH_DIM_H;
    } else if (io_layout == SBH) {
        // SBH
        dim_s = SBH_DIM_S;
        dim_b = SBH_DIM_B;
        dim_h = SBH_DIM_H;
    }
    TORCH_CHECK(
        query.dim() == dim_num,
        "Input Q dim num ", query.dim(), " invalid, should be ", dim_num);
    TORCH_CHECK(
        key.dim() == dim_num,
        "Input K dim num ", key.dim(), " invalid, should be ", dim_num);
    TORCH_CHECK(
        value.dim() == dim_num,
        "Input V dim num ", value.dim(), " invalid, should be ", dim_num);
    auto batch_size = query.size(dim_b);
    auto head_dim_size = query.size(dim_h);

    TORCH_CHECK(key.size(dim_b) == batch_size &&
                key.size(dim_h) == head_dim_size,
                "Shape of input Q and input K should be same in batch_size_dim and head_dim");
    TORCH_CHECK(value.size(dim_b) == batch_size &&
                value.size(dim_h) == head_dim_size,
                "Shape of input Q and input V should be same in batch_size_dim and head_dim");
    TORCH_CHECK(value.size(dim_s) == key.size(dim_s),
                "Shape of input K and input V should be same in batch_size_dim and head_dim");
}

void CheckFlashAttentionBackward(const at::Tensor &dy, const at::Tensor &softmax_log_max_sum,
                                 const at::Tensor &attention_out, const at::Tensor &query,
                                 const at::Tensor &key, const at::Tensor &value,
                                 int64_t head_num, int64_t io_layout)
{   CheckFlashAttention(query, key, value, head_num, io_layout);
    TORCH_CHECK(dy.scalar_type() == at::ScalarType::Half || query.scalar_type() == at::ScalarType::BFloat16,
                "Input dy dtype ", dy.scalar_type(), " invalid, should be float16 or bfloat16");
    TORCH_CHECK(softmax_log_max_sum.scalar_type() == at::ScalarType::Float,
                "Input softmax_log_max_sum dtype ", softmax_log_max_sum.scalar_type(),
                " invalid, should be float ");
    TORCH_CHECK(attention_out.scalar_type() == at::ScalarType::Half || query.scalar_type() == at::ScalarType::BFloat16,
                "Input attention_out dtype ", attention_out.scalar_type(),
                " invalid, should be float16 or bfloat16");
}
} // namespace

std::tuple<at::Tensor, at::Tensor> flash_attention(const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
                                                   const c10::optional<at::Tensor> &atten_mask, const c10::optional<at::Tensor> &alibi_mask,
                                                   const c10::optional<at::Tensor> &drop_mask, float scale_value, int64_t head_num,
                                                   int64_t io_layout, float keep_prob, int64_t pre_tokens, int64_t next_tokens,
                                                   int64_t precise_mode, int64_t groups)
{
#ifndef ENABLE_ATB
    TORCH_CHECK(false, "flash_attention not implemented");
#else 
    atb::train::FlashAttentionParam param;
    param.scaleValue = scale_value;
    param.headNum = head_num;
    param.preTokens = pre_tokens;
    param.nextTokens = next_tokens;
    param.preciseMode = precise_mode;
    param.ioLayout = (atb::train::FlashAttentionParam::IoLayout)io_layout;
    param.keepProb = keep_prob;
    param.groups = groups;

    //infer shape
    CheckFlashAttention(query, key, value, head_num, io_layout);
    c10::SmallVector<int64_t, N> tensor_softmax_shape;
    InferShapeFlashAttention(tensor_softmax_shape, io_layout, head_num, query);

    //apply tensor
    at::Tensor tensor_softmax = CreateAtTensor(tensor_softmax_shape, 
                                               at::ScalarType::Float);
    at::Tensor tensor_attention_out = CreateAtTensor(query.sizes(),
                                                     query.scalar_type());
    //set input and output
    ParamSetter paramsetter;
    paramsetter.Input(query)
               .Input(key)
               .Input(value)
               .Input(atten_mask)
               .Input(alibi_mask)
               .Input(drop_mask)
               .Output(tensor_attention_out)
               .Output(tensor_softmax);

    RUN_ATB_CMD(param, paramsetter, "fa_forward")
    return std::make_tuple(tensor_attention_out, tensor_softmax);
#endif
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> flash_attention_grad(const at::Tensor &dy, const at::Tensor &softmax_log_max_sum, const at::Tensor &attention_out,
                                                                    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
                                                                    const c10::optional<at::Tensor> &atten_mask, const c10::optional<at::Tensor> &alibi_mask,
                                                                    const c10::optional<at::Tensor> &drop_mask, float scale_value, int64_t head_num, int64_t io_layout,
                                                                    float keep_prob, int64_t pre_tokens, int64_t next_tokens, int64_t precise_mode, int64_t groups)
{
#ifndef ENABLE_ATB
    TORCH_CHECK(false, "flash_attention_grad not implemented");
#else
    atb::train::FlashAttentionBackwardParam param;
    param.scaleValue = scale_value;
    param.headNum = head_num;
    param.preTokens = pre_tokens;
    param.nextTokens = next_tokens;
    param.preciseMode = precise_mode;
    param.ioLayout = (atb::train::FlashAttentionBackwardParam::IoLayout)io_layout;
    param.keepProb = keep_prob;
    param.groups = groups;

    CheckFlashAttentionBackward(dy, softmax_log_max_sum,
                                attention_out, query,
                                key, value,
                                head_num, io_layout);

    at::Tensor tensor_query_grad = CreateAtTensor(query.sizes(), query.scalar_type());
    at::Tensor tensor_key_grad = CreateAtTensor(query.sizes(), query.scalar_type());
    at::Tensor tensor_value_grad = CreateAtTensor(query.sizes(), query.scalar_type());

    ParamSetter paramsetter;
    paramsetter.Input(dy)
               .Input(softmax_log_max_sum)
               .Input(attention_out)
               .Input(query)
               .Input(key)
               .Input(value)
               .Input(atten_mask)
               .Input(alibi_mask)
               .Input(drop_mask)
               .Output(tensor_query_grad)
               .Output(tensor_key_grad)
               .Output(tensor_value_grad);

    RUN_ATB_CMD(param, paramsetter, "fa_backward");
    return std::make_tuple(tensor_query_grad, tensor_key_grad, tensor_value_grad);
#endif
}

std::tuple<at::Tensor, int64_t, int64_t, int64_t> gen_mask(const at::Tensor &self, double keep_prob,
    int64_t head_num, std::string input_layout, bool gen_mask_parallel, bool sync)
{
    int64_t seed;
    int64_t offset;
    int64_t numels;
    at::Tensor drop_mask;
    if (input_layout == "BSH") {
        numels = self.size(0) * head_num * self.size(1) * self.size(1); // [B,N,S,S]
    } else if (input_layout == "SBH") {
        numels = self.size(1) * head_num * self.size(0) * self.size(0); // [B,N,S,S]
    } else if (input_layout == "BNSD") {
        numels = self.size(0) * self.size(1) * self.size(2) * self.size(2); // [B,N,S,S]
    }
    int64_t length = (numels + 256 - 1) / 256 * 256 / 8;
    length += 32;
    if (get_status(keep_prob) == DropOutStatus::DROPOUT_NORMAL) {
        const auto gen = at_npu::detail::getDefaultNPUGenerator();
        auto pair = at::check_generator<at_npu::NPUGeneratorImpl>(gen)->philox_engine_inputs(10);
        seed = pair.first;
        offset = pair.second;
        drop_mask = gen_mask_dispatch(self, at::Scalar(keep_prob), at::Scalar(seed),
            offset, numels, gen_mask_parallel, sync);
    } else if (get_status(keep_prob) == DropOutStatus::DROPOUT_ALL) {
        drop_mask = at::zeros(at::IntArrayRef{length}, self.options().dtype(at::kByte));
    }

    return std::make_tuple(drop_mask, seed, offset, numels);
}

at::Tensor exist_gen_mask(const at::Tensor &self, double keep_prob, bool gen_mask_parallel, bool sync,
                          int64_t seed, int64_t offset, int64_t numels)
{
    int64_t length = (numels + 256 - 1) / 256 * 256 / 8;
    length += 32;
    at::Tensor drop_mask;
    if (get_status(keep_prob) == DropOutStatus::DROPOUT_NORMAL) {
        drop_mask = gen_mask_dispatch(self, at::Scalar(keep_prob), at::Scalar(seed), offset, numels,
                                      gen_mask_parallel, sync);
    } else if (get_status(keep_prob) == DropOutStatus::DROPOUT_ALL) {
        drop_mask = at::zeros(at::IntArrayRef{length}, self.options().dtype(at::kByte));
    }
    return drop_mask;
}
