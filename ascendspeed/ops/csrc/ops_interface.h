/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef ASCENDSPEED_OPS_CSRC_COMMON_H
#define ASCENDSPEED_OPS_CSRC_COMMON_H

at::Tensor npu_flash_attention(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    const c10::optional<at::Tensor> &atten_mask, const c10::optional<at::Tensor> &alibi_mask,
    float scale_value = 1.0, float q_scale = 1.0, int64_t head_num = 1,
    std::string io_layout = "BNSD", float keep_prob = 1.0, int64_t pre_tokens = 2147483647, int64_t next_tokens = 1,
    int64_t precise_mode = 0, int64_t groups = -1,
    bool sync = false, bool parallel = true);

#endif
