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

"""Sparse Attention Score."""
from multiprocessing import Pool
import numpy as np
from scipy import stats

from mindspore import jit, JitConfig
from mindspore import Tensor, ops
import mindspore.common.dtype as mstype
from mindspore.ops.auto_generate.gen_ops_prim import FlashAttentionScore
from mindspore.ops.auto_generate import FlashAttentionScoreGrad

class Block:
    """Mask Block"""
    def __init__(self, x1, x2, y1, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

        self.a = abs(x2 - x1)
        self.b = abs(y2 - y1)

        self.p1 = np.array([x1, y1])
        self.p2 = np.array([x1, y2])
        self.p3 = np.array([x2, y1])
        self.p4 = np.array([x2, y2])

    def get_area(self):
        return self.a * self.b

    def get_side_ratio(self):
        if self.b == 0:
            return 0
        return self.a / self.b

class FindBlockArgs:
    """Find Block Args"""
    def __init__(self, data_2d_arg, sep_value_arg, init_y_arg, init_x_arg):
        self.data_2d_arg = data_2d_arg
        self.sep_value_arg = sep_value_arg
        self.init_y_arg = init_y_arg
        self.init_x_arg = init_x_arg

class CompileUtil():
    """Compile Util"""
    def __init__(self):
        self.skip_calc_flag = False

    def compile_mask(self, mask, q_blocksize=128, kv_blocksize=128, search_num=20):
        '''Compile mask to block list'''
        if mask is None:
            raise ValueError("The input mask can not be None for compile_mask.")

        self.skip_calc_flag = False

        coarse_mask = self.generate_coarse_mask(mask, q_blocksize=q_blocksize, kv_blocksize=kv_blocksize)
        data_2d = self.coarse_mask_to_decompose_data(coarse_mask)
        if self.skip_calc_flag:
            return None

        blocks = []
        searches = search_num
        (blocks, _) = self.decompose(data_2d, searches)

        # save blocks to file
        block_array = np.zeros((len(blocks), 4))
        # blocks: (x1, y1), (x2, y2); p1,p2,p3,p4
        for i in range(len(blocks)):
            block_array[i][0] = blocks[i].x1
            block_array[i][1] = blocks[i].y1
            block_array[i][2] = blocks[i].x2
            block_array[i][3] = blocks[i].y2

        blockpos_list = block_array.astype(int)

        block_num = len(blockpos_list)
        new_blockpos_list = []
        for i in range(block_num):
            x1, y1, x2, y2 = blockpos_list[i]
            #blocks: (x1, y1), (x2, y2); p1,p2,p3,p4
            #x1,x2是第x1, x2行，作用到q上，y1,y2是列，作用到kv上
            x1 = x1.item() * q_blocksize
            x2 = (x2.item() + 1) * q_blocksize
            y1 = y1.item() * kv_blocksize
            y2 = (y2.item() + 1) * kv_blocksize
            new_blockpos_list.append([x1, y1, x2, y2])

        return Tensor(new_blockpos_list)

    def coarse_mask_to_decompose_data(self, coarse_mask):
        '''get coarse mask'''
        array_data = []
        for x in range(coarse_mask.shape[0]):
            for y in range(coarse_mask.shape[1]):
                if coarse_mask[x][y] == 1:
                    array_data.append((x, y))

        array_data = np.array(array_data)
        if array_data.shape[0] == 0:
            self.skip_calc_flag = True
            return None

        data_2d = np.hstack((array_data, np.zeros((array_data.shape[0], 1))))

        return data_2d

    def generate_coarse_mask(self, mask, q_blocksize, kv_blocksize):
        '''generate coarse mask'''
        row_num = mask.shape[0]
        col_num = mask.shape[1]
        if row_num % q_blocksize != 0 or col_num % kv_blocksize != 0:
            raise ValueError("The row and column number of the input matrix must be divisible by the basic block size")

        mask = 1 - mask.copy().asnumpy()
        mask = mask.reshape(
            row_num // q_blocksize, q_blocksize, col_num // kv_blocksize, kv_blocksize
        )
        mask = np.transpose(mask, (0, 2, 1, 3))

        mask_block_sum = np.apply_over_axes(np.sum, mask, [-2, -1]) # [row_num//q_blocksize, col_num//kv_blocksize]
        coarse_mask = mask_block_sum > 0
        coarse_mask = coarse_mask.squeeze().astype(int)

        return coarse_mask

    def decompose(self, data_2d_global_arg, n_searches_per_step):
        '''decompose the input mask'''
        data_2d_global = data_2d_global_arg.copy()
        pool = Pool()
        sep_value = self.get_separation_value(data_2d_global)

        n_sqr = 0
        n_sqr_empty = 0
        recs = []

        #   Loop
        while True:
            # Select Data which is empty
            condition_sqr = data_2d_global[:, 2] == n_sqr_empty
            data_2d = data_2d_global[condition_sqr, :]

            # Break condition
            if data_2d.shape[0] == 0:
                break

            # Create args (SERIAL)
            r_args = []
            for i in range(n_searches_per_step):
                rand_point = int(np.random.rand() * len(data_2d))
                init_x = data_2d[rand_point][0]
                init_y = data_2d[rand_point][1]
                r_args.append(FindBlockArgs(data_2d, sep_value, init_y, init_x))

            recs_temp = pool.map(self.find_block, r_args)

            features = np.zeros(shape=[n_searches_per_step, 3])

            for i in range(n_searches_per_step):
                features[i, 0] = i
                features[i, 1] = recs_temp[i].get_area()
                features[i, 2] = recs_temp[i].get_side_ratio()
            # Max
            max_sqr_index = np.where(features[:, 1] == features[:, 1].max())[0][0]

            n_sqr += 1
            self.save_block(data_2d_global, recs_temp[max_sqr_index], n_sqr)

            recs.append(recs_temp[max_sqr_index])

        return recs, sep_value

    def get_separation_value(self, data_2d_global_arg):
        '''get speration value'''
        n_sample = 100
        x_data = np.unique(np.sort(data_2d_global_arg[:, 0]))
        y_data = np.unique(np.sort(data_2d_global_arg[:, 1]))

        diffs_x = np.zeros(shape=[n_sample])
        diffs_y = np.zeros(shape=[n_sample])

        for p in range(n_sample):
            x_rand_num = int(np.random.rand() * (len(x_data) - 1))
            y_rand_num = int(np.random.rand() * (len(y_data) - 1))
            diffs_x[p] = np.abs(x_data[x_rand_num] - x_data[x_rand_num + 1])
            diffs_y[p] = np.abs(y_data[y_rand_num] - y_data[y_rand_num + 1])

        sep_value_val = (stats.mode(diffs_x).mode + stats.mode(diffs_y).mode) / 2
        return sep_value_val

    def find_block(self, find_block_args: FindBlockArgs):
        '''find block'''
        # args:
        data_2d_arg = find_block_args.data_2d_arg
        sep_value_arg = find_block_args.sep_value_arg
        init_y_arg = find_block_args.init_y_arg
        init_x_arg = find_block_args.init_x_arg

        # work:
        all_x_points = np.sort(data_2d_arg[data_2d_arg[:, 1] == init_y_arg], axis=0)
        all_y_points = np.sort(data_2d_arg[data_2d_arg[:, 0] == init_x_arg], axis=0)

        init_x_index = np.where(all_x_points[:, 0] == init_x_arg)[0][0]
        init_y_index = np.where(all_y_points[:, 1] == init_y_arg)[0][0]

        dist_l = self.get_dist_left(all_x_points, init_x_index, sep_value_arg)
        dist_r = self.get_dist_right(all_x_points, init_x_index, sep_value_arg)

        f_index_l = init_x_index - dist_l
        f_index_r = dist_r + init_x_index

        lr_range = range(f_index_l, f_index_r)
        all_x_points = all_x_points[lr_range, :]

        dist_d = self.get_dist_down(all_y_points, init_y_index, sep_value_arg)
        dist_u = self.get_dist_up(all_y_points, init_y_index, sep_value_arg)

        f_index_d = init_y_index - dist_d
        f_index_u = dist_u + init_y_index

        du_range = range(f_index_d, f_index_u)
        all_y_points = all_y_points[du_range, :]

        # Re calc indexes
        init_x_index = np.where(all_x_points[:, 0] == init_x_arg)[0][0]
        init_y_index = np.where(all_y_points[:, 1] == init_y_arg)[0][0]

        # Has a hole? for each x vector > > >
        final_x_min = np.zeros(shape=[2])
        final_x_max = np.zeros(shape=[2])
        # # Down
        final_x_min[0], final_x_max[0], final_y_down = self.get_final_xy_index_down(data_2d_arg,
                                                                                    all_y_points,
                                                                                    init_y_index,
                                                                                    dist_l,
                                                                                    dist_r,
                                                                                    sep_value_arg)
        # # Up
        final_x_min[1], final_x_max[1], final_y_up = self.get_final_xy_index_up(data_2d_arg,
                                                                                all_y_points,
                                                                                init_y_index,
                                                                                dist_l,
                                                                                dist_r,
                                                                                sep_value_arg)
        # Square/Block Data
        x1_out = final_x_min.max()
        x2_out = final_x_max.min()
        y1_out = final_y_down
        y2_out = final_y_up

        return Block(x1_out, x2_out, y1_out, y2_out)

    def save_block(self, data_2d_global_arg, block: Block, block_id):
        '''Save block'''
        # Write Condition
        condition = (((data_2d_global_arg[:, 0] >= block.x1) & (data_2d_global_arg[:, 0] <= block.x2))
                     & ((data_2d_global_arg[:, 1] >= block.y1) & (data_2d_global_arg[:, 1] <= block.y2)))
        data_2d_global_arg[condition, 2] = block_id

    def get_dist_left(self, all_x_points_arg, init_x_index_arg, sep_value):
        '''Get dist left'''
        error_ratio = 0.05
        error_ratio_sup = sep_value * (1 + error_ratio)
        error_ratio_inf = sep_value * (1 - error_ratio)
        # Left
        l_lim = 0
        index_val = init_x_index_arg
        while index_val > l_lim:
            diff_bound_val = abs(all_x_points_arg[index_val, 0] - all_x_points_arg[index_val - 1, 0])
            if diff_bound_val >= error_ratio_sup or diff_bound_val <= error_ratio_inf:
                break
            index_val = index_val - 1

        f_index_l_val = index_val
        dist_l_val = init_x_index_arg - f_index_l_val
        return dist_l_val

    def get_dist_right(self, all_x_points_arg, init_x_index_arg, sep_value):
        '''get dist right'''
        error_ratio = 0.05
        error_ratio_sup = sep_value * (1 + error_ratio)
        error_ratio_inf = sep_value * (1 - error_ratio)
        # Right
        r_lim = len(all_x_points_arg) - 1
        index_val = init_x_index_arg
        while index_val < r_lim:
            diff_bound = abs(all_x_points_arg[index_val, 0] - all_x_points_arg[index_val + 1, 0])
            if diff_bound >= error_ratio_sup or diff_bound <= error_ratio_inf:
                break
            index_val = index_val + 1

        f_index_r_val = index_val + 1
        dist_r_val = f_index_r_val - init_x_index_arg
        return dist_r_val

    def get_dist_down(self, all_y_points_arg, init_y_index_arg, sep_value):
        '''get dist down'''
        error_ratio = 0.05
        error_ratio_sup = sep_value * (1 + error_ratio)
        error_ratio_inf = sep_value * (1 - error_ratio)
        # Left
        d_lim = 0
        index_val = init_y_index_arg
        while index_val > d_lim:
            diff_bound = abs(all_y_points_arg[index_val, 1] - all_y_points_arg[index_val - 1, 1])
            if diff_bound >= error_ratio_sup or diff_bound <= error_ratio_inf:
                break
            index_val = index_val - 1

        f_index_d_val = index_val
        dist_d_val = init_y_index_arg - f_index_d_val
        return dist_d_val

    def get_dist_up(self, all_y_points_arg, init_y_index_arg, sep_value):
        '''get dist up'''
        error_ratio = 0.05
        error_ratio_sup = sep_value * (1 + error_ratio)
        error_ratio_inf = sep_value * (1 - error_ratio)
        # Right
        u_lim = len(all_y_points_arg) - 1
        index_val = init_y_index_arg
        while index_val < u_lim:
            diff_bound = abs(all_y_points_arg[index_val, 1] - all_y_points_arg[index_val + 1, 1])
            if diff_bound >= error_ratio_sup or diff_bound <= error_ratio_inf:
                break
            index_val = index_val + 1

        f_index_u_val = index_val + 1
        dist_u_val = f_index_u_val - init_y_index_arg
        return dist_u_val

    def is_broken(self, vector_to_test, sep_value):
        '''is broken'''
        error_ratio = 0.05
        error_ratio_sup = sep_value * (1 + error_ratio)
        error_ratio_inf = sep_value * (1 - error_ratio)
        it_is = False

        for i in range(len(vector_to_test) - 1):
            diff_val = abs(vector_to_test[i] - vector_to_test[i + 1])
            if diff_val <= error_ratio_sup:
                if diff_val >= error_ratio_inf:
                    it_is = False
                else:
                    it_is = True
                    break
            else:
                it_is = True
                break

        return it_is

    def get_final_index_down(self, data_2d_arg, all_y_points_arg, init_y_index_arg, dist_l_arg, dist_r_arg, sep_value):
        '''get final index down'''
        # Down
        down_lim = 0
        index = init_y_index_arg
        while index >= down_lim:
            temp_y = all_y_points_arg[index, 1]
            all_x_points_arg = np.sort(data_2d_arg[data_2d_arg[:, 1] == temp_y], axis=0)

            temp_x = all_y_points_arg[index, 0]
            temp_x_index = np.where(all_x_points_arg[:, 0] == temp_x)[0][0]

            index_lim_sup = temp_x_index + dist_r_arg
            index_lim_inf = temp_x_index - dist_l_arg

            if index_lim_inf < 0:
                index_lim_inf = 0

            if index_lim_sup > len(all_x_points_arg):
                index_lim_sup = len(all_x_points_arg)

            temp_range_lr = range(index_lim_inf, index_lim_sup)

            just_x = all_x_points_arg[temp_range_lr, 0]
            if self.is_broken(just_x, sep_value):
                break
            index = index - 1

        final_index_val = index + 1
        return final_index_val

    def get_final_index_up(self, data_2d_arg, all_y_points_arg, init_y_index_arg, dist_l_arg, dist_r_arg, sep_value):
        '''get final index up'''
        # Up
        up_lim = len(all_y_points_arg) - 1
        index = init_y_index_arg
        while index <= up_lim:
            temp_y = all_y_points_arg[index, 1]
            all_x_points = np.sort(data_2d_arg[data_2d_arg[:, 1] == temp_y], axis=0)

            temp_x = all_y_points_arg[index, 0]
            temp_x_index = np.where(all_x_points[:, 0] == temp_x)[0][0]

            index_lim_sup = temp_x_index + dist_r_arg
            index_lim_inf = temp_x_index - dist_l_arg

            if index_lim_inf < 0:
                index_lim_inf = 0

            if index_lim_sup > len(all_x_points):
                index_lim_sup = len(all_x_points)

            temp_range_lr = range(index_lim_inf, index_lim_sup)

            just_x = all_x_points[temp_range_lr, 0]
            if self.is_broken(just_x, sep_value):
                break
            index = index + 1

        final_index_val = index - 1
        return final_index_val

    def get_final_xy_index_down(self, data_2d_arg, all_y_points_arg,
                                init_y_index_arg, dist_l_arg, dist_r_arg, sep_value):
        '''get final xy index down'''
        # Down
        final_index = self.get_final_index_down(data_2d_arg,
                                                all_y_points_arg,
                                                init_y_index_arg,
                                                dist_l_arg,
                                                dist_r_arg,
                                                sep_value)

        # ---- last step
        temp_y = all_y_points_arg[final_index, 1]
        all_x_points = np.sort(data_2d_arg[data_2d_arg[:, 1] == temp_y], axis=0)

        # ---- plot
        temp_x = all_y_points_arg[final_index, 0]
        temp_x_index = np.where(all_x_points[:, 0] == temp_x)[0][0]

        index_lim_sup = temp_x_index + dist_r_arg
        index_lim_inf = temp_x_index - dist_l_arg

        if index_lim_inf < 0:
            index_lim_inf = 0

        if index_lim_sup > len(all_x_points):
            index_lim_sup = len(all_x_points)

        temp_range_lr = range(index_lim_inf, index_lim_sup)

        final_x_min = all_x_points[temp_range_lr, 0].min()
        final_x_max = all_x_points[temp_range_lr, 0].max()
        final_y_down = temp_y
        return final_x_min, final_x_max, final_y_down

    def get_final_xy_index_up(self, data_2d_arg, all_y_points_arg,
                              init_y_index_arg, dist_l_arg, dist_r_arg, sep_value):
        '''get final xy index up'''
        # Up
        final_index = self.get_final_index_up(data_2d_arg,
                                              all_y_points_arg,
                                              init_y_index_arg,
                                              dist_l_arg,
                                              dist_r_arg,
                                              sep_value)
        # ---- last step
        temp_y = all_y_points_arg[final_index, 1]
        all_x_points = np.sort(data_2d_arg[data_2d_arg[:, 1] == temp_y], axis=0)

        # ---- plot
        temp_x = all_y_points_arg[final_index, 0]
        temp_x_index = np.where(all_x_points[:, 0] == temp_x)[0][0]

        index_lim_sup = temp_x_index + dist_r_arg
        index_lim_inf = temp_x_index - dist_l_arg

        if index_lim_inf < 0:
            index_lim_inf = 0

        if index_lim_sup > len(all_x_points):
            index_lim_sup = len(all_x_points)

        temp_range_lr = range(index_lim_inf, index_lim_sup)

        final_x_min = all_x_points[temp_range_lr, 0].min()
        final_x_max = all_x_points[temp_range_lr, 0].max()
        final_y_up = temp_y
        return final_x_min, final_x_max, final_y_up

class SparseAttentionScore():
    """Sequence parallelism with sparse attention score."""
    def __init__(self,
                 head_num,
                 keep_prob=1.0,
                 scale_value=1.0,
                 pre_tokens=2147483647,
                 next_tokens=2147483647,
                 inner_precise=0,
                 input_layout="SBH",
                 sparse_mode=0,
                 attn_mask=None):
        self.head_num = head_num
        self.keep_prob = keep_prob
        self.scale_value = scale_value
        self.pre_tokens = pre_tokens
        self.next_tokens = next_tokens
        self.inner_precise = inner_precise
        self.input_layout = input_layout
        self.sparse_mode = sparse_mode
        if self.sparse_mode != 0:
            raise ValueError('Only sparse_mode = 0 is supported for SparseAttentionScore.')
        self.attn_mask = attn_mask
        self.q_blocksize = None
        self.kv_blocksize = None
        self.search_num = None

        self.flash_attention_forward = FlashAttentionScore(head_num=self.head_num,
                                                           keep_prob=self.keep_prob,
                                                           scale_value=self.scale_value,
                                                           pre_tokens=self.pre_tokens,
                                                           next_tokens=self.next_tokens,
                                                           input_layout='BNSD',
                                                           inner_precise=self.inner_precise,
                                                           sparse_mode=self.sparse_mode)

        self.blockpos_list = None
        self.q_blocksize = None
        self.kv_blocksize = None
        self.search_num = None
        self.compile_util = CompileUtil()
        if self.attn_mask is not None:
            self.compile_mask(self.attn_mask)

    @jit(mode="PSJit", jit_config=JitConfig(jit_level="O1"))
    def forward_update(self, prev_attn_out, prev_softmax_max, prev_softmax_sum,
                       cur_attn_out, cur_softmax_max, cur_softmax_sum):
        '''Update ring attention output'''

        softmax_max = ops.maximum(prev_softmax_max, cur_softmax_max)
        prev_scale = ops.exp(prev_softmax_max - softmax_max)
        cur_scale = ops.exp(cur_softmax_max - softmax_max)

        prev_softmax_sum_scaled = prev_softmax_sum * prev_scale
        cur_softmax_sum_scaled = cur_softmax_sum * cur_scale
        softmax_sum = prev_softmax_sum_scaled + cur_softmax_sum_scaled

        prev_out_scale = prev_softmax_sum_scaled / softmax_sum
        cur_out_scale = cur_softmax_sum_scaled / softmax_sum

        d = prev_attn_out.shape[-1]

        prev_out_scale = prev_out_scale[..., 0].unsqueeze(3).tile((1, 1, 1, d))
        cur_out_scale = cur_out_scale[..., 0].unsqueeze(3).tile((1, 1, 1, d))

        attn_out = prev_attn_out * prev_out_scale + cur_attn_out * cur_out_scale

        return attn_out, softmax_max, softmax_sum

    def set_block_size(self, q_blocksize=128, kv_blocksize=128, search_num=20):
        self.q_blocksize = q_blocksize
        self.kv_blocksize = kv_blocksize
        self.search_num = search_num

    def compile_mask(self, attn_mask):
        '''compile mask for sparse attention'''
        if attn_mask is not None:
            if (self.q_blocksize and self.kv_blocksize) and self.search_num:
                self.blockpos_list = self.compile_util.compile_mask(attn_mask,
                                                                    self.q_blocksize,
                                                                    self.kv_blocksize,
                                                                    self.search_num)
            elif (self.q_blocksize and self.kv_blocksize) and (not self.search_num):
                self.blockpos_list = self.compile_util.compile_mask(attn_mask,
                                                                    self.q_blocksize,
                                                                    self.kv_blocksize)
            else:
                self.blockpos_list = self.compile_util.compile_mask(attn_mask)
        return self.blockpos_list

    def __call__(self, q, k, v, real_shift: Tensor = None, drop_mask: Tensor = None,
                 padding_mask: Tensor = None, attn_mask: Tensor = None, prefix: list = None,
                 actual_seq_qlen: list = None, actual_seq_kvlen: list = None):

        if self.attn_mask is None:
            if attn_mask is None:
                raise ValueError("The input attn_mask must not be None.")
            self.attn_mask = attn_mask
            self.compile_mask(self.attn_mask)

        if self.input_layout == 'SBH':
            out = Tensor(np.zeros((q.shape[1], self.head_num, q.shape[0], q.shape[2]//self.head_num)),
                         dtype=q.dtype)
            softmax_sum = Tensor(np.zeros((q.shape[1], self.head_num, q.shape[0], 8)),
                                 dtype=mstype.float32)
            softmax_max = Tensor(np.full((q.shape[1], self.head_num, q.shape[0], 8), float('-inf')),
                                 dtype=mstype.float32)
        elif self.input_layout == 'BSH':
            out = Tensor(np.zeros((q.shape[0], self.head_num, q.shape[1], q.shape[2]//self.head_num)),
                         dtype=q.dtype)
            softmax_sum = Tensor(np.zeros((q.shape[0], self.head_num, q.shape[1], 8)),
                                 dtype=mstype.float32)
            softmax_max = Tensor(np.full((q.shape[0], self.head_num, q.shape[1], 8), float('-inf')),
                                 dtype=mstype.float32)
        elif self.input_layout == 'BNSD':
            out = Tensor(np.zeros(q.shape), dtype=q.dtype)
            softmax_sum = Tensor(np.zeros((q.shape[0], self.head_num, q.shape[2], 8)),
                                 dtype=mstype.float32)
            softmax_max = Tensor(np.full((q.shape[0], self.head_num, q.shape[2], 8), float('-inf')),
                                 dtype=mstype.float32)
        else:
            raise ValueError(f"Only value SBH, BSH or BNSD is supported for input_layout. "
                             f"But found {self.input_layout}")

        if self.compile_util.skip_calc_flag:
            if self.input_layout == 'SBH':
                out = ops.transpose(out, (2, 0, 1, 3)) # bnsd to sbnd
                out = out.reshape(out.shape[0], out.shape[1], -1) # sbnd -> sbh
            elif self.input_layout == 'BSH':
                out = ops.transpose(out, (0, 2, 1, 3)) # bnsd to bsnd
                out = out.reshape(out.shape[0], out.shape[1], -1) # bsnd -> bsh
            return softmax_max, softmax_sum, None, out

        if self.input_layout == 'SBH':
            q = q.reshape(q.shape[0], q.shape[1], self.head_num, -1) # sbh->sbnd
            q = ops.transpose(q, (1, 2, 0, 3)).contiguous() # sbnd to bnsd
            k = k.reshape(k.shape[0], k.shape[1], self.head_num, -1) # sbh->sbnd
            k = ops.transpose(k, (1, 2, 0, 3)).contiguous() # sbnd to bnsd
            v = v.reshape(v.shape[0], v.shape[1], self.head_num, -1) # sbh->sbnd
            v = ops.transpose(v, (1, 2, 0, 3)).contiguous() # sbnd to bnsd
        elif self.input_layout == 'BSH':
            q = q.reshape(q.shape[0], q.shape[1], self.head_num, -1) # bsh->bsnd
            q = ops.transpose(q, (0, 2, 1, 3)).contiguous() # bsnd to bnsd
            k = k.reshape(k.shape[0], k.shape[1], self.head_num, -1) # bsh->bsnd
            k = ops.transpose(k, (0, 2, 1, 3)).contiguous() # bsnd to bnsd
            v = v.reshape(v.shape[0], v.shape[1], self.head_num, -1) # bsh->bsnd
            v = ops.transpose(v, (0, 2, 1, 3)).contiguous() # bsnd to bnsd
        elif self.input_layout == 'BNSD':
            pass
        else:
            raise ValueError(f"Only value SBH, BSH or BNSD is supported for input_layout. "
                             f"But found {self.input_layout}")

        block_num = len(self.blockpos_list)
        for i in range(block_num):
            x1, y1, x2, y2 = self.blockpos_list[i]
            block_mask = self.attn_mask[x1:x2, y1:y2]

            all_att_outs = self.flash_attention_forward(
                q[:, :, x1:x2], k[:, :, y1:y2], v[:, :, y1:y2],
                real_shift=real_shift, drop_mask=drop_mask, padding_mask=padding_mask,
                attn_mask=block_mask,
                prefix=prefix,
                actual_seq_qlen=actual_seq_qlen,
                actual_seq_kvlen=actual_seq_kvlen) #q/k/v.shape= s b h, shape for _flash_attn_forward
            cur_attn_out = all_att_outs[3] # (x2-x1, b, h)
            cur_softmax_max = all_att_outs[0] # (b n s 8)
            cur_softmax_sum = all_att_outs[1] # (b n s 8)

            slice_out, slice_softmax_max, slice_softmax_sum = self.forward_update(
                out[:, :, x1:x2],
                softmax_max[:, :, x1:x2],
                softmax_sum[:, :, x1:x2],
                cur_attn_out,
                cur_softmax_max,
                cur_softmax_sum,
            )
            out[:, :, x1:x2], softmax_max[:, :, x1:x2], softmax_sum[:, :, x1:x2] = (
                slice_out, slice_softmax_max, slice_softmax_sum)

        if self.input_layout == 'SBH':
            out = ops.transpose(out, (2, 0, 1, 3)) # bnsd to sbnd
            out = out.reshape(out.shape[0], out.shape[1], -1) # sbnd -> sbh
        elif self.input_layout == 'BSH':
            out = ops.transpose(out, (0, 2, 1, 3)) # bnsd to bsnd
            out = out.reshape(out.shape[0], out.shape[1], -1) # bsnd -> bsh

        return softmax_max, softmax_sum, None, out

class SparseAttentionScoreGrad():
    """Backward of sparse attention."""
    def __init__(self,
                 head_num,
                 keep_prob=1.0,
                 scale_value=1.0,
                 pre_tokens=2147483647,
                 next_tokens=2147483647,
                 inner_precise=0,
                 input_layout="SBH",
                 sparse_mode=0,
                 atten_mask=None,
                 blockpos_list=None):
        self.head_num = head_num
        self.keep_prob = keep_prob
        self.scale_value = scale_value
        self.pre_tokens = pre_tokens
        self.next_tokens = next_tokens
        self.inner_precise = inner_precise
        self.input_layout = input_layout
        self.sparse_mode = sparse_mode
        if self.sparse_mode != 0:
            raise ValueError('Only sparse_mode = 0 is supported for SparseAttentionScore.')
        self.blockpos_list = blockpos_list
        self.attn_mask = atten_mask
        self.q_blocksize = None
        self.kv_blocksize = None
        self.search_num = None

        self.flash_attention_grad = FlashAttentionScoreGrad(head_num=self.head_num,
                                                            keep_prob=self.keep_prob,
                                                            scale_value=self.scale_value,
                                                            pre_tokens=self.pre_tokens,
                                                            next_tokens=self.next_tokens,
                                                            input_layout='BNSD',
                                                            inner_precise=self.inner_precise,
                                                            sparse_mode=self.sparse_mode)
        self.compile_util = CompileUtil()

        if (self.attn_mask is not None) and (blockpos_list is None):
            self.compile_mask(self.attn_mask)

    def backward_update(self, cur_dq, cur_dk, cur_dv, dq, dk, dv):
        dq = dq.add(cur_dq)
        dk = dk.add(cur_dk)
        dv = dv.add(cur_dv)
        return dq, dk, dv

    def set_block_size(self, q_blocksize=128, kv_blocksize=128, search_num=20):
        self.q_blocksize = q_blocksize
        self.kv_blocksize = kv_blocksize
        self.search_num = search_num

    def compile_mask(self, attn_mask):
        '''compile mask for sparse attention'''
        if attn_mask is not None:
            if (self.q_blocksize and self.kv_blocksize) and self.search_num:
                self.blockpos_list = self.compile_util.compile_mask(attn_mask,
                                                                    self.q_blocksize,
                                                                    self.kv_blocksize,
                                                                    self.search_num)
            elif (self.q_blocksize and self.kv_blocksize) and (not self.search_num):
                self.blockpos_list = self.compile_util.compile_mask(attn_mask,
                                                                    self.q_blocksize,
                                                                    self.kv_blocksize)
            else:
                self.blockpos_list = self.compile_util.compile_mask(attn_mask)
        return self.blockpos_list

    def __call__(self, q, k, v, dout, pse_shift: Tensor = None, drop_mask: Tensor = None,
                 padding_mask: Tensor = None, atten_mask: Tensor = None, softmax_max: Tensor = None,
                 softmax_sum: Tensor = None, softmax_in: Tensor = None, attention_in: Tensor = None,
                 prefix: list = None, actual_seq_qlen: list = None, actual_seq_kvlen: list = None):

        if self.attn_mask is None:
            if atten_mask is None:
                raise ValueError("The input attn_mask must not be None.")
            self.attn_mask = atten_mask
            if self.blockpos_list is None:
                self.compile_mask(self.attn_mask)

        if self.blockpos_list is None:
            self.compile_mask(self.attn_mask)

        if self.compile_util.skip_calc_flag:
            dq = Tensor(np.zeros(q.shape), dtype=q.dtype)
            dk = Tensor(np.zeros(k.shape), dtype=k.dtype)
            dv = Tensor(np.zeros(v.shape), dtype=v.dtype)
            return dq, dk, dv, None

        if self.input_layout == 'SBH':
            dq = Tensor(np.zeros((q.shape[1], self.head_num, q.shape[0], q.shape[2]//self.head_num)), dtype=q.dtype)
            dk = Tensor(np.zeros((k.shape[1], self.head_num, k.shape[0], k.shape[2]//self.head_num)), dtype=k.dtype)
            dv = Tensor(np.zeros((v.shape[1], self.head_num, v.shape[0], v.shape[2]//self.head_num)), dtype=v.dtype)
        elif self.input_layout == 'BSH':
            dq = Tensor(np.zeros((q.shape[0], self.head_num, q.shape[1], q.shape[2]//self.head_num)), dtype=q.dtype)
            dk = Tensor(np.zeros((k.shape[0], self.head_num, k.shape[1], k.shape[2]//self.head_num)), dtype=k.dtype)
            dv = Tensor(np.zeros((v.shape[0], self.head_num, v.shape[1], v.shape[2]//self.head_num)), dtype=v.dtype)
        elif self.input_layout == 'BNSD':
            dq = Tensor(np.zeros(q.shape), dtype=q.dtype)
            dk = Tensor(np.zeros(k.shape), dtype=k.dtype)
            dv = Tensor(np.zeros(v.shape), dtype=v.dtype)
        else:
            raise ValueError(f"Only value SBH or BNSD is supported for input_layout."
                             f"But found {self.input_layout}")

        if self.input_layout == 'SBH':
            q = q.reshape(q.shape[0], q.shape[1], self.head_num, -1) # sbh->sbnd
            q = ops.transpose(q, (1, 2, 0, 3)).contiguous() # sbnd to bnsd
            k = k.reshape(k.shape[0], k.shape[1], self.head_num, -1) # sbh->sbnd
            k = ops.transpose(k, (1, 2, 0, 3)).contiguous() # sbnd to bnsd
            v = v.reshape(v.shape[0], v.shape[1], self.head_num, -1) # sbh->sbnd
            v = ops.transpose(v, (1, 2, 0, 3)).contiguous() # sbnd to bnsd
            dout = dout.reshape(dout.shape[0], dout.shape[1], self.head_num, -1) # sbh->sbnd
            dout = ops.transpose(dout, (1, 2, 0, 3)).contiguous() # sbnd to bnsd
            attention_in = attention_in.reshape(attention_in.shape[0], attention_in.shape[1], self.head_num, -1) # sbh->sbnd
            attention_in = ops.transpose(attention_in, (1, 2, 0, 3)).contiguous() # sbnd to bnsd
        elif self.input_layout == 'BSH':
            q = q.reshape(q.shape[0], q.shape[1], self.head_num, -1) # bsh->bsnd
            q = ops.transpose(q, (0, 2, 1, 3)).contiguous() # bsnd to bnsd
            k = k.reshape(k.shape[0], k.shape[1], self.head_num, -1) # bsh->bsnd
            k = ops.transpose(k, (0, 2, 1, 3)).contiguous() # bsnd to bnsd
            v = v.reshape(v.shape[0], v.shape[1], self.head_num, -1) # bsh->bsnd
            v = ops.transpose(v, (0, 2, 1, 3)).contiguous() # bsnd to bnsd
            dout = dout.reshape(dout.shape[0], dout.shape[1], self.head_num, -1) # bsh->bsnd
            dout = ops.transpose(dout, (0, 2, 1, 3)).contiguous() # bsnd to bnsd
            attention_in = attention_in.reshape(attention_in.shape[0], attention_in.shape[1], self.head_num, -1) # bsh->bsnd
            attention_in = ops.transpose(attention_in, (0, 2, 1, 3)).contiguous() # bsnd to bnsd
        elif self.input_layout == 'BNSD':
            pass
        else:
            raise ValueError(f"Only value SBH, BSH or BNSD is supported for input_layout. "
                             f"But found {self.input_layout}")

        block_num = len(self.blockpos_list)
        for i in range(block_num):
            x1, y1, x2, y2 = self.blockpos_list[i]
            block_mask = self.attn_mask[x1:x2, y1:y2]

            slice_q = q[:, :, x1:x2]
            slice_k = k[:, :, y1:y2]
            slice_v = v[:, :, y1:y2]
            slice_dout = dout[:, :, x1:x2]
            slice_softmax_max = softmax_max[:, :, x1:x2]
            slice_softmax_sum = softmax_sum[:, :, x1:x2]
            slice_attention_in = attention_in[:, :, x1:x2]

            attn_grad_outs = self.flash_attention_grad(
                slice_q, slice_k, slice_v, slice_dout,
                pse_shift=prefix, drop_mask=drop_mask, padding_mask=padding_mask,
                atten_mask=block_mask,
                softmax_max=slice_softmax_max,
                softmax_sum=slice_softmax_sum,
                attention_in=slice_attention_in,
                prefix=prefix,
                actual_seq_qlen=actual_seq_qlen,
                actual_seq_kvlen=actual_seq_kvlen
                )
            cur_dq, cur_dk, cur_dv = attn_grad_outs[0], attn_grad_outs[1], attn_grad_outs[2]
            slice_dq = dq[:, :, x1:x2]
            slice_dk = dk[:, :, y1:y2]
            slice_dv = dv[:, :, y1:y2]

            dq[:, :, x1:x2] = slice_dq.add(cur_dq)
            dk[:, :, y1:y2] = slice_dk.add(cur_dk)
            dv[:, :, y1:y2] = slice_dv.add(cur_dv)

        if self.input_layout == 'SBH':
            dq = ops.transpose(dq, (2, 0, 1, 3)) # bnsd to sbnd
            dq = dq.reshape(dq.shape[0], dq.shape[1], -1) # sbnd -> sbh
            dk = ops.transpose(dk, (2, 0, 1, 3)) # bnsd to sbnd
            dk = dk.reshape(dk.shape[0], dk.shape[1], -1) # sbnd -> sbh
            dv = ops.transpose(dv, (2, 0, 1, 3)) # bnsd to sbnd
            dv = dv.reshape(dv.shape[0], dv.shape[1], -1) # sbnd -> sbh
        elif self.input_layout == 'BSH':
            dq = ops.transpose(dq, (0, 2, 1, 3)) # bnsd to bsnd
            dq = dq.reshape(dq.shape[0], dq.shape[1], -1) # bsnd -> bsh
            dk = ops.transpose(dk, (0, 2, 1, 3)) # bnsd to bsnd
            dk = dk.reshape(dk.shape[0], dk.shape[1], -1) # bsnd -> bsh
            dv = ops.transpose(dv, (0, 2, 1, 3)) # bnsd to bsnd
            dv = dv.reshape(dv.shape[0], dv.shape[1], -1) # bsnd -> bsh

        return dq, dk, dv, None
