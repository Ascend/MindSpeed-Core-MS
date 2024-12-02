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
"""Checkpoint related classes and functions."""

import os
import time
from mindspore import context
from mindspore.common.api import _pynative_executor
from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor
from mindspore.communication import get_rank, get_group_size
from mindspore import log as logger
from mindspore._c_expression import _repair_device, _stop_device, _tft_sem_post
from mindspore._c_expression import clean_tdt_channel
from mindspore._c_expression import send_recv
from mindspore._c_expression import CollectiveManager
from mindspore._c_expression import _get_uce_process_strategy, _get_uce_mem_info
import mindspore
import mindspore.common.dtype as mstype
from mindspeed_ms.core.dist_checkpointing import save_checkpoint

class _CtxParam(dict):
    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value


def _get_ckpt_dir(step, ckpt_save_path, is_tmp_file):
    """ Common func to generate ckpt dir name."""
    tmp = "_tmp" if is_tmp_file else ""
    mid_dir = f"tft_saved_checkpoints-step_{str(step)}{tmp}"
    return os.path.join(ckpt_save_path, mid_dir)

def _save_checkpoint_on_failure(step, save_info, args, cb_ctx):   # pylint: disable=W0613
    """ Callback used for TFT save ckpt function when errors occur."""
    logger.warning("Enter _save_checkpoint_on_failure function")

    ckpt_save_path = cb_ctx.ckpt_save_path
    cb_params = args
    cur_step_num = cb_params.cur_step_num
    cur_epoch_num = cb_params.cur_epoch_num
    model_config = cb_params.model_config
    training_config = cb_params.training_config
    train_one_step_cell = cb_params.train_one_step_cell

    cur_ckpt_dir = _get_ckpt_dir(step, ckpt_save_path, True)
    os.makedirs(cur_ckpt_dir, exist_ok=True)
    while True:
        if cb_ctx.starting_sync:
            break
        time.sleep(0.1)
    save_checkpoint(model_config,
                    train_one_step_cell.network_with_loss,
                    train_one_step_cell.optimizer,
                    train_one_step_cell.opt_param_scheduler,
                    cur_ckpt_dir,
                    format=training_config.ckpt_format,
                    prefix=training_config.prefix,
                    epoch_num=cur_epoch_num,
                    step_num=cur_step_num,
                    crc_check=training_config.crc_check,
                    keep_checkpoint_max=training_config.keep_checkpoint_max + 1,
                    enable_high_availability=True)
    logger.warning("Finish _save_checkpoint_on_failure function")

def _rename_save_result(step, cb_ctx):
    """ Callback used for TFT rename function after ckpt save callback was finished and successful."""
    logger.warning("Enter _rename_save_result function")
    tmp_dir = _get_ckpt_dir(step, cb_ctx.ckpt_save_path, True)
    fin_dir = _get_ckpt_dir(step, cb_ctx.ckpt_save_path, False)

    os.rename(tmp_dir, fin_dir)
    logger.warning("Finish _rename_save_result function")

def _tft_exit_cb(ctx):
    logger.error("Enter mindio ttp exit process, which means other ranks occur exception, check other ranks' logs!")
    _tft_sem_post()
    os._exit(1)   # pylint: disable=W0212

def _tft_repair_callback(step, need_rebuild, error_ranks, repair_info, args, cb_ctx):  # pylint: disable=W0613
    """ Callback used for TFT repair function."""
    logger.info("Enter _tft_repair_callback repair type: {}".format(repair_info["repair_type"]))
    if(repair_info["repair_type"] == cb_ctx.tft.RepairType.RT_UCE_HIGHLEVEL.value\
or repair_info["repair_type"] == cb_ctx.tft.RepairType.RT_UCE_LOWLEVEL.value):
        logger.info("Enter _tft_repair_callback uce REPARI_DEVICE device_id : {}".format(cb_ctx.device_id))
        _repair_device(cb_ctx.device_id)

    if(repair_info["repair_type"] == cb_ctx.tft.RepairType.RT_UCE_HIGHLEVEL.value\
       or repair_info["repair_type"] == cb_ctx.tft.RepairType.RT_SEND.value):
        logger.info("Enter _tft_repair_callback SEND_RECV repair type: \
{}, src_rank:{}, dst_rank: {}".format(repair_info["repair_type"], repair_info["src"], repair_info["dst"]))
        cb_params = args
        src_rank = repair_info["src"][0]
        dst_rank = repair_info["dst"][0]
        send_recv(cb_params.train_network.trainable_params(), src_rank, dst_rank)
    logger.info("Finish _tft_repair_callback")


def _tft_clean_callback(is_uce_error, args, ctx):  # pylint: disable=W0613
    """ Callback used for TFT clean function."""
    logger.info("Enter _tft_clean_callback")
    ret = 0
    if is_uce_error:
        _get_uce_mem_info(ctx.device_id)
        err_strategy = _get_uce_process_strategy()
        logger.info("_tft_clean_callback err_strategy: {}".format(err_strategy))
        if err_strategy == "RS_UCE_HIGHLEVEL":
            ret = 0
        elif err_strategy == "RS_UCE_LOWLEVEL":
            ret = 2
        else:
            ret = 1
    clean_tdt_channel()
    logger.info("Enter _tft_clean_callback resume_hccl_comm")
    CollectiveManager.get_instance().resume_hccl_comm()
    logger.info("Finish _tft_clean_callback, ret: {}".format(ret))
    return ret

def _tft_stop_callback(args, cb_ctx):  # pylint: disable=W0613
    """ Callback used for TFT stop function."""
    logger.info("Enter _tft_stop_callback device_id: {}".format(cb_ctx.device_id))
    _stop_device(cb_ctx.device_id)
    logger.info("Finish _tft_stop_callback")

class MindIOAdapter:
    """
        Used for Mindspore High availability manager
    """
    def __init__(self, **kwargs):
        """
        Do at train start.
        """
        logger.info("Begin to init MindIO TFT controller and processor.")
        device_target = context.get_context("device_target")
        if device_target != "Ascend":
            raise ValueError("MindIO TFT feature only support on Ascend device!")
        # used for hold mindio context common params, such as tft reference,  init ckpt path.
        self.ctx_params = _CtxParam()

        from mindio_ttp import framework_ttp as tft
        self.tft = tft
        self.py_exec = _pynative_executor

        self.g_one = Parameter(Tensor([1], dtype=mstype.int32))
        self.allreduce = mindspore.ops.AllReduce()
        self.device_id = context.get_context("device_id")
        self.starting_sync = False

        self.ckpt_save_path = None
        world_size = get_group_size()
        cur_rank = get_rank()
        controller_ip = self.get_value_from_env_and_args("CONTROLLER_ADDR", "controller_ip", "127.0.0.1", **kwargs,)
        controller_port = int(self.get_value_from_env_and_args("CONTROLLER_PORT", "controller_port", '8000', **kwargs))
        enable_tls = self.get_value_from_env_and_args("ENABLE_TLS", "enable_tls", 'False', **kwargs)
        enable_tls = enable_tls.lower() in ['true', '1']
        tls_cert_key_dir = self.get_value_from_env_and_args("TLS_CERT_KEY_DIR", "tls_cert_key_dir", '', **kwargs)
        controller_rank = int(self.get_value_from_env_and_args("CONTROLLER_RANK", "controller_rank", '0', **kwargs))

        enable_local_copy = False
        self.controller_rank = controller_rank

        if cur_rank == controller_rank:
            logger.info(f"Begin to start MindIO TFT controller on rank_id:{cur_rank}")
            self.tft.tft_init_controller(cur_rank, world_size, enable_local_copy)
            self.tft.tft_start_controller(controller_ip, controller_port, enable_tls, tls_cert_key_dir)
            logger.info("Finish start MindIO TFT controller.")

        self.tft.tft_init_processor(cur_rank, world_size, enable_local_copy, enable_tls, tls_cert_key_dir)
        # it seems mindio tft check reboot node after start processor
        self.tft.tft_start_processor(controller_ip, controller_port)

        logger.info("Finish init MindIO TFT controller and processor.")

    def get_value_from_env_and_args(self, env_name, arg_name, default_value, **kwargs):
        if os.getenv(env_name):
            return os.getenv(env_name)
        return kwargs.get(arg_name, default_value)

    def report_tft_status(self):
        self.tft.tft_report_error(self.tft.ReportState.RS_UNKNOWN.value)

    def _set_tft_optimizer_replica(self, dp):
        """ set Mindio TFT optimizer replica info, used internal. """
        if not isinstance(dp, list):
            raise ValueError("dp must be a list!")
        cur_rank = get_rank()
        logger.warning(f"Set TFT replica with dp: {dp}.")
        replica_info = [
            {
                "type": 1,
                "rank_list": dp,
                "replica_cnt": len(dp),
                "replica_shift": 0
            }
        ]
        self.tft.tft_set_optimizer_replica(cur_rank, replica_info)

    def register_processor(self, **kwargs):
        """
        Do at tarin start.
        """

        self.tft.tft_register_save_ckpt_handler(_save_checkpoint_on_failure, self)
        self.tft.tft_register_rename_handler(_rename_save_result, self)
        self.tft.tft_register_exit_handler(_tft_exit_cb, self)
        self.tft.tft_register_stop_handler(_tft_stop_callback, self)
        self.tft.tft_register_clean_handler(_tft_clean_callback, self)
        self.tft.tft_register_repair_handler(_tft_repair_callback, self)

        self.tft.tft_set_step_args(self.ctx_params)
        self.ctx_params.train_one_step_cell = kwargs.get("train_one_step_cell")
        self.ctx_params.training_config = kwargs.get("training_config")
        self.ctx_params.model_config = kwargs.get("model_config")
        if self.ckpt_save_path is None:
            self.ckpt_save_path = self.ctx_params.training_config.output_dir

        self._set_tft_optimizer_replica(kwargs.get("dp_ranks", []))

    def train_end(self):
        """
        Do at train end.
        """
        cur_rank = get_rank()
        if cur_rank == self.controller_rank:
            self.tft.tft_destroy_controller()
        self.tft.tft_destroy_processor()

    def set_start_updating_opt_state(self, step=-1):
        """
        Send updating state to mindio before optimizer updating.
        """
        self.allreduce(self.g_one)
        self.starting_sync = True
        self.py_exec.sync()
        self.tft.tft_start_updating_os(step)


    def train_step_end(self, cur_epoch_num, cur_step_num, global_step):
        """
        And report status to MindIO TFT after every step finished.

        Args:
            run_context (RunContext): Context of the train running. Refer to
                                      :class:`mindspore.train.RunContext` for detail.
        """
        self.ctx_params.cur_step_num = cur_step_num
        self.ctx_params.cur_epoch_num = cur_epoch_num
        self.ctx_params.global_step = global_step
        logger.debug("START Set optimizer finish step status to TFT. step: {}".format(cur_step_num))
        self.tft.tft_end_updating_os(global_step)
        # if we locked in mindio, then we can do ckpt save.
        self.starting_sync = False
        logger.debug("END Set optimizer finish step status to TFT.")
