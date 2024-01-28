import os

import torch


@torch.no_grad()
def mixed_precision_optimizer_step(self, args, timers):
    # Copy gradients from model params to main params.
    timers('optimizer-copy-to-main-grad', log_level=1).start(
        barrier=args.barrier_with_L1_time)
    self._copy_model_grads_to_main_grads()
    timers('optimizer-copy-to-main-grad').stop()

    # Do unscale, check for inf, and update grad scaler only for
    # the case that grad scaler is provided.
    if self.grad_scaler:

        # Unscale and check for inf/nan.
        timers('optimizer-unscale-and-check-inf', log_level=1).start(
            barrier=args.barrier_with_L1_time)
        found_inf_flag = self._unscale_main_grads_and_check_for_nan()
        timers('optimizer-unscale-and-check-inf').stop()

        # We are done with scaling gradients
        # so we can update the loss scale.
        self.grad_scaler.update(found_inf_flag)

        # If we found inf/nan, skip the update.
        if found_inf_flag:
            if int(os.getenv('NPU_DETECT', '0')):
                from torch_npu.utils.silent_error import clear_hookmodule_list
                clear_hookmodule_list()
            return False, None, None

    # Clip the main gradients.
    timers('optimizer-clip-main-grad', log_level=1).start(
        barrier=args.barrier_with_L1_time)
    grad_norm = None
    if self.clip_grad > 0.0:
        grad_norm = self.clip_grad_norm(self.clip_grad,
                                        self.check_for_nan_in_grad)
    timers('optimizer-clip-main-grad').stop()

    found_silent_flag = False
    if int(os.getenv('NPU_DETECT', '0')):
        from torch_npu.utils.silent_error import silent_fault_check
        loss_scale = 1.0 if self.grad_scaler is None else self.grad_scaler.inv_scale.item()
        silent_error = silent_fault_check(loss_scale)
        silent_error = torch.tensor(silent_error, dtype=torch.float32).npu()
        torch.distributed.all_reduce(silent_error, op=torch.distributed.ReduceOp.MAX)
        found_silent_flag = (silent_error.item() > 0)

    if not found_silent_flag or not (int(os.getenv('NPU_RECOVERY', '0'))):

        # Count the zeros in the grads.
        timers('optimizer-count-zeros', log_level=1).start(
            barrier=args.barrier_with_L1_time)
        num_zeros_in_grad = self.count_zeros() if \
                            self.log_num_zeros_in_grad else None
        timers('optimizer-count-zeros').stop()

        # Step the optimizer.
        timers('optimizer-inner-step', log_level=1).start(
            barrier=args.barrier_with_L1_time)
        self.optimizer.step()
        timers('optimizer-inner-step').stop()

        # Update params from main params.
        timers('optimizer-copy-main-to-model-params', log_level=1).start(
            barrier=args.barrier_with_L1_time)
        self._copy_main_params_to_model_params()
        timers('optimizer-copy-main-to-model-params').stop()

    else:
        # The silent error is found, and skip the step, then call print_error_plog api to print log in plog.
        import torch_npu
        if hasattr(torch_npu.npu, "print_error_plog"):
            torch_npu.npu.print_error_plog("NPUCheckEvent:AICore Numerical error happen, skip this step!")
        return False, None, None

    # Successful update.
    return True, grad_norm, num_zeros_in_grad


@torch.no_grad()
def fp32_optimizer_step(self, args, timers):
    """Clip gradients (if needed) and step the base optimizer.
    Always return successful since there is no overflow."""

    # Copy main_grads to grads.
    timers('optimizer-copy-to-main-grad', log_level=1).start(
        barrier=args.barrier_with_L1_time)
    if self.params_have_main_grad:
        for param_group in self.optimizer.param_groups:
            for param in param_group['params']:
                param.grad = param.main_grad

    timers('optimizer-copy-to-main-grad').stop()

    # Clip gradients.
    timers('optimizer-clip-main-grad', log_level=1).start(
        barrier=args.barrier_with_L1_time)
    grad_norm = None
    if self.clip_grad > 0.0:
        grad_norm = self.clip_grad_norm(self.clip_grad,
                                        self.check_for_nan_in_grad)
    timers('optimizer-clip-main-grad').stop()

    found_silent_flag = False
    if int(os.getenv('NPU_DETECT', '0')):
        from torch_npu.utils.silent_error import silent_fault_check
        silent_error = silent_fault_check(1.0)
        silent_error = torch.tensor(silent_error, dtype=torch.float32).npu()
        torch.distributed.all_reduce(silent_error, op=torch.distributed.ReduceOp.MAX)
        found_silent_flag = (silent_error.item() > 0)

    if not found_silent_flag or not (int(os.getenv('NPU_RECOVERY', '0'))):
        # count the zeros in the grads
        timers('optimizer-count-zeros', log_level=1).start(
            barrier=args.barrier_with_L1_time)
        num_zeros_in_grad = self.count_zeros() if \
                            self.log_num_zeros_in_grad else None
        timers('optimizer-count-zeros').stop()

        # Update parameters.
        timers('optimizer-inner-step', log_level=1).start(
            barrier=args.barrier_with_L1_time)
        self.optimizer.step()
        timers('optimizer-inner-step').stop()
    else:
        # The silent error is found, and skip the step, then call print_error_plog api to print log in plog.
        import torch_npu
        if hasattr(torch_npu.npu, "print_error_plog"):
            torch_npu.npu.print_error_plog("NPUCheckEvent:AICore Numerical error happen, skip this step!")
        return False, None, None

    # No overflow for FP32 optimizer.
    return True, grad_norm, num_zeros_in_grad