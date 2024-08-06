#
# This file contains the utils for Surrogate Gradient Descent method.
#

import _init_paths

import torch

from consts.exp_consts import EXC
from utils.base_utils.exp_utils import ExpUtils

class SRGT_GD_Utils(object):
  def __init__(self, rtc):
    self._rtc = rtc
    self._exu = ExpUtils(rtc)

  def init_output_layer_all_ts_logits_dict(self):
    otp_lyr_all_ts_logits = {
        "dense_lyr_%s" % self._exu.num_lyrs["dense_lyrs"] : torch.zeros(
        self._rtc.DCFG["batch_size"], self._rtc.DCFG["presentation_ts"],
        self._rtc.DCFG["num_clss"], dtype=EXC.PT_DTYPE, device=self._rtc.DEVICE)
        }

    return otp_lyr_all_ts_logits

  def get_dset_otp_lyr_pred_and_true_cls_dict(self):
    otp_lyr_pred_and_true_cls = {
        "dense_lyr_%s" % self._exu.num_lyrs["dense_lyrs"]: [],
        "true_cls": []
        }
    return otp_lyr_pred_and_true_cls

  def get_output_layer_all_ts_pr_y_dict(self, otp_lyr_all_ts_logits):
    otp_lyr_all_ts_pr_y = {}

    for key in otp_lyr_all_ts_logits.keys():
      otp_lyr_all_ts_pr_y[key] = torch.zeros_like(otp_lyr_all_ts_logits[key])

      for t in range(self._rtc.DCFG["presentation_ts"]):
        otp_lyr_all_ts_pr_y[key][:, t] = self._exu.get_output_lyr_pred_y(
            otp_lyr_all_ts_logits[key][:, t])

    return otp_lyr_all_ts_pr_y

  def get_loss_for_otp_lyr_all_ts(self, otp_lyr_all_ts_pr_y, true_y):
    output_err_func = self._exu.get_output_lyr_loss_func()
    ret_loss = 0

    for key in otp_lyr_all_ts_pr_y.keys():
      otp_loss = 0
      if self._rtc.DEBUG:
        assert otp_lyr_all_ts_pr_y[key].requires_grad == True
        assert true_y.requires_grad == False
        assert otp_lyr_all_ts_pr_y[key][:, 0].shape == true_y.shape

      for t in range(self._rtc.DCFG["presentation_ts"]):
        otp_loss += output_err_func(otp_lyr_all_ts_pr_y[key][:, t], true_y)
      otp_loss /= self._rtc.DCFG["presentation_ts"]

      ret_loss += otp_loss

    return ret_loss
