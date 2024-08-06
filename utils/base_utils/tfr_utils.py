#
# This file contains the utils to support the plain TFR nets.
#

import _init_paths

from consts.exp_consts import EXC
from utils.base_utils import log
from utils.base_utils.exp_utils import ExpUtils

import torch

class TFR_Utils(object):

  def __init__(self, rtc):
    self._rtc = rtc
    self._exu = ExpUtils(rtc)
    self.error_func_itm = self._exu.get_intermediate_lyr_loss_func()
    self.error_func_otp = self._exu.get_output_lyr_loss_func()

  def _get_presentation_ts_averaged_loss(self, all_ts_pr_y, true_y, err_func):
    if self._rtc.DEBUG:
      log.DEBUG("Shape of all_ts_pr_y: {} and of true_y: {}".format(
                all_ts_pr_y.shape, true_y.shape))
      assert all_ts_pr_y.shape[1] == self._rtc.DCFG["presentation_ts"]
      assert all_ts_pr_y.requires_grad == True
      assert true_y.requires_grad == False

    all_ts_loss = 0
    for t in range(self._rtc.DCFG["presentation_ts"]):
      pr_y = all_ts_pr_y[:, t]
      all_ts_loss += err_func(pr_y, true_y)

    all_ts_loss /= self._rtc.DCFG["presentation_ts"]
    return all_ts_loss

  def get_tfr_loss_for_all_rol_all_ts(self, all_rol_all_ts_pr_y, true_y):
    all_rol_loss = 0
    # Get the final layer or Output Layer loss function.
    output_err_func = self._exu.get_output_lyr_loss_func()
    # Get the intermediate Readout Layers loss function.
    itrmdt_err_func = self._exu.get_intermediate_lyr_loss_func()

    # Get the loss for Conv Readout Layers.
    for rol in range(1, self._exu.num_lyrs["conv_lyrs"]+1):
      conv_rol_all_ts_pr_y = all_rol_all_ts_pr_y["conv_lyr_%s" % rol]
      rol_loss = self._get_presentation_ts_averaged_loss(
          conv_rol_all_ts_pr_y, true_y, itrmdt_err_func)
      all_rol_loss += rol_loss
      if self._rtc.DEBUG:
        log.DEBUG("Loss for Conv Readout Layer: %s obtained." % rol)

    # Get the loss for Dense Readout Layers.
    for rol in range(1, self._exu.num_lyrs["dense_lyrs"]+1):
      dense_rol_all_ts_pr_y = all_rol_all_ts_pr_y["dense_lyr_%s" % rol]
      if rol != self._exu.num_lyrs["dense_lyrs"]: # Intermediate Layers.
        rol_loss = self._get_presentation_ts_averaged_loss(
            dense_rol_all_ts_pr_y, true_y, itrmdt_err_func)
        if self._rtc.DEBUG:
          log.DEBUG("Loss for Dense Readout Layer: %s obtained." % rol)
      elif rol == self._exu.num_lyrs["dense_lyrs"]: # Final output layer.
        rol_loss = self._get_presentation_ts_averaged_loss(
            dense_rol_all_ts_pr_y, true_y, output_err_func)
        if self._rtc.DEBUG:
          log.DEBUG("Loss for Dense Readout Layer: %s obtained." % rol)

      all_rol_loss += rol_loss

    return all_rol_loss

  def init_all_rol_all_ts_logits_dict(self):
    all_rol_all_ts_logits = {}
    # For the Readout Layers of the Spiking Conv Layers which are all Dense.
    for i in range(1, self._exu.num_lyrs["conv_lyrs"]+1):
      all_rol_all_ts_logits["conv_lyr_%s" % i] = torch.zeros(
          self._rtc.DCFG["batch_size"], self._rtc.DCFG["presentation_ts"],
          self._rtc.DCFG["num_clss"], dtype=EXC.PT_DTYPE,
          device=self._rtc.DEVICE)

    # For the Readout Layers of the Spiking Dense Layers which are all Dense.
    for i in range(1, self._exu.num_lyrs["dense_lyrs"]+1):
      all_rol_all_ts_logits["dense_lyr_%s" % i] = torch.zeros(
          self._rtc.DCFG["batch_size"], self._rtc.DCFG["presentation_ts"],
          self._rtc.DCFG["num_clss"], dtype=EXC.PT_DTYPE,
          device=self._rtc.DEVICE)

    return all_rol_all_ts_logits

  def get_dset_all_rol_pred_and_true_cls_dict(self):
    all_rol_pred_and_true_cls = {}
    for lyr in range(1, self._exu.num_lyrs["conv_lyrs"]+1):
      all_rol_pred_and_true_cls["conv_lyr_%s" % lyr] = []
    for lyr in range(1, self._exu.num_lyrs["dense_lyrs"]+1):
      all_rol_pred_and_true_cls["dense_lyr_%s" % lyr] = []

    all_rol_pred_and_true_cls["true_cls"] = []

    return all_rol_pred_and_true_cls
  ##############################################################################
  ################## U T I L S    F O R    R T R L    E X P S###################
  ##############################################################################

  def _get_loss_for_1_ts(self, one_ts_pr_y, one_ts_tr_y, err_func):
    if self._rtc.DEBUG:
      log.DEBUG("Shape of one_ts_pr_y: {} and of one_ts_tr_y: {}".format(
                one_ts_pr_y.shape, one_ts_tr_y.shape))
      assert one_ts_pr_y.shape == one_ts_tr_y.shape

    return err_func(one_ts_pr_y, one_ts_tr_y)

  def get_tfr_loss_for_all_rol_1_ts(self, all_rol_1_ts_pr_y, true_y):
    all_rol_loss = 0

    # Get the loss for Conv Readout Layers.
    for rol in range(1, self._exu.num_lyrs["conv_lyrs"]+1):
      conv_rol_1_ts_pr_y = all_rol_1_ts_pr_y["conv_lyr_%s" % rol]
      rol_loss = self._get_loss_for_1_ts(
          conv_rol_1_ts_pr_y, true_y, self.error_func_itm)
      if self._rtc.DEBUG:
        log.DEBUG(
            "Loss: %s for Conv Readout Layer: %s obtained." % (rol_loss, rol))
      all_rol_loss += rol_loss

    # Get the loss for Dense Readout Layers.
    for rol in range(1, self._exu.num_lyrs["dense_lyrs"]+1):
      dense_rol_1_ts_pr_y = all_rol_1_ts_pr_y["dense_lyr_%s" % rol]
      if rol != self._exu.num_lyrs["dense_lyrs"]: # Intermediate Layers.
        rol_loss = self._get_loss_for_1_ts(
            dense_rol_1_ts_pr_y, true_y, self.error_func_itm)
        if self._rtc.DEBUG:
          log.DEBUG("Loss: %s for Dense Readout Layer: %s obtained."
                    % (rol_loss, rol))
      elif rol == self._exu.num_lyrs["dense_lyrs"]: # Final output layer.
        rol_loss = self._get_loss_for_1_ts(
            dense_rol_1_ts_pr_y, true_y, self.error_func_otp)
        if self._rtc.DEBUG:
          log.DEBUG("Loss: %s for Dense Readout Layer: %s obtained."
                    % (rol_loss, rol))

      all_rol_loss += rol_loss

    return all_rol_loss

  def init_all_rol_1_ts_logits_dict(self):
    all_rol_1_ts_logits = {}
    # For the Readout Layers of the Spiking Conv Layers which are all Dense.
    for i in range(1, self._exu.num_lyrs["conv_lyrs"]+1):
      all_rol_1_ts_logits["conv_lyr_%s" % i] = torch.zeros(
          self._rtc.DCFG["batch_size"], self._rtc.DCFG["num_clss"],
          dtype=EXC.PT_DTYPE, device=self._rtc.DEVICE)

    # For the Readout Layers of the Spiking Dense Layers which are all Dense.
    for i in range(1, self._exu.num_lyrs["dense_lyrs"]+1):
      all_rol_1_ts_logits["dense_lyr_%s" % i] = torch.zeros(
          self._rtc.DCFG["batch_size"], self._rtc.DCFG["num_clss"],
          dtype=EXC.PT_DTYPE, device=self._rtc.DEVICE)

    return all_rol_1_ts_logits
