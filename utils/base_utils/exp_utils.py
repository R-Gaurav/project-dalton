#
# This file contains the general utils for the experiments and code common to
# all the types of SNNs experimente here.
#

import _init_paths

import datetime
import h5py
import sys
import torch
import numpy as np

from consts.dir_consts import DRC
from consts.exp_consts import EXC
from consts.runtime_consts import RTC

class ExpUtils(object):
  def __init__(self, rtc):
    self._rtc = rtc
    if not hasattr(self, "num_lyrs"):
      self.get_num_layers_dict(rtc.SCNN_ARCH)

  def get_timestamp(self):
    now = datetime.datetime.now()
    now = "%s" % now
    return "T".join(now.split())

  def get_pyt_seed(self):
    gen = torch.Generator()
    return gen.manual_seed(self._rtc.SEED)

  @staticmethod
  def spike_func(x): # Heaviside Step Function.
    spikes = torch.zeros_like(x, dtype=EXC.PT_DTYPE)
    spikes[x > 0] = 1.0 # Spike when v crosses the v_thrld, `x = v - v_thrld`
    return spikes

  @staticmethod
  def srgt_drtv_func(x):
    return 1.0/((1.0 + torch.abs(x)*RTC.SRGT_DRTV_SCALE)**2)

  def reset_v(self, v, spikes):
    mask = spikes > 0
    if EXC.HARD_RESET:
      v[mask] = 0.
    else:
      v[mask] = v[mask] - v_thr

    return v

  def get_class_predictions(self, all_ts_pr_y):
    assert all_ts_pr_y.shape[-1] == self._rtc.DCFG["num_clss"]
    # Get mean over time-steps.
    mean_val = torch.mean(all_ts_pr_y, dim=1)
    # Get the max over predicted class logits.
    pred_cls = mean_val.argmax(dim=-1).cpu().numpy()

    return pred_cls

  def get_all_rol_class_predictions(self, all_rol_all_ts_pr_y):
    all_rol_pred_cls = {}
    for key in all_rol_all_ts_pr_y.keys():
      all_rol_pred_cls[key] = self.get_class_predictions(
          #all_rol_all_ts_pr_y[key][:, :, :].detach())
          all_rol_all_ts_pr_y[key].detach())

    return all_rol_pred_cls

  def save_hdf5_file(self, f_dict, f_name, f_path):
    with h5py.File(f_path+"/"+f_name, "w") as f:
      for key in f_dict.keys():
        f.create_dataset(key, data=f_dict[key], compression="gzip",
                         compression_opts=9, chunks=True)

  def get_num_layers_dict(self, arch_dict):
    self.num_lyrs = dict.fromkeys(["conv_lyrs", "dense_lyrs"], 0)

    for lyr in arch_dict.keys():
      if "conv" in lyr:
        self.num_lyrs["conv_lyrs"] += 1
      elif "dense" in lyr:
        self.num_lyrs["dense_lyrs"] += 1
      elif "inp" in lyr:
        print("Found one input layer")
      else:
        sys.exit("Layer which is neither conv, dense or inp found, exiting...")

    return self.num_lyrs

  # Loss function for the final output layer.
  def get_output_lyr_loss_func(self):
    if self._rtc.OTP_LOSS_FUNC == "MSE":
      loss_func = torch.nn.MSELoss()
    elif self._rtc.OTP_LOSS_FUNC == "CE":
      loss_func = torch.nn.CrossEntropyLoss()

    return loss_func

  # Loss function for the intermediate layers.
  def get_intermediate_lyr_loss_func(self):
    if self._rtc.ITM_LOSS_FUNC == "MSE":
      loss_func = torch.nn.MSELoss()
    elif self._rtc.ITM_LOSS_FUNC == "CE":
      loss_func = torch.nn.CrossEntropyLoss()

    return loss_func

  def get_output_lyr_true_y(self, tr_y):
    if self._rtc.OTP_LOSS_FUNC == "MSE":
      tr_y = torch.eye(
          self._rtc.DCFG["num_clss"], device=self._rtc.DEVICE)[
          tr_y.to(self._rtc.DEVICE)].to(EXC.PT_DTYPE)
      return tr_y
    elif self._rtc.OTP_LOSS_FUNC == "CE":
      return tr_y

  def get_output_lyr_pred_y(self, pred_logits):
    # Note: pred_logits are of shape: (batch_size, num_clss).
    if self._rtc.OTP_LOSS_FUNC == "MSE":
      #pr_y = torch.softmax(pred_logits, dim=1).to(EXC.PT_DTYPE).requires_grad_()
      #pr_y = torch.sigmoid(pred_logits).to(EXC.PT_DTYPE).requires_grad_()
      pr_y = pred_logits
    elif self._rtc.OTP_LOSS_FUNC == "CE":
      #pr_y = torch.softmax(pred_logits, dim=1).to(EXC.PT_DTYPE).requires_grad_()
      pr_y = pred_logits

    return pr_y

  def get_intermediate_lyr_pred_y(self, pred_logits):
    if self._rtc.ITM_LOSS_FUNC == "MSE":
      pr_y = pred_logits
    elif self._rtc.ITM_LOSS_FUNC == "CE":
      #pr_y = torch.softmax(pred_logits, dim=1).to(EXC.PT_DTYPE).requires_grad_()
      pr_y = pred_logits

    return pr_y

  def get_all_rol_all_ts_pr_y_dict(self, all_rol_all_ts_logits):
    all_rol_all_ts_pr_y = {}

    for lyr in range(1, self.num_lyrs["conv_lyrs"]+1): # WRT Conv Layers.
      rol = "conv_lyr_%s" % lyr
      all_rol_all_ts_pr_y[rol] = torch.zeros_like(all_rol_all_ts_logits[rol])

      for t in range(self._rtc.DCFG["presentation_ts"]):
        all_rol_all_ts_pr_y[rol][:, t] = self.get_intermediate_lyr_pred_y(
            all_rol_all_ts_logits[rol][:, t])

    for lyr in range(1, self.num_lyrs["dense_lyrs"]+1): # WRT Dense Layers.
      rol = "dense_lyr_%s" % lyr
      all_rol_all_ts_pr_y[rol] = torch.zeros_like(all_rol_all_ts_logits[rol])

      if lyr != self.num_lyrs["dense_lyrs"]:
        for t in range(self._rtc.DCFG["presentation_ts"]):
          all_rol_all_ts_pr_y[rol][:, t] = self.get_intermediate_lyr_pred_y(
              all_rol_all_ts_logits[rol][:, t])
      elif lyr == self.num_lyrs["dense_lyrs"]: # Output layer.
        for t in range(self._rtc.DCFG["presentation_ts"]):
          all_rol_all_ts_pr_y[rol][:, t] = self.get_output_lyr_pred_y(
              all_rol_all_ts_logits[rol][:, t])

    return all_rol_all_ts_pr_y

  def init_all_hdl_spk_conv_dense_all_ts_delta_v_dict(self, net):
    all_hlr_all_ts_delta_v = {}
    for i in range(1, self.num_lyrs["conv_lyrs"]+1):
      spk_conv_lyr = net._spk_conv_lyrs[i-1] # List indexing at 0 onwards.
      all_hlr_all_ts_delta_v["conv_lyr_%s" % i] = torch.zeros(
          net._bsize, net._rtc.DCFG["presentation_ts"], spk_conv_lyr.otp_chnls,
          spk_conv_lyr.otp_dmx, spk_conv_lyr.otp_dmy, dtype=EXC.PT_DTYPE,
          device=net._rtc.DEVICE)

    for i in range(1, self.num_lyrs["dense_lyrs"]+1):
      all_hlr_all_ts_delta_v["dense_lyr_%s" % i] = torch.zeros(
          net._bsize, net._rtc.DCFG["presentation_ts"],
          net._rtc.SCNN_ARCH["dense_lyr_%s" % i], dtype=EXC.PT_DTYPE,
          device=net._rtc.DEVICE)

    return all_hlr_all_ts_delta_v

  def reinitialize_all_lyrs_neurons_states(self, net):
    net._i2s.re_initialize_v() # Current does not have a state in encoding lyr.
    for c_lyr in net._spk_conv_lyrs:
      c_lyr.re_initialize_states()
    for d_lyr in net._spk_dense_lyrs:
      d_lyr.re_initialize_states()

  def get_conv_layer_kwargs(self, prev_inp_chnls, prev_inp_dmx, prev_inp_dmy,
                            conv_lyr_cfg):
    kwargs = {}
    kwargs["inp_chnls"] = prev_inp_chnls
    kwargs["inp_dmx"] = prev_inp_dmx
    kwargs["inp_dmy"] = prev_inp_dmy
    kwargs["pool_size"] = conv_lyr_cfg.get("pool_size", None)
    kwargs["drpt_prob"] = conv_lyr_cfg.get("drpt_prob", None)
    kwargs["otp_chnls"] = conv_lyr_cfg["otp_chnls"]
    kwargs["krnl_size"] = conv_lyr_cfg["krnl_size"]
    kwargs["stride"] = conv_lyr_cfg["stride"]

    return kwargs

  ##############################################################################
  ################## U T I L S   F O R   R T R L   E X P S #####################
  ##############################################################################
  def get_all_rol_1_ts_pr_y_dict(self, all_rol_1_ts_logits):
    all_rol_1_ts_pr_y = {}

    for lyr in range(1, self.num_lyrs["conv_lyrs"]+1): # WRT Conv Layers.
      rol = "conv_lyr_%s" % lyr
      all_rol_1_ts_pr_y[rol] = self.get_intermediate_lyr_pred_y(
            all_rol_1_ts_logits[rol])

    for lyr in range(1, self.num_lyrs["dense_lyrs"]+1): # WRT Dense Layers.
      rol = "dense_lyr_%s" % lyr
      if lyr != self.num_lyrs["dense_lyrs"]:
        all_rol_1_ts_pr_y[rol] = self.get_intermediate_lyr_pred_y(
            all_rol_1_ts_logits[rol])
      elif lyr == self.num_lyrs["dense_lyrs"]: # Output layer.
        all_rol_1_ts_pr_y[rol] = self.get_output_lyr_pred_y(
            all_rol_1_ts_logits[rol])

    return all_rol_1_ts_pr_y

  def init_all_hdl_spk_conv_dense_1_ts_delta_v_dict(self, net):
    all_hlr_1_ts_delta_v = {}
    for i in range(1, self.num_lyrs["conv_lyrs"]+1):
      spk_conv_lyr = net._spk_conv_lyrs[i-1] # List indexing at 0 onwards.
      all_hlr_1_ts_delta_v["conv_lyr_%s" % i] = torch.zeros(
          net._bsize, spk_conv_lyr.otp_chnls, spk_conv_lyr.otp_dmx,
          spk_conv_lyr.otp_dmy, dtype=EXC.PT_DTYPE, device=net._rtc.DEVICE)

    for i in range(1, self.num_lyrs["dense_lyrs"]+1):
      all_hlr_1_ts_delta_v["dense_lyr_%s" % i] = torch.zeros(
          net._bsize, net._rtc.SCNN_ARCH["dense_lyr_%s" % i],
          dtype=EXC.PT_DTYPE, device=net._rtc.DEVICE)

    return all_hlr_1_ts_delta_v

  def init_bwise_last_rol_all_ts_pr_y(self, n_ts):
    """
    This is useful to storing the predictions for each time-step during the RTRL
    updates, such that predicted classes can be calculated over the mean of the
    predicted_y (e.g., logits) values.
    """
    return torch.zeros(
        self._rtc.DCFG["batch_size"], n_ts,
        self._rtc.DCFG["num_clss"], dtype=EXC.PT_DTYPE, device=self._rtc.DEVICE)

  def init_bwise_all_rol_all_ts_pr_y_dict(self, n_ts):
    all_rol_all_ts_pr_y = {}
    for i in range(1, self.num_lyrs["conv_lyrs"]+1):
      all_rol_all_ts_pr_y["conv_lyr_%s" % i] = torch.zeros(
          self._rtc.DCFG["batch_size"], n_ts, self._rtc.DCFG["num_clss"],
          dtype=EXC.PT_DTYPE, device=self._rtc.DEVICE)

    for i in range(1, self.num_lyrs["dense_lyrs"]+1):
      all_rol_all_ts_pr_y["dense_lyr_%s" % i] = torch.zeros(
          self._rtc.DCFG["batch_size"], n_ts, self._rtc.DCFG["num_clss"],
          dtype=EXC.PT_DTYPE, device=self._rtc.DEVICE)

    return all_rol_all_ts_pr_y

  def get_dvs_gesture_batch_and_n_ts(self, inp_x, is_train):
    if is_train:
      n_ts = inp_x.shape[1] # b_size, n_ts, (n_ch, dm_x, dm_y).
      st_idx = (np.random.randint(
                0, n_ts-EXC.DVS_GES_NTS) if n_ts > EXC.DVS_GES_NTS else 0)

      return (inp_x[:, st_idx:st_idx+EXC.DVS_GES_NTS].to(self._rtc.DEVICE),
              EXC.DVS_GES_NTS)
    else:
      # Minimum recording time in test set is around 1.79 million microseconds.
      n_ts = 1790000//self._rtc.DCFG["time_window"]
      return inp_x[:, :n_ts].to(self._rtc.DEVICE), n_ts

  def get_dvs_cifar10_batch_and_n_ts(self, inp_x, is_train):
    if is_train:
      n_ts = inp_x.shape[1] # b_size, n_ts, (n_ch, dm_x, dm_y).
      st_idx = (np.random.randint(
                0, n_ts-EXC.DVS_C10_NTS) if n_ts > EXC.DVS_C10_NTS else 0)
      return (inp_x[:, st_idx:st_idx+EXC.DVS_C10_NTS].to(self._rtc.DEVICE),
              EXC.DVS_C10_NTS)
    else:
      # Minimum recording time in test set is definitely 1141 milliseconds.
      n_ts = 1141000//self._rtc.DCFG["time_window"]
      return inp_x[:, :n_ts].to(self._rtc.DEVICE), n_ts
