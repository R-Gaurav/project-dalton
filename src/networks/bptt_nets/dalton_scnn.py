import _init_paths

import torch

from consts.exp_consts import EXC
from utils.base_utils import log
from utils.base_utils.exp_utils import ExpUtils
from utils.base_utils.dalton_utils import DALTON_Utils
from utils.tensor_encoder import ImageToSpike
from utils.spk_conv_layer import SpkConvLayer
from utils.spk_dense_layer import SpkDenseLayer
from utils.readout_layer import DenseReadoutLayer
from utils.readout_layer import ConvReadoutLayer

class DALTON_SCNN(torch.nn.Module):
  def __init__(self, rtc):
    """
    Args:
      rtc <RTC>: The Run Time Constants class.
    """
    super().__init__()

    self._rtc = rtc
    self._exu = ExpUtils(rtc)
    self._tgu = DALTON_Utils(rtc)
    self._bsize = rtc.DCFG["batch_size"]
    self._otpdm = rtc.DCFG["num_clss"]
    self._scnn_arch = rtc.SCNN_ARCH

    log.INFO("Creating the DALTON Spiking CNN... Architecture below:")
    log.INFO(self._scnn_arch)

    log.INFO("Creating the Input Conv Layer...")
    self._i2s = ImageToSpike(
        self._scnn_arch["inp_lyr"]["inp_chnls"],
        self._scnn_arch["inp_lyr"]["inp_dmx"],
        self._scnn_arch["inp_lyr"]["inp_dmy"],
        self._bsize, rtc)
    log.INFO("Creating the intermediate Spiking Conv and Dense Layers and their"
             "corresponding local Readout Layers...")
    (self._spk_conv_lyrs, self._conv_rdt_lyrs, self._spk_dense_lyrs,
     self._dense_rdt_lyrs) = self._get_conv_and_dense_and_rdt_layers()
    self._initialize_all_readout_layers_weights()
    log.INFO("Following are the layers in this DALTON SCNN...")
    log.INFO("Spiking Conv Layers: {}".format(self._spk_conv_lyrs))
    log.INFO("Spiking Dense Layers: {}".format(self._spk_dense_lyrs))
    log.INFO("Conv Readout Layers: {}".format(self._conv_rdt_lyrs))
    log.INFO("Dense Readout Layers: {}".format(self._dense_rdt_lyrs))

  def _get_spk_conv_lyrs_and_conv_rdt_lyrs(self, prev_inp_chnls, prev_inp_dmx,
                                           prev_inp_dmy):
    """
    Returns the lists of Spiking Conv Layer and the associated Conv Readout
    Layers.

    Args:
      prev_inp_chnls <int>: Number of `inp_chnls` in the Input Conv Layer.
      prev_inp_dmx <int>: Dimension of the input image in x axis.
      prev_inp_dmy <int>: Dimension of the input image in y axis.
    """
    spk_conv_lyrs_lst = torch.nn.ModuleList()
    conv_rdt_lyrs_lst = torch.nn.ModuleList()
    n_prev = prev_inp_chnls * prev_inp_dmx * prev_inp_dmy

    for l_num in range(1, self._exu.num_lyrs["conv_lyrs"]+1):
      kwargs = self._exu.get_conv_layer_kwargs(
          prev_inp_chnls, prev_inp_dmx, prev_inp_dmy,
          self._scnn_arch["conv_lyr_%s" % l_num])

      spk_conv_lyr = SpkConvLayer(self._bsize, self._rtc, **kwargs)

      # Get the associated Conv Readout Layer, note requires_grad = False.
      # If it is the last Conv Layer, then the associated Readout Layer is a
      # Dense Readout Layer.
      if l_num == self._exu.num_lyrs["conv_lyrs"]:
        n_prev = (spk_conv_lyr.otp_chnls *
                  spk_conv_lyr.otp_dmx * spk_conv_lyr.otp_dmy)
        rdt_lyr = DenseReadoutLayer(
            n_prev, self._scnn_arch["dense_lyr_1"], self._rtc)
      else:
        kwargs = self._exu.get_conv_layer_kwargs(
            self._scnn_arch["conv_lyr_%s" % l_num]["otp_chnls"],
            spk_conv_lyr.otp_dmx, spk_conv_lyr.otp_dmy,
            self._scnn_arch["conv_lyr_%s" % (l_num+1)])

        rdt_lyr = ConvReadoutLayer(self._rtc, **kwargs)

      spk_conv_lyrs_lst.append(spk_conv_lyr)
      conv_rdt_lyrs_lst.append(rdt_lyr)

      prev_inp_dmx, prev_inp_dmy = spk_conv_lyr.otp_dmx, spk_conv_lyr.otp_dmy
      prev_inp_chnls = self._scnn_arch["conv_lyr_%s" % l_num]["otp_chnls"]

    return spk_conv_lyrs_lst, conv_rdt_lyrs_lst, n_prev

  def _get_spk_dense_lyrs_and_dense_rdt_lyrs(self, n_prev):
    """
    Returns the list of Spiking Dense Layers and the associated Dense Readout
    Layers.

    Args:
      n_prev <int>: Number of neurons in the previous layer.
    """
    spk_dense_lyrs_lst = torch.nn.ModuleList()
    dense_rdt_lyrs_lst = torch.nn.ModuleList()

    for l_num in range(1, self._exu.num_lyrs["dense_lyrs"]+1):
      spk_dense_lyr = SpkDenseLayer(
          n_prev, self._scnn_arch["dense_lyr_%s" % l_num], self._bsize,
          self._rtc)
      # If it is the last Dense Spiking Layer, then the Dense Readout Layer is
      # the final output layer.
      if l_num == self._exu.num_lyrs["dense_lyrs"]:
        rdt_lyr = DenseReadoutLayer(
            self._scnn_arch["dense_lyr_%s" % l_num], self._otpdm, self._rtc)
      else:
        rdt_lyr = DenseReadoutLayer(
            self._scnn_arch["dense_lyr_%s" % l_num],
            self._scnn_arch["dense_lyr_%s" % (l_num+1)], self._rtc)

      spk_dense_lyrs_lst.append(spk_dense_lyr)
      dense_rdt_lyrs_lst.append(rdt_lyr)
      n_prev = self._scnn_arch["dense_lyr_%s" % l_num]

    return spk_dense_lyrs_lst, dense_rdt_lyrs_lst

  def _get_conv_and_dense_and_rdt_layers(self):
    """
    Get the ConvLayer, DenseLayers, and the corresponding ReadoutLayers. Note
    that each ConvLayer and DenseLayer has an associate ReadoutLayer to train
    the Conv and Dense weights. The ReadoutLayer weights are non-trainable.
    """
    prev_inp_chnls, prev_inp_dmx, prev_inp_dmy = (
        self._scnn_arch["inp_lyr"]["inp_chnls"],
        self._scnn_arch["inp_lyr"]["inp_dmx"],
        self._scnn_arch["inp_lyr"]["inp_dmy"]
        )
    ###########################################################################
    # Get the Spiking Conv Layers and Conv Readout Layers.
    spk_conv_lyrs_lst, conv_rdt_lyrs_lst, n_prev = (
        self._get_spk_conv_lyrs_and_conv_rdt_lyrs(
        prev_inp_chnls, prev_inp_dmx, prev_inp_dmy))
    ###########################################################################
    # Get the Spiking Dense Layers and Dense Readout Layers.
    spk_dense_lyrs_lst, dense_rdt_lyrs_lst  = (
        self._get_spk_dense_lyrs_and_dense_rdt_lyrs(n_prev))
    ###########################################################################

    return (spk_conv_lyrs_lst, conv_rdt_lyrs_lst, spk_dense_lyrs_lst,
            dense_rdt_lyrs_lst)

  def _initialize_all_readout_layers_weights(self):
    self._tgu.update_all_readout_layers_weights(self)

  def _check_norm_and_requires_grad_of_conv_lyrs(self):
    assert self._spk_conv_lyrs[0]._conv.weight.requires_grad == True
    log.DEBUG("Spiking Conv Layer: {}, Norm: {}, Requires Grad: {}".format(
              self._spk_conv_lyrs[0].__str__,
              torch.norm(self._spk_conv_lyrs[0]._conv.weight.data),
              self._spk_conv_lyrs[0]._conv.weight.requires_grad))

    for cl, rl in zip(self._spk_conv_lyrs[1:], self._conv_rdt_lyrs[:-1]):
      assert (
          torch.norm(cl._conv.weight.data) == torch.norm(rl._conv.weight.data))
      assert cl._conv.weight.requires_grad == True
      assert rl._conv.weight.requires_grad == False
      log.DEBUG("Spiking Conv Layer: {}, Norm = {}, Requires Grad: {}".format(
                cl.__str__, torch.norm(cl._conv.weight.data),
              cl._conv.weight.requires_grad))
      log.DEBUG("Readout Conv Layer: {}, Norm = {}, Requires Grad: {}".format(
                rl.__str__, torch.norm(rl._conv.weight.data),
                rl._conv.weight.requires_grad))

  def _check_norm_and_requires_grad_of_last_conv_rdt_and_first_spk_dense(self):
    # Check the norm of the last Readout Layer in self._conv_rdt_lyrs which is
    # actually a Dense Readout Layer.
    assert(
        torch.norm(self._spk_dense_lyrs[0]._fc.weight.data) == torch.norm(
        self._conv_rdt_lyrs[-1]._fc.weight.data))
    assert self._spk_dense_lyrs[0]._fc.weight.requires_grad == True
    assert self._conv_rdt_lyrs[-1]._fc.weight.requires_grad == False

    log.DEBUG("Spiking Dense Layer: {}, Norm = {}, Requires Grad: {}".format(
              self._spk_dense_lyrs[0].__str__,
              torch.norm(self._spk_dense_lyrs[0]._fc.weight.data),
              self._spk_dense_lyrs[0]._fc.weight.requires_grad))
    log.DEBUG("Readout Dense Layer: {}, Norm = {}, Requires Grad: {}".format(
              self._conv_rdt_lyrs[-1].__str__,
              torch.norm(self._conv_rdt_lyrs[-1]._fc.weight.data),
              self._conv_rdt_lyrs[-1]._fc.weight.requires_grad))

  def _check_norm_and_requires_grad_of_dense_lyrs(self):
    for dl, rl in zip(self._spk_dense_lyrs[1:], self._dense_rdt_lyrs[:-1]):
      assert torch.norm(dl._fc.weight.data) == torch.norm(rl._fc.weight.data)
      assert dl._fc.weight.requires_grad == True
      assert rl._fc.weight.requires_grad == False

      log.DEBUG("Spiking Dense Layer: {}, Norm = {}, Requires Grad: {}".format(
                dl.__str__, torch.norm(dl._fc.weight.data),
                dl._fc.weight.requires_grad))
      log.DEBUG("Readout Dense Layer: {}, Norm = {}, Requires Grad: {}".format(
                rl.__str__, torch.norm(rl._fc.weight.data),
                rl._fc.weight.requires_grad))

    assert self._dense_rdt_lyrs[-1]._fc.weight.requires_grad == False
    log.DEBUG("Readout Dense Layer: {}, Norm: {}. Requires Grad: {}".format(
              self._dense_rdt_lyrs[-1].__str__,
              torch.norm(self._dense_rdt_lyrs[-1]._fc.weight.data),
              self._dense_rdt_lyrs[-1]._fc.weight.requires_grad))

  def _check_values_stored_in_delta_v_and_lgts_dict(self, dltv_dict, lgts_dict):
    num_conv_lyrs, num_dense_lyrs = 0, 0
    for key in dltv_dict.keys():
      log.DEBUG("Checking delta_v and logits values of key: %s" % key)
      assert torch.norm(dltv_dict[key].detach()) != 0
      assert torch.norm(lgts_dict[key].detach()) != 0
      if "conv" in key:
        num_conv_lyrs += 1
      elif "dense" in key:
        num_dense_lyrs += 1

    assert self._exu.num_lyrs["conv_lyrs"] == num_conv_lyrs
    assert self._exu.num_lyrs["dense_lyrs"] == num_dense_lyrs

    for key in self._scnn_arch.keys():
      if "conv" in key or "dense" in key:
        assert key in dltv_dict.keys()
        assert key in lgts_dict.keys()

  def _forward_through_time(self, x, mode):
    """
    Implements the forward pass through all time-steps.

    Args:
      x <Tensor>: Standardized floating point pixel values (all >= 0)
    """
    all_hdl_all_ts_delta_v = (
        self._exu.init_all_hdl_spk_conv_dense_all_ts_delta_v_dict(self))
    all_rol_all_ts_logits = self._tgu.init_all_rol_all_ts_logits_dict(self)

    for t in range(self._rtc.DCFG["presentation_ts"]):
      spikes = self._i2s.encode(x)

      for i, (c_lyr, r_lyr) in enumerate(
          zip(self._spk_conv_lyrs, self._conv_rdt_lyrs)):
        spikes, delta_v = c_lyr(spikes.detach()) # Detach spks to avoid SurrGD.

        if i == self._exu.num_lyrs["conv_lyrs"]-1: # Last SpkConv Layer.
          spikes = spikes.flatten(start_dim=1) # Flatten the spikes.

        logits = r_lyr(spikes)
        all_hdl_all_ts_delta_v["conv_lyr_%s" % (i+1)][:, t] = delta_v
        all_rol_all_ts_logits["conv_lyr_%s" % (i+1)][:, t] = logits

      for i, (d_lyr, r_lyr) in enumerate(
          zip(self._spk_dense_lyrs, self._dense_rdt_lyrs)):
        spikes, delta_v = d_lyr(spikes.detach()) # Detach spks to avoid SurrGD.
        logits = r_lyr(spikes)
        all_hdl_all_ts_delta_v["dense_lyr_%s" %(i+1)][:, t] = delta_v
        all_rol_all_ts_logits["dense_lyr_%s" % (i+1)][:, t] = logits

    return all_hdl_all_ts_delta_v, all_rol_all_ts_logits

  def forward(self, x, mode):
    """
    Implements the forward method on the batch input `x`. Note that for each
    batch input `x`, the necessary neuron states are reset.

    Args:
      x <Tensor>: Standardized floating point pixel values (all >= 0) of shape
                  (batch_size, inp_chnls, inp_dmx, inp_dmy). Note that this
                  input remains the same for all the presentation time-steps.
                  Hence, this DALTON_SCNN is suitable for static images only.
    """
    # Check and log the norm of all the SpkConv/SpkDense and Readout Layers.
    if self._rtc.DEBUG:
      self._check_norm_and_requires_grad_of_conv_lyrs()
      self._check_norm_and_requires_grad_of_last_conv_rdt_and_first_spk_dense()
      self._check_norm_and_requires_grad_of_dense_lyrs()

    # Do forward pass through time.
    all_hdl_all_ts_delta_v, all_rol_all_ts_logits = (
        self._forward_through_time(x, mode))

    if self._rtc.DEBUG: # Check if values w.r.t. all the layers are obtained.
      self._check_values_stored_in_delta_v_and_lgts_dict(
          all_hdl_all_ts_delta_v, all_rol_all_ts_logits)

    return all_hdl_all_ts_delta_v, all_rol_all_ts_logits
