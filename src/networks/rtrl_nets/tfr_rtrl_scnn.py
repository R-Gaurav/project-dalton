import _init_paths

import torch

from consts.exp_consts import EXC
from utils.base_utils import log
from utils.base_utils.exp_utils import ExpUtils
from utils.base_utils.tfr_utils import TFR_Utils
from utils.tensor_encoder import ImageToSpike
from utils.spk_conv_layer import SpkConvLayer
from utils.spk_dense_layer import SpkDenseLayer
from utils.readout_layer import DenseReadoutLayer

class TFR_RTRL_SCNN(torch.nn.Module):
  def __init__(self, rtc):
    """
    Args:
      rtc <RTC>: The Run Time Constants class.
    """
    super().__init__()

    self._rtc = rtc
    self._exu = ExpUtils(rtc)
    self._tfu = TFR_Utils(rtc)
    self._bsize = rtc.DCFG["batch_size"]
    self._otpdm = rtc.DCFG["num_clss"]
    self._scnn_arch = rtc.SCNN_ARCH

    log.INFO("Creating the TFR RTRL Spiking CNN... Architecture below:")
    log.INFO(self._scnn_arch)

    log.INFO("Creating the Input Conv Layer...")
    self._i2s = ImageToSpike(
        self._scnn_arch["inp_lyr"]["inp_chnls"],
        self._scnn_arch["inp_lyr"]["inp_dmx"],
        self._scnn_arch["inp_lyr"]["inp_dmy"],
        self._bsize, rtc)
    log.INFO("Creating the intermdiate Spiking Conv and Dense Layers and their"
             "corresponding local Readout Layers...")
    (self._spk_conv_lyrs, self._conv_rdt_lyrs, self._spk_dense_lyrs,
     self._dense_rdt_lyrs) = self._get_spk_conv_and_spk_dense_and_rdt_layers()
    log.INFO("Following are the layers in the TFR SCNN...")
    log.INFO("Spiking Conv Layers: {}".format(self._spk_conv_lyrs))
    log.INFO("Spiking Dense Layers: {}".format(self._spk_dense_lyrs))
    log.INFO("Conv Readout Layers: {}".format(self._conv_rdt_lyrs))
    log.INFO("Dense Readout Layers: {}".format(self._dense_rdt_lyrs))

  def _get_spk_conv_and_spk_dense_and_rdt_layers(self):
    """
    Get the ConvLayer, DenseLayers, and the corresponding ReadoutLayers. Note
    that each ConvLayer and DenseLayer has an associate ReadoutLayer to train
    the Conv and Dense weights. The ReadoutLayer weights are non-trainable.
    """
    spk_conv_lyrs = torch.nn.ModuleList()
    spk_dense_lyrs = torch.nn.ModuleList()
    conv_rdt_lyrs = torch.nn.ModuleList()
    dense_rdt_lyrs = torch.nn.ModuleList()

    prev_inp_chnls, prev_inp_dmx, prev_inp_dmy = (
        self._scnn_arch["inp_lyr"]["inp_chnls"],
        self._scnn_arch["inp_lyr"]["inp_dmx"],
        self._scnn_arch["inp_lyr"]["inp_dmy"]
        )
    n_prev = prev_inp_chnls * prev_inp_dmx * prev_inp_dmy

    for l_num in range(1, self._exu.num_lyrs["conv_lyrs"]+1):
      # Get the Spiking Conv Layer.
      kwargs = self._exu.get_conv_layer_kwargs(
          prev_inp_chnls, prev_inp_dmx, prev_inp_dmy,
          self._scnn_arch["conv_lyr_%s" % l_num])

      spk_conv_lyr = SpkConvLayer(self._bsize, self._rtc, **kwargs)
      spk_conv_lyrs.append(spk_conv_lyr)

      # Get the associated Linear Readout Layer, note requires_grad = False.
      n_prev = (spk_conv_lyr.otp_chnls * # Flatten the output from SpkConvLayer.
                spk_conv_lyr.otp_dmx * spk_conv_lyr.otp_dmy)

      rdt_lyr = DenseReadoutLayer(n_prev, self._otpdm, self._rtc)
      conv_rdt_lyrs.append(rdt_lyr)

      prev_inp_dmx, prev_inp_dmy = spk_conv_lyr.otp_dmx, spk_conv_lyr.otp_dmy
      prev_inp_chnls = self._scnn_arch["conv_lyr_%s" % l_num]["otp_chnls"]

    for l_num in range(1, self._exu.num_lyrs["dense_lyrs"]+1):
      spk_dense_lyrs.append(
          SpkDenseLayer(n_prev, self._scnn_arch["dense_lyr_%s" % l_num],
                         self._bsize, self._rtc)
          )
      dense_rdt_lyrs.append(DenseReadoutLayer(
          self._scnn_arch["dense_lyr_%s" % l_num], self._otpdm, self._rtc))

      n_prev = self._scnn_arch["dense_lyr_%s" % l_num]

    return spk_conv_lyrs, conv_rdt_lyrs, spk_dense_lyrs, dense_rdt_lyrs

  def _check_all_spk_conv_and_dense_and_readout_layers_requires_grad(self):
    for cl, rl in zip(self._spk_conv_lyrs, self._conv_rdt_lyrs):
      assert cl._conv.weight.requires_grad == True
      assert rl._fc.weight.requires_grad == False
      log.DEBUG("Spiking Conv Layer: {} and Requires Grad: {}, Norm: {}".format(
                cl.__str__, cl._conv.weight.requires_grad,
                torch.norm(cl._conv.weight.data)))
      log.DEBUG("Conv Readout Layer: {} and Requires Grad: {}, Norm: {}".format(
                rl.__str__, rl._fc.weight.requires_grad,
                torch.norm(rl._fc.weight.data)))

    for dl, rl in zip(self._spk_dense_lyrs, self._dense_rdt_lyrs):
      assert dl._fc.weight.requires_grad == True
      assert rl._fc.weight.requires_grad == False
      log.DEBUG("Spiking Dense Layer: {}, Requires Grad: {}, Norm: {}".format(
                dl.__str__, dl._fc.weight.requires_grad,
                torch.norm(dl._fc.weight.data)))
      log.DEBUG("Dense Readout Layer: {}, Requires Grad: {}".format(
                rl.__str__, rl._fc.weight.requires_grad,
                torch.norm(rl._fc.weight.data)))

  def _check_values_stored_in_lgts_dict(self, lgts_dict):
    num_conv_lyrs, num_dense_lyrs = 0, 0
    for key in lgts_dict.keys():
      #log.DEBUG("Checking values of the logits with key: %s" % key)
      #assert torch.norm(lgts_dict[key].detach()) != 0
      if "conv" in key:
        num_conv_lyrs += 1
      elif "dense" in key:
        num_dense_lyrs += 1

    assert self._exu.num_lyrs["conv_lyrs"] == num_conv_lyrs
    assert self._exu.num_lyrs["dense_lyrs"] == num_dense_lyrs

    for key in self._scnn_arch.keys():
      if "conv" in key or "dense" in key:
        assert key in lgts_dict.keys()

  def _forward_through_1_ts(self, x):
    """
    Implements the forward pass through one time-step.

    Args:
      x <Tensor>: Standardized floating point pixel values (all >= 0)
    """
    all_rol_1_ts_logits = self._tfu.init_all_rol_1_ts_logits_dict()
    all_hdl_1_ts_delta_v = (
        self._exu.init_all_hdl_spk_conv_dense_1_ts_delta_v_dict(self))

    spikes = self._i2s.encode(x)

    for i, (c_lyr, r_lyr) in enumerate(
        zip(self._spk_conv_lyrs, self._conv_rdt_lyrs)):
      # Detach the spikes to avoid SurrGD.
      spikes, delta_v = c_lyr(spikes.detach())
      logits = r_lyr(spikes.flatten(start_dim=1)) # Flatten the spikes.

      all_hdl_1_ts_delta_v["conv_lyr_%s" % (i+1)] = delta_v
      all_rol_1_ts_logits["conv_lyr_%s" % (i+1)] = logits

    spikes = spikes.flatten(start_dim=1)
    for i, (d_lyr, r_lyr) in enumerate(
        zip(self._spk_dense_lyrs, self._dense_rdt_lyrs)):
      # Detach the spikes to avoid SurrGD.
      spikes, delta_v = d_lyr(spikes.detach())
      logits = r_lyr(spikes)

      all_hdl_1_ts_delta_v["dense_lyr_%s" % (i+1)] = delta_v
      all_rol_1_ts_logits["dense_lyr_%s" % (i+1)] = logits

    return all_hdl_1_ts_delta_v, all_rol_1_ts_logits

  def forward(self, x):
    """
    Implements the forward method on the batch input `x`. Note that for each
    input `x`, the necessary neuron states are reset.

    Args:
      x <Tensor>: Standardized floating point pixel values (all >= 0) of shape
                  (batch_size, inp_chnls, inp_dmx, inp_dmy)
    """
    if self._rtc.DEBUG: # Check the requires_grad of all the layers.
      self._check_all_spk_conv_and_dense_and_readout_layers_requires_grad()

    # Do forward pass through time.
    all_hdl_1_ts_delta_v, all_rol_1_ts_logits = self._forward_through_1_ts(x)

    if self._rtc.DEBUG: # Check if values w.r.t. all the layers are obtained.
      self._check_values_stored_in_lgts_dict(all_rol_1_ts_logits)

    return all_hdl_1_ts_delta_v, all_rol_1_ts_logits
