import _init_paths

import torch

from consts.exp_consts import EXC
from utils.base_utils import log
from utils.base_utils.exp_utils import ExpUtils
from utils.base_utils.srgt_gd_utils import SRGT_GD_Utils
from utils.tensor_encoder import ImageToSpike
from utils.spk_conv_layer import SpkConvLayer
from utils.spk_dense_layer import SpkDenseLayer
from utils.readout_layer import DenseReadoutLayer

class SRGT_GD_SCNN(torch.nn.Module):
  def __init__(self, rtc):
    """
    Args:
      rtc <RTC>: The Run Time Constants class.
    """
    super().__init__()

    self._rtc = rtc
    self._exu = ExpUtils(rtc)
    self._sgu = SRGT_GD_Utils(rtc)
    self._otpdm = rtc.DCFG["num_clss"]
    self._bsize = rtc.DCFG["batch_size"]
    self._scnn_arch = rtc.SCNN_ARCH

    log.INFO("Creating the SRGT_GD Spiking CNN..., Architecture below:")
    log.INFO(self._scnn_arch)

    log.INFO("Creaing the Input Conv Layer...")
    self._i2s = ImageToSpike(
        self._scnn_arch["inp_lyr"]["inp_chnls"],
        self._scnn_arch["inp_lyr"]["inp_dmx"],
        self._scnn_arch["inp_lyr"]["inp_dmy"],
        self._bsize, rtc)
    log.INFO("Creating the intermediate Spiking Conv and Dense Layers and the "
             "final Output Layer...")
    self._spk_conv_lyrs, self._spk_dense_lyrs, self._otp_lyr = (
        self._get_conv_and_dense_and_otp_layers())
    log.INFO("Following are the layers in this SRGT GD SCNN...")
    log.INFO("Spiking Conv Layers: {}".format(self._spk_conv_lyrs))
    log.INFO("Spiking Dense Layers: {}".format(self._spk_dense_lyrs))
    log.INFO("Output Layer: {}".format(self._otp_lyr))

  def _get_conv_and_dense_and_otp_layers(self):
    """
    Get the ConvLayers, DenseLayers, and the final Output Layer. All layers are
    trainable.
    """
    spk_conv_lyrs = torch.nn.ModuleList()
    spk_dense_lyrs = torch.nn.ModuleList()
    prev_inp_chnls, prev_inp_dmx, prev_inp_dmy = (
        self._scnn_arch["inp_lyr"]["inp_chnls"],
        self._scnn_arch["inp_lyr"]["inp_dmx"],
        self._scnn_arch["inp_lyr"]["inp_dmy"]
        )
    for l_num in range(1, self._exu.num_lyrs["conv_lyrs"]+1):
      # Get the Spiking Conv Layer.
      kwargs = self._exu.get_conv_layer_kwargs(
          prev_inp_chnls, prev_inp_dmx, prev_inp_dmy,
          self._scnn_arch["conv_lyr_%s" % l_num])

      spk_conv_lyr = SpkConvLayer(self._bsize, self._rtc, **kwargs)
      spk_conv_lyrs.append(spk_conv_lyr)
      prev_inp_dmx, prev_inp_dmy = spk_conv_lyr.otp_dmx, spk_conv_lyr.otp_dmy
      prev_inp_chnls = self._scnn_arch["conv_lyr_%s" % l_num]["otp_chnls"]

    n_prev = ( # Last Conv Layer output is flattened.
        spk_conv_lyr.otp_chnls * spk_conv_lyr.otp_dmx * spk_conv_lyr.otp_dmy)

    for l_num in range(1, self._exu.num_lyrs["dense_lyrs"]+1):
      spk_dense_lyrs.append(
          SpkDenseLayer(
          n_prev, self._scnn_arch["dense_lyr_%s" % l_num], self._bsize,
          self._rtc)
          )
      n_prev = self._scnn_arch["dense_lyr_%s" % l_num]

    # Get final Output Layer. It's a ReadoutLayer but with requires_grad = True.
    otp_lyr = DenseReadoutLayer(n_prev, self._otpdm, self._rtc)
    otp_lyr._fc.weight.requires_grad = True

    return spk_conv_lyrs, spk_dense_lyrs, otp_lyr

  def _check_all_spk_conv_and_dense_and_otp_layers_requires_grad(self):
    for cl in self._spk_conv_lyrs:
      assert cl._conv.weight.requires_grad == True
      log.DEBUG("Spiking Conv Layer: {} and Requires Grad: {}".format(
                cl.__str__, cl._conv.weight.requires_grad))

    for dl in self._spk_dense_lyrs:
      assert dl._fc.weight.requires_grad == True
      log.DEBUG("Spiking Dense Layer: {} and Requires Grad: {}".format(
                 dl.__str__, dl._fc.weight.requires_grad))

    assert self._otp_lyr._fc.weight.requires_grad == True
    log.DEBUG("Final Output Layer: {} and Requires Grad: {}".format(
              self._otp_lyr.__str__, self._otp_lyr._fc.weight.requires_grad))

  def _forward_through_time(self, x):
    """
    Implements the forward pass through all time-steps.

    Args:
      x <Tensor>: Standardized floating point pixel values (all >= 0)
    """
    all_hdl_all_ts_delta_v = (
        self._exu.init_all_hdl_spk_conv_dense_all_ts_delta_v_dict(self))
    otp_lyr_all_ts_logits = self._sgu.init_output_layer_all_ts_logits_dict()

    for t in range(self._rtc.DCFG["presentation_ts"]):
      spikes = self._i2s.encode(x)

      for i, c_lyr in enumerate(self._spk_conv_lyrs):
        spikes, delta_v = c_lyr(spikes)
        all_hdl_all_ts_delta_v["conv_lyr_%s" % (i+1)][:, t] = delta_v

      spikes = spikes.flatten(start_dim=1) # Flatten the spikes.

      for i, d_lyr in enumerate(self._spk_dense_lyrs):
        spikes, delta_v = d_lyr(spikes)
        all_hdl_all_ts_delta_v["dense_lyr_%s" % (i+1)][:, t] = delta_v

      logits = self._otp_lyr(spikes)
      otp_lyr_all_ts_logits[
          "dense_lyr_%s" % self._exu.num_lyrs["dense_lyrs"]][:, t] = logits

    return all_hdl_all_ts_delta_v, otp_lyr_all_ts_logits

  def forward(self, x):
    """
    Implements the forward method on the batch input `x`. Note that for each
    batch input `x`, the necessary neuron states are reset.

    Args:
      x <Tensor>: Standardized floating point pixel values (all >= 0) of shape
                  (batch_size, inp_chnls, inp_dmx, inp_dmy)
    """
    # Reinitialize neuron states.
    #self._i2s.re_initialize_v() # Current does not have a state in encoding lyr.
    #for c_lyr in self._spk_conv_lyrs:
    #  c_lyr.re_initialize_states()
    #for d_lyr in self._spk_dense_lyrs:
    #  d_lyr.re_initialize_states()

    if self._rtc.DEBUG: # Check and log the requires grad of all layers.
      self._check_all_spk_conv_and_dense_and_otp_layers_requires_grad()

    # Do forward pass through time.
    all_hdl_all_ts_delta_v, otp_lyr_all_ts_logits = (
        self._forward_through_time(x))

    return all_hdl_all_ts_delta_v, otp_lyr_all_ts_logits
