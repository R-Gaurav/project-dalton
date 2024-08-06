#
# This file implements the spiking Convolutional layer.
#

import _init_paths

import numpy as np
import sys
import torch

from consts.exp_consts import EXC
from utils.base_utils import log
from utils.base_utils.spc_utils import SPCUtils
from utils.base_utils.exp_utils import ExpUtils
from utils.base_utils.srgt_spike_derivative import srgt_spike_func

class ConvLayer(torch.nn.Module):
  """ Implementation for one time-step. """

  def __init__(self, device, **kwargs):
    """
    Args:
      kwargs:
        inp_chnls <int>: Number of input channels.
        otp_chnls <int>: Number of output channels.
        krnl_size <int>: Size of the convolving kernel.
        stride <int>: Stride of the convolution.
        padding <int>: Padding around the input, default 0.
        dilation <int>: Dilation of the kernel, default 1.
    """
    super().__init__()

    self.inp_chnls = kwargs["inp_chnls"]
    self.otp_chnls = kwargs["otp_chnls"]
    self._krnl_size = kwargs["krnl_size"]
    self._stride = kwargs["stride"]
    self._padding = kwargs.get("padding", 0)
    self._dilation = kwargs.get("dilation", 1)

    self._conv = torch.nn.Conv2d(
        self.inp_chnls, self.otp_chnls, self._krnl_size, stride=self._stride,
        padding=self._padding, dilation=self._dilation, dtype=EXC.PT_DTYPE,
        device=device, bias=False)

class SpkConvLayer(ConvLayer):
  """ Implementation for one time-step. """

  def __init__(self, batch_size, rtc, dt=1e-3, **kwargs):
    """
    Args:
      batch_size <int>: Batch Size of the input.
      rtc <RTC>: Run Time Constants class.
      dt <float>: Delta t for calculating decay constants.

      kwargs:
        inp_dmx <int>: Width of input.
        inp_dmy <int>: Height of input.
        krnl_size <int>: Size of the convolving kernel.
        stride <int>: Stride of the convolution, default 1.
        padding <int>: Padding around the input, default 0.
        dilation <int>: Dilation of the kernel, default 1.
    """
    super().__init__(rtc.DEVICE, **kwargs)
    log.INFO("Spiking Conv Layer initialized with shape: {} and requires_grad:"
             " {}".format(self._conv.weight.shape,
                          self._conv.weight.requires_grad))

    self._rtc = rtc
    self._scu = SPCUtils()
    self._exu = ExpUtils(rtc)
    self._bsize = batch_size
    self._pool_krsz = kwargs.get("pool_size", None)
    self._drpt_prob = kwargs.get("drpt_prob", None)
    if rtc.TAU_CUR is not None:
      self._c_decay = torch.as_tensor(np.exp(-dt/rtc.TAU_CUR))
    else:
      self._c_decay = 0
    log.INFO("Spk Conv Layer current decay value: %s" % self._c_decay)
    self._v_thr = torch.as_tensor(rtc.V_THRESHOLD)

    self.otp_dmx, self.otp_dmy = self._scu.get_output_shape(
        (kwargs["inp_dmx"], kwargs["inp_dmy"]),
        (self._krnl_size, self._krnl_size), (self._stride, self._stride),
        (self._padding, self._padding), (self._dilation, self._dilation))
    log.INFO("Output of Spiking Conv Layer: (otp_dmx, otp_dmy) = {}".format(
             (self.otp_dmx, self.otp_dmx)))

    #p=0.50 => 42.xx % acc in 100 epochs. CIFAR10 Arch 1
    #p=0.25 => 47.xx % acc in 100 epochs. CIFAR10 Arch 1
    #p=0.10 => 50.xx % acc in 100 epochs. CIFAR10 Arch 1
    #p=0.05 => 52.xx % acc in 100 epochs. CIFAR10 Arch 1 (Trn Acc: 60%).
    #p=0.00 => 51.xx % acc in 100 epochs. CIFAR10 Arch 1 (Trn Acc: 68%)

    if self._drpt_prob is not None:
      self._dropout = torch.nn.Dropout(self._drpt_prob)
      log.INFO("Dropout Layer intialized with dropout probability: {}".format(
               self._drpt_prob))

    if self._pool_krsz is not None:
      if self._rtc.DCFG["pool_type"] == "MaxPool":
        self._pool_lyr = torch.nn.MaxPool2d(kernel_size=self._pool_krsz)
        log.INFO("MaxPooling initialized in Spiking Conv Layer.")
      elif self._rtc.DCFG["pool_type"] == "AvgPool":
        self._pool_lyr = torch.nn.AvgPool2d(kernel_size=self._pool_krsz)
        log.INFO("AvgPooling initialized in Spiking Conv Layer.")
      else:
        sys.exit("pool_type should be either MaxPool/AvgPool. Found: %s"
                 % self._rtc.DCFG["pool_type"])


      # Default pooling stride = kernel_size passed, i.e. the pool_size here.
      # Halve the output dimensions as Pooling will be applied before spiking.
      self.otp_dmx //= self._pool_krsz
      self.otp_dmy //= self._pool_krsz
      log.INFO("Pooling in effect with final output shape: (otp_dmx, otp_dmy) "
               "= {}".format((self.otp_dmx, self.otp_dmy)))

    # Note that for batch_size = bsize, the output from the Max/AvgPooling Layer
    # after convolution is of shape: (bsize, otp_chnls, otp_dmx, otp_dmy).
    self._v = torch.zeros(
        batch_size, self.otp_chnls, self.otp_dmx, self.otp_dmy,
        dtype=EXC.PT_DTYPE, device=rtc.DEVICE)
    self._c = torch.zeros(
        batch_size, self.otp_chnls, self.otp_dmx, self.otp_dmy,
        dtype=EXC.PT_DTYPE, device=rtc.DEVICE)
    log.INFO("Spiking Conv Layer neurons' Voltage shape: {} & Current shape: {}"
             "".format(self._v.shape, self._c.shape))

  def re_initialize_states(self):
    self._v = torch.zeros_like(self._v)
    self._c = torch.zeros_like(self._c)

  def _get_spikes_and_reset_v(self, v):
    delta_v = v - self._v_thr
    spikes = srgt_spike_func(delta_v) # Get spikes which have a surrogate derivative.
    v = self._exu.reset_v(v, spikes.detach()) # Reset V.

    return spikes, delta_v, v

  def forward(self, x):
    """
    Forward computation for one time-step.

    Args:
      x <torch.Tensor>: Spikes from the previous layer of shape:
                        (batch_size, inp_chnls, inp_dmx, inp_dmy).
    """
    x = self._conv(x) # Do convolution.
    if self._pool_krsz is not None:
      if self._rtc.DEBUG:
        log.DEBUG("Executing Pooling after the Conv operation in SpkConv Layer")
      x = self._pool_lyr(x) # Do MaxPooling or AvgPooling based on `pool_type`.

    # Neuron dynamics.
    c = self._c_decay*self._c + x
    v = self._v + c
    # Rectify the voltage if negative.
    mask = v < 0.0
    v[mask] = 0.0

    spikes, delta_v, v = self._get_spikes_and_reset_v(v)
    self._c = c
    self._v = v

    if "RTRL" in self._rtc.MODEL_NAME: # Detach neuron states from graph.
      self._c.detach_()
      self._v.detach_()

    if self._drpt_prob is not None:
      spikes = self._dropout(spikes)

    return spikes, delta_v
