import _init_paths

import numpy as np
import sys
import torch

from consts.exp_consts import EXC
from utils.base_utils import log
from utils.base_utils.exp_utils import ExpUtils

class DenseReadoutLayer(torch.nn.Module):
  """ Implementation for one time-step. """

  def __init__(self, n_hdn, n_out, rtc):
    """
    Args:
      n_hdn <int>: Number of neurons in the hidden layer on which readout is
                   applied.
      n_out <int>: Number of units in the output layer where the local Loss is
                   calculated.
      rtc <RTC>: Run Time Constant class.
    """
    super().__init__()
    self._exu = ExpUtils(rtc)
    self._fc = torch.nn.Linear(n_hdn, n_out, bias=False, dtype=EXC.PT_DTYPE)

    # Initialize the Readout Layer weights.
    if rtc.UNIFORM_RO:
      self._fc.weight.data = torch.empty(
          n_out, n_hdn, dtype=EXC.PT_DTYPE).uniform_(
          -np.sqrt(1/n_hdn), np.sqrt(1/n_hdn)).to(rtc.DEVICE)
    elif rtc.NORMAL_RO:
      self._fc.weight.data = torch.empty(
          n_out, n_hdn, dtype=EXC.PT_DTYPE, device=rtc.DEVICE).normal_(mean=0,
          std=EXC.WEIGHT_SCALE/np.sqrt(n_hdn)).to(rtc.DEVICE)

    # Make the Readout Layer weights untrainable.
    self._fc.weight.requires_grad = False
    log.INFO("Dense Readout Layer initialized with shape: {} and requires_grad:"
             " {}".format(self._fc.weight.shape, self._fc.weight.requires_grad))

  def forward(self, x):
    """
    Forward computation for one time-step.

    Args:
      x <torch.Tensor>: Spikes from the previous layer of shape:
                        (batch_size, n_hdn).
    """
    x = self._fc(x)
    return x

class ConvReadoutLayer(torch.nn.Module):
  """
  Implementation for one time-step.
  """
  def __init__(self, rtc, **kwargs):
    """
    Args:
      kwargs:
        inp_chnls <int>: Number of input channels.
        otp_chnls <int>: Number of output channels.
        krnl_size <int>: Size of the convolving kernels.
        stride <int>: Stride of the convolution.
        padding <int>: Padding around the input, default 0.
        dilation <int>: Dilation of the kernel, default 1.
    """
    super().__init__()

    self._rtc = rtc
    self.inp_chnls = kwargs["inp_chnls"]
    self.otp_chnls = kwargs["otp_chnls"]
    self._krnl_size = kwargs["krnl_size"]
    self._stride = kwargs["stride"]
    self._padding = kwargs.get("padding", 0)
    self._dilation = kwargs.get("dilation", 1)
    self._pool_krsz = kwargs.get("pool_size", None)

    self._conv = torch.nn.Conv2d(
        self.inp_chnls, self.otp_chnls, self._krnl_size, stride=self._stride,
        padding=self._padding, dilation=self._dilation, dtype=EXC.PT_DTYPE,
        bias=False, device=rtc.DEVICE)

    # Make the Conv Readout Layer weight untrainable.
    self._conv.weight.requires_grad = False
    log.INFO("Conv Readout Layer initialized with shape: {} and requires_grad:"
             " {}".format(self._conv.weight.shape,
                          self._conv.weight.requires_grad))

    if self._pool_krsz is not None:
      if self._rtc.DCFG["pool_type"] == "MaxPool":
        self._pool_lyr = torch.nn.MaxPool2d(kernel_size=self._pool_krsz)
        log.INFO("MaxPooling layer initialized in Conv Readout Layer.")
      elif self._rtc.DCFG["pool_type"] == "AvgPool":
        self._pool_lyr = torch.nn.AvgPool2d(kernel_size=self._pool_krsz)
        log.INFO("AvgPooling layer initialized in Conv Readout Layer.")
      else:
        sys.exit("pool_type should be either MaxPool/AvgPool. Found: %s"
                 % self._rtc.DCFG["pool_type"])

  def forward(self, x):
    """
    Forward computation for one time-step.

    Args:
      x <torch.Tensor>: Spikes from the previous layer of shape:
                        (batch_size, inp_chnls, dim_x, dim_y).
    """
    x = self._conv(x)
    if self._pool_krsz is not None:
      if self._rtc.DEBUG:
        log.DEBUG("Executing Pooling after the Conv operation in Readout Layer")
      x = self._pool_lyr(x)

    return x
