#
# All Hidden Layer operations here are for one time-step.
#

import _init_paths

import numpy as np
import torch

from consts.exp_consts import EXC
from utils.base_utils import log
from utils.base_utils.exp_utils import ExpUtils
from utils.base_utils.srgt_spike_derivative import srgt_spike_func

class SpkDenseLayer(torch.nn.Module):
  """ Implementation for one time-step. """

  def __init__(self, n_prev, n_hdn, batch_size, rtc, dt=1e-3):
    """
    Args:
      n_prev <int>: Number of neurons in the previous layer.
      n_hdn <int>: Number of neurons in the hidden layer.
      batch_size <int>: Batch Size.
      rtc <RTC> Run Time Constants class.
      dt <float> Delta t for calculating decay constants.
    """
    super().__init__()
    self._rtc = rtc
    self._exu = ExpUtils(rtc)
    self._v = torch.zeros(batch_size, n_hdn, dtype=EXC.PT_DTYPE).to(rtc.DEVICE)
    self._c = torch.zeros(batch_size, n_hdn, dtype=EXC.PT_DTYPE).to(rtc.DEVICE)
    if rtc.TAU_CUR is not None:
      self._c_decay = torch.as_tensor(np.exp(-dt/rtc.TAU_CUR))
    else:
      self._c_decay = 0
    log.INFO("Spk Dense Layer current decay value: %s" % self._c_decay)
    self._v_thr = torch.as_tensor(rtc.V_THRESHOLD)

    self._fc = torch.nn.Linear(n_prev, n_hdn, bias=False, dtype=EXC.PT_DTYPE)
    # Transpose the weight dimensions, because internally, the Linear Layer
    # transposes the weights when multiplying to the input.
    self._fc.weight.data = torch.empty(
        n_hdn, n_prev, dtype=EXC.PT_DTYPE).normal_(mean=0.0,
        std=EXC.WEIGHT_SCALE/np.sqrt(n_prev)).to(rtc.DEVICE)
    log.INFO("SpkDenseLayer initialized with weight shape: {} and requires_grad"
             ": {}".format(self._fc.weight.shape,
                           self._fc.weight.requires_grad))

  def re_initialize_states(self):
    self._v = torch.zeros_like(self._v, dtype=EXC.PT_DTYPE)
    self._c = torch.zeros_like(self._c, dtype=EXC.PT_DTYPE)

  def _get_spikes_and_reset_v(self, v):
    """
    Spikes and resets the membrane potential `v`.
    """
    delta_v = v - self._v_thr
    spikes = srgt_spike_func(delta_v) # Get spikes which have a surrogate derivative.
    v = self._exu.reset_v(v, spikes.detach()) # Reset V.

    return spikes, delta_v, v

  def forward(self, x):
    """
    Forward computation for one time-step.

    Args:
      x <torch.Tensor>: Spikes from the previous layer of shape:
                        (batch_size, n_prev)
    """
    x = self._fc(x)

    # Neuron dynamics.
    c = self._c_decay*self._c + x
    v = self._v + c
    # Rectify the voltage if negative.
    mask = v < 0.0
    v[mask] = 0.0

    spikes, delta_v, volt = self._get_spikes_and_reset_v(v)
    self._c = c
    self._v = v

    if "RTRL" in self._rtc.MODEL_NAME: # Detach neuron states from graph.
      self._c.detach_()
      self._v.detach_()

    return spikes, delta_v
