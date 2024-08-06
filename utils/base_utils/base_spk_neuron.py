import _init_paths
import sys
import torch

from consts.exp_consts import EXC
from utils.base_utils.exp_utils import ExpUtils

class BaseSpikingNeuron(object):
  def __init__(self, tensor_size, rtc):
    self._exu = ExpUtils(rtc)
    self._v = torch.zeros(tensor_size, dtype=EXC.PT_DTYPE)
    self._v_thr = torch.as_tensor(rtc.V_THRESHOLD, dtype=EXC.PT_DTYPE)

  def get_spikes_and_reset_v(self, v):
    """
    Returns spike and resets the neuron.
    """
    delta_v = v - self._v_thr
    # Following function doesn't have spike derivative as it is not required.
    spikes = self._exu.spike_func(delta_v)
    #spikes = ExpUtils.spike_func(delta_v)
    v = self._exu.reset_v(v, spikes.detach())

    return spikes, v
