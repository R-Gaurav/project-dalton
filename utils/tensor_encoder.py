import _init_paths

import torch

from consts.exp_consts import EXC
from utils.base_utils.base_spk_neuron import BaseSpikingNeuron

class ImageToSpike(BaseSpikingNeuron): # Encodes an Image to Spike.
  def __init__(self, inp_chnls, inp_dmx, inp_dmy, batch_size, rtc):
    super().__init__((batch_size, inp_chnls, inp_dmx, inp_dmy), rtc)

    self._rtc = rtc
    self._gain = rtc.GAIN
    self._bias = rtc.BIAS
    # Move the voltage tensor to GPU (it is defined in `BaseSpikingNeuron`).
    self._v = self._v.to(rtc.DEVICE)
    self._v_thr = self._v_thr.to(rtc.DEVICE)

  def encode(self, x, e=1):
    """
    Encodes the input x. Note that x is assumed to be positive all the time for
    images/videos.

    Args:
      x <Tensor>: The tensor input x at time t.
      e <int>: Encoder value. Keept it always 1 for images/videos.
    """
    self._J = self._gain*e*x + self._bias
    v = self._v + self._J
    spikes, v = self.get_spikes_and_reset_v(v)
    self._v = v
    if "RTRL" in self._rtc.MODEL_NAME:
      self._v.detach_()

    return spikes

  def re_initialize_v(self):
    self._v = torch.zeros_like(self._v)
