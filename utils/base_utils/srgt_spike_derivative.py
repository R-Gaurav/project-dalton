import _init_paths

import torch

from consts.exp_consts import EXC
from utils.base_utils.exp_utils import ExpUtils

class SrgtSpkDrtv(torch.autograd.Function):
  """
  This class implements the spiking function in the forward pass and calculates
  its surrogate derivative in the backward pass.
  """
  @staticmethod
  def forward(ctx, x):
    """
    Computes and returns the spikes.

    Args:
      ctx: is the context object to be used later in backward pass.
      x: is the input to the spiking function, i.e. the Heaviside Step function.
         Here `x` = v[t] - v_thr`, i.e. S(v[t]) = H(v[t] - v_thr).
    """
    ctx.save_for_backward(x)
    spikes = ExpUtils.spike_func(x)

    return spikes

  @staticmethod
  def backward(ctx, grad_output):
    """
    Computes the local gradient to be propagated back. Note that the local
    gradient = gradient of the `forward` function * grad_output. Here the
    `forward` function is estimated by the fast sigmoid function: x/(1+|x|).

    Args:
      ctx: is the local context object whose stored values would be used to
           calculate local gradient.
      grad_output: is the gradient output received from the previous layer.
    """
    x, = ctx.saved_tensors
    local_grad = grad_output * ExpUtils.srgt_drtv_func(x)
    return local_grad

srgt_spike_func = SrgtSpkDrtv.apply
