import torch

class EXC(object):
  PT_DTYPE = torch.float32
  MNIST = "MNIST"
  CIFAR10 = "CIFAR10"
  FMNIST = "FMNIST"
  DVS_GESTURE = "DVS_GESTURE"
  DVS_MNIST = "DVS_MNIST"
  DVS_CIFAR10 = "DVS_CIFAR10"

  HARD_RESET = True
  WEIGHT_SCALE = 2.0

  # RTRL exp consts.
  DVS_GES_NTS = 500 # time-steps in milliseconds DVS128-Gesture.
  DVS_MNT_NTS = 300 # time-steps in milliseconds N-MNIST.
  DVS_C10_NTS = 500 # time-steps in milliseconds for DVS-CIFAR10.
  BURN_IN_NTS = 50 # time-steps in milliseconds for event and static images.

  TAU_CUR_LIST = [1e-3, 5e-3] #[None, 1e-3, 5e-3, 10e-3] # 20e-3
  GAIN_LIST = [1, 2] #, 4
  BIAS_LIST = [0]
