class RTC(object):
  UNIFORM_RO = True
  NORMAL_RO = False
  assert UNIFORM_RO ^ NORMAL_RO # Make sure only one is True.

  V_THRESHOLD = 1
  LR_DCY_EPOCH = 30
  ITM_LOSS_FUNC = "MSE"
  OTP_LOSS_FUNC = "MSE"
  USE_TRANSFORMS=False # For experiments with Arch-5 set it True.

  # Frequent changes.
  #MODEL_NAME = "SRGT_GD_SCNN"
  #MODEL_NAME = "TFR_SCNN"
  MODEL_NAME = "DALTON_SCNN"

  #MODEL_NAME = "DALTON_RTRL_SCNN"
  #MODEL_NAME = "TFR_RTRL_SCNN"

  SRGT_DRTV_SCALE = 5 # Fixed.
