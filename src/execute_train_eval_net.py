import _init_paths
import argparse
import os
import sys
import time
import torch
import numpy as np
import random

from consts.dir_consts import DRC
from consts.exp_consts import EXC
from consts.runtime_consts import RTC
from utils.base_utils import log
from utils.base_utils.data_prep_utils import DataPrepUtils
from utils.base_utils.exp_utils import ExpUtils
from src.bptt_trev.train_eval_tfr_scnn import TREV_TFR_SCNN
from src.bptt_trev.train_eval_dalton_scnn import TREV_DALTON_SCNN
from src.bptt_trev.train_eval_srgt_gd_scnn import TREV_SRGT_GD_SCNN
from src.rtrl_trev.train_eval_dalton_rtrl_scnn import TREV_DALTON_RTRL_SCNN
from src.rtrl_trev.train_eval_tfr_rtrl_scnn import TREV_TFR_RTRL_SCNN
from utils.base_utils.data_prep_utils import DataPrepUtils

def setup_logging(rtc):
  exu = ExpUtils(rtc)
  log.configure_log_handler(
      "%s/train_eval_tfr_net_%s_%s.log"
      % (rtc.OTP_DIR, rtc.MODEL_NAME, exu.get_timestamp()))
  keys = list(vars(rtc).keys())
  log.INFO("#"*30 + " RUN TIME CONSTANTS " + "#"*30)
  for key in keys:
    log.INFO("{0}: {1}".format(key, getattr(rtc, key)))
  log.INFO("#"*80)

def setup_otp_dir(rtc):
  otp_dir = DRC.RESULTS_DIR + "%s/%s/%s/SEED_%s/" % (
      rtc.MODEL_NAME, rtc.META_CFG_DIR, rtc.DATASET, rtc.SEED)

  rtc.OTP_DIR = otp_dir
  os.makedirs(otp_dir, exist_ok=True)

def execute_net(rtc):
  setup_otp_dir(rtc)
  setup_logging(rtc)
  log.INFO("Obtaining the network ... %s" % rtc.MODEL_NAME)
  if rtc.MODEL_NAME == "SRGT_GD_SCNN":
    trev_net = TREV_SRGT_GD_SCNN(rtc)
  elif rtc.MODEL_NAME == "TFR_SCNN":
    trev_net = TREV_TFR_SCNN(rtc)
  elif rtc.MODEL_NAME == "DALTON_SCNN":
    trev_net = TREV_DALTON_SCNN(rtc)
  elif rtc.MODEL_NAME == "DALTON_RTRL_SCNN":
    trev_net = TREV_DALTON_RTRL_SCNN(rtc)
  elif rtc.MODEL_NAME == "TFR_RTRL_SCNN":
    trev_net = TREV_TFR_RTRL_SCNN(rtc)
  else:
    sys.exit("Enter valid MODEL_NAME. Exiting...")

  log.INFO("Starting the model training and evaluation (each epoch)...")
  trev_net.train_model()

def construct_runtime_constants_and_run_exp(rtc, args, debug=False):
  dpu = DataPrepUtils(args.dataset)
  rtc.DRC = DRC
  rtc.DEBUG = debug
  rtc.DATASET = args.dataset
  rtc.DCFG = dpu.load_data_yaml_file()
  rtc.DCFG["batch_size"] = args.batch_size
  rtc.SCNN_ARCH = RTC.DCFG[args.scnn_arch]
  rtc.DEVICE = (torch.device("cuda:%s" % args.gpu_id)
                if torch.cuda.is_available() else torch.device("cpu"))
  rtc.DCFG["epochs"] = args.epochs

  for tau_cur in EXC.TAU_CUR_LIST:
    for gain in EXC.GAIN_LIST:
      for bias in EXC.BIAS_LIST:
        rtc.TAU_CUR = tau_cur
        rtc.GAIN = gain
        rtc.BIAS = bias

        rtc.META_CFG_DIR = (
            "num_lyrs_%s/srgt_drtv_scale_%s/tau_cur_%s/gain_%s/bias_%s/" % (
                len(rtc.SCNN_ARCH), rtc.SRGT_DRTV_SCALE, rtc.TAU_CUR, rtc.GAIN,
                rtc.BIAS)
            )

        start = time.time()
        execute_net(rtc)
        log.INFO(
            "Experiment Done! Time Taken: {}h".format((time.time()-start)/3600))
        log.RESET()


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset", type=str, required=True)
  parser.add_argument("--epochs", type=int, required=True)
  parser.add_argument("--gpu_id", type=int, default=0, required=False)
  parser.add_argument("--scnn_arch", type=int, required=True)
  parser.add_argument("--batch_size", type=int, required=True)
  parser.add_argument("--seed", type=int, required=True)

  args = parser.parse_args()
  args.scnn_arch = "spk_cnn_arch_%s" % args.scnn_arch

  RTC.SEED = args.seed
  torch.manual_seed(RTC.SEED)
  np.random.seed(RTC.SEED)
  random.seed(RTC.SEED)
  start = time.time()
  construct_runtime_constants_and_run_exp(RTC, args)
  print("All Experiments Done, Time Taken: {}".format((time.time()-start)/3600))
