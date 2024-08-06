import _init_paths

import hickle
import pickle
import numpy as np
import torch

from consts.exp_consts import EXC
from src.networks.rtrl_nets.dalton_rtrl_scnn import DALTON_RTRL_SCNN
from utils.base_utils import log
from utils.base_utils.data_prep_utils import DataPrepUtils
from utils.base_utils.exp_utils import ExpUtils
from utils.base_utils.dalton_utils import DALTON_Utils

class TREV_DALTON_RTRL_SCNN(object):
  def __init__(self, rtc):
    self._rtc = rtc
    self._exu = ExpUtils(rtc)
    self._tgu = DALTON_Utils(rtc)
    self._dpu = DataPrepUtils(rtc.DATASET)
    self._train_dl, self._test_dl = self._dpu.load_dataset(
        rtc.DCFG["batch_size"], rtc.USE_TRANSFORMS)
    self._net = DALTON_RTRL_SCNN(rtc).to(rtc.DEVICE)
    self._best_test_acc = 0
    log.INFO("Experimenting with DALTON RTRL Spiking CNN...")

  def train_model(self):
    learning_rate = self._rtc.DCFG["learning_rate"]
    for epoch in range(1, self._rtc.DCFG["epochs"]+1):
      log.INFO("Starting training for epoch: %s" % epoch)
      pred_clss = [] # For the last dense layer.
      true_clss = []
      dset_last_rol_p_t_cls = (
          self._tgu.get_dset_last_rol_pred_and_true_cls_dict())

      self._net.train()
      # Define the optimizer here to reinitialize the internal states of
      # the Adam optimizer (otherwise the frozen weights still get updated).
      if epoch % self._rtc.LR_DCY_EPOCH == 0:
        learning_rate *= 0.5

      log.INFO("Epoch: %s, Learning rate value: %s" % (epoch, learning_rate))
      optimizer = torch.optim.Adam(self._net.parameters(), lr=learning_rate)

      for trn_x, trn_y in self._train_dl: # Batch Input and Batch Output.
        if self._rtc.DCFG["event_type"]:
          if self._rtc.DCFG["dataset"] == EXC.DVS_GESTURE:
            trn_x, n_ts = self._exu.get_dvs_gesture_batch_and_n_ts(trn_x, True)
          elif self._rtc.DCFG["dataset"] == EXC.DVS_CIFAR10:
            trn_x, n_ts = self._exu.get_dvs_cifar10_batch_and_n_ts(trn_x, True)
          elif self._rtc.DCFG["dataset"] == EXC.DVS_MNIST:
            # N-MNIST (n_ts = trn_x.shape[1] otherwise).
            trn_x, n_ts = trn_x.to(self._rtc.DEVICE), EXC.DVS_MNT_NTS
        else:
          trn_x, n_ts = (
              trn_x.to(self._rtc.DEVICE), self._rtc.DCFG["presentation_ts"])

        trn_y = trn_y.to(self._rtc.DEVICE)

        last_rol_all_ts_pr_y = self._exu.init_bwise_last_rol_all_ts_pr_y(
            n_ts-EXC.BURN_IN_NTS)

        # Reinitialize the neurons' states for every input batch of samples.
        self._exu.reinitialize_all_lyrs_neurons_states(self._net)

        for t in range(n_ts):
          ######################################################################
          # Do forward pass and get predictions logits for one time-step.
          if not self._rtc.DCFG["event_type"]:
            all_hdl_1_ts_delta_v, all_rol_1_ts_logits = self._net(trn_x)
          else:
            all_hdl_1_ts_delta_v, all_rol_1_ts_logits = self._net(trn_x[:, t])

          ######################################################################
          # Get the y_pred values for all the Readout Layers.
          all_rol_1_ts_pr_y = self._exu.get_all_rol_1_ts_pr_y_dict(
              all_rol_1_ts_logits) # Output Layer shape: (batch_size, otp_clss).
          ######################################################################
          # Get the y_true values for all the Readout Layers.
          all_rol_1_ts_tr_y = self._tgu.get_all_rol_1_ts_tr_y_dict(
              all_rol_1_ts_pr_y, all_hdl_1_ts_delta_v, self._net, trn_y)
          ######################################################################
          # Calculate all Readout Layers loss now.
          loss = self._tgu.get_dalton_loss_for_all_rol_1_ts(
              all_rol_1_ts_pr_y, all_rol_1_ts_tr_y)

          # Update the network weights post the BURN_IN time-steps.
          if t >= EXC.BURN_IN_NTS:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update all the Readout Layer weights.
            self._tgu.update_all_readout_layers_weights(self._net)
            last_rol_all_ts_pr_y[:, t-EXC.BURN_IN_NTS] = all_rol_1_ts_pr_y[
                "dense_lyr_%s" % self._exu.num_lyrs["dense_lyrs"]].detach()
          else:
            if self._rtc.DEBUG:
              log.DEBUG(
                  "Burning in for t/BURN_IN_NTS: %s/%s" % (t, EXC.BURN_IN_NTS))

        # Get the predicted training classes over all the time-steps.
        pred_clss.extend(
            self._exu.get_class_predictions(last_rol_all_ts_pr_y).tolist())

        true_clss.extend(trn_y.cpu().tolist())


      log.INFO("Epoch %s training done, Average Training Accuracy: %s" % (
               epoch, np.mean(np.array(pred_clss) == np.array(true_clss))))
      log.INFO("Saving the pred and true clss obtained from all readout lyrs")
      dset_last_rol_p_t_cls["true_cls"].extend(true_clss)
      dset_last_rol_p_t_cls[
          "dense_lyr_%s" % self._exu.num_lyrs["dense_lyrs"]].extend(pred_clss)
      self._exu.save_hdf5_file(
          dset_last_rol_p_t_cls,
          "epoch_%s_training_all_rol_pred_and_true_clss.hdf5" % epoch,
          self._rtc.OTP_DIR)
      #self.eval_model()
      self._exu.save_hdf5_file(
          self.eval_model(),
          "epoch_%s_test_all_rol_pred_and_true_clss.hdf5" % epoch,
          self._rtc.OTP_DIR)

      if epoch == self._rtc.DCFG["epochs"]: # Save model.
        log.INFO("Saving model checkpoint now...")
        torch.save({
            "epoch": epoch,
            "model_state_dict": self._net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss},
            self._rtc.OTP_DIR+"saved_model_epoch_%s.pth" % epoch)

    log.INFO("All Training Epochs Done!")

  def eval_model(self):
    pred_clss = []
    true_clss = []
    dset_last_rol_p_t_cls = self._tgu.get_dset_last_rol_pred_and_true_cls_dict()
    log.INFO("Starting evaluation on test dataset ...")
    self._net.eval()
    with torch.no_grad():
      for tst_x, tst_y in self._test_dl:
        if self._rtc.DCFG["event_type"]:
          if self._rtc.DCFG["dataset"] == EXC.DVS_GESTURE:
            tst_x, n_ts = self._exu.get_dvs_gesture_batch_and_n_ts(tst_x, False)
          elif self._rtc.DCFG["dataset"] == EXC.DVS_CIFAR10:
            tst_x, n_ts = self._exu.get_dvs_cifar10_batch_and_n_ts(tst_x, False)
          elif self._rtc.DCFG["dataset"] == EXC.DVS_MNIST:
            # DVS_MNIST (n_ts = tst_x.shape[1] otherwise.)
            tst_x, n_ts = tst_x.to(self._rtc.DEVICE), EXC.DVS_MNT_NTS
        else:
          tst_x, n_ts = (
              tst_x.to(self._rtc.DEVICE), self._rtc.DCFG["presentation_ts"])

        last_rol_all_ts_pr_y = self._exu.init_bwise_last_rol_all_ts_pr_y(
            n_ts-EXC.BURN_IN_NTS)

        # Reinitialize the neurons' states for every input batch of samples.
        self._exu.reinitialize_all_lyrs_neurons_states(self._net)

        for t in range(n_ts):
          # Do forward pass for 1 time-step and get predicted logits.
          if not self._rtc.DCFG["event_type"]:
            _, all_rol_1_ts_logits = self._net(tst_x)
          else:
            _, all_rol_1_ts_logits = self._net(tst_x[:, t])
          # Get the y_pred values from all the Readout Layers.
          all_rol_1_ts_pr_y = self._exu.get_all_rol_1_ts_pr_y_dict(
            all_rol_1_ts_logits)
          # Store the y_pred values at `t` time-step of the last Readout Layer.
          if t >= EXC.BURN_IN_NTS:
            last_rol_all_ts_pr_y[:, t-EXC.BURN_IN_NTS] = all_rol_1_ts_pr_y[
                "dense_lyr_%s" % self._exu.num_lyrs["dense_lyrs"]].detach()

        dset_last_rol_p_t_cls[
            "dense_lyr_%s" % self._exu.num_lyrs["dense_lyrs"]].extend(
            last_rol_all_ts_pr_y.cpu().numpy())

        pred_clss.extend(
            self._exu.get_class_predictions(last_rol_all_ts_pr_y).tolist())
        true_clss.extend(tst_y.cpu().tolist())

      dset_last_rol_p_t_cls["true_cls"].extend(true_clss)

    test_acc = np.mean(np.array(pred_clss) == np.array(true_clss))
    if test_acc > self._best_test_acc:
      self._best_test_acc = test_acc
    log.INFO("Current Test Accuracy: {}, Best Test Accuracy: {}".format(
             test_acc, self._best_test_acc))

    return dset_last_rol_p_t_cls
