import _init_paths

import hickle
import pickle
import numpy as np
import torch

from src.networks.bptt_nets.dalton_scnn import DALTON_SCNN
from utils.base_utils import log
from utils.base_utils.data_prep_utils import DataPrepUtils
from utils.base_utils.exp_utils import ExpUtils
from utils.base_utils.dalton_utils import DALTON_Utils

class TREV_DALTON_SCNN(object):
  def __init__(self, rtc):
    self._rtc = rtc
    self._exu = ExpUtils(rtc)
    self._tgu = DALTON_Utils(rtc)
    self._dpu = DataPrepUtils(rtc.DATASET)
    self._train_dl, self._test_dl = self._dpu.load_dataset(
        rtc.DCFG["batch_size"], rtc.USE_TRANSFORMS)
    self._net = DALTON_SCNN(rtc).to(rtc.DEVICE)
    self._best_test_acc = 0
    log.INFO("Experimenting with DALTON Spiking CNN...")

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

      log.INFO("Epoch: %s, Learning Rate value: %s" % (epoch, learning_rate))
      optimizer = torch.optim.Adam(self._net.parameters(), lr=learning_rate)

      for trn_x, trn_y in self._train_dl:
        trn_x = trn_x.to(self._rtc.DEVICE)
        trn_y = trn_y.to(self._rtc.DEVICE)

        ########################################################################
        # Reinitialize the neurons states in all the layers.
        self._exu.reinitialize_all_lyrs_neurons_states(self._net)

        ########################################################################
        # Do forward pass and get predictions logits.
        all_hdl_all_ts_delta_v, all_rol_all_ts_logits = self._net(trn_x, "train")

        ########################################################################
        # Get the y_pred values for all the Readout Layers.
        all_rol_all_ts_pr_y = self._exu.get_all_rol_all_ts_pr_y_dict(
            all_rol_all_ts_logits)

        ########################################################################
        # Get the y_true values for all the Readout Layers.
        all_rol_all_ts_tr_y = self._tgu.get_all_rol_all_ts_tr_y_dict(
            all_rol_all_ts_pr_y, all_hdl_all_ts_delta_v, self._net, trn_y)

        ########################################################################
        # Calculate all Readout Layers loss now.
        loss = self._tgu.get_dalton_loss_for_all_rol_all_ts(
            all_rol_all_ts_pr_y, all_rol_all_ts_tr_y, self._exu.num_lyrs)

        # Update the network weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update all the Readout Layer weights.
        self._tgu.update_all_readout_layers_weights(self._net)

        # Get the predicted training classes.
        pred_clss.extend(
            self._exu.get_class_predictions(all_rol_all_ts_pr_y["dense_lyr_%s"
            % self._exu.num_lyrs["dense_lyrs"]].detach()).tolist())

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
        tst_x = tst_x.to(self._rtc.DEVICE)
        true_clss.extend(tst_y)

        # Reinitialize the neurons states in all the layers.
        self._exu.reinitialize_all_lyrs_neurons_states(self._net)
        # Do forward pass.
        _, all_rol_all_ts_logits = self._net(tst_x, "test") # Returns logits
        # Get the predicted test classes from the last Readout Layer.
        all_rol_all_ts_pr_y = self._exu.get_all_rol_all_ts_pr_y_dict(
            all_rol_all_ts_logits)
        pred_clss.extend(
            self._exu.get_class_predictions(all_rol_all_ts_pr_y["dense_lyr_%s"
            % self._exu.num_lyrs["dense_lyrs"]].detach()).tolist())

        dset_last_rol_p_t_cls[
            "dense_lyr_%s" % self._exu.num_lyrs["dense_lyrs"]].extend(
            all_rol_all_ts_pr_y[
            "dense_lyr_%s" % self._exu.num_lyrs["dense_lyrs"]
            ].detach().cpu().numpy())

      dset_last_rol_p_t_cls["true_cls"].extend(true_clss)

    test_acc = np.mean(np.array(pred_clss) == np.array(true_clss))
    if test_acc > self._best_test_acc:
      self._best_test_acc = test_acc
    log.INFO("Current Test Accuracy: {}, Best Test Accuracy: {}".format(
             test_acc, self._best_test_acc))

    return dset_last_rol_p_t_cls
