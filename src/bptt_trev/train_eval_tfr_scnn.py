import _init_paths

import hickle
import pickle
import numpy as np
import torch

from src.networks.bptt_nets.tfr_scnn import TFR_SCNN
from utils.base_utils import log
from utils.base_utils.data_prep_utils import DataPrepUtils
from utils.base_utils.exp_utils import ExpUtils
from utils.base_utils.tfr_utils import TFR_Utils

class TREV_TFR_SCNN(object):
  def __init__(self, rtc):
    self._rtc = rtc
    self._exu = ExpUtils(rtc)
    self._tfu = TFR_Utils(rtc)
    self._dpu = DataPrepUtils(rtc.DATASET)
    self._train_dl, self._test_dl = self._dpu.load_dataset(
        rtc.DCFG["batch_size"], rtc.USE_TRANSFORMS)
    self._net = TFR_SCNN(rtc).to(rtc.DEVICE)
    self._best_test_acc = 0
    log.INFO("Experimenting with TFR Spiking CNN...")

  def train_model(self):
    learning_rate = self._rtc.DCFG["learning_rate"]
    for epoch in range(1, self._rtc.DCFG["epochs"]+1):
      log.INFO("Starting training for epoch: %s" % epoch)
      pred_clss = []
      true_clss = []
      dset_all_rol_p_t_cls = self._tfu.get_dset_all_rol_pred_and_true_cls_dict()

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
        # Do forward pass and get predictions.
        all_hdl_all_ts_delta_v, all_rol_all_ts_logits = self._net(trn_x)

        ########################################################################
        # Get y_pred values from all the Readout Layers.
        all_rol_all_ts_pr_y = self._exu.get_all_rol_all_ts_pr_y_dict(
            all_rol_all_ts_logits)

        ########################################################################
        # Get y_true values applicable to all the Readout Layers. Note that in
        # TFR the final output layer "Global" true y values are used for each
        # intermediate/local Readout layer.
        true_y = self._exu.get_output_lyr_true_y(trn_y)

        ########################################################################
        # Get the loss from all the Readout Layers.
        loss = self._tfu.get_tfr_loss_for_all_rol_all_ts(
            all_rol_all_ts_pr_y, true_y)

        # Update the network weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Get the predicted training classes from the last Readout Layer.
        pred_clss.extend(
            self._exu.get_class_predictions(all_rol_all_ts_pr_y["dense_lyr_%s"
            % self._exu.num_lyrs["dense_lyrs"]].detach()).tolist())

        bw_all_rol_pred_clss = self._exu.get_all_rol_class_predictions(
            all_rol_all_ts_pr_y)

        for key in bw_all_rol_pred_clss.keys():
          dset_all_rol_p_t_cls[key].extend(bw_all_rol_pred_clss[key])

        true_clss.extend(trn_y.cpu().tolist())

      log.INFO("Epoch %s training done, Average Training Accuracy: %s" % (
               epoch, np.mean(np.array(pred_clss) == np.array(true_clss))))
      log.INFO(
          "Saving the pred cls obtained from all readout lyrs and true cls")
      dset_all_rol_p_t_cls["true_cls"].extend(true_clss)
      self._exu.save_hdf5_file(
          dset_all_rol_p_t_cls,
          "epoch_%s_training_all_rol_pred_and_true_clss.hdf5" % epoch,
          self._rtc.OTP_DIR)
      #self.eval_model()
      self._exu.save_hdf5_file(
          self.eval_model(), "epoch_%s_test_all_rol_pred_and_true_clss.hdf5"
          % epoch, self._rtc.OTP_DIR)

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
    dset_all_rol_p_t_cls = self._tfu.get_dset_all_rol_pred_and_true_cls_dict()

    log.INFO("Starting evaluation on test dataset ...")
    self._net.eval()
    with torch.no_grad():
      for tst_x, tst_y in self._test_dl:
        tst_x = tst_x.to(self._rtc.DEVICE)
        true_clss.extend(tst_y)

        # Reinitialize the neurons states in all the layers.
        self._exu.reinitialize_all_lyrs_neurons_states(self._net)
        # Do forward pass.
        _, all_rol_all_ts_logits = self._net(tst_x) # Returns logits
        # Get the y_pred values from all the Readout Layers.
        all_rol_all_ts_pr_y = self._exu.get_all_rol_all_ts_pr_y_dict(
            all_rol_all_ts_logits)
        # Get the predicted test classes from the last Readout Layer.
        pred_clss.extend(
            self._exu.get_class_predictions(all_rol_all_ts_pr_y["dense_lyr_%s"
            % self._exu.num_lyrs["dense_lyrs"]].detach()).tolist())

        # Get the predicted test classes from all the Readout Layers.
        bw_all_rol_pred_clss = self._exu.get_all_rol_class_predictions(
            all_rol_all_ts_pr_y)

        for key in all_rol_all_ts_pr_y.keys():
            dset_all_rol_p_t_cls[key].extend(
                all_rol_all_ts_pr_y[key].detach().cpu().numpy())

      dset_all_rol_p_t_cls["true_cls"].extend(true_clss)

    test_acc = np.mean(np.array(pred_clss) == np.array(true_clss))
    if test_acc > self._best_test_acc:
      self._best_test_acc = test_acc
    log.INFO("Current Test Accuracy: {}, Best Test Accuracy: {}".format(
             test_acc, self._best_test_acc))

    return dset_all_rol_p_t_cls
