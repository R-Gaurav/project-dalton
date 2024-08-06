#
# This file contains the utils to support the DALTON nets.
#

import _init_paths

from consts.exp_consts import EXC
from utils.base_utils import log
from utils.base_utils.exp_utils import ExpUtils

import torch

class DALTON_Utils(object):

  def __init__(self, rtc):
    self._rtc = rtc
    self._exu = ExpUtils(rtc)
    self.error_func_itm = self._exu.get_intermediate_lyr_loss_func()
    self.error_func_otp = self._exu.get_output_lyr_loss_func()

  def _get_presentation_ts_averaged_loss(self, all_ts_pr_y, all_ts_tr_y,
                                         err_func):
    if self._rtc.DEBUG:
      log.DEBUG("Shape of all_ts_pr_y: {}, and of all_ts_tr_y: {}".format(
            all_ts_pr_y.shape, all_ts_tr_y.shape))
      assert all_ts_pr_y.shape == all_ts_tr_y.shape

    all_ts_loss = 0
    n_ts = all_ts_pr_y.shape[1] # batch_size x n_ts x (n_dim or nc dx dy).
    for t in range(n_ts):
      pr_y = all_ts_pr_y[:, t]
      tr_y = all_ts_tr_y[:, t]
      if self._rtc.DEBUG:
        assert pr_y.requires_grad == True
        assert tr_y.requires_grad == False

      all_ts_loss += err_func(pr_y, tr_y)

    all_ts_loss/= n_ts
    return all_ts_loss

  def _get_jacobian_of_conv_wrt_spks(self, l_spk, l_rol_flts, l_stride):
    def _conv2d(l_spk):
      return torch.nn.functional.conv2d(l_spk, l_rol_flts, stride=l_stride)

    return torch.autograd.functional.jacobian(_conv2d, l_spk)

  def _get_jacobian_of_maxpool_wrt_single_inp(self, l_inp, pool_size):
    def _maxpool_2d(inp):
      return torch.nn.functional.max_pool2d(inp, kernel_size=pool_size)

    return torch.autograd.functional.jacobian(_maxpool_2d, l_inp)

  def _get_jacobian_of_avgpool_wrt_single_inp(self, l_inp, pool_size):
    def _avgpool_2d(inp):
      return torch.nn.functional.avg_pool2d(inp, kernel_size=pool_size)

    return torch.autograd.functional.jacobian(_avgpool_2d, l_inp)

  def _get_jacobian_of_maxpool_wrt_batch_inp(self, batch_inp, pool_size):
    b_sz, n_ch, dm_x, dm_y = batch_inp.shape
    batch_jcobn_mp = torch.zeros(
        b_sz, n_ch, dm_x//pool_size, dm_y//pool_size, n_ch, dm_x, dm_y,
        dtype=EXC.PT_DTYPE, device=self._rtc.DEVICE)
    for i in range(b_sz):
      batch_jcobn_mp[i] = self._get_jacobian_of_maxpool_wrt_single_inp(
          batch_inp[i], pool_size=pool_size)

    return batch_jcobn_mp

  def _get_tensordot_of_jcbn_and_diff_y(self, l_dcnv_dspk, l_diff_y):
    # dims = [0, 1, 2] are considered below for both `l_dcnv_dspk` and
    # `l_diff_y`, because first three dimensions are the ones with derivative
    # of conv w.r.t. spikes for each filter's dx and dy.
    if self._rtc.DEBUG:
      assert l_dcnv_dspk.shape[:3] == l_diff_y.shape
    return torch.tensordot(l_dcnv_dspk, l_diff_y, dims=([0, 1, 2], [0, 1, 2]))

  def _get_local_y_true_for_l_conv_rol_at_t_ts_wout_pool(self, pr_y_l, pr_y_lp1,
                                                         tr_y_lp1, jcobn_conv,
                                                         srgt_drtv_lp1):
    if self._rtc.ITM_LOSS_FUNC == "MSE":
      diff_y = pr_y_lp1 - tr_y_lp1
      batch_tdot = torch.einsum("abcdefg,abcd -> aefg", jcobn_conv, diff_y)
      return pr_y_l - batch_tdot * srgt_drtv_lp1

  def _get_local_y_true_for_l_conv_rol_at_t_ts_with_pool(self, pr_y_l,
                                                         pr_y_lp1, tr_y_lp1,
                                                         jcobn_conv_lp1,
                                                         jcobn_pool_lp1,
                                                         srgt_drtv_lp1):
    if self._rtc.ITM_LOSS_FUNC == "MSE":
      diff_y = pr_y_lp1 - tr_y_lp1
      batch_tdot_with_mp_jcobn = torch.einsum(
          "abcdefg,abcd -> aefg", jcobn_pool_lp1, diff_y)
      batch_tdot_with_conv_jcobn = torch.einsum(
          "abcdefg,abcd -> aefg", jcobn_conv_lp1, batch_tdot_with_mp_jcobn)

      return pr_y_l - batch_tdot_with_conv_jcobn * srgt_drtv_lp1

  def _get_local_y_true_for_l_dense_rol_at_t_ts(self, pr_y_l, pr_y_lp1,
                                                tr_y_lp1, wt_rol_lp1_T,
                                                srgt_drtv_lp1):
    if self._rtc.ITM_LOSS_FUNC == "MSE":
      return pr_y_l - (
          wt_rol_lp1_T.matmul((pr_y_lp1 - tr_y_lp1).T) * srgt_drtv_lp1.T).T

  def init_all_rol_all_ts_logits_dict(self, net):
    all_rol_all_ts_logits = {}
    # Readout Layer logits for the Spiking Conv Layers.
    for i in range(1, self._exu.num_lyrs["conv_lyrs"]):
      spk_conv_lyr = net._spk_conv_lyrs[i] # Logits stored correspond to next l.
      all_rol_all_ts_logits["conv_lyr_%s" % i] = torch.zeros(
          net._bsize, self._rtc.DCFG["presentation_ts"], spk_conv_lyr.otp_chnls,
          spk_conv_lyr.otp_dmx, spk_conv_lyr.otp_dmy, dtype=EXC.PT_DTYPE,
          device=self._rtc.DEVICE)

    # Create the entry for the last Readout Layer of the Conv Readout Layer list
    # which is actually a Dense Readout layer.
    all_rol_all_ts_logits["conv_lyr_%s" % self._exu.num_lyrs["conv_lyrs"]] = (
        torch.zeros(net._bsize, self._rtc.DCFG["presentation_ts"],
        net._rtc.SCNN_ARCH["dense_lyr_1"], dtype=EXC.PT_DTYPE,
        device= self._rtc.DEVICE)
        )

    # Readout Layer logits for the Spiking Dense Layers.
    for i in range(1, self._exu.num_lyrs["dense_lyrs"]):
      all_rol_all_ts_logits["dense_lyr_%s" % i] = torch.zeros(
          net._bsize, self._rtc.DCFG["presentation_ts"],
          net._rtc.SCNN_ARCH["dense_lyr_%s" % (i+1)], dtype=EXC.PT_DTYPE,
          device=self._rtc.DEVICE)

    # Create the entry for the last Readout Layer of the Dense Readout Layer
    # list which is actually the final output Readout Layer.
    all_rol_all_ts_logits["dense_lyr_%s" % self._exu.num_lyrs["dense_lyrs"]] = (
        torch.zeros(net._bsize, self._rtc.DCFG["presentation_ts"], net._otpdm,
        dtype=EXC.PT_DTYPE, device=self._rtc.DEVICE)
        )

    return all_rol_all_ts_logits

  def update_all_readout_layers_weights(self, net):
    num_lyrs_dict = self._exu.num_lyrs

    # Update the weights of the Conv Readout Layer, except for the last one, as
    # the last one is Dense Readout Layer and will be updated separtely below.
    for i in range(num_lyrs_dict["conv_lyrs"]-1):
      if self._rtc.DEBUG:
        assert (net._conv_rdt_lyrs[i]._conv.weight.shape ==
                net._spk_conv_lyrs[i+1]._conv.weight.shape)
      net._conv_rdt_lyrs[i]._conv.weight.data = (
          net._spk_conv_lyrs[i+1]._conv.weight.data)
    # Update the weight of the last Dense Readout Layer in the Conv Readout
    # Layers list.
    if self._rtc.DEBUG:
      assert (net._conv_rdt_lyrs[num_lyrs_dict["conv_lyrs"]-1]._fc.weight.shape
              == net._spk_dense_lyrs[0]._fc.weight.shape)
    net._conv_rdt_lyrs[num_lyrs_dict["conv_lyrs"]-1]._fc.weight.data = (
        net._spk_dense_lyrs[0]._fc.weight.data)

    # Update the weights of the Dense Readout Layers, except the last one, as it
    # is the final Output Layer.
    for i in range(num_lyrs_dict["dense_lyrs"]-1):
      if self._rtc.DEBUG:
        assert (net._dense_rdt_lyrs[i]._fc.weight.shape ==
                net._spk_dense_lyrs[i+1]._fc.weight.shape)
      net._dense_rdt_lyrs[i]._fc.weight.data = (
          net._spk_dense_lyrs[i+1]._fc.weight.data)

  def get_dalton_loss_for_all_rol_all_ts(self, all_rol_all_ts_pr_y,
                                           all_rol_all_ts_tr_y, num_lyrs_dict):
    all_rol_loss = 0
    error_func_itm = self._exu.get_intermediate_lyr_loss_func()
    error_func_otp = self._exu.get_output_lyr_loss_func()

    num_conv_rdt_lyrs = num_lyrs_dict["conv_lyrs"]
    num_dense_rdt_lyrs = num_lyrs_dict["dense_lyrs"]

    # Get the loss for Conv Readout Layers.
    for rol in range(1, num_conv_rdt_lyrs+1):
      # The Readout Layer of the penultimate Spiking Conv Layer is Conv with
      # weights of the last Spiking Conv Layer. Since the output of the last
      # Spiking Conv Layer is flattened, same has to be done for the Readout
      # layer of the penultimate Spiking Conv Layer. Note that the Readout Layer
      # of the last Spiking Conv Layer is already a Dense layer.
      if rol == (num_conv_rdt_lyrs - 1):
        conv_rol_all_ts_pr_y = all_rol_all_ts_pr_y["conv_lyr_%s" % rol].flatten(
            start_dim=2)
      else:
        conv_rol_all_ts_pr_y = all_rol_all_ts_pr_y["conv_lyr_%s" % rol]

      conv_rol_all_ts_tr_y = all_rol_all_ts_tr_y["conv_lyr_%s" % rol]

      all_rol_loss += self._get_presentation_ts_averaged_loss(
          conv_rol_all_ts_pr_y, conv_rol_all_ts_tr_y, error_func_itm)
      if self._rtc.DEBUG:
        log.DEBUG("Loss for Conv Readout Layer: %s obtained." % rol)

    # Get the loss for Dense Readout Layers.
    for rol in range(1, num_dense_rdt_lyrs+1):
      dense_rol_all_ts_pr_y = all_rol_all_ts_pr_y["dense_lyr_%s" % rol]
      dense_rol_all_ts_tr_y = all_rol_all_ts_tr_y["dense_lyr_%s" % rol]

      if rol != num_dense_rdt_lyrs: # Intermediate Layers.
        all_rol_loss += self._get_presentation_ts_averaged_loss(
            dense_rol_all_ts_pr_y, dense_rol_all_ts_tr_y, error_func_itm)
        if self._rtc.DEBUG:
          log.DEBUG("Loss for Dense Readout Layer: %s obtained." % rol)
      elif rol == num_dense_rdt_lyrs: # Final output layer.
        all_rol_loss += self._get_presentation_ts_averaged_loss(
            dense_rol_all_ts_pr_y, dense_rol_all_ts_tr_y, error_func_otp)
        if self._rtc.DEBUG:
          log.DEBUG("Loss for Dense Readout Layer: %s obtaiend." % rol)

    return all_rol_loss

  def _get_all_dense_lyrs_all_ts_tr_y(self, all_rol_all_ts_pr_y,
                                      all_hdl_all_ts_delta_v, net,
                                      global_y_true, all_rol_all_ts_tr_y):
    # The local y_true values for each Readout Layer has to be created backwards
    # starting from `global_y_true` values which are the actual ground truths.
    num_dense_rdt_lyrs = self._exu.num_lyrs["dense_lyrs"]
    for lyr in range(num_dense_rdt_lyrs, 0, -1):
      rol = "dense_lyr_%s" % lyr
      b_sz, n_ts, n_dm = all_rol_all_ts_pr_y[rol].shape
      all_rol_all_ts_tr_y[rol] = torch.zeros(
          b_sz, n_ts, n_dm, dtype=EXC.PT_DTYPE, device=self._rtc.DEVICE)

      if lyr == num_dense_rdt_lyrs: # Last Readout Layer => Global y_true.
        for t in range(n_ts):
          all_rol_all_ts_tr_y[rol][:, t] = self._exu.get_output_lyr_true_y(
              global_y_true)

      # Readout layers previous to the final Readout Layer.
      else: # Generate local y_true.
        detd_hdl_all_ts_srgt_drtv = self._exu.srgt_drtv_func(
            all_hdl_all_ts_delta_v["dense_lyr_%s" % (lyr+1)].detach())
        for t in range(n_ts):
          all_rol_all_ts_tr_y[rol][:, t] = (
              self._get_local_y_true_for_l_dense_rol_at_t_ts(
                  all_rol_all_ts_pr_y[rol][:, t].detach(),
                  all_rol_all_ts_pr_y["dense_lyr_%s" % (lyr+1)][:, t].detach(),
                  all_rol_all_ts_tr_y["dense_lyr_%s" % (lyr+1)][:, t],
                  net._dense_rdt_lyrs[lyr]._fc.weight.detach().T,
                  detd_hdl_all_ts_srgt_drtv[:, t]
                  )
              )

  def _get_all_conv_lyrs_all_ts_tr_y(self, all_rol_all_ts_pr_y,
                                     all_hdl_all_ts_delta_v, net,
                                     all_rol_all_ts_tr_y):
    num_conv_rdt_lyrs = self._exu.num_lyrs["conv_lyrs"]
    for lyr in range(num_conv_rdt_lyrs, 0, -1):
      rol = "conv_lyr_%s" % lyr

      if lyr == num_conv_rdt_lyrs:
        # The last Readout Layer of the Spiking Conv Layers is a Dense Readout
        # Layer, with Readout Layer weights = weights of the first Dense Layer.
        b_sz, n_ts, n_dm = all_rol_all_ts_pr_y[rol].shape
        all_rol_all_ts_tr_y[rol] = torch.zeros(
            b_sz, n_ts, n_dm, dtype=EXC.PT_DTYPE, device=self._rtc.DEVICE)
        detd_hdl_all_ts_srgt_drtv = self._exu.srgt_drtv_func(
            all_hdl_all_ts_delta_v["dense_lyr_1"].detach())

        for t in range(n_ts):
          all_rol_all_ts_tr_y[rol][:, t] = (
              self._get_local_y_true_for_l_dense_rol_at_t_ts(
                  all_rol_all_ts_pr_y[rol][:, t].detach(),
                  all_rol_all_ts_pr_y["dense_lyr_1"][:, t].detach(),
                  all_rol_all_ts_tr_y["dense_lyr_1"][:, t],
                  net._dense_rdt_lyrs[0]._fc.weight.detach().T,
                  detd_hdl_all_ts_srgt_drtv[:, t]
                  )
              )

      elif lyr == num_conv_rdt_lyrs - 1:
        # The penultimate Readout Layer of the Spiking Conv Layers has to work
        # with next Readout Layer's predicted and true y values which are all
        # flat, as well as the next Readout Layer's weights are dense. It also
        # has to mimic the flattening of the output from the last Spiking Conv.
        b_sz, n_ts, n_ch, dm_x, dm_y = all_rol_all_ts_pr_y[rol].shape
        all_rol_all_ts_tr_y[rol] = torch.zeros(
            b_sz, n_ts, n_ch * dm_x * dm_y, dtype=EXC.PT_DTYPE,
            device=self._rtc.DEVICE)
        detd_hdl_all_ts_srgt_drtv = self._exu.srgt_drtv_func(
            all_hdl_all_ts_delta_v["conv_lyr_%s" % (lyr+1)].detach())

        for t in range(n_ts):
          all_rol_all_ts_tr_y[rol][:, t] = (
              self._get_local_y_true_for_l_dense_rol_at_t_ts(
                  all_rol_all_ts_pr_y[rol][:, t].detach().flatten(start_dim=1),
                  all_rol_all_ts_pr_y["conv_lyr_%s" % (lyr+1)][:, t].detach(),
                  all_rol_all_ts_tr_y["conv_lyr_%s" % (lyr+1)][:, t],
                  net._conv_rdt_lyrs[lyr]._fc.weight.detach().T,
                  detd_hdl_all_ts_srgt_drtv[:, t].flatten(start_dim=1)
                  )
              )

      else:
        if "pool_size" in self._rtc.SCNN_ARCH["conv_lyr_%s" % (lyr+1)]:
          is_pool = True
        else:
          is_pool = False

        # Rest of the readout layers of the Conv Readout Layers are Conv.
        b_sz, n_ts, n_ch, dm_x, dm_y = all_rol_all_ts_pr_y[rol].shape
        all_rol_all_ts_tr_y[rol] = torch.zeros(
            b_sz, n_ts, n_ch, dm_x, dm_y, dtype=EXC.PT_DTYPE,
            device= self._rtc.DEVICE)

        py_b_sz, py_n_ch, py_dm_x, py_dm_y = all_rol_all_ts_pr_y[
            "conv_lyr_%s" % (lyr+1)][:, t].shape

        detd_hdl_all_ts_delta_v = all_hdl_all_ts_delta_v[
            "conv_lyr_%s" % (lyr+1)].detach()
        detd_hdl_all_ts_srgt_drtv = self._exu.srgt_drtv_func(
            detd_hdl_all_ts_delta_v) # Here `delta_v` implies `v - v_thr`.
        detd_hdl_all_ts_spikes = self._exu.spike_func(detd_hdl_all_ts_delta_v)

        # Jacobian of Conv(Input, Filters) with respect to Input remains the same
        # for all the Inputs at different time-steps -- Empirically Checked here.
        # This also makes sense because the Jacobian is composed of just filters
        # in each time-step t. Therefore, I choose t=0 arbitrarily here. Also,
        # calculating Jacobian is a compute-costly operation, hence following
        # calculated one is used for all the time-steps.
        # Note that Jacobian of Conv w.r.t. spikes for one input sample in batch
        # is same as that of the other samples. To save time, calculate Jacobian
        # of one sample and expand it on the batch dimension. Also, calculation
        # of Jacobian for the entire batch is GPU memory intensive.
        jcobn_conv = self._get_jacobian_of_conv_wrt_spks(
            detd_hdl_all_ts_spikes[0, 0], # Jacobian for sample 0 at t=0.
            net._conv_rdt_lyrs[lyr]._conv.weight.detach(), # Indexing starts 0.
            net._scnn_arch["conv_lyr_%s" % (lyr+1)]["stride"]
            )
        jcobn_conv = jcobn_conv.clone().expand(
            self._rtc.DCFG["batch_size"], -1, -1, -1, -1, -1, -1)

        # Jacobian of AvgPool(input) with respect to `input` remains the same
        # for all the inputs and for all the time-steps. Therefore, I choose t=0
        # aribitrarily and compute Jacobian of AvgPool for first input sample.
        # After this, I expand the computed Jacobian for one input sample across
        # the batch dimension.
        jcobn_pool = None
        if is_pool and self._rtc.DCFG["pool_type"] == "AvgPool":
          jcobn_pool = self._get_jacobian_of_avgpool_wrt_single_inp(
              torch.nn.functional.conv2d(
                  detd_hdl_all_ts_spikes[0, 0], # Jacobian for sample 0 at t=0.
                  net._conv_rdt_lyrs[lyr]._conv.weight.detach(),
                  stride=net._scnn_arch["conv_lyr_%s" % (lyr+1)]["stride"]),
              self._rtc.SCNN_ARCH["conv_lyr_%s" % (lyr+1)]["pool_size"])
          jcobn_pool = jcobn_pool.clone().expand(
              self._rtc.DCFG["batch_size"], -1, -1, -1, -1, -1, -1)

        for t in range(n_ts):
          # If the next to next Conv SpkLayer has a MaxPool layer => the next
          # Conv ReadoutLayer will also have a MaxPooling layer (in
          # correspondance with l_th readout layer wts = (l+1)_th spiking layer
          # wts. Therefore, obtain the Jacobian of the MaxPooling op w.r.t. all
          # the inputs in the batch. Note that this jacobian varies for each
          # input sample in the batch, as well as for all the time-steps. Hence,
          # it is calculated for each sample and each time-step.
          if is_pool and self._rtc.DCFG["pool_type"] == "MaxPool":
            mp_batch_inp = torch.nn.functional.conv2d(
                detd_hdl_all_ts_spikes[:, t],
                net._conv_rdt_lyrs[lyr]._conv.weight.detach(),
                stride=net._scnn_arch["conv_lyr_%s" % (lyr+1)]["stride"])
            jcobn_pool = self._get_jacobian_of_maxpool_wrt_batch_inp(
                mp_batch_inp,
                self._rtc.SCNN_ARCH["conv_lyr_%s" % (lyr+1)]["pool_size"])

          if jcobn_pool is None:
            all_rol_all_ts_tr_y[rol][:, t] = (
                self._get_local_y_true_for_l_conv_rol_at_t_ts_wout_pool(
                    all_rol_all_ts_pr_y[rol][:, t].detach(),
                    all_rol_all_ts_pr_y["conv_lyr_%s" % (lyr+1)][:, t].detach(),
                    # Local tr_y obtained for `conv_lyr_(n-1)` in previous i
                    # `elif` block was already flattened due to dense. So,
                    # reshape it here for the Conv operation in this `else`
                    # block.
                    all_rol_all_ts_tr_y["conv_lyr_%s" % (lyr+1)][:, t].reshape(
                        py_b_sz, py_n_ch, py_dm_x, py_dm_y),
                    jcobn_conv,
                    detd_hdl_all_ts_srgt_drtv[:, t],
                    )
                )
          else:
            all_rol_all_ts_tr_y[rol][:, t] = (
                self._get_local_y_true_for_l_conv_rol_at_t_ts_with_pool(
                    all_rol_all_ts_pr_y[rol][:, t].detach(),
                    all_rol_all_ts_pr_y["conv_lyr_%s" % (lyr+1)][:, t].detach(),
                    # Local tr_y obtained for `conv_lyr_(n-1)` in previous i
                    # `elif` block was already flattened due to dense. So,
                    # reshape it here for the Conv operation in this `else`
                    # block.
                    all_rol_all_ts_tr_y["conv_lyr_%s" % (lyr+1)][:, t].reshape(
                        py_b_sz, py_n_ch, py_dm_x, py_dm_y),
                    jcobn_conv,
                    jcobn_pool,
                    detd_hdl_all_ts_srgt_drtv[:, t],
                    )
                )


  def get_all_rol_all_ts_tr_y_dict(self, all_rol_all_ts_pr_y,
                                   all_hdl_all_ts_delta_v, net, trn_y):
    all_rol_all_ts_tr_y = {}

    # First get the local y_true values for dense layers backwards.
    self._get_all_dense_lyrs_all_ts_tr_y(all_rol_all_ts_pr_y,
                                         all_hdl_all_ts_delta_v, net, trn_y,
                                         all_rol_all_ts_tr_y)
    # Now get the local y_true values for the conv layers backwards.
    self._get_all_conv_lyrs_all_ts_tr_y(all_rol_all_ts_pr_y,
                                        all_hdl_all_ts_delta_v,
                                        net, all_rol_all_ts_tr_y)

    return all_rol_all_ts_tr_y

  def get_dset_last_rol_pred_and_true_cls_dict(self):
    last_rol_pred_and_true_cls = {
        "dense_lyr_%s" % self._exu.num_lyrs["dense_lyrs"]: [],
        "true_cls": []
        }
    return last_rol_pred_and_true_cls

  ##############################################################################
  ################### U T I L S   F O R   R T R L   E X P S ####################
  ##############################################################################

  def _get_loss_for_1_ts(self, one_ts_pr_y, one_ts_tr_y, err_func):
    if self._rtc.DEBUG:
      log.DEBUG("Shape of one_ts_pr_y: {}, and of one_ts_tr_y: {}".format(
                one_ts_pr_y.shape, one_ts_tr_y.shape))
      # Shape should be: batch_size x (n_dim or n_ch dm_x dm_y).
      assert one_ts_pr_y.shape == one_ts_tr_y.shape

    return err_func(one_ts_pr_y, one_ts_tr_y)

  def init_all_rol_1_ts_logits_dict(self, net):
    all_rol_1_ts_logits = {}
    # Readout Layer logits for the Spiking Conv Layers.
    for i in range(1, self._exu.num_lyrs["conv_lyrs"]):
      spk_conv_lyr = net._spk_conv_lyrs[i] # Logits stored correspond to next l.
      all_rol_1_ts_logits["conv_lyr_%s" % i] = torch.zeros(
          net._bsize, spk_conv_lyr.otp_chnls, spk_conv_lyr.otp_dmx,
          spk_conv_lyr.otp_dmy, dtype=EXC.PT_DTYPE, device=self._rtc.DEVICE)

    # Create the entry for the last Readout Layer of the Conv Readout Layer list
    # which is actually a Dense Readout layer.
    all_rol_1_ts_logits["conv_lyr_%s" % self._exu.num_lyrs["conv_lyrs"]] = (
        torch.zeros(net._bsize, net._rtc.SCNN_ARCH["dense_lyr_1"],
                    dtype=EXC.PT_DTYPE, device= self._rtc.DEVICE)
        )

    # Readout Layer logits for the Spiking Dense Layers.
    for i in range(1, self._exu.num_lyrs["dense_lyrs"]):
      all_rol_1_ts_logits["dense_lyr_%s" % i] = torch.zeros(
          net._bsize, net._rtc.SCNN_ARCH["dense_lyr_%s" % (i+1)],
          dtype=EXC.PT_DTYPE, device=self._rtc.DEVICE)

    # Create the entry for the last Readout Layer of the Dense Readout Layer
    # list which is actually the final output Readout Layer.
    all_rol_1_ts_logits["dense_lyr_%s" % self._exu.num_lyrs["dense_lyrs"]] = (
        torch.zeros(net._bsize, net._otpdm, dtype=EXC.PT_DTYPE,
                    device=self._rtc.DEVICE)
        )

    return all_rol_1_ts_logits

  def get_dalton_loss_for_all_rol_1_ts(self, all_rol_1_ts_pr_y,
                                         all_rol_1_ts_tr_y):
    all_rol_loss = 0
    # Function arugment dict values shape: (bsize, n_ch, dm_x, dm_y).
    # Get the loss for Conv Readout Layers.
    for rol in range(1, self._exu.num_lyrs["conv_lyrs"]+1):
      # The Readout Layer of the penultimate Spiking Conv Layer is Conv with
      # weights of the last Spiking Conv Layer. Since the output of the last
      # Spiking Conv Layer is flattened, same has to be done for the Readout
      # layer of the penultimate Spiking Conv Layer. Note that the Readout Layer
      # of the last Spiking Conv Layer is already a Dense layer.
      if rol == (self._exu.num_lyrs["conv_lyrs"]-1):
        conv_rol_1_ts_pr_y = all_rol_1_ts_pr_y["conv_lyr_%s" % rol].flatten(
            start_dim=1)
      else:
        conv_rol_1_ts_pr_y = all_rol_1_ts_pr_y["conv_lyr_%s" % rol]

      conv_rol_1_ts_tr_y = all_rol_1_ts_tr_y["conv_lyr_%s" % rol]

      #all_rol_loss += self._get_presentation_ts_averaged_loss(
      #    conv_rol_all_ts_pr_y, conv_rol_all_ts_tr_y, error_func_itm)
      all_rol_loss += self._get_loss_for_1_ts(
          conv_rol_1_ts_pr_y, conv_rol_1_ts_tr_y, self.error_func_itm)
      if self._rtc.DEBUG:
        log.DEBUG("Loss for Conv Readout Layer: %s obtained." % rol)

    # Get the loss for Dense Readout Layers.
    for rol in range(1, self._exu.num_lyrs["dense_lyrs"]+1):
      dense_rol_1_ts_pr_y = all_rol_1_ts_pr_y["dense_lyr_%s" % rol]
      dense_rol_1_ts_tr_y = all_rol_1_ts_tr_y["dense_lyr_%s" % rol]

      if rol != self._exu.num_lyrs["dense_lyrs"]: # Intermediate Layers.
        #all_rol_loss += self._get_presentation_ts_averaged_loss(
        #    dense_rol_all_ts_pr_y, dense_rol_all_ts_tr_y, error_func_itm)
        all_rol_loss += self._get_loss_for_1_ts(
            dense_rol_1_ts_pr_y, dense_rol_1_ts_tr_y, self.error_func_itm)
        if self._rtc.DEBUG:
          log.DEBUG("Loss for Dense Readout Layer: %s obtained." % rol)
      elif rol == self._exu.num_lyrs["dense_lyrs"]: # Final output layer.
        #all_rol_loss += self._get_presentation_ts_averaged_loss(
        #    dense_rol_all_ts_pr_y, dense_rol_all_ts_tr_y, error_func_otp)
        all_rol_loss += self._get_loss_for_1_ts(
            dense_rol_1_ts_pr_y, dense_rol_1_ts_tr_y, self.error_func_otp)
        if self._rtc.DEBUG:
          log.DEBUG("Loss for Dense Readout Layer: %s obtaiend." % rol)

    return all_rol_loss

  def _get_all_dense_lyrs_1_ts_tr_y(self, all_rol_1_ts_pr_y,
                                    all_hdl_1_ts_delta_v, net,
                                    global_y_true, all_rol_1_ts_tr_y):
    # The local y_true values for each Readout Layer has to be created backwards
    # starting from `global_y_true` values which are the actual ground truths.
    num_dense_rdt_lyrs = self._exu.num_lyrs["dense_lyrs"]
    for lyr in range(num_dense_rdt_lyrs, 0, -1):
      rol = "dense_lyr_%s" % lyr
      if lyr == num_dense_rdt_lyrs: # Last Readout Layer => Global y_true.
        all_rol_1_ts_tr_y[rol] = self._exu.get_output_lyr_true_y(
              global_y_true)

      # Readout layers previous to the final Readout Layer.
      else: # Generate local y_true.
        detd_hdl_1_ts_srgt_drtv = self._exu.srgt_drtv_func(
            all_hdl_1_ts_delta_v["dense_lyr_%s" % (lyr+1)].detach())
        all_rol_1_ts_tr_y[rol] = (
            self._get_local_y_true_for_l_dense_rol_at_t_ts(
                all_rol_1_ts_pr_y[rol].detach(),
                all_rol_1_ts_pr_y["dense_lyr_%s" % (lyr+1)].detach(),
                all_rol_1_ts_tr_y["dense_lyr_%s" % (lyr+1)],
                net._dense_rdt_lyrs[lyr]._fc.weight.detach().T,
                detd_hdl_1_ts_srgt_drtv
                )
            )

  def _get_all_conv_lyrs_1_ts_tr_y(self, all_rol_1_ts_pr_y,
                                   all_hdl_1_ts_delta_v, net,
                                   all_rol_1_ts_tr_y):
    num_conv_rdt_lyrs = self._exu.num_lyrs["conv_lyrs"]
    for lyr in range(num_conv_rdt_lyrs, 0, -1):
      rol = "conv_lyr_%s" % lyr

      if lyr == num_conv_rdt_lyrs:
        # The last Readout Layer of the Spiking Conv Layers is a Dense Readout
        # Layer, with Readout Layer weights = weights of the first Dense Layer.
        detd_hdl_1_ts_srgt_drtv = self._exu.srgt_drtv_func(
            all_hdl_1_ts_delta_v["dense_lyr_1"].detach())

        all_rol_1_ts_tr_y[rol] = self._get_local_y_true_for_l_dense_rol_at_t_ts(
              all_rol_1_ts_pr_y[rol].detach(),
              all_rol_1_ts_pr_y["dense_lyr_1"].detach(),
              all_rol_1_ts_tr_y["dense_lyr_1"],
              net._dense_rdt_lyrs[0]._fc.weight.detach().T,
              detd_hdl_1_ts_srgt_drtv
              )

      elif lyr == num_conv_rdt_lyrs - 1:
        # The penultimate Readout Layer of the Spiking Conv Layers has to work
        # with next Readout Layer's predicted and true y values which are all
        # flat, as well as the next Readout Layer's weights are dense. It also
        # has to mimic the flattening of the output from the last Spiking Conv.
        detd_hdl_1_ts_srgt_drtv = self._exu.srgt_drtv_func(
            all_hdl_1_ts_delta_v["conv_lyr_%s" % (lyr+1)].detach())

        all_rol_1_ts_tr_y[rol] = self._get_local_y_true_for_l_dense_rol_at_t_ts(
                all_rol_1_ts_pr_y[rol].detach().flatten(start_dim=1),
                all_rol_1_ts_pr_y["conv_lyr_%s" % (lyr+1)].detach(),
                all_rol_1_ts_tr_y["conv_lyr_%s" % (lyr+1)],
                net._conv_rdt_lyrs[lyr]._fc.weight.detach().T,
                detd_hdl_1_ts_srgt_drtv.flatten(start_dim=1)
                )

      else:
        if "pool_size" in self._rtc.SCNN_ARCH["conv_lyr_%s" % (lyr+1)]:
          is_pool = True
        else:
          is_pool = False

        # Rest of the readout layers of the Conv Readout Layers are Conv.
        py_b_sz, py_n_ch, py_dm_x, py_dm_y = all_rol_1_ts_pr_y[
            "conv_lyr_%s" % (lyr+1)].shape

        detd_hdl_1_ts_srgt_drtv = self._exu.srgt_drtv_func(
            all_hdl_1_ts_delta_v["conv_lyr_%s" % (lyr+1)].detach())
        detd_hdl_1_ts_spikes = self._exu.spike_func(
            all_hdl_1_ts_delta_v["conv_lyr_%s" % (lyr+1)].detach())

        # For one time-step Jacobian of Conv(Input, Filters) with respect to the
        # Input remains the same for all the samples in a batch, because the
        # same Filters are used. Also, calculating Jacobian is a compute-costly
        # operation. To save time, calculate Jacobian of one sample in the batch
        # and expand it on the batch dimension for other samples. Also,
        # calculation of Jacobian for the entire batch is GPU memory intensive.
        jcobn_conv = self._get_jacobian_of_conv_wrt_spks(
            detd_hdl_1_ts_spikes[0], # Calculating Jacobian for one sample.
            net._conv_rdt_lyrs[lyr]._conv.weight.detach(),
            net._scnn_arch["conv_lyr_%s" % (lyr+1)]["stride"]
            )
        jcobn_conv = jcobn_conv.clone().expand(
            self._rtc.DCFG["batch_size"], -1, -1, -1, -1, -1, -1)

        jcobn_pool = None
        if is_pool and self._rtc.DCFG["pool_type"] == "AvgPool":
          jcobn_pool = self._get_jacobian_of_avgpool_wrt_single_inp(
              torch.nn.functional.conv2d(
                  detd_hdl_1_ts_spikes[0], # Jacobian for sample 0.
                  net._conv_rdt_lyrs[lyr]._conv.weight.detach(),
                  stride=net._scnn_arch["conv_lyr_%s" % (lyr+1)]["stride"]),
              self._rtc.SCNN_ARCH["conv_lyr_%s" % (lyr+1)]["pool_size"])
          jcobn_pool = jcobn_pool.clone().expand(
              self._rtc.DCFG["batch_size"], -1, -1, -1, -1, -1, -1)

        if jcobn_pool is None:
          all_rol_1_ts_tr_y[rol] = (
              self._get_local_y_true_for_l_conv_rol_at_t_ts_wout_pool(
                  all_rol_1_ts_pr_y[rol].detach(),
                  all_rol_1_ts_pr_y["conv_lyr_%s" % (lyr+1)].detach(),
                  # Local tr_y obtained for `conv_lyr_(n-1)` in previous `elif`
                  # block was already flattened due to dense. So, reshape it
                  # here for the Conv operation in this `else` block.
                  all_rol_1_ts_tr_y["conv_lyr_%s" % (lyr+1)].reshape(
                      py_b_sz, py_n_ch, py_dm_x, py_dm_y),
                  jcobn_conv,
                  detd_hdl_1_ts_srgt_drtv
                  )
              )
        else:
          all_rol_1_ts_tr_y[rol] = (
              self._get_local_y_true_for_l_conv_rol_at_t_ts_with_pool(
                  all_rol_1_ts_pr_y[rol].detach(),
                  all_rol_1_ts_pr_y["conv_lyr_%s" % (lyr+1)].detach(),
                  # Local tr_y obtained for `conv_lyr_(n-1)` in previous `elif`
                  # block was already flattened due to dense. So, reshape it
                  # here for the Conv operation in this `else` block.
                  all_rol_1_ts_tr_y["conv_lyr_%s" % (lyr+1)].reshape(
                      py_b_sz, py_n_ch, py_dm_x, py_dm_y),
                  jcobn_conv,
                  jcobn_pool,
                  detd_hdl_1_ts_srgt_drtv
                  )
              )

  def get_all_rol_1_ts_tr_y_dict(self, all_rol_1_ts_pr_y,
                                 all_hdl_1_ts_delta_v, net, trn_y):
    all_rol_1_ts_tr_y = {}

    # First get the local y_true values for dense layers backwards.
    self._get_all_dense_lyrs_1_ts_tr_y(all_rol_1_ts_pr_y,
                                       all_hdl_1_ts_delta_v, net, trn_y,
                                       all_rol_1_ts_tr_y)
    # Now get the local y_true values for the conv layers backwards.
    self._get_all_conv_lyrs_1_ts_tr_y(all_rol_1_ts_pr_y,
                                      all_hdl_1_ts_delta_v, net,
                                      all_rol_1_ts_tr_y)

    return all_rol_1_ts_tr_y
