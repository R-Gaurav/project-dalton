#
# This file contains the utils code for Spiking Convolutional Layer.
#

import numpy as np

class SPCUtils(object):
  def get_output_shape(self, dims, krnl_size, stride, padding, dilation):
    dim_x, ksz_x, str_x, pdg_x, dln_x = (
        dims[0], krnl_size[0], stride[0], padding[0], dilation[0])
    dim_y, ksz_y, str_y, pdg_y, dln_y = (
        dims[1], krnl_size[1], stride[1], padding[1], dilation[1])

    out_dmx = int(np.floor((dim_x + 2*pdg_x - dln_x * (ksz_x-1) - 1)/str_x + 1))
    out_dmy = int(np.floor((dim_y + 2*pdg_y - dln_y * (ksz_y-1) - 1)/str_y + 1))

    return out_dmx, out_dmy
