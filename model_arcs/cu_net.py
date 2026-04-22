from dataclasses import dataclass
import enum

import torch
import torch.nn as nn

from model_arcs.base_layers import SparseDownSampleClosest


class CU_Net(nn.Module):

  """
  Cu-Net architecture for LiDAR depth completion. 
  You can find a detailed description of the CU-Net architecture in "model_arcs/CU-Net_arcitecture.md".
  """

  def __init__(self):
    super(CU_Net, self).__init__()

    self.initial_conv 
