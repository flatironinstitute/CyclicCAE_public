#!/usr/bin/env python
#
# @author: Andrew Powers
# @brief: Classes and Functions needed for autoencoder model
#

########################
######  Packages  ######
########################

# base python packages
import os, sys
from collections import defaultdict

# ml imports
import torch
import torch.nn as nn

########################
######  Classes  #######
########################

class EncoderLayer(nn.Modules):
    def __init__(
        self,
        input_size: int = 128,
        output_size: int = 3,
        ):
        """ A Basic MLP layer for the autoencoder model

        PARAMS
        ------
            input_size: int
                The size of the input vector from the last layer

            output_size: int
                The size of the vector at the end of this layer
        """
        super(EncoderLayer, self).__init__()
        # Save inputs
        self.input_size = input_size
        self.output_size = output_size

        # method
        self.linear = nn.Linear(self.input_size, self.output_size)

        # activation funciton
        self.gelu = nn.GELU()

    def forward(x: torch.Tensor):
        """ forward pass of Encoderlayer

        PARAMS
        ------
            x: torch.Tensor
                The DoF input vector of size (B, N) where B is
                the batch size and N is the number of elements
        """
        return self.gelu(self.linear(x))
