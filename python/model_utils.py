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

    def forward(self, x: torch.Tensor):
        """ forward pass of Encoderlayer

        PARAMS
        ------
            x: torch.Tensor
                The DoF input vector of size (B, N) where B is
                the batch size and N is the number of elements
        """
        return self.gelu(self.linear(x))

class Encoder(nn.Modules):
    def __init__(
        self,
        num_layers: int = 3,
        start_size: int = 512,
        end_size: int = 3,
        ):
        """ A General Encoder Block 

        PARAMS
        ------
            num_layers: int
                Number of EncoderLayers to include within the model

            start_size: int
                The beginning last dimension size of the input vector 
                going into the block

            end_size: int
                The ending last dimension size of the output vector going
                out of the block
        """
        # Save input parameters
        self.num_layers = num_layers
        self.start_size = start_size
        self.end_size = end_size

        # make layer sizes based upon num_layers and varying expanding and contracting
        self.layer_size = defaultdict(int)

        # Init this value
        current_end = self.start_size

        # loop through and pull out attributes for each layer
        for i in range(1, self.num_layers+1):
            # set the number that will be the next start value and this end
            new_start = current_end
            current_end = self.start_size//i
            # If this isn't the last layer
            if i != self.num_layers:
                self.layer_size[i] = (new_start, current_end)
            # Last layer use end_size
            else: 
                self.layer_size[i] = (new_start, self.end_size)

        # Make list of encoder layers that make up the encoder block
        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer(
                    input_size = self.layer_size[l],
                    output_size = self.layer_size[l],
                ) for l in range(1,self.num_layers+1)
            ]
        )

    def forward(self, x: torch.Tensor):
        """ Forward of Encoder Block

        PARAMS
        ------
            x: torch.Tensor
                The DoF input vector of size (B, N) where B is
                the batch size and N is the number of elements
        """
        for layer in self.encoder_layers:
            x = layer(x)
        return x


class DecoderLayer(nn.Modules):
    def __init__(
        self,
        input_size: int = 128,
        output_size: int = 3,
        ):
        """ A Basic MLP Decoder layer for the autoencoder model

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

    def forward(self, x: torch.Tensor):
        """ forward pass of Encoderlayer

        PARAMS
        ------
            x: torch.Tensor
                The DoF input vector of size (B, N) where B is
                the batch size and N is the number of elements
        """
        return self.gelu(self.linear(x))

class Decoder(nn.Modules):
    def __init__(
        self,
        num_layers: int = 3,
        start_size: int = 3,
        end_size: int = 512,
        ):
        """ A General Decoder Block 

        PARAMS
        ------
            num_layers: int
                Number of EncoderLayers to include within the model

            start_size: int
                The beginning last dimension size of the input vector 
                going into the block

            end_size: int
                The ending last dimension size of the output vector going
                out of the block
        """
        # Save input parameters
        self.num_layers = num_layers
        self.start_size = start_size
        self.end_size = end_size

        # make layer sizes based upon num_layers and varying expanding and contracting
        self.layer_size = defaultdict(int)

        # Init this value
        current_end = self.start_size

        # loop through and pull out attributes for each layer
        for i in range(self.num_layers, 0, -1):
            # set the number that will be the next start value and this end
            new_start = current_end
            current_end = self.end_size//i
            # If this isn't the last layer
            if i != 1:
                self.layer_size[i-(self.num_layers-1)] = (new_start, current_end)
            # Last layer use end_size
            else: 
                self.layer_size[i-(self.num_layers-1)] = (new_start, self.end_size)

        # Make list of encoder layers that make up the encoder block
        self.decoder_layers = nn.ModuleList(
            [
                DecoderLayer(
                    input_size = self.layer_size[l],
                    output_size = self.layer_size[l],
                ) for l in range(1,self.num_layers+1)
            ]
        )

    def forward(self, x: torch.Tensor):
        """ Forward of Encoder Block

        PARAMS
        ------
            x: torch.Tensor
                The DoF input vector of size (B, N) where B is
                the batch size and N is the number of elements
        """
        for layer in self.decoder_layers:
            x = layer(x)
        return x

class LatentSpace(nn.Modules):
    def __init__(self,):


class OutputBlock(nn.Modules):
    def __init__(self,):

class InputDecomposition(nn.Modules):
    def __init__(
        self,
        protein_len: int,
        ):
        """Decomposes DoF input portion into Sin and Cos

        PARAMS
        ------
            protein_len: int
                The length of the protein, so we know where to cutoff the
                DoF decomposition process.
        """
        # save passed variables
        self.protein_len = protein_len

    def forward(self, x: torch.Tensor):
        """ Forward of Input Decompose
        PARAMS
        ------
            x: torch.Tensor
                Input vector of at least N (residue number of DoFs).

        RETURNS
        -------
            torch.Tensor
                Decomposed Dofs in Sin/Cos and all extra information
        """
        # grab out non-dof if necessary
        if x.size(1) > self.protein_len:
            non_dof_x = x[:, self.protein_len:]
        else:
            non_dof_x = None

        # decompose dofs into both sin and cos
        sin_x = torch.sin(torch.deg2rad(x[:, :self.protein_len]))
        cos_x = torch.cos(torch.deg2rad(x[:, :self.protein_len]))

        # stack new decomposed vectors
        stacked_sincos_x = torch.vstack((sin_x, cos_x))

        # Now add back in the non dof information if necessary
        return stacked_sincos_x if non_dof_x == None else torch.vstack((stacked_sincos_x, non_dof_x))


