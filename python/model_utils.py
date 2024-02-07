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

        # normalize the inputs coming in to the layer (could possibly not be needed)
        self.norm = nn.BatchNorm1d(self.input_size)

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
        return self.gelu(self.linear(self.norm(x)))

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
        super(Encoder, self).__init__()
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
        super(DecoderLayer, self).__init__()
        # Save inputs
        self.input_size = input_size
        self.output_size = output_size

        # method
        self.linear = nn.Linear(self.input_size, self.output_size)

        # normalize before input (this could be taken out possibly)
        self.norm = self.BatchNorm1d(self.input_size)

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
        return self.gelu(self.linear(self.norm(x)))

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
        super(Decoder, self).__init__()
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
    def __init__(
        self,
        latent_space_dim: int = 3,
        ):
        """ Modular Latent Space Block

        PARAMS
        ------
            latent_space_dim: int
                What dimension the latent space will be in
        """
        super(LatentSpace, self).__init__()
        # save input variables
        self.latent_space_dim = latent_space_dim

        # linear function to keep it in the latent space
        self.latent = nn.Linear(latent_space_dim, latent_space_dim)

        # activation function
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor):
        """ Forward of Latent Space Block

        PARAMS
        ------
            x: torch.Tensor
                Reduced input vector (B, latent_space_dim)
        """
        return self.gelu(self.latent(x))


class OutputBlock(nn.Modules):
    def __init__(
        self,
        protein_len: int,
        normalize: bool = True,
        in_size: int = 128,
        output_size: int = 512,
        ):
        """ Output Block with normalization and Conversion to Angle

        PARAMS
        ------
            protein_len: int
                The length of the protein/peptide, so we can know how many
                nodes to combine.

            normalize: bool
                Determines if normalization happens or not before combination.

            output_size: int
                Final output of the model size. Should be the same size as the
                input.
        """
        super(OutputBlock, self).__init__()
        # store values
        self.protein_len = protein_len
        self.normalize = normalize

        # Final linear layer
        self.output_linear = nn.Linear(in_size, output_size)

        # batch normalization
        self.norm = nn.BatchNorm1d(in_size)

    def forward(self, x: torch.Tensor):
        """ Forward for OutputBlock which normalizes and outputs

        PARAMS
        ------
            x: torch.Tensor
                Reconstructed x vector post-decomposition (B, D)
        """
        # grab out non-dof if necessary
        if x.size(1) > self.protein_len*2:
            non_dof_x = x[:, self.protein_len*2:]
        else:
            non_dof_x = None

        # normalize the sin/cos vector output
        sincos_norm = self.norm(x[:, :self.protein_len*2])

        # perform arctan2 
        recomputed_dofs = torch.atan2(sincos_norm[:, :self.protein_len], sincos_norm[:, self.protein_len:])

        # restack computed dofs with non_dof_x if necessary
        return torch.vstack((recomputed_dofs, non_dof_x), dim=-1) if non_dof_x != None else recomputed_dofs

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
        super(InputDecomposition, self).__init__()
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
        stacked_sincos_x = torch.vstack((sin_x, cos_x), dim=-1)

        # Now add back in the non dof information if necessary
        return stacked_sincos_x if non_dof_x == None else torch.vstack((stacked_sincos_x, non_dof_x), dim=-1)


