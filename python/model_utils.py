#!/usr/bin/env python
#
# @author: Andrew Powers
# @brief: Classes and Functions needed for autoencoder model
#

##################################################################################################################
#                                               Packages 
##################################################################################################################

# base python packages
import os, sys
from collections import defaultdict

# ml imports
import torch
import torch.nn as nn

##################################################################################################################
#                                                Classes
##################################################################################################################

class ConformationalMLPEncoderLayer(nn.Module):
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
        super().__init__()
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

class ConformationalMLPEncoderBlock(nn.Module):
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
        super().__init__()
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
            current_end = self.start_size//(i+1)
            # If this isn't the last layer
            if i != self.num_layers:
                self.layer_size[i] = (new_start, current_end)
            # Last layer use end_size
            else: 
                self.layer_size[i] = (new_start, self.end_size)

        # Make list of encoder layers that make up the encoder block
        self.encoder_layers = nn.ModuleList(
            [
                ConformationalMLPEncoderLayer(
                    input_size = self.layer_size[l][0],
                    output_size = self.layer_size[l][1],
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


class ConformationalMLPDecoderLayer(nn.Module):
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
        super().__init__()
        # Save inputs
        self.input_size = input_size
        self.output_size = output_size

        # method
        self.linear = nn.Linear(self.input_size, self.output_size)

        # normalize before input (this could be taken out possibly)
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

class ConformationalMLPDecoderBlock(nn.Module):
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
        super().__init__()
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
                self.layer_size[i] = (new_start, current_end)
            # Last layer use end_size
            else: 
                self.layer_size[i] = (new_start, self.end_size)

        # Make list of encoder layers that make up the encoder block
        self.decoder_layers = nn.ModuleList(
            [
                ConformationalMLPDecoderLayer(
                    input_size = self.layer_size[l][0],
                    output_size = self.layer_size[l][1],
                ) for l in range(self.num_layers, 0, -1)
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

class ConformationalLatentSpace(nn.Module):
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
        super().__init__()
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


class ConformationalOutputBlock(nn.Module):
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

            in_size: int
                Vector output size from decoder block

            output_size: int
                Final output of the model size. Should be the same size as the
                input.
        """
        super().__init__()
        # store values
        self.protein_len = protein_len
        self.normalize = normalize

        # Final linear layer
        self.output_linear = nn.Linear(in_size, output_size)

        # batch normalization
        self.norm = nn.BatchNorm1d(self.protein_len*2)

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
        return torch.cat((recomputed_dofs, non_dof_x), dim=-1) if non_dof_x != None else recomputed_dofs

class ConformationalInputDecomposition(nn.Module):
    def __init__(
        self,
        protein_len: int,
        torsion_n: int = 0,
        bond_angle_n: int = 0,
        bond_length_n: int = 0,
        main_chain_n: int = 0,
        side_chain_n: int = 0,
        ):
        """Decomposes DoF input portion into Sin and Cos

        PARAMS
        ------
            protein_len: int
                The length of the protein, so we know where to cutoff the
                DoF decomposition process.

            torsion_n: int
                Strictly specify the number of torsion angles present within the dataset

            bond_angle_n: int
                Strictly specify the number of bond angles present within the dataset

            bond_length_n: int
                Strictly specify the number of bond lengths present within the dataset

            main_chain_n: int
                Number of main chain atoms for each torsion, bond_angle, and bond_length

            side_chain_n: int
                Number of side chain atoms for each torsion, bond_angle, and bond_length
        """
        super().__init__()
        # save passed variables
        self.protein_len   = protein_len
        self.torsion_n     = torsion_n
        self.bond_angle_n  = bond_angle_n
        self.bond_length_n = bond_length_n
        self.main_chain_n  = main_chain_n
        self.side_chain_n  = side_chain_n
        

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
        # running total for the amount to index down
        index_in = 0
        # grab out torsion
        if self.torsion_n != 0:
            # check to make sure that main chain or side chain isn't
            # specified. Currently if main is set then side chain will be set too.
            if self.main_chain_n == 0:
                # pul out the specific numbers from the passed params
                torsion = x[:, self.torsion_n]
            else:
                # Calculate the number of torsion angles based upon the addition of the main and side chains
                torsion_plus_n = self.torsion_n*self.main_chain_n + self.torsion_n*self.side_chain_n 
                # pull out the torsion angles
                torsion = x[:, :torsion_plus_n]
        else:
            torsion = 0

        # update index_in
        index_in += torsion.size(-1) if torsion != 0 else torsion

        # grab out bond_angle
        if self.bond_angle_n != 0:
            # check to make sure that main chain or side chain isn't
            # specified. Currently if main is set then side chain will be set too.
            if self.main_chain_n == 0:
                # pull out the specific numbers from the passed params
                bond_angle = x[:, index_in : index_in + self.bond_angle_n]
            else:
                # now calculate the number of bond angles based upon the addition of main and side chains
                bond_angle_plus_n = self.bond_angle_n*self.main_chain_n + self.bond_angle_n*self.side_chain_n
                # pull out these angles
                bond_angle = x[:, index_in : index_in + bond_angle_plus_n]
        else:
            bond_angle = 0

        # update index_in
        index_in += bond_angle.size(-1) if bond_angle != 0 else bond_angle

        # grab out bond length
        if self.bond_length_n != 0:
            if self.main_chain_n == 0:
                # pull out number of passed params
                bond_length = x[: index_in : index_in + self.bond_length_n]
            else:
                bond_length_plus_n = self.bond_length_n*self.main_chain_n + self.bond_length_n*self.side_chain_n
                # pull out the bond lengths specifically
                bond_length = x[: index_in : index_in + bond_length_plus_n]
        else:
            bond_length = 0

        # update index_in
        index_in += bond_length.size(-1) if bond_length != 0 else bond_length

        # now pull out extra information if available
        if x.size(-1) > index_in:
            extra_scalar_info = x[:, index_in:]
        else:
            extra_scalar_info = 0

        # decompose pulled out angles into both sin and cos
        if torsion != 0:
            torsion_decompose = self.decompose_angle_(torsion)
        if bond_angle != 0:
            bond_angle_decompose = self.decompose_angle_(bond_angle)

        # Some function that does something to bond_length. Not decided yet
        # bond_length_blah = self.bond_length_adjust(bond_length)

        # combine all information
        out = torch.cat((torsion_decompose, bond_angle_decompose), dim=-1)
        out = torch.cat((out, bond_length), dim=-1)
        if extra_scalar_info != 0:
            out = torch.cat((out, extra_scalar_info), dim=-1)

        return out

    def decompose_angle_(self, x: torch.Tensor, idx: torch.Tensor = None):
        """Break the angle up into sin & cos components"""
        # if idx is provided pull out only angle idx components
        # Assumption is made that the input is 2d (B, N)
        if idx != None:
            # Grab out specific angle information
            x = x[:, idx]

        # decompose angle into both sin and cos
        sin_x = torch.sin(torch.deg2rad(x))
        cos_x = torch.cos(torch.deg2rad(x))

        # stack new decomposed vectors
        stacked_sincos_x = torch.cat((sin_x, cos_x), dim=-1)

        return stacked_sincos_x




##################################################################################################################
#                                               Architecture
##################################################################################################################

class ConformationalMLPAutoEncoder(nn.Module):
    def __init__(
        self,
        protein_len: int,
        output_size: int,
        input_size: int,
        normalize: bool,
        decoder_layers: int,
        encoder_layers: int,
        latent_space_dim: int,
        non_dof_info: bool = False,
        torsion_n: int = 0,
        bond_angle_n: int = 0,
        bond_length_n: int = 0,
        main_chain_n: int = 0,
        side_chain_n: int = 0,
        extra_info_n: int = 0,
        ):
        """MLP Base Auto-Encoder Model (Base Model Architecture)

        PARAMS
        ------
            protein_len: int
                Size of the input protein/peptide. This is a stagnent variable, but we
                could probably make a vector with this information to get around the
                single scalar value problem. Essentially an array of peptide/protein
                sizes.

            output_size: int
                The size of the final output from the model

            input_size: int
                The size of the input vector

            normalize: bool
                Specify normalization of values or not

            decoder_layers: int
                The number of layers the DecoderBlock contains

            encoder_layers: int
                The number of layers the EncoderBlock contains

            latent_space_dim: int
                $R$ space we want our latent representation to be in

            non_dof_info: bool
                True/False non-DoF information is passed with the input

            torsion_n: int
                Strictly specify the number of torsion angles present within the dataset

            bond_angle_n: int
                Strictly specify the number of bond angles present within the dataset

            bond_length_n: int
                Strictly specify the number of bond lengths present within the dataset

            main_chain_n: int
                Number of main chain atoms for each torsion, bond_angle, and bond_length

            side_chain_n: int
                Number of side chain atoms for each torsion, bond_angle, and bond_length

            extra_info_n: int
                Number of passed additional information after torsion, bond_angle, and bond
                length.
        """
        super().__init__()
        # Save inputs
        self.protein_len      = protein_len
        self.output_size      = output_size
        self.input_size       = input_size
        self.normalize        = normalize
        self.decoder_layers   = decoder_layers
        self.encoder_layers   = encoder_layers
        self.latent_space_dim = latent_space_dim
        self.non_dof_info     = non_dof_info
        self.torsion_n        = torsion_n
        self.bond_angle_n     = bond_angle_n
        self.bond_length_n    = bond_length_n
        self.main_chain_n     = main_chain_n
        self.side_chain_n     = side_chain_n
        self.extra_info_n     = extra_info_n

        # Generate Values
        # if non_dof_info:
        #     self.extra_info = input_size - protein_len
        # else:
        #     self.extra_info = 0

        # Size after decomposition
        # self.after_decomposition_size = self.protein_len*2 + self.extra_info

        # Calculate the size of the input vector
        if self.main_chain_n == 0:
            self.after_decomposition_size = self.torsion_n*2 + self.bond_angle_n*2 + self.bond_length_n*2 + self.extra_info_n
        else:
            torsion = (self.torsion_n*self.main_chain_n + self.torsion_n*self.main_chain_n)*2
            bond_angle = (self.bond_angle_n*self.main_chain_n + self.bond_angle_n*self.main_chain_n)*2
            bond_length = (self.bond_length_n*self.main_chain_n + self.bond_length_n*self.main_chain_n)*2
            self.after_decomposition_size = torsion + bond_angle + bond_length

        # Init Model Blocks
        self.input_decomposition = ConformationalInputDecomposition(protein_len=self.protein_len)
        self.encoder = ConformationalMLPEncoderBlock(
            num_layers = self.encoder_layers,
            start_size = self.after_decomposition_size,
            end_size   = self.latent_space_dim,
        )
        self.latent = ConformationalLatentSpace(latent_space_dim=self.latent_space_dim)
        self.decoder = ConformationalMLPDecoderBlock(
            num_layers = self.decoder_layers,
            start_size = self.latent_space_dim,
            end_size   = self.after_decomposition_size,
        )
        self.output = ConformationalOutputBlock(
            protein_len = self.protein_len,
            normalize   = self.normalize,
            in_size     = self.after_decomposition_size,
            output_size = self.input_size,
        )

    def forward(self, x: torch.Tensor):
        """Train and Test Forward Method"""
        # decompose
        x = self.input_decomposition(x)
        # encode the information into latent
        x = self.encoder(x)
        # pass the latent to itself D -> D
        x = self.latent(x)
        # Decode the information back to post-decomposition input size
        x = self.decoder(x)
        # Recompute original input information
        return self.output(x)


##################################################################################################################
#                                               Main
##################################################################################################################
if __name__ == "__main__":
    #### This is for testing the models functionality purposes
    # Import
    from model_functions import train

    # Create dummy input values
    x = torch.randint(-180, 180, size=(10000, 8))
    l = torch.rand(size=(10000, 120))
    # concat the pseudo-angles and pseudo-extra info together
    y = torch.cat((x, l), dim=-1)
    dataload = torch.utils.data.DataLoader(y, batch_size=1000)

    # init model
    model = ConformationalMLPAutoEncoder(
        8, 128, 128, True, 3, 3, 3, True
    )

    # init general Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Run Model
    print("***MODEL TRAINING IS STARTING***")
    final_loss = train(
        model = model,
        dataset = dataload,
        epochs = 1000,
        device = "cpu",
        optimizer = optimizer,
    )
