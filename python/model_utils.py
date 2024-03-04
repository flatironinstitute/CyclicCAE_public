#!/usr/bin/env python
#
# @author: Andrew Powers (apowers@flatironinstitute.org) Guest Researcher
# @brief: Classes and Functions needed for autoencoder model
#

##################################################################################################################
#                                               Packages 
##################################################################################################################

# base python packages
import os, sys
from collections import defaultdict
import pdb
import random

# local import
from model_functions import sklearn_scale

# ml imports
import torch
import torch.nn as nn




##################################################################################################################
#                                                Classes
##################################################################################################################

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: torch.Tensor):
        """Custom Class to return indices when accessing items"""
        self.dataset = dataset

    def __getitem__(self, index):
        """Extracts sample `index` that is passed"""
        data = self.dataset[index]
        return data, index

    def __len__(self):
        """Returns the length of the dataset object"""
        return len(self.dataset)


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
        self.linear = nn.Linear(self.input_size, self.output_size, bias=False)

        # activation funciton
        # self.leakyrelu = nn.LeakyReLU()
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor, final_layer:bool = False):
        """ forward pass of Encoderlayer

        PARAMS
        ------
            x: torch.Tensor
                The DoF input vector of size (B, N) where B is
                the batch size and N is the number of elements
        """
        if final_layer:
            return self.linear(x)
        else:
            # return self.leakyrelu(self.linear(x))
            return self.tanh(self.linear(x))

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
        for i, layer in enumerate(self.encoder_layers):
            # If last layer then pass True
            if i+1 == self.num_layers:
                x = layer(x, True)

            else:
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
        self.linear = nn.Linear(self.input_size, self.output_size, bias=False)

        # activation funciton
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor, final_layer:bool = False):
        """ forward pass of Encoderlayer

        PARAMS
        ------
            x: torch.Tensor
                The DoF input vector of size (B, N) where B is
                the batch size and N is the number of elements
        """
        if final_layer:
            return self.linear(x)
        else:
            return self.tanh(self.linear(x))

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
        for i, layer in enumerate(self.decoder_layers):
            # If last layer then pass True
            if i+1 == self.num_layers:
                x = layer(x, True)

            else:
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

        # activation func

        # linear function to keep it in the latent space
        # self.latent = nn.Linear(latent_space_dim, latent_space_dim)

    def forward(self, x: torch.Tensor):
        """ Forward of Latent Space Block
        with a Skip Connnection for BackProp

        PARAMS
        ------
            x: torch.Tensor
                Reduced input vector (B, latent_space_dim)
        """
        # return torch.add(self.latent(x), x)
        return x


class ConformationalOutputBlock(nn.Module):
    def __init__(
        self,
        protein_len: int,
        normalize: bool = True,
        in_size: int = 128,
        output_size: int = 512,
        torsion_main_n: int = 0,
        torsion_side_n: int = 0,
        bond_angle_main_n: int = 0,
        bond_angle_side_n: int = 0,
        bond_length_main_n: int = 0,
        bond_length_side_n: int = 0,
        jump_n: int = 0,
        energy_n: int = 0,
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

            torsion_main_n: int
                Strictly specify the number of main chain torsion angles present within the dataset

            torsion_side_n: int
                Strictly specify the number of side chain torsion angles present within the dataset

            bond_angle_main_n: int
                Strictly specify the number of main chain bond angles present within the dataset

            bond_angle_side_n: int
                Strictly specify the number of side chain bond angles present within the dataset

            bond_length_main_n: int
                Strictly specify the number of main chain bond lengths present within the dataset

            bond_length_side_n: int
                Strictly specify the number of side chain bond lengths present within the dataset

            jump_n: int
                Strictly specify the number of jumps within the protein/peptide

            energy_n: int
                Strictly specify the number of energy terms for the protein/peptide

        """
        super().__init__()
        # store values
        self.protein_len         = protein_len
        self.normalize           = normalize
        self.torsion_main_n      = torsion_main_n
        self.torsion_side_n      = torsion_side_n
        self.bond_angle_main_n   = bond_angle_main_n
        self.bond_angle_side_n   = bond_angle_side_n
        self.bond_length_main_n  = bond_length_main_n
        self.bond_length_side_n  = bond_length_side_n
        self.jump_n              = jump_n
        self.energy_n            = energy_n

        # Final linear layer
        # self.output_linear = nn.Linear(in_size, in_size)

        # batch normalization by keeping in (-1, 1) range
        self.tanh_norm = nn.Hardtanh(min_val=-1.0, max_val=1.0)

        # activation function
        self.tanh = nn.Tanh()

        # lastout
        self.output_layers = nn.Sequential(
            nn.Linear(in_size, in_size, bias=False),
            nn.LeakyReLU(),
            nn.Linear(in_size, in_size, bias=False),
            nn.LeakyReLU(),
        )

    def forward(self, x: torch.Tensor):
        """ Forward for OutputBlock which normalizes and outputs

        PARAMS
        ------
            x: torch.Tensor
                Reconstructed x vector post-decomposition (B, D)
        """
        # layer output linear layer
        # x = self.output_linear(x)

        # grab appropriate sections if main/side chain included
        torsion_num = 2 * ( self.torsion_main_n + self.torsion_side_n ) 
        bond_angle_num = 2 * ( self.bond_angle_main_n + self.bond_angle_side_n ) 
        bond_length_num = self.bond_length_main_n + self.bond_length_side_n

        # Now grab values
        torsion = x[:, :torsion_num]
        bond_angle = x[:, torsion_num : torsion_num + bond_angle_num]

        bond_length = x[:, torsion_num + bond_angle_num : torsion_num + bond_angle_num + bond_length_num]
        extra = x[:, torsion_num + bond_angle_num + bond_length_num:]

        #linear term combine
        linear_combo = torch.cat((bond_length, extra), dim=-1)

        # compute large values
        linear_out = self.output_layers(linear_combo)

        # normalize to keep the values on the unit circle vector output
        # torsion_norm = self.tanh_norm(torsion)
        # bond_angle_norm = self.tanh_norm(bond_angle)

        # perform arctan2 
        # recomputed_torsion = torch.rad2deg(torch.atan2(torsion_norm[:, ::2], torsion_norm[:, 1::2]))
        # recomputed_bond_angle = torch.rad2deg(torch.atan2(bond_angle_norm[:, ::2], bond_angle_norm[:, 1::2]))
        # recomputed_torsion = torch.rad2deg(torch.atan2(torsion[:, ::2], torsion[:, 1::2]))
        # recomputed_bond_angle = torch.rad2deg(torch.atan2(bond_angle[:, ::2], bond_angle[:, 1::2]))

        # restack computed dofs with non_dof_x if necessary
        # out = torch.cat((recomputed_torsion, recomputed_bond_angle), dim=-1)
        out = torch.cat((torsion, bond_angle), dim=-1)
        out = torch.cat((out, linear_out), dim=-1)
        # out = torch.cat((out, bond_length), dim=-1)
        # out = torch.cat((out, extra), dim=-1)
        return out #, torch.cat((recomputed_torsion, recomputed_bond_angle), dim=-1)

class ConformationalInputDecomposition(nn.Module):
    def __init__(
        self,
        protein_len: int,
        torsion_main_n: int = 0,
        torsion_side_n: int = 0,
        bond_angle_main_n: int = 0,
        bond_angle_side_n: int = 0,
        bond_length_main_n: int = 0,
        bond_length_side_n: int = 0,
        jump_n: int = 0,
        energy_n: int = 0,
        ):
        """Decomposes DoF input portion into Sin and Cos

        PARAMS
        ------
            protein_len: int
                The length of the protein, so we know where to cutoff the
                DoF decomposition process.

            torsion_main_n: int
                Strictly specify the number of main chain torsion angles present within the dataset

            torsion_side_n: int
                Strictly specify the number of side chain torsion angles present within the dataset

            bond_angle_main_n: int
                Strictly specify the number of main chain bond angles present within the dataset

            bond_angle_side_n: int
                Strictly specify the number of side chain bond angles present within the dataset

            bond_length_main_n: int
                Strictly specify the number of main chain bond lengths present within the dataset

            bond_length_side_n: int
                Strictly specify the number of side chain bond lengths present within the dataset

            jump_n: int
                Strictly specify the number of jumps within the protein/peptide

            energy_n: int
                Strictly specify the number of energy terms for the protein/peptide
        """
        super().__init__()
        # save passed variables
        self.protein_len         = protein_len
        self.torsion_main_n      = torsion_main_n
        self.torsion_side_n      = torsion_side_n
        self.bond_angle_main_n   = bond_angle_main_n
        self.bond_angle_side_n   = bond_angle_side_n
        self.bond_length_main_n  = bond_length_main_n
        self.bond_length_side_n  = bond_length_side_n
        self.jump_n              = jump_n
        self.energy_n            = energy_n
        

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

        # Sum up the amount of torsion
        torsion_size = self.torsion_main_n + self.torsion_side_n
        # grab out torsion
        if torsion_size != 0:
            # pul out the specific numbers from the passed params
            torsion = x[:, :torsion_size]
            # decompose pulled out angles into both sin and cos
            torsion_decompose = self.decompose_angle_(torsion)

        # update index_in
        index_in += torsion_size

        # sum up the amount of bond angles
        bond_angle_size = self.bond_angle_main_n + self.bond_angle_side_n
        # grab out bond_angle
        if bond_angle_size != 0:
            # pull out the specific numbers from the passed params
            bond_angle = x[:, index_in : index_in + bond_angle_size]
            # decompose pulled out angles into both sin and cos
            bond_angle_decompose = self.decompose_angle_(bond_angle)

        # update index_in
        index_in += bond_angle_size

        # sum up the amount of bond lengths
        bond_length_size = self.bond_length_main_n + self.bond_length_side_n
        # grab out bond length
        if bond_length_size != 0:
            # pull out number of passed params
            bond_length = x[:, index_in : index_in + bond_length_size]

        # update index_in
        index_in += bond_length_size

        # Pull out jumps
        if self.jump_n != 0:
            jump = x[:, index_in : index_in + self.jump_n]

        # update index_in
        index_in += self.jump_n

        # pull out energy
        if self.energy_n != 0:
            energy = x[:, index_in : index_in + self.energy_n]

        # update index_in
        index_in += self.energy_n

        # combine all information
        out = torch.cat((torsion_decompose, bond_angle_decompose), dim=-1)
        out = torch.cat((out, bond_length), dim=-1)
        if self.jump_n != 0:
            out = torch.cat((out, jump), dim=-1)
        if self.energy_n != 0:
            out = torch.cat((out, energy), dim=-1)

        return out

    def decompose_angle_(self, x: torch.Tensor, idx: torch.Tensor = None):
        """Break the angle up into sin & cos components"""
        # if idx is provided pull out only angle idx components
        # Assumption is made that the input is 2d (B, N)
        if idx != None:
            # Grab out specific angle information
            x = x[:, idx]

        # decompose angle into both sin and cos
        sin_x = torch.sin(torch.pi * self.circular_scale(x))
        cos_x = torch.cos(torch.pi * self.circular_scale(x))

        # stack new decomposed vectors
        stacked_sincos_x = torch.cat((sin_x, cos_x), dim=-1)

        return stacked_sincos_x

    def circular_scale(self, angle: torch.tensor):
        """Convert angles to be circular and continuous"""
        scaled_angle = (angle + 180) / 360 # Map to [0, 1]
        return scaled_angle * 2 - 1 # Map to [-1, 1]


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
        torsion_main_n: int = 0,
        torsion_side_n: int = 0,
        bond_angle_main_n: int = 0,
        bond_angle_side_n: int = 0,
        bond_length_main_n: int = 0,
        bond_length_side_n: int = 0,
        jump_n: int = 0,
        energy_n: int = 0,
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

            torsion_main_n: int
                Strictly specify the number of main chain torsion angles present within the dataset

            torsion_side_n: int
                Strictly specify the number of side chain torsion angles present within the dataset

            bond_angle_main_n: int
                Strictly specify the number of main chain bond angles present within the dataset

            bond_angle_side_n: int
                Strictly specify the number of side chain bond angles present within the dataset

            bond_length_main_n: int
                Strictly specify the number of main chain bond lengths present within the dataset

            bond_length_side_n: int
                Strictly specify the number of side chain bond lengths present within the dataset

            jump_n: int
                Strictly specify the number of jumps within the protein/peptide

            energy_n: int
                Strictly specify the number of energy terms for the protein/peptide
        """
        super().__init__()
        # Save inputs
        self.protein_len         = protein_len
        self.output_size         = output_size
        self.input_size          = input_size
        self.normalize           = normalize
        self.decoder_layers      = decoder_layers
        self.encoder_layers      = encoder_layers
        self.latent_space_dim    = latent_space_dim
        self.torsion_main_n      = torsion_main_n
        self.torsion_side_n      = torsion_side_n
        self.bond_angle_main_n   = bond_angle_main_n
        self.bond_angle_side_n   = bond_angle_side_n
        self.bond_length_main_n  = bond_length_main_n
        self.bond_length_side_n  = bond_length_side_n
        self.jump_n              = jump_n
        self.energy_n            = energy_n

        # combine terms
        self.torsion = self.torsion_main_n + self.torsion_side_n
        self.bond_angle = self.bond_angle_main_n + self.bond_angle_side_n
        self.bond_length = self.bond_length_main_n + self.bond_length_side_n

        # Combine for infromation (Also, the jump_n might need to be muptlied by 6)
        self.after_decomposition_size = self.torsion*2 + self.bond_angle*2 + self.bond_length + self.jump_n + self.energy_n
        self.linear_out_size = self.after_decomposition_size - (self.torsion*2 + self.bond_angle*2)

        # Init Model Blocks
        self.input_decomposition = ConformationalInputDecomposition(
            protein_len         = self.protein_len,
            torsion_main_n      = self.torsion_main_n,
            torsion_side_n      = self.torsion_side_n,
            bond_angle_main_n   = self.bond_angle_main_n,
            bond_angle_side_n   = self.bond_angle_side_n,
            bond_length_main_n  = self.bond_length_main_n,
            bond_length_side_n  = self.bond_length_side_n,
            jump_n              = self.jump_n,
            energy_n            = self.energy_n,
        )
        self.encoder = ConformationalMLPEncoderBlock(
            num_layers = self.encoder_layers,
            start_size = self.after_decomposition_size,
            end_size   = self.latent_space_dim,
        )
        self.latent = ConformationalLatentSpace(latent_space_dim=self.latent_space_dim)
        # self.latent = ConformationalLatentSpace(latent_space_dim=self.after_decomposition_size) # check bottleneck

        self.decoder = ConformationalMLPDecoderBlock(
            num_layers = self.decoder_layers,
            start_size = self.latent_space_dim,
            end_size   = self.after_decomposition_size,
        )

        self.output = ConformationalOutputBlock(
            protein_len         = self.protein_len,
            normalize           = self.normalize,
            in_size             = self.linear_out_size,
            output_size         = self.input_size,
            torsion_main_n      = self.torsion_main_n,
            torsion_side_n      = self.torsion_side_n,
            bond_angle_main_n   = self.bond_angle_main_n,
            bond_angle_side_n   = self.bond_angle_side_n,
            bond_length_main_n  = self.bond_length_main_n,
            bond_length_side_n  = self.bond_length_side_n,
            jump_n              = self.jump_n,
            energy_n            = self.energy_n,
        )

    def forward(self, x: torch.Tensor):
        """Train and Test Forward Method"""
        # decompose
        x = self.input_decomposition(x)

        # Extract out initial decomposed input for MSE Loss calc
        initial_input = x.clone().detach()

        # encode the information into latent
        x = self.encoder(x)

        # pass the latent to itself (basically just returns passed in tensor)
        x = self.latent(x)

        # compute latent rmsd
        latent_clone = x.clone()
        latent_euclid_dist = self.compute_euclidean_distance(latent_clone)

        # Decode the information back to post-decomposition input size
        # Currently we are not using the OutputBlock
        # Return decoded x (decomposition size), latent_euclidean_distance, decomposed input
        out = self.output(self.decoder(x))
        return out, latent_euclid_dist, latent_clone, initial_input

    def compute_euclidean_distance(self, xyz_mat: torch.Tensor):
        """Compute the latent space RMSD"""
        # compute pairwise diff
        differences = xyz_mat.unsqueeze(1) - xyz_mat.unsqueeze(0)

        # squared_distances along last dim
        distances = torch.sum( torch.pow( differences, 2), dim=-1)

        # distance matrix
        # distance_matrix = torch.sqrt(squared_distances)
        return distances

##################################################################################################################
#                                               Main
##################################################################################################################
if __name__ == "__main__":
    #### This is for testing the models functionality purposes
    # Import
    from model_functions import train as mse_train


    # Create dummy input values
    torsion = torch.randint(-180, 180, size=(10000, 58))
    bond_angle = torch.randint(-180, 180, size=(10000, 58))
    bond_length = torch.rand(size=(10000, 58))
    l = torch.rand(size=(10000, 6))
    # concat the pseudo-angles and pseudo-extra info together
    input_vector = torch.cat((torsion, bond_angle), dim=-1)
    input_vector = torch.cat((input_vector, bond_length), dim=-1)
    input_vector = torch.cat((input_vector, l), dim=-1)
    dataload = torch.utils.data.DataLoader(input_vector, batch_size=1000)

    # init model
    model = ConformationalMLPAutoEncoder(
        protein_len         = 8,
        output_size         = 58*3 + 6,
        input_size          = 58*3 + 6,
        normalize           = True,
        decoder_layers      = 3,
        encoder_layers      = 3,
        latent_space_dim    = 3,
        torsion_main_n      = 58,
        torsion_side_n      =  0,
        bond_angle_main_n   = 58,
        bond_angle_side_n   =  0,
        bond_length_main_n  = 58,
        bond_length_side_n  =  0,
        jump_n              =  0,
        energy_n            =  6,
    )

    # init AdamW optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-8)

    # init scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max = 10000,
                                                           eta_min = 1e-6,
                                                           )

    # Model Run specific number classifier for output files
    random_num = random.randint(0, 10000)

    # Run Model
    print("***MODEL TRAINING IS STARTING***")
    final_loss = mse_train(
        model = model,
        dataset = dataload,
        epochs = 1000,
        device = "cuda",
        optimizer = optimizer,
        scheduler = scheduler,
        angle_n = 58*2,
        linear_n = 58+6,
        file_out_track = None,
        output_dir = None,
        weight_save_n = None,
        file_num = random_num,
    )
