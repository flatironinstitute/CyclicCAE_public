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
import torch.nn.functional as F

# numpy
import numpy as np


##################################################################################################################
#                                               Functions 
##################################################################################################################

def subset_dataset(dataset, indices):
    """Create a subset of the dataset while preserving the original indices."""
    # This function returns a Subset of the original dataset but keeps track of the indices.
    return torch.utils.data.Subset(dataset, indices)

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

class CustomScheduler:
    def __init__(
        self,
        warmup_steps: int,
        base_lr: float,
        max_lr: float,
        patience: int,
        cooldown: int,
        optimizer: torch.optim,
        threshold: float = 1e-3,
        factor: float = 0.5,
        low_lr_cutoff: float = 1e-7,
        ):
        """Custom Warmup with ReduceLROnPlateau scheduler"""
        # Assign params
        self.warmup_steps = warmup_steps
        self.continue_warmup = 0
        self.base_lr = base_lr
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.low_lr_cutoff = low_lr_cutoff

        # set schedulers
        # self.cycleLR = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr = max_lr, total_steps = warmup_steps*2)
        self.reduceLR = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            min_lr = base_lr,
            patience = patience,
            mode = "min",
            factor = factor,
            threshold = threshold,
            cooldown = cooldown,
        )

    def increase_lr(self, step, warmup_steps):
        """Increase the lr back to max_lr slowly"""
        # Calculate new lr
        update_lr = self.base_lr + (self.max_lr - self.base_lr) * (step / warmup_steps)

        # set lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = update_lr

    def step_cycleLR(self, step):
        """Step cycleLR"""
        current_lr = self.optimizer.param_groups[0]['lr']

        if (step < self.warmup_steps):
            self.increase_lr(step, self.warmup_steps)

        elif (current_lr < self.low_lr_cutoff) and (step > self.warmup_steps):
            # update step to check now.
            self.warmup_steps = step + self.warmup_steps
            self.increase_lr(step, self.warmup_steps)

    def step_reduceLR(self, val_loss):
        """Step ReduceLR"""
        return self.reduceLR.step(val_loss)

class ConvPEInput(nn.Module):
    def __init__(
        self,
        channel_in: int = 116,
        channel_out: int = 58,
        d_model: int = 58,
        stride: int = 1,
        padding: int = 0,
        kernel_size: int = 24, 
        dropout: float = 0.2,
        ):
        super().__init__()
        """Apply Convolution to the PE based decomposition of the inputs."""
        ###### NOTE FOR MYSELF LATER
        ###### Might want to think about padding sequences, so I don't have to do the linear layer at the end of 
        ###### this block. !!!!
        ###### Paper: https://arxiv.org/pdf/1705.03122.pdf
        # conv PE
        self.conv1 = nn.Conv1d(channel_in, channel_in, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2 = nn.Conv1d(channel_in, channel_out, kernel_size=kernel_size//2, stride=stride, padding=padding)

        # calculate d size
        d_1 = d_model+(2* (kernel_size)) - kernel_size + 1
        d_2 = d_1 - kernel_size//2 + 1

        # batch norm per L
        self.batch_norm1 = nn.BatchNorm1d(num_features=channel_in)
        self.batch_norm2 = nn.BatchNorm1d(num_features=channel_out)
        self.batch_norm3 = nn.BatchNorm1d(d_model*2)

        # Linear Layer
        self.linear_out = nn.Linear(d_2*channel_out, d_model*2)

    def forward(self, x: torch.Tensor):
        """Forward convPE"""
        # extract shape (remember that the shape is permuted for pytorch)
        B, D, L = x.shape

        # padding size
        padding_size = self.conv1.kernel_size[0] - 1

        # cyclic padd
        x = torch.cat(
            (
                x[:, :, -padding_size:],
                x,
                x[:, :, :-padding_size],
            ),
            dim = -1,
        )

        # Apply convolutions
        conv1 = F.tanh(self.batch_norm1(self.conv1(x)))
        conv2 = F.tanh(self.batch_norm2(self.conv2(conv1)))
        return F.tanh(self.batch_norm3(self.linear_out(conv2.reshape(B, -1))))

class EncoderLayerUnitCircle(nn.Module):
    def __init__(
        self,
        input_size: int = 128,
        output_size: int = 3,
        dropout: float = 0.2,
        ):
        super().__init__()
        """Enforce Output to UnitCircle and add to information"""

        # Expand the inputs 
        self.linear_expand = nn.Linear(input_size, 128)

        # Shrink from last layer
        self.linear_intermediate = nn.Linear(128, 64)

        # Generate sin/cos values
        self.angle_information = nn.Linear(64, 2)

        # Add the angle computed from the forced unit circle above
        self.angle_aware_layer = nn.Linear(64+1, 32)

        # output last layer
        self.final_layer = nn.Linear(32, output_size)

        # dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        x = self.dropout(F.tanh(self.linear_expand(x)))
        x = self.dropout(F.tanh(self.linear_intermediate(x)))

        sin_cos = self.angle_information(x)
        unit_circle = F.normalize(sin_cos, p=2, dim=-1)
        computed_angles = torch.atan2(unit_circle[:, 0], unit_circle[:, 1]).unsqueeze(-1)

        x = F.tanh(self.angle_aware_layer(
            torch.cat((x, computed_angles), dim=-1)
        ))

        return self.final_layer(x)

class EncoderBlock(nn.Module):
    def __init__(
        self,
        num_layers: int = 0,
        input_size: int = 3,
        middle_size: int = 64,
        output_size: int = 128,
        dropout: int = 0.1,
        ):
        super().__init__()
        """Specify Number of DecoderLayerUnitCircle"""
        # layer input size and out
        self.num_layers = num_layers
        insize = [input_size] + [middle_size]*(num_layers-1)
        outsize = [middle_size]*(num_layers-1) + [output_size]


        self.decoder_layers = nn.ModuleList(
            [
                EncoderLayerUnitCircle(insize[i], outsize[i], dropout) for i in range(num_layers)
            ]
        )

    def forward(self, x:torch.Tensor):
        # iter through layres
        for n, layer in enumerate(self.decoder_layers):

            # if first layer we need to compute skip connection
            if n == 0:

                # Compute first layer
                x = layer(x)
                # compute skip connection
                skip = x.clone()

            elif n+1 == self.num_layers:
                x = layer(x+skip)

            else:
                x = layer(x)

        return x

class EncoderBlockList(nn.Module):
    def __init__(
        self,
        num_layers: int,
        input_size: int,
        output_size: int,
        middle_size: int = 64,
        dropout: float = 0.2,
        ):
        super().__init__()
        """Encoder Block for Cyclic PE and encoding"""
        # layer input size and out
        self.num_layers = num_layers
        insize = [input_size] + [middle_size]*(num_layers-1)
        outsize = [middle_size]*(num_layers-1) + [output_size]


        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer64(insize[i], outsize[i], dropout) for i in range(num_layers)
            ]
        )

    def forward(self, x:torch.Tensor):
        # iter through layers
        for layer in self.encoder_layers:
            x = layer(x)
        return x


class DecoderLayerUnitCircle(nn.Module):
    def __init__(
        self,
        input_size: int = 128,
        output_size: int = 3,
        dropout: float = 0.2,
        ):
        super().__init__()
        """Enforce Output to UnitCircle and add to information"""

        # Expand the inputs 
        self.linear_expand = nn.Linear(input_size, 128)

        # Shrink from last layer
        self.linear_intermediate = nn.Linear(128, 64)

        # Generate sin/cos values
        self.angle_information = nn.Linear(64, 2)

        # Add the angle computed from the forced unit circle above
        self.angle_aware_layer = nn.Linear(64+1, 32)

        # output last layer
        self.final_layer = nn.Linear(32, output_size)

        # dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        x = self.dropout(F.tanh(self.linear_expand(x)))
        x = self.dropout(F.tanh(self.linear_intermediate(x)))

        sin_cos = self.angle_information(x)
        unit_circle = F.normalize(sin_cos, p=2, dim=-1)
        computed_angles = torch.atan2(unit_circle[:, 0], unit_circle[:, 1]).unsqueeze(-1)

        x = F.tanh(self.angle_aware_layer(
            torch.cat((x, computed_angles), dim=-1)
        ))

        return self.final_layer(x)

class DecoderBlock(nn.Module):
    def __init__(
        self,
        num_layers: int = 0,
        input_size: int = 3,
        output_size: int = 128,
        dropout: float = 0.1,
        ):
        super().__init__()
        """Specify Number of DecoderLayerUnitCircle"""
        # layer input size and out
        self.num_layers = num_layers
        insize = [3] + [output_size]*(num_layers-1)
        outsize = [output_size]*num_layers


        self.decoder_layers = nn.ModuleList(
            [
                DecoderLayerUnitCircle(insize[i], outsize[i], dropout) for i in range(num_layers)
            ]
        )

    def forward(self, x:torch.Tensor):
        # iter through layres
        for n, layer in enumerate(self.decoder_layers):

            # if first layer we need to compute skip connection
            if n == 0:

                # Compute first layer
                x = layer(x)
                # compute skip connection
                skip = x.clone()

            elif n+1 == self.num_layers:
                x = layer(x+skip)

            else:
                x = layer(x)

        return x

class PositionalEncoding(nn.Module):
    def __init__(
        self,
        device: str,
        seq_len: int = 58,
        d_model: int = 1,
        ):
        super().__init__()
        """Add in positional encodings for atom positions"""
        self.embed_dim = nn.Embedding(seq_len, d_model, device=device)

    def forward(self, x_pos: torch.Tensor):
        """Compute PE"""
        return self.embed_dim(x_pos)

class CyclicPE(nn.Module):
    def __init__(
        self,
        device: str,
        seq_len: int = 58,
        d_model: int = 1,
        ):
        """Apply cyclic offset PE for atom positions"""
        super().__init__()
        # set param
        self.atom_n = seq_len

        # Generate the ith column index
        i = torch.arange(seq_len)

        # Take make a 2d tensor of size (L, 2) Where we can (i, i+L), so we have to flip the stack
        ij = torch.stack((i, i+seq_len), dim=-1)

        # We make the same symmetric matrix from before
        offset = i[:, None] - i[None, :]

        # Now we use a dimension trick to compute the absolute distances between ij by adding ij.unsqueeze(1, 2) and ij.unsqueeze(0,1)
        # This allows us to compute the cyclic position in the resulting tensors 2nd dimension. 
        c_offset = torch.abs(ij[:, None,:, None] - ij[None, :, None, :])

        # Then we an compute the distances by taking the min from that 2nd and 3rd resulting dimension
        R = torch.tensor(c_offset.numpy().min((2,3)))

        # Now we need to flip the sign of values that are less than the offset distance 
        a = R < torch.abs(offset)
        R[a] = -R[a]

        # Fully we reflip the signs 
        self.R = (R * torch.sign(offset)).to(device)

        # setup embeddings
        self.offset_embed = nn.Embedding(seq_len + 1, d_model)

    def forward(self):
        """Generate cyclic pe"""
        offset_cyc = self.offset_embed(self.R + self.atom_n//2)
        return offset_cyc


class EncoderLayer64(nn.Module):
    def __init__(
        self,
        input_size: int = 128,
        output_size: int = 3,
        layer_num: int = 0,
        dropout: float = 0.0,
        ):
        """ A Basic MLP layer for the autoencoder model

        PARAMS
        ------
            input_size: int
                The size of the input vector from the last layer

            output_size: int
                The size of the vector at the end of this layer

            layer_num: (int) The specific layer of our Block
            feature_channel_in: (int) The feature in channel

        """
        super().__init__()
        # Save inputs
        self.input_size = input_size
        self.output_size = output_size

        self.linear = nn.Linear(input_size, output_size, bias=False)

        # self.leakyrelu = nn.LeakyReLU()
        self.relu = nn.ReLU()

        self.batch_norm = nn.BatchNorm1d(output_size)

        self.dropout = nn.Dropout(dropout)

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
            return self.relu(self.batch_norm(self.dropout(self.linear(x))))
            # return self.relu(self.batch_norm(self.linear(x)))

class EncoderBlock64(nn.Module):
    def __init__(
        self,
        num_layers: int = 3,
        start_size: int = 512,
        d_model: int = 64,
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

        self.layer_size = [start_size] + [d_model]*(num_layers-1)
        self.out_size = [d_model] * (num_layers-1) + [end_size]


        # Make list of encoder layers that make up the encoder block
        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer64(
                    input_size = self.layer_size[l],
                    output_size = self.out_size[l],
                ) for l in range(0,self.num_layers)
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
        # Inital ski
        # skip = x.clone()

        # for i, layer in enumerate(self.encoder_layers):
        #     # If last layer then pass True
        #     if i+1 == self.num_layers:
        #         x = layer(torch.add(x, skip), True)
        #     elif (i+1 != self.num_layers) or (i+1 != 1):
        #         skip_connect = x.clone()
        #         x = layer(x, True)
        #     else:
        #         x = layer(torch.add(x, skip_connect))
        for layer in self.encoder_layers:
            x = layer(x)
        return x

##### Base ENCODER
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

        # batch normalize
        self.batch_norm = nn.BatchNorm1d(self.output_size)

        # activation funciton
        self.leakyrelu = nn.LeakyReLU()
        # self.gelu = nn.GELU()

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
            return self.leakyrelu(self.batch_norm(self.linear(x)))

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

        # batch normalize
        self.batch_norm = nn.BatchNorm1d(self.output_size)

        # activation funciton
        # self.sigmoid = nn.Sigmoid()
        self.leakyrelu = nn.LeakyReLU()

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
            return self.leakyrelu(self.batch_norm(self.linear(x)))

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
        self.latent = nn.Linear(latent_space_dim, latent_space_dim)

    def forward(self, x: torch.Tensor):
        """ Forward of Latent Space Block
        with a Skip Connnection for BackProp

        PARAMS
        ------
            x: torch.Tensor
                Reduced input vector (B, latent_space_dim)
        """
        return torch.add(self.latent(x), x)
        # return x


class ConformationalOutputBlock(nn.Module):
    def __init__(
        self,
        protein_len: int,
        normalize: bool = True,
        in_size_linear: int = 128,
        in_size_angle: int = 128,
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

        # batch normalization by keeping in (-1, 1) range
        # self.tanh_norm = nn.Hardtanh(min_val=-1.0, max_val=1.0)

        # last angle
        self.out_angle = nn.Sequential(
            nn.Linear(in_size_angle, in_size_angle, bias=False),
            nn.Tanh(),
            nn.Linear(in_size_angle, in_size_angle, bias=False),
            # nn.Tanh(),
        )

        # Uncomment This
        # # lastout
        # self.output_layers = nn.Sequential(
        #     nn.Linear(in_size_linear, in_size_linear, bias=False),
        #     nn.BatchNorm1d(in_size_linear),
        #     nn.LeakyReLU(),
        #     nn.Linear(in_size_linear, in_size_linear, bias=False),
        #     # nn.LeakyReLU(),
        # )

    def forward(self, x: torch.Tensor):
        """ Forward for OutputBlock which normalizes and outputs

        PARAMS
        ------
            x: torch.Tensor
                Reconstructed x vector post-decomposition (B, D)
        """
        # layer output linear layer
        # x = self.output_linear(x)

        #### Uncomment This
        # # grab appropriate sections if main/side chain included
        # torsion_num = 2 * ( self.torsion_main_n + self.torsion_side_n ) 
        # bond_angle_num = 2 * ( self.bond_angle_main_n + self.bond_angle_side_n ) 
        # bond_length_num = self.bond_length_main_n + self.bond_length_side_n
        #
        # # Now grab values
        # torsion = x[:, :torsion_num]
        # bond_angle = x[:, torsion_num : torsion_num + bond_angle_num]
        # angle_combo = torch.cat((torsion, bond_angle), dim=-1)
        #
        # # compute angles
        # angle_out = self.out_angle(angle_combo)
        #
        # bond_length = x[:, torsion_num + bond_angle_num : torsion_num + bond_angle_num + bond_length_num]
        # extra = x[:, torsion_num + bond_angle_num + bond_length_num:]
        #
        # #linear term combine
        # linear_combo = torch.cat((bond_length, extra), dim=-1)
        #
        # # compute large values
        # linear_out = self.output_layers(linear_combo)

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
        # out = torch.cat((torsion, bond_angle), dim=-1)
        # out = torch.cat((torsion_norm, bond_angle_norm), dim=-1)
        # out = torch.cat((out, linear_out), dim=-1)

        # Uncomment This
        # out = torch.cat((angle_out, linear_out), dim=-1)
        # out_layers = self.out_angle(x)

        # N, D = x.shape
        x = self.out_angle(x)
        # out_layers = x.reshape(-1, 2)
        # out = F.normalize(out_layers, p=2, dim=-1)
        return x #, out.reshape(N, D)

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
        decomposition_n: int = 1,
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

            decomposition_n: int
                Number of times to decompose x
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
        self.decomposition_n     = decomposition_n
        

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
            torsion_decompose = self.decompose_angle_(torsion, self.decomposition_n)
            # torsion_decompose = torsion * torch.pi / 180

        # update index_in
        index_in += torsion_size

        # sum up the amount of bond angles
        bond_angle_size = self.bond_angle_main_n + self.bond_angle_side_n
        # grab out bond_angle
        if bond_angle_size != 0:
            # pull out the specific numbers from the passed params
            bond_angle = x[:, index_in : index_in + bond_angle_size]
            # decompose pulled out angles into both sin and cos
            bond_angle_decompose = self.decompose_angle_(bond_angle, self.decomposition_n)
            # bond_angle_decompose = bond_angle * torch.pi / 180

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

        return torsion_decompose, energy  #, out

    def decompose_angle_(self, x: torch.Tensor, decomposition_n: int, idx: torch.Tensor = None):
        """Break the angle up into sin & cos components"""
        # if idx is provided pull out only angle idx components
        # Assumption is made that the input is 2d (B, N)
        if idx != None:
            # Grab out specific angle information
            x = x[:, idx]

        # decompose angle into both sin and cos
        # sin_x = torch.sin(torch.pi * self.circular_scale(x))
        # cos_x = torch.cos(torch.pi * self.circular_scale(x))
        if decomposition_n == 1:
            sin_x = torch.sin(torch.deg2rad(x))
            cos_x = torch.cos(torch.deg2rad(x))
        else:
            sin_x, cos_x = torch.zeros(x.size(0), decomposition_n, x.size(-1), device="cuda"), torch.zeros(x.size(0), decomposition_n, x.size(-1), device="cuda").double()
            for n in range(1, decomposition_n+1):
                sin_x[:, n-1, :] = torch.sin(n*torch.deg2rad(x))
                cos_x[:, n-1, :] = torch.cos(n*torch.deg2rad(x))
            sin_x = sin_x.reshape(x.size(0), -1)
            cos_x = cos_x.reshape(x.size(0), -1)

        # stack new decomposed vectors
        stacked_sincos_x = torch.cat((sin_x, cos_x), dim=-1)

        return stacked_sincos_x

    def circular_scale(self, angle: torch.tensor):
        """Convert angles to be circular and continuous"""
        scaled_angle = (angle + 180) / 360 # Map to [0, 1]
        return scaled_angle * 2 - 1 # Map to [-1, 1]

class InputDecomposition(nn.Module):
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
        decomposition_n: int = 1,
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

            decomposition_n: int
                Number of times to decompose x
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
        self.decomposition_n     = decomposition_n

        torsion_size = 2 * (torsion_main_n + torsion_side_n)
        bond_angle_size = 2 * (bond_angle_main_n + bond_angle_side_n)
        bond_length_size = bond_length_main_n + bond_length_side_n

        self.torsion_conv = nn.Sequential(
            EncoderConvBlock(torsion_size, 64),
            EncoderConvBlock(64, 64),
            EncoderConvBlock(64, 32),
        ).double()

        self.bond_angle_conv = nn.Sequential(
            EncoderConvBlock(bond_angle_size, 64),
            EncoderConvBlock(64, 64),
            EncoderConvBlock(64, 32),
        ).double()

        self.bond_length_conv = nn.Sequential(
            EncoderConvBlock(bond_length_size, 64),
            EncoderConvBlock(64, 64),
            EncoderConvBlock(64, 32),
        ).double()
        

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
            torsion_decompose = self.decompose_angle_(torsion, self.decomposition_n).double()
            # torsion_conv = self.torsion_conv(torsion_decompose) # 12

        # update index_in
        index_in += torsion_size

        # sum up the amount of bond angles
        bond_angle_size = self.bond_angle_main_n + self.bond_angle_side_n
        # grab out bond_angle
        if bond_angle_size != 0:
            # pull out the specific numbers from the passed params
            bond_angle = x[:, index_in : index_in + bond_angle_size]
            # decompose pulled out angles into both sin and cos
            bond_angle_decompose = self.decompose_angle_(bond_angle, self.decomposition_n).double()
            # bond_angle_conv = self.bond_angle_conv(bond_angle_decompose)

        # update index_in
        index_in += bond_angle_size

        # sum up the amount of bond lengths
        bond_length_size = self.bond_length_main_n + self.bond_length_side_n
        # grab out bond length
        if bond_length_size != 0:
            # pull out number of passed params
            bond_length = x[:, index_in : index_in + bond_length_size]
            # bond_length_conv = self.bond_length_conv(bond_length)

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
        # out = torch.cat((torsion_decompose, bond_angle_decompose), dim=-1)
        # out = torch.cat((out, bond_length), dim=-1)
        # out = torch.cat((torsion_conv, bond_angle_conv), dim=-1)
        # out = torch.cat((out, bond_length_conv), dim=-1)
        if (self.jump_n != 0) and (self.energy_n != 0):
            out = torch.cat((jump, energy), dim=-1)
        if (self.energy_n != 0) and (self.jump_n == 0):
            out = energy

        return torsion_decompose, bond_angle_decompose, bond_length, out

    def decompose_angle_(self, x: torch.Tensor, decomposition_n: int):
        """Break the angle up into sin & cos components"""
        if decomposition_n == 1:
            sin_x = torch.sin(torch.deg2rad(x))
            cos_x = torch.cos(torch.deg2rad(x))
        else:
            sin_x, cos_x = torch.zeros(x.size(0), decomposition_n, x.size(-1), device="cuda"), torch.zeros(x.size(0), decomposition_n, x.size(-1), device="cuda")
            for n in range(1, decomposition_n+1):
                sin_x[:, n-1, :] = torch.sin(n*torch.deg2rad(x))
                cos_x[:, n-1, :] = torch.cos(n*torch.deg2rad(x))
            sin_x = sin_x.reshape(x.size(0), -1)
            cos_x = cos_x.reshape(x.size(0), -1)

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
        decomposition_n: int = 1,
        torsion_main_n: int = 0,
        torsion_side_n: int = 0,
        bond_angle_main_n: int = 0,
        bond_angle_side_n: int = 0,
        bond_length_main_n: int = 0,
        bond_length_side_n: int = 0,
        jump_n: int = 0,
        energy_n: int = 0,
        dropout: float = 0.1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        filetype_major: int = 0,
        filetype_minor: int = 1,
        reference_pose: int = None,
        ref_pose_len: int = 128,
        noise_std: int = 5,
        cyclic: bool = False,
        pe_linear: bool = False,
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

            decomposition_n: int
                Series of decomposing x into n number of sin/cos values
                sin(x), sin(n_1*x), ..., sin(n*x)
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
        self.decomposition_n     = decomposition_n
        self.dropout             = dropout
        self.device              = device
        self.noise_std           = noise_std
        self.cyclic              = cyclic
        self.pe_linear           = pe_linear

        # combine terms
        self.torsion = self.torsion_main_n + self.torsion_side_n
        self.bond_angle = self.bond_angle_main_n + self.bond_angle_side_n
        self.bond_length = self.bond_length_main_n + self.bond_length_side_n

        # Combine for infromation (Also, the jump_n might need to be muptlied by 6)
        # Actual decomposition
        # self.after_decomposition_size = self.torsion*2 + self.bond_angle*2 + self.bond_length + self.jump_n + self.energy_n
        # self.linear_out_size = self.after_decomposition_size - (self.torsion*2 + self.bond_angle*2)

        ### Uncomment This
        self.after_decomposition_size = self.torsion*self.decomposition_n*2 # + self.bond_angle*self.decomposition_n*2 + self.bond_length + self.jump_n + self.energy_n
        self.linear_out_size = self.after_decomposition_size - (self.torsion*2 + self.bond_angle*2)
        ### Uncomment This
        self.angle_out_size = self.torsion*2 #+ self.bond_angle*2

        self.after_conv_input_size = (12*3) + jump_n + energy_n
        # self.linear_out_size = self.after_decomposition_size - (self.torsion*self.decomposition_n*2 + self.bond_angle*self.decomposition_n*2)

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
            decomposition_n     = self.decomposition_n,
        ).double()
        self.input_single_decomp = ConformationalInputDecomposition(
            protein_len         = self.protein_len,
            torsion_main_n      = self.torsion_main_n,
            torsion_side_n      = self.torsion_side_n,
            bond_angle_main_n   = self.bond_angle_main_n,
            bond_angle_side_n   = self.bond_angle_side_n,
            bond_length_main_n  = self.bond_length_main_n,
            bond_length_side_n  = self.bond_length_side_n,
            jump_n              = self.jump_n,
            energy_n            = self.energy_n,
            decomposition_n     = 1,
        ).double()

        for param in self.input_single_decomp.parameters():
            param.requires_grad = False

        ##### Basic Pyramid Shape Encoder Decoder
        # self.encoder = ConformationalMLPEncoderBlock(
        #     num_layers = self.encoder_layers,
        #     start_size = self.after_decomposition_size,
        #     end_size   = self.latent_space_dim,
        # ).double()
        # self.decoder = ConformationalMLPDecoderBlock(
        #     num_layers = self.decoder_layers,
        #     start_size = self.latent_space_dim,
        #     end_size   = self.after_decomposition_size,
        # ).double()
        #####

        ##### Convolutions
        # self.input_decomposition = InputDecomposition(
        #     protein_len         = self.protein_len,
        #     torsion_main_n      = self.torsion_main_n,
        #     torsion_side_n      = self.torsion_side_n,
        #     bond_angle_main_n   = self.bond_angle_main_n,
        #     bond_angle_side_n   = self.bond_angle_side_n,
        #     bond_length_main_n  = self.bond_length_main_n,
        #     bond_length_side_n  = self.bond_length_side_n,
        #     jump_n              = self.jump_n,
        #     energy_n            = self.energy_n,
        #     decomposition_n     = self.decomposition_n,
        # ).double()
        #
        # self.torsion_conv = nn.Sequential(
        #     EncoderConvBlock(self.torsion*2, 64),
        #     nn.Tanh(),
        #     EncoderConvBlock(64, 32),
        #     nn.Tanh(),
        #     EncoderConvBlock(32, 12),
        #     nn.Tanh(),
        # ).double()
        #
        # self.bond_angle_conv = nn.Sequential(
        #     EncoderConvBlock(self.bond_angle*2, 64),
        #     nn.Tanh(),
        #     EncoderConvBlock(64, 32),
        #     nn.Tanh(),
        #     EncoderConvBlock(32, 12),
        #     nn.Tanh(),
        # ).double()
        #
        # self.bond_length_conv = nn.Sequential(
        #     EncoderConvBlock(self.bond_length, 32),
        #     nn.Tanh(),
        #     EncoderConvBlock(32, 12),
        #     nn.Tanh(),
        # ).double()
        #####

        # For Identity Function
        # self.decoder = nn.Sequential(
        #         nn.Linear(self.after_decomposition_size, self.after_decomposition_size, bias=True),
        #         # nn.LeakyReLU(),
        #         # nn.Linear(self.after_decomposition_size, self.after_decomposition_size, bias=True),
        #         # nn.LeakyReLU(),
        #         # nn.Linear(self.after_decomposition_size, self.after_decomposition_size, bias=True),
        #         # nn.LeakyReLU(),
        #         # nn.Linear(self.after_decomposition_size, self.after_decomposition_size, bias=True),
        # )

        ### FOR ENCODER BLOCK 64 WITH SKIP CONNECTION THESE TWO ARE IMPORTANT

        ####### 0.3 Model Architecture
        # Refer to https://docs.google.com/presentation/d/108BtzY6V326gqdxCST71rjD37zmHl9wsrWfiC_LQ2RY/edit#slide=id.g2c4e6a2ebe4_0_23
        # for architecture image
        if self.cyclic:
            input_size = self.after_decomposition_size * self.torsion
            self.encoder_skip_create = EncoderBlock64(
                num_layers = 1,
                # start_size = self.after_decomposition_size*self.torsion,
                start_size = 64, ### This is for model 0.4 after conv1d
                end_size   = 64,
            ).double()
        else:
            input_size = self.after_decomposition_size
            self.encoder_skip_create = EncoderBlock64(
                num_layers = 1,
                start_size = input_size//32,
                end_size   = 64,
            ).double()

        # encoder 64 
        self.encoder_n_64 = EncoderBlockList(
            num_layers = self.encoder_layers,
            input_size = 64,
            output_size = 64,
            dropout = self.dropout,
        ).double()

        self.decreasing_encoder = nn.Sequential(
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16,8),
            nn.Tanh(),
            nn.Linear(8, self.latent_space_dim),
        ).double()

        ##### DECODER
        self.decoder_4_64_skip = EncoderBlock64(
            num_layers = 1,
            start_size = self.latent_space_dim,
            end_size   = 64,
        ).double()

        self.decoder_n_64 = EncoderBlockList(
            num_layers = self.decoder_layers,
            input_size = 64,
            output_size = 64,
            dropout = self.dropout,
        ).double()

        self.decoder = DecoderLayerUnitCircle(
            input_size = 64,
            output_size = self.angle_out_size,
        ).double()
        ####### 0.3 Model Architecture END

        ####### 0.3.1 Includes above and pyramid encode for reducing space
        self.pyramid_encode = nn.Sequential(
            nn.Linear(input_size, input_size//8),
            nn.Tanh(),
            nn.Linear(input_size//8, input_size//32),
            nn.Tanh(),
        ).double()

        ###### END

        ###### 0.4 Model Conv1d
        self.convPE = ConvPEInput(
            channel_in = self.after_decomposition_size,
            channel_out = self.after_decomposition_size//4,
            d_model = self.torsion,
            stride = 1,
            padding = 0,
            kernel_size = self.torsion//2,
        ).double()

        ###### END

        # self.energy_encode = EncoderBlock64(
        #     num_layers = 1,
        #     start_size = self.energy_n,
        #     end_size = 1,
        # ).double()
        #
        # self.energy_decode = EncoderBlock64(
        #     num_layers = 1,
        #     start_size = self.latent_space_dim,
        #     end_size = self.energy_n,
        # ).double()
        #
        # self.output_energy = nn.Sequential(
        #     nn.Linear(self.energy_n, self.energy_n, bias=False),
        #     nn.BatchNorm1d(self.energy_n),
        #     nn.LeakyReLU(),
        #     nn.Linear(self.energy_n, self.energy_n, bias=False),
        # ).double()

        # self.encoder = EncoderBlock(
        #     num_layers = self.encoder_layers,
        #     input_size = self.after_decomposition_size,
        #     middle_size = 32,
        #     output_size = self.latent_space_dim,
        #     dropout = self.dropout,
        # ).double()

        self.latent = ConformationalLatentSpace(latent_space_dim=self.latent_space_dim).double()

        # self.decoder = DecoderBlock(
        #     num_layers = self.decoder_layers,
        #     input_size = self.latent_space_dim,
        #     output_size = self.angle_out_size,
        #     dropout = self.dropout,
        # ).double()
        # self.decoder = DecoderLayerUnitCircle(
        #     input_size = self.latent_space_dim,
        #     output_size = self.angle_out_size,
        # ).double()



        self.output = ConformationalOutputBlock(
            protein_len         = self.protein_len,
            normalize           = self.normalize,
            in_size_linear      = self.linear_out_size,
            in_size_angle       = self.angle_out_size,
            output_size         = self.input_size,
            torsion_main_n      = self.torsion_main_n,
            torsion_side_n      = self.torsion_side_n,
            bond_angle_main_n   = self.bond_angle_main_n,
            bond_angle_side_n   = self.bond_angle_side_n,
            bond_length_main_n  = self.bond_length_main_n,
            bond_length_side_n  = self.bond_length_side_n,
            jump_n              = self.jump_n,
            energy_n            = self.energy_n,
        ).double()

        self.embedding_dim = PositionalEncoding(
            device = self.device,
            seq_len = self.torsion+1,
            d_model = 1,
        ).double()

        self.cyclic_embed = CyclicPE(
            device = self.device,
            seq_len = self.torsion,
            d_model = 1,
        ).double()

        self.linear_conv_to_encoder = nn.Linear(self.torsion*2, 64).double()

    def forward(self, x: torch.Tensor, ):
        """Train and Test Forward Method"""
        # Grab dimensions
        B, D = x.shape

        # Extract out initial decomposed input for MSE Loss calc
        initial_input = self.input_single_decomp(x.clone().detach())[0]

        ############## CONVOLUTION
        # decompose for conv
        # torsion, bond_angle, bond_length, energy = self.input_decomposition(x)

        # Convolution layers 
        # torsion_conv = self.torsion_conv(torsion)
        # bond_angle_conv = self.bond_angle_conv(bond_angle)
        # bond_length_conv = self.bond_length_conv(bond_length)
        #
        # x = torch.cat((torsion_conv, bond_angle_conv), dim=-1)
        # x = torch.cat((x, bond_length_conv), dim=-1)
        # x = torch.cat((x, energy), dim=-1)
        ##############

        # Add noise for training
        if self.training:
            noise = torch.randn_like(x, device="cuda" if torch.cuda.is_available() else "cpu") * self.noise_std
            x = x + noise

        # decompose for non conv
        x, energy = self.input_decomposition(x)
        x_skip = x.clone()

        # Cyclic PE layer
        if self.cyclic:
            # Compute Embeddings
            embeddings = self.cyclic_embed().repeat_interleave(self.decomposition_n, dim=-2).unsqueeze(0).repeat(B, 1, 1, 1)
            B, L, LD, _ = embeddings.shape
            embeddings = embeddings.reshape(B, L, -1)
            # grab sin/cos (N, LD) 
            sin_x, cos_x = x[:, :LD], x[:, LD:]
            # reshape x (N, LD) -> (N, L, LD)
            sin_x, cos_x = sin_x.unsqueeze(1).repeat_interleave(L, dim=1), cos_x.unsqueeze(1).repeat_interleave(L, dim=1)
            sin_x_pe, cos_x_pe = sin_x + embeddings, cos_x + embeddings
            # This just added for reducing param num
            # sin_x_sum, cos_x_sum = sin_x_pe.sum(dim=1), cos_x_pe.sum(dim=1)
            x = torch.cat(
                (
                    # sin_x_sum.reshape(B, -1),
                    # cos_x_sum.reshape(B, -1),
                    sin_x_pe.reshape(B, L, -1),
                    cos_x_pe.reshape(B, L, -1),
                ),
                dim=-1,
            )
        # Positional Encoding Linear Layers
        elif self.pe_linear:
            # positional encodings
            positions = torch.arange(1,self.torsion+1, device=self.device)
            embeddings = self.embedding_dim(positions).repeat_interleave(self.decomposition_n)
            embeddings = torch.cat((embeddings, embeddings), dim=-1)
            x = torch.add(x, embeddings)
        # no positional Encodings
        else:
            pass
            

        # if self.pe_linear:
            # This is for linear peptides/proteins. This should be a reduction in size
        # x = self.pyramid_encode(x)
        x = self.convPE(x.permute(0, 2, 1))
        
        # skip connection from intial decomposition to after PE and Conv.
        x = self.linear_conv_to_encoder(torch.add(x, x_skip))

        # Encoder for angles Unit Circle + PE Embeddings
        skip = self.encoder_skip_create(x)

        encoder_4_64_x = self.encoder_n_64(skip)

        encoder_to_latent = self.decreasing_encoder(
            torch.add(encoder_4_64_x, skip)
        )

        # Encode the energy information 
        # energy = self.energy_encode(energy)

        # sum together the energy and angle information
        # x = self.latent(torch.add(x, energy))
        # x = self.latent(x)
        x = self.latent(encoder_to_latent)

        # compute latent rmsd
        latent_clone = x.clone()
        latent_euclid_dist = self.compute_euclidean_distance(latent_clone)

        # Decode the information back to post-decomposition input size
        post_latent_skip = self.decoder_4_64_skip(x)

        post_latent = self.decoder_n_64(post_latent_skip)

        out = self.output(self.decoder(torch.add(post_latent, post_latent_skip)))

        # Decode the energy information out
        # energy_out = self.output_energy(self.energy_decode(x))

        # concat together 
        # out = torch.cat((out, energy_out), dim=-1)

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



#### MODEL SPECIFIC STOP CLASS

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False




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
