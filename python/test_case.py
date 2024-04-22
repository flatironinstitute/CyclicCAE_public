import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import argparse

# from model_utils import EncoderBlockList, EncoderBlock64, DecoderLayerUnitCircle, ConvPEInput, ConformationalLatentSpace

##### MODULES
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

        return torsion_decompose, bond_angle_decompose, bond_length, energy  #, out

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

    def forward(self, x: torch.Tensor):
        """ Forward of Latent Space Block
        with a Skip Connnection for BackProp

        PARAMS
        ------
            x: torch.Tensor
                Reduced input vector (B, latent_space_dim)
        """
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

class EncoderLayer64_leaky(nn.Module):
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
        self.leakyrelu = nn.LeakyReLU()
        # self.tanh = nn.Tanh()
        # self.relu = nn.ReLU()
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
            return self.leakyrelu(self.linear(x))
        else:
            return self.leakyrelu(self.dropout(self.linear(x)))
           # return self.tanh(self.batch_norm(self.dropout(self.linear(x))))

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
        self.tanh = nn.Tanh()
        # self.relu = nn.ReLU()
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
            return self.tanh(self.linear(x))
        else:
            return self.tanh(self.dropout(self.linear(x)))
           # return self.tanh(self.batch_norm(self.dropout(self.linear(x))))

class EncoderBlock64_leaky(nn.Module):
    def __init__(
        self,
        num_layers: int = 3,
        start_size: int = 512,
        d_model: int = 64,
        end_size: int = 3,
        dropout: float = 0.0,
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
                EncoderLayer64_leaky(
                    input_size = self.layer_size[l],
                    output_size = self.out_size[l],
                    dropout = dropout,
                ) for l in range(0, self.num_layers)
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
            if i+1 != self.num_layers:
                x = layer(x, False)
            else:
                x = layer(x, True)
        
        return x

class EncoderBlock64(nn.Module):
    def __init__(
        self,
        num_layers: int = 3,
        start_size: int = 512,
        d_model: int = 64,
        end_size: int = 3,
        dropout: float = 0.0,
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
                    dropout = dropout,
                ) for l in range(0, self.num_layers)
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
            if i+1 != self.num_layers:
                x = layer(x, False)
            else:
                x = layer(x, True)
        
        return x

# class EncoderBlock64(nn.Module):
#     def __init__(
#         self,
#         num_layers: int = 3,
#         start_size: int = 512,
#         d_model: int = 64,
#         end_size: int = 3,
#         dropout: float = 0.0,
#         ):
#         """ A General Encoder Block 
#
#         PARAMS
#         ------
#             num_layers: int
#                 Number of EncoderLayers to include within the model
#
#             start_size: int
#                 The beginning last dimension size of the input vector 
#                 going into the block
#
#             end_size: int
#                 The ending last dimension size of the output vector going
#                 out of the block
#         """
#         super().__init__()
#         # Save input parameters
#         self.num_layers = num_layers
#         self.start_size = start_size
#         self.end_size = end_size
#
#         self.layer_size = [start_size] + [d_model]*(num_layers-1)
#         self.out_size = [d_model] * (num_layers-1) + [end_size]
#
#
#         # Make list of encoder layers that make up the encoder block
#         self.encoder_layers = nn.ModuleList(
#             [
#                 EncoderLayer64(
#                     input_size = self.layer_size[l],
#                     output_size = self.out_size[l],
#                     dropout = dropout,
#                 ) for l in range(0, self.num_layers)
#             ]
#         )
#
#     def forward(self, x: torch.Tensor):
#         """ Forward of Encoder Block
#
#         PARAMS
#         ------
#             x: torch.Tensor
#                 The DoF input vector of size (B, N) where B is
#                 the batch size and N is the number of elements
#         """
#         # Inital ski
#         skip = x.clone()
#
#         for i, layer in enumerate(self.encoder_layers):
#             # If last layer then pass True
#             if i+1 == self.num_layers:
#                 x = layer(torch.add(x, skip), True)
#             elif (i+1 != self.num_layers) or (i+1 != 1):
#                 skip_connect = x.clone()
#                 x = layer(x, True)
#             else:
#                 x = layer(torch.add(x, skip_connect))
#         return x


class TestCase1(nn.Module):
    def __init__(
        self,
        protein_len: int,
        output_size: int,
        input_size: int,
        normalize: bool,
        decoder_layers: int,
        encoder_layers: int,
        latent_space_dim: int,
        dof_residue_metadata: torch.Tensor,
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
        convolution_layer: bool = False,
        ):
        super().__init__()
        """Only an Encoder/Decoder Block"""
        # Grab needed values for forward
        self.noise_std = noise_std
        self.dof_metadata = dof_residue_metadata

        # encoder first set
        self.encoder_one = EncoderBlock64(
            num_layers = 1,
            start_size = torsion_main_n*2,
            end_size = 64,
            dropout = dropout,
        )
        
        # encoder block
        self.encoder = EncoderBlock64(
            num_layers = encoder_layers,
            start_size = 64,
            # d_model = 232,
            end_size = 64,
            dropout = dropout,
        )
        self.encoder_n2 = EncoderBlock64(
            num_layers = encoder_layers,
            start_size = 64,
            end_size = 64,
            dropout = dropout,
        )
        self.encoder_n3 = EncoderBlock64(
            num_layers = encoder_layers,
            start_size = 64,
            # d_model = 64,
            end_size = 64,
            dropout = dropout,
        )

        # Decreasing encoder block
        self.encoder_pyramid = nn.Sequential(
            # layer 1
            nn.Linear(64, 32),
            nn.Tanh(),
            # layer 2
            nn.Linear(32, 16),
            nn.Tanh(),
            # layer 3
            nn.Linear(16, 8),
            nn.Tanh(),
            # layer 4
            nn.Linear(8, latent_space_dim),
            nn.Tanh(),
        )

        ##### DECODER PORTION
        self.decoder_skip = EncoderBlock64(
            num_layers = 1,
            start_size = latent_space_dim,
            end_size = 64,
            dropout = dropout,
        )

        self.decoder_n = EncoderBlock64(
            num_layers = decoder_layers,
            start_size = 64,
            # d_model = 120,
            end_size = 64,
            dropout = dropout,
        )
        self.decoder_n2 = EncoderBlock64(
            num_layers = decoder_layers,
            start_size = 64,
            # d_model = 120,
            end_size = 64,
            dropout = dropout,
        )
        self.decoder_n3 = EncoderBlock64(
            num_layers = decoder_layers,
            start_size = 64,
            # d_model = 116,
            end_size = 64,
            dropout = dropout,
        )

        self.final_layer = nn.Sequential(
            nn.Linear(64, torsion_main_n*2),
            # nn.Tanh(),
        )

        self.input_decompose = ConformationalInputDecomposition(
            protein_len = protein_len,
            torsion_main_n = torsion_main_n,
            torsion_side_n = torsion_side_n,
            bond_angle_main_n = bond_angle_main_n,
            bond_angle_side_n = bond_angle_side_n,
            bond_length_main_n = bond_length_main_n,
            bond_length_side_n = bond_length_side_n,
            jump_n = jump_n,
            energy_n = energy_n,
            decomposition_n = 1,
        )

    def forward(self, x: torch.Tensor):
        """Forward"""
        # Add noise if we are training and not validating
        if self.training:
            noise = torch.normal(torch.zeros(x.shape), self.noise_std).to("cuda" if torch.cuda.is_available() else "cpu")
            x = x + noise
            x = torch.clamp(x, -180, 180)

        # decompose
        x, energy = self.input_decompose(x)
        initial_input = x.clone().detach()

        ##### ENCODER
        # Convert 116 -> 64 for encoder block training
        x = self.encoder_one(x)
        skip_one = x.clone()

        # First Encoder block 64 -> 64
        x = self.encoder(x)
        skip_n1 = x.clone()

        # Second Encoder Block 64 -> 64 
        # x = self.encoder_n2(torch.add(x, skip_one))
        x = self.encoder_n2(x)
        skip_n2 = x.clone() 

        # Third Encoder Block 64 -> 64
        # x = self.encoder_n3(torch.add(x, skip_n1))
        x = self.encoder_n3(x)

        # Pyramid Encoder 64 -> 32 -> 16 -> 8 -> 3
        # x = self.encoder_pyramid(torch.add(x, skip_n2))
        x = self.encoder_pyramid(x)
        ##### ENCODER END



        ##### LATENT SPACE
        # Copy and compute distance for latent space
        latent_space = x.clone()
        latent_euclid_dist = self.compute_euclidean_distance(latent_space)
        ##### LATENT SPACE END



        ##### DECODER
        # Decoder latent to decoder blocks size 3 -> 64
        x = self.decoder_skip(x)
        decode_skip1 = x.clone()

        # First Decoder Block 64 -> 64
        x = self.decoder_n(x)
        dskip_n1 = x.clone()

        # Second Decoder Block 64 -> 64
        # x = self.decoder_n2(torch.add(x, decode_skip1))
        x = self.decoder_n2(x)
        dskip_n2 = x.clone()

        # Third Decoder Block 64 -> 64
        # x = self.decoder_n3(torch.add(x, dskip_n1))
        x = self.decoder_n3(x)

        # FINAL decoder layer 64 -> 116 out size
        # out = self.final_layer(torch.add(x, dskip_n2))
        out = self.final_layer(x) # allo of these are for testing without residual connection
        ##### DECODER END

        return out, latent_euclid_dist, latent_space, initial_input 

    def compute_euclidean_distance(self, xyz_mat: torch.Tensor):
        """Compute the latent space RMSD"""
        # compute pairwise diff
        differences = xyz_mat.unsqueeze(1) - xyz_mat.unsqueeze(0)

        # squared_distances along last dim
        distances = torch.sum( torch.pow( differences, 2), dim=-1)
        return distances

class ConvCycTest2d(nn.Module):
    def __init__(
        self,
        channel_in: int,
        channel_out: int,
        out_size: int,
        kernel_seq: int,
        kernel_feats: int,
        kernel_max: int,
        ):
        super().__init__()
        """Apply Convolution to the cyclic peptides"""
        # Temporal Convolution
        self.conv_temporal1 = nn.Conv2d(channel_in, channel_out, kernel_size=(kernel_seq, kernel_feats), stride=1, padding=1)
        # self.conv_temporal2 = nn.Conv2d(channel_in, channel_out, kernel_size=(kernel_seq, kernel_feats), stride=1, padding=1)

        output_size = out_size // channel_out
        # pooling output
        # self.average_pooling = nn.AdaptiveAvgPool1d(output_size=int(output_size))
        self.adaptive_maxpool = nn.AdaptiveMaxPool2d(8)

        # Layer Norm
        # self.layernorm = nn.LayerNorm([channel_out, output_size])

    def forward(self, x: torch.Tensor):
        """Forward Cyclic Conv"""
        B, C, L, D = x.shape

        # padding size extract for cyclic convolution
        padding_cyc_size = self.conv_temporal1.kernel_size[0] - 1

        # cyclic padding
        x = torch.cat(
            (
                x[:, :, -padding_cyc_size:, :],
                x,
                x[:, :, :-padding_cyc_size, :],
            ),
            dim = -2,
        )

        # Apply convolution
        # print('x', x.shape)
        # x = F.relu(self.batch_temporal1(self.conv_temporal1(x)))
        x = F.tanh(self.conv_temporal1(x))
        # print('conv1', x.shape)
        # x = F.tanh(self.conv_temporal2(x))
        # print('conv2', x.shape)
        # x = self.average_pooling(x)
        x = self.adaptive_maxpool(x)
        # print('pool', x.shape)
        # x = self.layernorm(x)
        return x.view(B, -1)

class ConvCycDecoder(nn.Module):
    def __init__(
        self,
        channel_in: int,
        channel_out: int,
        out_size: int,
        kernel_size1: int,
        kernel_size2: int,
        ):
        super().__init__()
        """Apply Convolution to the cyclic peptides"""

        self.channel_in = channel_in

        # Convolution
        self.conv1 = nn.ConvTranspose1d(channel_in, channel_in//2, kernel_size=kernel_size1, stride=2, padding=(kernel_size1-1)//2, output_padding=0)
        self.conv2 = nn.ConvTranspose1d(channel_in//2, channel_out, kernel_size=kernel_size2, stride=2, padding=(kernel_size2-1)//2, output_padding=1)

        # pooling output
        # self.maxpool = nn.MaxPool1d(2)
        
        #activation
        # self.glu = nn.GLU(dim=1)

        outsize = channel_in*36
        # Convert to 64
        self.linear = nn.Sequential(
            nn.Linear(64, outsize//4),
            nn.Tanh(),
            nn.Linear(outsize//4, outsize//2),
            nn.Tanh(),
            nn.Linear(outsize//2, outsize),
        )

    def forward(self, x: torch.Tensor):
        """Forward Cyclic Conv"""
        B, D = x.shape
        # print("decoder x", x.shape)
        # Apply convolution
        x = F.tanh(self.linear(x))
        # print('linear', x.shape)
        # reshape x
        x = x.view(B, self.channel_in, -1)
        # print('reshape', x.shape)
        x = F.tanh(self.conv1(x))
        # print("conv1", x.shape)
        # print("conv1", x.shape)
        x = F.tanh(self.conv2(x))
        # print("conv2", x.shape)
        # print("conv2", x.shape)
        # print("linear", x.shape)
        return x

class ConvCycEncoder(nn.Module):
    def __init__(
        self,
        channel_in: int,
        channel_out: int,
        out_size: int,
        kernel_size1: int,
        kernel_size2: int,
        ):
        super().__init__()
        """Apply Convolution to the cyclic peptides"""
        # Convolution
        self.conv1 = nn.Conv1d(channel_in, channel_out//2, kernel_size=kernel_size1, stride=1, padding=kernel_size1//2)
        self.conv2 = nn.Conv1d(channel_out//2, channel_out, kernel_size=kernel_size2, stride=1, padding=kernel_size2//2)

        # pooling output
        self.maxpool = nn.MaxPool1d(2)
        
        #activation
        # self.glu = nn.GLU(dim=1)

        outsize = channel_out*36
        # Convert to 64
        self.linear = nn.Sequential(
            nn.Linear(outsize, (outsize)//2),
            nn.Tanh(),
            nn.Linear(outsize//2, outsize//4),
            nn.Tanh(),
            nn.Linear(outsize//4, 64),
        )

    def forward(self, x: torch.Tensor):
        """Forward Cyclic Conv"""
        B, L, D = x.shape

        # padding size extract for cyclic convolution
        padding_cyc_size = self.conv1.kernel_size[0] - 1

        # cyclic padding
        x = torch.cat(
            (
                x[:, :, -padding_cyc_size:],
                x,
                x[:, :, :-padding_cyc_size],
            ),
            dim = -1,
        )

        # Apply convolution
        # print("Encoder x", x.shape)
        x = self.maxpool(F.tanh(self.conv1(x)))
        # print("conv1", x.shape)
        x = self.maxpool(F.tanh(self.conv2(x)))
        # print("conv2", x.shape)
        x = F.tanh(self.linear(x.view(B, -1)))
        # print("linear", x.shape)
        return x

class ConvCase1(nn.Module):
    def __init__(
        self,
        protein_len: int,
        output_size: int,
        input_size: int,
        normalize: bool,
        decoder_layers: int,
        encoder_layers: int,
        latent_space_dim: int,
        dof_residue_metadata: torch.Tensor,
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
        convolution_layer: bool = False,
        ):
        super().__init__()
        """Only an Encoder/Decoder Block"""
        # Grab needed values for forward
        self.noise_std = noise_std
        self.dof_residue_metadata = dof_residue_metadata
        self.device = device
        self.convolution = convolution_layer
        self.torsion_main_n = torsion_main_n

        # grab atom Length residue length
        F = self.dof_residue_metadata[:, 1].max().long()


        in_channel = 2
        out_channel = 32
        # Convolution 
        self.conv_cyc_encoder = ConvCycEncoder(
            channel_in = in_channel,
            channel_out = out_channel,
            kernel_size1 = 9,
            kernel_size2 = 18,
            out_size = 64,
        )
        self.conv_cyc_decoder = ConvCycDecoder(
            channel_in = out_channel,
            channel_out = in_channel,
            kernel_size1 = 18,
            kernel_size2 = 9,
            out_size = (8*9)*2,
        )
        # self.conv_cyc2d = ConvCycTest2d(
        #     channel_in = 2,
        #     channel_out = 1,
        #     kernel_seq = 5,
        #     kernel_feats = 3,
        #     kernel_max = 3,
        #     out_size = 64//2,
        # )

        # encoder first set
        self.encoder_one = EncoderBlock64(
            num_layers = 1,
            start_size = torsion_main_n*2,
            end_size = 64,
            dropout = dropout,
        )
        
        # encoder block
        self.encoder = EncoderBlock64(
            num_layers = encoder_layers,
            start_size = 64,
            end_size = 64,
            dropout = dropout,
        )
        self.encoder_n2 = EncoderBlock64(
            num_layers = encoder_layers,
            start_size = 64,
            end_size = 64,
            dropout = dropout,
        )
        self.encoder_n3 = EncoderBlock64(
            num_layers = encoder_layers,
            start_size = 64,
            end_size = 64,
            dropout = dropout,
        )

        # Decreasing encoder block
        self.encoder_pyramid = nn.Sequential(
            # layer 1
            nn.Linear(64, 32),
            nn.Tanh(),
            # layer 2
            nn.Linear(32, 16),
            nn.Tanh(),
            # layer 3
            nn.Linear(16, 8),
            nn.Tanh(),
            # layer 4
            nn.Linear(8, latent_space_dim),
            nn.Tanh(),
        )

        ##### DECODER PORTION

        # Decreasing encoder block
        self.decoder_pyramid = nn.Sequential(
            # layer 1
            nn.Linear(latent_space_dim, 8),
            nn.Tanh(),
            # layer 2
            nn.Linear(8, 16),
            nn.Tanh(),
            # layer 3
            nn.Linear(16, 32),
            nn.Tanh(),
            # layer 4
            nn.Linear(32, 64),
            nn.Tanh(),
        )
        # self.decoder_skip = EncoderBlock64(
        #     num_layers = 1,
        #     start_size = 
        #     end_size = 64,
        #     dropout = dropout,
        # )

        self.decoder_n = EncoderBlock64(
            num_layers = decoder_layers,
            start_size = 64,
            end_size = 64,
            dropout = dropout,
        )
        self.decoder_n2 = EncoderBlock64(
            num_layers = decoder_layers,
            start_size = 64,
            end_size = 64,
            dropout = dropout,
        )
        self.decoder_n3 = EncoderBlock64(
            num_layers = decoder_layers,
            start_size = 64,
            end_size = 64,
            dropout = dropout,
        )

        self.final_layer = nn.Sequential(
            nn.Linear((8*9)*4, torsion_main_n*2),
        )

        self.input_decompose = ConformationalInputDecomposition(
            protein_len = protein_len,
            torsion_main_n = torsion_main_n,
            torsion_side_n = torsion_side_n,
            bond_angle_main_n = bond_angle_main_n,
            bond_angle_side_n = bond_angle_side_n,
            bond_length_main_n = bond_length_main_n,
            bond_length_side_n = bond_length_side_n,
            jump_n = jump_n,
            energy_n = energy_n,
            decomposition_n = 1,
        )

    def forward(self, x: torch.Tensor, noise: bool):
        """Forward"""
        # Add noise if we are training and not validating
        if self.training and noise:
            noise = torch.normal(torch.zeros(x.shape), self.noise_std).to("cuda" if torch.cuda.is_available() else "cpu")
            x = x + noise
            x = torch.clamp(x, -180, 180)

        # decompose
        x, _, _,  energy = self.input_decompose(x)
        initial_input = x.clone().detach()

        ##### ENCODER
        # If convolution is true then reformat and apply convolution
        if self.convolution:
            sin_x = self.reformat_convolution2(x[:, :self.torsion_main_n], self.dof_residue_metadata)
            cos_x = self.reformat_convolution2(x[:, self.torsion_main_n:], self.dof_residue_metadata)
            x_combo = torch.cat(
                (
                    sin_x.unsqueeze(-1),
                    cos_x.unsqueeze(-1),
                ),
                dim=-1,
            )
            # reshape from (N, L, D, 2) -> (N, 2, D*L)
            B, L, D, C = x_combo.shape
            x = self.conv_cyc_encoder(x_combo.reshape(B, C, D*L)) # 1D
            # x = self.conv_cyc2d(x_combo.permute(0, 3, 1, 2)) # 2D
            # sin_conv = self.conv_cyc(sin_x.permute(0, 2, 1))
            # cos_conv = self.conv_cyc(cos_x.permute(0, 2, 1))
            # x = torch.cat((sin_conv, cos_conv), dim=-1)
            # x = torch.add(sin_conv, cos_conv)
        else:
        # Convert 116 -> 64 for encoder block training
            x = self.encoder_one(x)
        skip_one = x.clone()

        # First Encoder block 64 -> 64
        x = self.encoder(x)
        skip_n1 = x.clone()

        # Second Encoder Block 64 -> 64 
        x = self.encoder_n2(torch.add(x, skip_one))
        skip_n2 = x.clone() 

        # Third Encoder Block 64 -> 64
        x = self.encoder_n3(torch.add(x, skip_n1))

        # Pyramid Encoder 64 -> 32 -> 16 -> 8 -> 3
        x = self.encoder_pyramid(torch.add(x, skip_n2))
        ##### ENCODER END



        ##### LATENT SPACE
        # Copy and compute distance for latent space
        latent_space = x.clone()
        latent_euclid_dist = self.compute_euclidean_distance(latent_space)
        ##### LATENT SPACE END



        ##### DECODER
        # Decoder latent to decoder blocks size 3 -> 64
        x = self.decoder_pyramid(x)
        decode_skip1 = x.clone()

        # First Decoder Block 64 -> 64
        x = self.decoder_n(x)
        dskip_n1 = x.clone()

        # Second Decoder Block 64 -> 64
        x = self.decoder_n2(torch.add(x, decode_skip1))
        dskip_n2 = x.clone()

        # Third Decoder Block 64 -> 64
        x = self.decoder_n3(torch.add(x, dskip_n1))

        # FINAL decoder layer 64 -> 116 out size
        if self.convolution:
            B, D = x.shape
            x = self.conv_cyc_decoder(x).view(B, -1)
        # out = self.final_layer(torch.add(x, dskip_n2))
        out = self.final_layer(x)
        ##### DECODER END

        return out, latent_euclid_dist, latent_space, initial_input 

    def compute_euclidean_distance(self, xyz_mat: torch.Tensor):
        """Compute the latent space RMSD"""
        # compute pairwise diff
        differences = xyz_mat.unsqueeze(1) - xyz_mat.unsqueeze(0)

        # squared_distances along last dim
        distances = torch.sum( torch.pow( differences, 2), dim=-1)
        return distances

    def reformat_convolution2(self, x: torch.Tensor, format: torch.Tensor):
        """Format x for cyclic convolution"""
        # grab shape
        B, D = x.shape
        format = format.long()
        # grab residue and index amount
        L = format[:D, 0].max()
        F = format[:D, 1].max()

        # generate holding tensor
        atom_tensor = torch.zeros(B, L, F, dtype=torch.float64, device=self.device)

        # grab indices
        residue_indices = format[:D, 0] - 1
        atom_indicies = format[:D, 1] - 1

        # create a gride for batch indices
        batch_indices = torch.arange(B, device=self.device)[:, None].expand(B, D)

        # advance indexing to fill in atom tensor
        atom_tensor[batch_indices, residue_indices, atom_indicies] = x

        return atom_tensor

    def reformat_convolution(self, x: torch.Tensor, format: torch.Tensor):
        """Format x for cyclic convolution"""
        # grab shape
        B, D = x.shape
        # grab residue and index amount
        L = format[:, 0].max().to(torch.long)
        F = format[:, 1].max().to(torch.long) - 1

        # input extract
        cut_points = x[:, [F, -1]]
        indices = torch.arange(x.size(-1), device=self.device)
        mask = (indices != F) & (indices != x.size(-1)-1)
        main_atoms = x[:, indices[mask]]

        # generate holding tensor
        atom_tensor = torch.zeros(B, L, F, dtype=torch.float64, device=self.device)

        # reshaep main atoms and fill atom tensor
        atom_tensor[:, :, :-1] = main_atoms.reshape(B, L, -1)

        # fill cutpoints
        atom_tensor[:,  0, -1] = cut_points[:, 0]
        atom_tensor[:, -1, -1] = cut_points[:,-1]

        return atom_tensor


class ConvCase3(nn.Module):
    def __init__(
        self,
        protein_len: int,
        output_size: int,
        input_size: int,
        normalize: bool,
        decoder_layers: int,
        encoder_layers: int,
        latent_space_dim: int,
        dof_residue_metadata: torch.Tensor,
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
        convolution_layer: bool = False,
        ):
        super().__init__()
        """Only an Encoder/Decoder Block"""
        # Grab needed values for forward
        self.noise_std = noise_std
        self.dof_residue_metadata = dof_residue_metadata
        self.device = device
        self.convolution = convolution_layer
        self.torsion_main_n = torsion_main_n
        self.bond_angle_main_n = bond_angle_main_n
        self.bond_length_main_n = bond_length_main_n

        # grab atom Length residue length
        F = self.dof_residue_metadata[:, 1].max().long()


        in_channel = 5
        out_channel = 32
        # Convolution 
        self.conv_cyc_encoder = ConvCycEncoder(
            channel_in = in_channel,
            channel_out = out_channel,
            kernel_size1 = 9,
            kernel_size2 = 18,
            out_size = 64,
        )
        self.conv_cyc_decoder = ConvCycDecoder(
            channel_in = out_channel,
            channel_out = in_channel,
            kernel_size1 = 18,
            kernel_size2 = 9,
            out_size = (8*9)*2,
        )

        self.bond_length_trans = nn.Sequential(
            nn.Linear(bond_length_main_n, F*8),
            nn.LeakyReLU(),
        )
        # self.conv_cyc2d = ConvCycTest2d(
        #     channel_in = 2,
        #     channel_out = 1,
        #     kernel_seq = 5,
        #     kernel_feats = 3,
        #     kernel_max = 3,
        #     out_size = 64//2,
        # )

        # encoder first set
        self.encoder_one = EncoderBlock64(
            num_layers = 1,
            start_size = torsion_main_n*2,
            end_size = 64,
            dropout = dropout,
        )
        
        # encoder block
        self.encoder = EncoderBlock64(
            num_layers = encoder_layers,
            start_size = 64,
            end_size = 64,
            dropout = dropout,
        )
        self.encoder_n2 = EncoderBlock64(
            num_layers = encoder_layers,
            start_size = 64,
            end_size = 64,
            dropout = dropout,
        )
        self.encoder_n3 = EncoderBlock64(
            num_layers = encoder_layers,
            start_size = 64,
            end_size = 64,
            dropout = dropout,
        )

        # Decreasing encoder block
        self.encoder_pyramid = nn.Sequential(
            # layer 1
            nn.Linear(64, 32),
            nn.Tanh(),
            # layer 2
            nn.Linear(32, 16),
            nn.Tanh(),
            # layer 3
            nn.Linear(16, 8),
            nn.Tanh(),
            # layer 4
            nn.Linear(8, latent_space_dim),
            nn.Tanh(),
        )

        ##### DECODER PORTION

        # Decreasing encoder block
        self.decoder_pyramid = nn.Sequential(
            # layer 1
            nn.Linear(latent_space_dim, 8),
            nn.Tanh(),
            # layer 2
            nn.Linear(8, 16),
            nn.Tanh(),
            # layer 3
            nn.Linear(16, 32),
            nn.Tanh(),
            # layer 4
            nn.Linear(32, 64),
            nn.Tanh(),
        )
        # self.decoder_skip = EncoderBlock64(
        #     num_layers = 1,
        #     start_size = 
        #     end_size = 64,
        #     dropout = dropout,
        # )

        self.decoder_n = EncoderBlock64(
            num_layers = decoder_layers,
            start_size = 64,
            end_size = 64,
            dropout = dropout,
        )
        self.decoder_n2 = EncoderBlock64(
            num_layers = decoder_layers,
            start_size = 64,
            end_size = 64,
            dropout = dropout,
        )
        self.decoder_n3 = EncoderBlock64(
            num_layers = decoder_layers,
            start_size = 64,
            end_size = 64,
            dropout = dropout,
        )

        self.head_torsion_angle = nn.Sequential(
            nn.Linear(720, 720//2),
            nn.Tanh(),
            nn.Linear(720//2, 720//4),
            nn.Tanh(),
            nn.Linear(720//4, torsion_main_n*2),
        )

        self.head_bond_length = nn.Sequential(
            nn.Linear(720, bond_length_main_n),
        )

        self.head_bond_angle = nn.Sequential(
            nn.Linear(720, bond_angle_main_n*2),
        )

        self.input_decompose = ConformationalInputDecomposition(
            protein_len = protein_len,
            torsion_main_n = torsion_main_n,
            torsion_side_n = torsion_side_n,
            bond_angle_main_n = bond_angle_main_n,
            bond_angle_side_n = bond_angle_side_n,
            bond_length_main_n = bond_length_main_n,
            bond_length_side_n = bond_length_side_n,
            jump_n = jump_n,
            energy_n = energy_n,
            decomposition_n = 1,
        )

    def forward(self, x: torch.Tensor, noise: bool):
        """Forward"""
        # Add noise if we are training and not validating
        if self.training and noise:
            noise = torch.normal(torch.zeros(x.shape), self.noise_std).to("cuda" if torch.cuda.is_available() else "cpu")
            x = x + noise
            x = torch.clamp(x, -180, 180)

        # decompose angles/ return linear terms
        x, bond_angle, bond_length, energy = self.input_decompose(x)
        init_x = torch.cat((x, bond_angle, bond_length), dim=-1)
        initial_input = init_x.clone().detach()

        ##### ENCODER
        # If convolution is true then reformat and apply convolution
        if self.convolution:
            sin_x = self.reformat_convolution2(x[:, :self.torsion_main_n], self.dof_residue_metadata[:self.torsion_main_n,:])
            cos_x = self.reformat_convolution2(x[:, self.torsion_main_n:], self.dof_residue_metadata[:self.torsion_main_n,:])
            sin_x_bond = self.reformat_convolution2(bond_angle[:, :self.bond_angle_main_n], self.dof_residue_metadata[self.torsion_main_n:self.torsion_main_n + self.bond_angle_main_n, :])
            cos_x_bond = self.reformat_convolution2(bond_angle[:, self.bond_angle_main_n:], self.dof_residue_metadata[self.torsion_main_n+self.bond_angle_main_n : self.torsion_main_n+self.bond_angle_main_n+self.bond_length_main_n, :])
            bond_length_transform = self.bond_length_trans(bond_length)
            B, L, D = sin_x_bond.shape
            x_combo = torch.cat(
                (
                    sin_x.unsqueeze(-1),
                    cos_x.unsqueeze(-1),
                    sin_x_bond.unsqueeze(-1),
                    cos_x_bond.unsqueeze(-1),
                    bond_length_transform.reshape(B, L, D).unsqueeze(-1),
                ),
                dim=-1,
            )
            # reshape from (N, L, D, 2) -> (N, 2, D*L)
            B, L, D, C = x_combo.shape
            x = self.conv_cyc_encoder(x_combo.reshape(B, C, D*L)) # 1D
            # x = self.conv_cyc2d(x_combo.permute(0, 3, 1, 2)) # 2D
            # sin_conv = self.conv_cyc(sin_x.permute(0, 2, 1))
            # cos_conv = self.conv_cyc(cos_x.permute(0, 2, 1))
            # x = torch.cat((sin_conv, cos_conv), dim=-1)
            # x = torch.add(sin_conv, cos_conv)
        else:
        # Convert 116 -> 64 for encoder block training
            x = self.encoder_one(x)
        skip_one = x.clone()

        # First Encoder block 64 -> 64
        x = self.encoder(x)
        skip_n1 = x.clone()

        # Second Encoder Block 64 -> 64 
        x = self.encoder_n2(torch.add(x, skip_one))
        skip_n2 = x.clone() 

        # Third Encoder Block 64 -> 64
        x = self.encoder_n3(torch.add(x, skip_n1))

        # Pyramid Encoder 64 -> 32 -> 16 -> 8 -> 3
        x = self.encoder_pyramid(torch.add(x, skip_n2))
        ##### ENCODER END



        ##### LATENT SPACE
        # Copy and compute distance for latent space
        latent_space = x.clone()
        latent_euclid_dist = self.compute_euclidean_distance(latent_space)
        ##### LATENT SPACE END



        ##### DECODER
        # Decoder latent to decoder blocks size 3 -> 64
        x = self.decoder_pyramid(x)
        decode_skip1 = x.clone()

        # First Decoder Block 64 -> 64
        x = self.decoder_n(x)
        dskip_n1 = x.clone()

        # Second Decoder Block 64 -> 64
        x = self.decoder_n2(torch.add(x, decode_skip1))
        dskip_n2 = x.clone()

        # Third Decoder Block 64 -> 64
        x = self.decoder_n3(torch.add(x, dskip_n1))

        # FINAL decoder layer 64 -> 116 out size
        if self.convolution:
            B, D = x.shape
            x = self.conv_cyc_decoder(x).view(B, -1)
        # out = self.final_layer(torch.add(x, dskip_n2))
        torsion_angle_pred = self.head_torsion_angle(x)
        bond_angle_pred = self.head_bond_angle(x)
        bond_length_pred = self.head_bond_length(x)
        ##### DECODER END

        out = torch.cat(
            (
                torsion_angle_pred,
                bond_angle_pred,
                bond_length_pred,
            ),
            dim=-1
        )

        return out, latent_euclid_dist, latent_space, initial_input 

    def compute_euclidean_distance(self, xyz_mat: torch.Tensor):
        """Compute the latent space RMSD"""
        # compute pairwise diff
        differences = xyz_mat.unsqueeze(1) - xyz_mat.unsqueeze(0)

        # squared_distances along last dim
        distances = torch.sum( torch.pow( differences, 2), dim=-1)
        return distances

    def reformat_convolution2(self, x: torch.Tensor, format: torch.Tensor):
        """Format x for cyclic convolution"""
        # grab shape
        B, D = x.shape
        format = format.long()
        # grab residue and index amount
        # L = format[:D, 0].max()
        # F = format[:D, 1].max()
        L = format[:, 0].max() # since slicing format tensors this should work fine
        F = format[:, 1].max()

        # generate holding tensor
        atom_tensor = torch.zeros(B, L, F, dtype=torch.float64, device=self.device)

        # grab indices
        residue_indices = format[:D, 0] - 1
        atom_indicies = format[:D, 1] - 1

        # create a gride for batch indices
        batch_indices = torch.arange(B, device=self.device)[:, None].expand(B, D)

        # advance indexing to fill in atom tensor
        atom_tensor[batch_indices, residue_indices, atom_indicies] = x

        return atom_tensor

    def reformat_convolution(self, x: torch.Tensor, format: torch.Tensor):
        """Format x for cyclic convolution"""
        # grab shape
        B, D = x.shape
        # grab residue and index amount
        L = format[:, 0].max().to(torch.long)
        F = format[:, 1].max().to(torch.long) - 1

        # input extract
        cut_points = x[:, [F, -1]]
        indices = torch.arange(x.size(-1), device=self.device)
        mask = (indices != F) & (indices != x.size(-1)-1)
        main_atoms = x[:, indices[mask]]

        # generate holding tensor
        atom_tensor = torch.zeros(B, L, F, dtype=torch.float64, device=self.device)

        # reshaep main atoms and fill atom tensor
        atom_tensor[:, :, :-1] = main_atoms.reshape(B, L, -1)

        # fill cutpoints
        atom_tensor[:,  0, -1] = cut_points[:, 0]
        atom_tensor[:, -1, -1] = cut_points[:,-1]

        return atom_tensor


class HingeModel(nn.Module):
    def __init__(
        self,
        protein_len: int,
        output_size: int,
        input_size: int,
        normalize: bool,
        decoder_layers: int,
        encoder_layers: int,
        latent_space_dim: int,
        # dof_residue_metadata: torch.Tensor,
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
        convolution_layer: bool = False,
        ):
        super().__init__()
        """Hinge Protein Model"""
        # Grab needed values for forward
        self.noise_std = noise_std
        # self.dof_metadata = dof_residue_metadata
        print((torsion_main_n*2)//4)

        self.reduce_input = nn.Sequential(
            nn.Linear(torsion_main_n*2, (torsion_main_n*2)//2),
            nn.Tanh(),
            nn.Linear((torsion_main_n*2)//2, (torsion_main_n*2)//4),
            nn.Tanh(),
            nn.Linear((torsion_main_n*2)//4, (torsion_main_n*2)//6),
            nn.Tanh(),
            nn.Linear((torsion_main_n*2)//6, (torsion_main_n*2)//8),
            nn.Tanh(),
            nn.Linear((torsion_main_n*2)//8, (torsion_main_n*2)//10),
            nn.Tanh(),
            nn.Linear((torsion_main_n*2)//10, (torsion_main_n*2)//12),
            nn.Tanh(),
            nn.Linear((torsion_main_n*2)//12, (torsion_main_n*2)//14),
            nn.Tanh(),
            nn.Linear((torsion_main_n*2)//14, (torsion_main_n*2)//16),
            nn.Tanh(),
        )

        # encoder first set
        self.encoder_one = EncoderBlock64(
            num_layers = 1,
            start_size = (torsion_main_n*2)//16,
            end_size = 64,
            dropout = dropout,
        )
        
        # encoder block
        self.encoder = EncoderBlock64(
            num_layers = encoder_layers,
            start_size = 64,
            # d_model = 232,
            end_size = 64,
            dropout = dropout,
        )
        self.encoder_n2 = EncoderBlock64(
            num_layers = encoder_layers,
            start_size = 64,
            end_size = 64,
            dropout = dropout,
        )
        self.encoder_n3 = EncoderBlock64(
            num_layers = encoder_layers,
            start_size = 64,
            # d_model = 5,
            end_size = 64,
            dropout = dropout,
        )

        # Decreasing encoder block
        self.encoder_pyramid = nn.Sequential(
            # layer 1
            nn.Linear(64, 32),
            nn.Tanh(),
            # layer 2
            nn.Linear(32, 16),
            nn.Tanh(),
            # layer 3
            nn.Linear(16, 8),
            nn.Tanh(),
            # layer 4
            nn.Linear(8, latent_space_dim),
            nn.Tanh(),
        )

        ##### DECODER PORTION
        self.decoder_skip = EncoderBlock64(
            num_layers = 1,
            start_size = latent_space_dim,
            end_size = 64,
            dropout = dropout,
        )

        self.decoder_n = EncoderBlock64(
            num_layers = decoder_layers,
            start_size = 64,
            # d_model = 120,
            end_size = 64,
            dropout = dropout,
        )
        self.decoder_n2 = EncoderBlock64(
            num_layers = decoder_layers,
            start_size = 64,
            # d_model = 120,
            end_size = 64,
            dropout = dropout,
        )
        self.decoder_n3 = EncoderBlock64(
            num_layers = decoder_layers,
            start_size = 64,
            # d_model = 120,
            end_size = 64,
            dropout = dropout,
        )

        self.final_layer = nn.Sequential(
            nn.Linear(64, (torsion_main_n*2)//16),
            nn.Tanh(),
            nn.Linear((torsion_main_n*2)//16, (torsion_main_n*2)//14),
            nn.Tanh(),
            nn.Linear((torsion_main_n*2)//14, (torsion_main_n*2)//12),
            nn.Tanh(),
            nn.Linear((torsion_main_n*2)//12, (torsion_main_n*2)//10),
            nn.Tanh(),
            nn.Linear((torsion_main_n*2)//10, (torsion_main_n*2)//8),
            nn.Tanh(),
            nn.Linear((torsion_main_n*2)//8, (torsion_main_n*2)//6),
            nn.Tanh(),
            nn.Linear((torsion_main_n*2)//6, (torsion_main_n*2)//4),
            nn.Tanh(),
            nn.Linear((torsion_main_n*2)//4, (torsion_main_n*2)//2),
            nn.Tanh(),
            nn.Linear((torsion_main_n*2)//2, torsion_main_n*2),
            # nn.Tanh(),
        )

        self.input_decompose = ConformationalInputDecomposition(
            protein_len = protein_len,
            torsion_main_n = torsion_main_n,
            torsion_side_n = torsion_side_n,
            bond_angle_main_n = bond_angle_main_n,
            bond_angle_side_n = bond_angle_side_n,
            bond_length_main_n = bond_length_main_n,
            bond_length_side_n = bond_length_side_n,
            jump_n = jump_n,
            energy_n = energy_n,
            decomposition_n = 1,
        )

    def forward(self, x: torch.Tensor):
        """Forward"""
        # Add noise if we are training and not validating
        if self.training:
            noise = torch.normal(torch.zeros(x.shape), self.noise_std).to("cuda" if torch.cuda.is_available() else "cpu")
            x = x + noise
            x = torch.clamp(x, -180, 180)

        # decompose
        x, energy = self.input_decompose(x)
        initial_input = x.clone().detach()

        ##### ENCODER
        # Convert 116 -> 64 for encoder block training
        x = self.encoder_one(self.reduce_input(x))
        skip_one = x.clone()

        # First Encoder block 64 -> 64
        x = self.encoder(x)
        skip_n1 = x.clone()

        # Second Encoder Block 64 -> 64 
        x = self.encoder_n2(torch.add(x, skip_one))
        skip_n2 = x.clone() 

        # Third Encoder Block 64 -> 64
        x = self.encoder_n3(torch.add(x, skip_n1))

        # Pyramid Encoder 64 -> 32 -> 16 -> 8 -> 3
        x = self.encoder_pyramid(torch.add(x, skip_n2))
        ##### ENCODER END



        ##### LATENT SPACE
        # Copy and compute distance for latent space
        latent_space = x.clone()
        latent_euclid_dist = self.compute_euclidean_distance(latent_space)
        ##### LATENT SPACE END



        ##### DECODER
        # Decoder latent to decoder blocks size 3 -> 64
        x = self.decoder_skip(x)
        decode_skip1 = x.clone()

        # First Decoder Block 64 -> 64
        x = self.decoder_n(x)
        dskip_n1 = x.clone()

        # Second Decoder Block 64 -> 64
        x = self.decoder_n2(torch.add(x, decode_skip1))
        dskip_n2 = x.clone()

        # Third Decoder Block 64 -> 64
        x = self.decoder_n3(torch.add(x, dskip_n1))

        # FINAL decoder layer 64 -> 116 out size
        out = self.final_layer(torch.add(x, dskip_n2))
        ##### DECODER END

        return out, latent_euclid_dist, latent_space, initial_input 

    def compute_euclidean_distance(self, xyz_mat: torch.Tensor):
        """Compute the latent space RMSD"""
        # compute pairwise diff
        differences = xyz_mat.unsqueeze(1) - xyz_mat.unsqueeze(0)

        # squared_distances along last dim
        distances = torch.sum( torch.pow( differences, 2), dim=-1)
        return distances


        



############# WORK IN PROGRESSS
class ConvCycDecoder2(nn.Module):
    def __init__(
        self,
        channel_in: int,
        channel_out: int,
        out_size: int,
        kernel_size1: int,
        kernel_size2: int,
        ):
        super().__init__()
        """Apply Convolution to the cyclic peptides"""

        self.channel_in = channel_in

        # Convolution
        self.conv1_2d = nn.ConvTranspose2d(channel_in, channel_in//2, kernel_size=(3,2), stride=(3,3), padding=(3//2, 1), output_padding=(0, 1))
        self.conv2_2d = nn.ConvTranspose2d(channel_in//2, channel_out, kernel_size=(kernel_size1, kernel_size2), stride=(2, 2), padding=((kernel_size1//2), (kernel_size2//2)-1), output_padding=(1,0))

        # pooling output
        # self.maxpool = nn.MaxPool1d(2)
        
        #activation
        # self.glu = nn.GLU(dim=1)

        outsize = channel_in*6
        # Convert to 64
        self.linear = nn.Sequential(
            nn.Linear(64, outsize//4),
            nn.Tanh(),
            nn.Linear(outsize//4, outsize//2),
            nn.Tanh(),
            nn.Linear(outsize//2, outsize),
        )

    def forward(self, x: torch.Tensor):
        """Forward Cyclic Conv"""
        B, D = x.shape
        # print("pre decoder convolution Transpose", x.shape)
        # print("decoder x", x.shape)
        # Apply convolution
        x = F.tanh(self.linear(x))
        # print(x.shape)
        # print('linear', x.shape)
        # reshape x
        x = x.view(B, self.channel_in, 3, -1)
        # print('reshape', x.shape)
        # print('reshape', x.shape)
        x = F.tanh(self.conv1_2d(x))
        # print('conv1 decoder', x.shape)
        # print("conv1", x.shape)
        # print("conv1", x.shape)
        x = F.tanh(self.conv2_2d(x))
        # print('conv2 decoder', x.shape)
        # print("conv2", x.shape)
        # print("conv2", x.shape)
        # print("linear", x.shape)
        return x

class ConvCycEncoder2(nn.Module):
    def __init__(
        self,
        channel_in: int,
        channel_out: int,
        out_size: int,
        kernel_size1: int,
        kernel_size2: int,
        ):
        super().__init__()
        """Apply Convolution to the cyclic peptides"""
        self.kernel_size1 = kernel_size1

        # Convolution
        self.conv1_2d = nn.Conv2d(channel_in, channel_out//2, kernel_size=(kernel_size1, kernel_size2), stride=1, padding=(kernel_size1//2, kernel_size2//2))
        self.conv2_2d = nn.Conv2d(channel_out//2, channel_out, kernel_size=(3, 2), stride=1, padding=(3//2, 2//2))
        
        # pooling output
        self.maxpool = nn.MaxPool2d(2)
        

        outsize = channel_out * 6
        # Convert to 64
        self.linear = nn.Sequential(
            nn.Linear(outsize, (outsize)//2),
            nn.Tanh(),
            nn.Linear((outsize//2),(outsize//4)),
            nn.Tanh(),
            nn.Linear(outsize//4, 64),
        )

    def forward(self, x: torch.Tensor):
        """Forward Cyclic Conv"""
        B = x.size(0)

        # cyclic padding
        # print('pre padding', x.shape)
        x = F.pad(x, (0,0,self.kernel_size1, self.kernel_size1,0, 0), mode="circular") # pad the L dimension
        # print('padded', x.shape)

        # Apply convolution
        # print("Encoder x", x.shape)
        x = self.conv1_2d(x)
        print('conv1', x.shape)
        # print("conv1 nan check", x.isnan().sum())
        # print("conv1 inf check", (x == float("inf")).sum())
        # print('max/min', x.min(), x.max())
        x = self.maxpool(F.tanh(x))
        # print("maxpool1", x.shape)
        # print('max/min', x.min(), x.max())
        # print("conv1", x.shape)
        x = F.tanh(self.conv2_2d(x))
        print('conv2', x.shape)
        x = self.maxpool(x)
        print("maxpool2", x.shape)
        # print('reshape', x.view(B, -1).shape)
        # print("conv2", x.shape)
        x = F.tanh(self.linear(x.view(B, -1)))
        # print('linear max/min', x.min(), x.max())
        # print("linear", x.shape)
        return x

class ConvCase2(nn.Module):
    def __init__(
        self,
        protein_len: int,
        output_size: int,
        input_size: int,
        normalize: bool,
        decoder_layers: int,
        encoder_layers: int,
        latent_space_dim: int,
        dof_residue_metadata: torch.Tensor,
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
        convolution_layer: bool = False,
        ):
        super().__init__()
        """Only an Encoder/Decoder Block"""
        # Grab needed values for forward
        self.noise_std = noise_std
        self.dof_residue_metadata = dof_residue_metadata
        self.device = device
        self.convolution = convolution_layer
        self.torsion_main_n = torsion_main_n

        # grab atom Length residue length
        F = self.dof_residue_metadata[:, 1].max().long()


        in_channel = 2
        out_channel = 32
        # Convolution 
        self.conv_cyc_encoder = ConvCycEncoder2(
            channel_in = in_channel,
            channel_out = out_channel,
            kernel_size1 = 3,
            kernel_size2 = 3,
            out_size = 64,
        )
        self.conv_cyc_decoder = ConvCycDecoder2(
            channel_in = out_channel,
            channel_out = in_channel,
            kernel_size1 = 3,
            kernel_size2 = 3,
            out_size = (8*9)*2,
        )
        # self.conv_cyc2d = ConvCycTest2d(
        #     channel_in = 2,
        #     channel_out = 1,
        #     kernel_seq = 5,
        #     kernel_feats = 3,
        #     kernel_max = 3,
        #     out_size = 64//2,
        # )

        # encoder first set
        self.encoder_one = EncoderBlock64(
            num_layers = 1,
            start_size = torsion_main_n*2,
            end_size = 64,
            dropout = dropout,
        )
        
        # encoder block
        self.encoder = EncoderBlock64(
            num_layers = encoder_layers,
            start_size = 64,
            end_size = 64,
            dropout = dropout,
        )
        self.encoder_n2 = EncoderBlock64(
            num_layers = encoder_layers,
            start_size = 64,
            end_size = 64,
            dropout = dropout,
        )
        self.encoder_n3 = EncoderBlock64(
            num_layers = encoder_layers,
            start_size = 64,
            end_size = 64,
            dropout = dropout,
        )

        # Decreasing encoder block
        self.encoder_pyramid = nn.Sequential(
            # layer 1
            nn.Linear(64, 32),
            nn.Tanh(),
            # layer 2
            nn.Linear(32, 16),
            nn.Tanh(),
            # layer 3
            nn.Linear(16, 8),
            nn.Tanh(),
            # layer 4
            nn.Linear(8, latent_space_dim),
            nn.Tanh(),
        )

        ##### DECODER PORTION

        # Decreasing encoder block
        self.decoder_pyramid = nn.Sequential(
            # layer 1
            nn.Linear(latent_space_dim, 8),
            nn.Tanh(),
            # layer 2
            nn.Linear(8, 16),
            nn.Tanh(),
            # layer 3
            nn.Linear(16, 32),
            nn.Tanh(),
            # layer 4
            nn.Linear(32, 64),
            nn.Tanh(),
        )
        # self.decoder_skip = EncoderBlock64(
        #     num_layers = 1,
        #     start_size = 
        #     end_size = 64,
        #     dropout = dropout,
        # )

        self.decoder_n = EncoderBlock64(
            num_layers = decoder_layers,
            start_size = 64,
            end_size = 64,
            dropout = dropout,
        )
        self.decoder_n2 = EncoderBlock64(
            num_layers = decoder_layers,
            start_size = 64,
            end_size = 64,
            dropout = dropout,
        )
        self.decoder_n3 = EncoderBlock64(
            num_layers = decoder_layers,
            start_size = 64,
            end_size = 64,
            dropout = dropout,
        )

        self.final_layer = nn.Sequential(
            nn.Linear(252, torsion_main_n*2),
        )

        self.input_decompose = ConformationalInputDecomposition(
            protein_len = protein_len,
            torsion_main_n = torsion_main_n,
            torsion_side_n = torsion_side_n,
            bond_angle_main_n = bond_angle_main_n,
            bond_angle_side_n = bond_angle_side_n,
            bond_length_main_n = bond_length_main_n,
            bond_length_side_n = bond_length_side_n,
            jump_n = jump_n,
            energy_n = energy_n,
            decomposition_n = 1,
        )

    def forward(self, x: torch.Tensor, noise: bool):
        """Forward"""
        # Add noise if we are training and not validating
        if self.training and noise:
            noise = torch.normal(torch.zeros(x.shape), self.noise_std).to("cuda" if torch.cuda.is_available() else "cpu")
            x = x + noise
            x = torch.clamp(x, -180, 180)

        # decompose
        x, bond_angle, bond_length, energy = self.input_decompose(x)
        initial_input = x.clone().detach()

        ##### ENCODER
        # If convolution is true then reformat and apply convolution
        if self.convolution:
            sin_x = self.reformat_convolution2(x[:, :self.torsion_main_n], self.dof_residue_metadata)
            cos_x = self.reformat_convolution2(x[:, self.torsion_main_n:], self.dof_residue_metadata)
            x_combo = torch.cat(
                (
                    sin_x.unsqueeze(-1),
                    cos_x.unsqueeze(-1),
                ),
                dim=-1,
            )
            # reshape from (N, L, D, 2) -> (N, 2, D*L)
            B, L, D, C = x_combo.shape
            x = self.conv_cyc_encoder(x_combo.permute(0, 3, 1, 2)) # 2D
            # x = self.conv_cyc2d(x_combo.permute(0, 3, 1, 2)) # 2D
            # sin_conv = self.conv_cyc(sin_x.permute(0, 2, 1))
            # cos_conv = self.conv_cyc(cos_x.permute(0, 2, 1))
            # x = torch.cat((sin_conv, cos_conv), dim=-1)
            # x = torch.add(sin_conv, cos_conv)
        else:
        # Convert 116 -> 64 for encoder block training
            x = self.encoder_one(x)
        skip_one = x.clone()

        # First Encoder block 64 -> 64
        x = self.encoder(x)
        skip_n1 = x.clone()

        # Second Encoder Block 64 -> 64 
        x = self.encoder_n2(torch.add(x, skip_one))
        skip_n2 = x.clone() 

        # Third Encoder Block 64 -> 64
        x = self.encoder_n3(torch.add(x, skip_n1))

        # Pyramid Encoder 64 -> 32 -> 16 -> 8 -> 3
        x = self.encoder_pyramid(torch.add(x, skip_n2))
        ##### ENCODER END



        ##### LATENT SPACE
        # Copy and compute distance for latent space
        latent_space = x.clone()
        latent_euclid_dist = self.compute_euclidean_distance(latent_space)
        ##### LATENT SPACE END



        ##### DECODER
        # Decoder latent to decoder blocks size 3 -> 64
        x = self.decoder_pyramid(x)
        decode_skip1 = x.clone()

        # First Decoder Block 64 -> 64
        x = self.decoder_n(x)
        dskip_n1 = x.clone()

        # Second Decoder Block 64 -> 64
        x = self.decoder_n2(torch.add(x, decode_skip1))
        dskip_n2 = x.clone()

        # Third Decoder Block 64 -> 64
        x = self.decoder_n3(torch.add(x, dskip_n1))

        # FINAL decoder layer 64 -> 116 out size
        if self.convolution:
            B, D = x.shape
            x = self.conv_cyc_decoder(x).view(B, -1)
        # out = self.final_layer(torch.add(x, dskip_n2))
        out = self.final_layer(x)
        ##### DECODER END

        return out, latent_euclid_dist, latent_space, initial_input 

    def compute_euclidean_distance(self, xyz_mat: torch.Tensor):
        """Compute the latent space RMSD"""
        # compute pairwise diff
        differences = xyz_mat.unsqueeze(1) - xyz_mat.unsqueeze(0)

        # squared_distances along last dim
        distances = torch.sum( torch.pow( differences, 2), dim=-1)
        return distances

    def reformat_convolution2(self, x: torch.Tensor, format: torch.Tensor):
        """Format x for cyclic convolution"""
        # grab shape
        B, D = x.shape
        format = format.long()
        # grab residue and index amount
        L = format[:D, 0].max()
        F = format[:D, 1].max()

        # generate holding tensor
        atom_tensor = torch.zeros(B, L, F, dtype=torch.float64, device=self.device)

        # grab indices
        residue_indices = format[:D, 0] - 1
        atom_indicies = format[:D, 1] - 1

        # create a gride for batch indices
        batch_indices = torch.arange(B, device=self.device)[:, None].expand(B, D)

        # advance indexing to fill in atom tensor
        atom_tensor[batch_indices, residue_indices, atom_indicies] = x

        return atom_tensor

    def reformat_convolution(self, x: torch.Tensor, format: torch.Tensor):
        """Format x for cyclic convolution"""
        # grab shape
        B, D = x.shape
        # grab residue and index amount
        L = format[:, 0].max().to(torch.long)
        F = format[:, 1].max().to(torch.long) - 1

        # input extract
        cut_points = x[:, [F, -1]]
        indices = torch.arange(x.size(-1), device=self.device)
        mask = (indices != F) & (indices != x.size(-1)-1)
        main_atoms = x[:, indices[mask]]

        # generate holding tensor
        atom_tensor = torch.zeros(B, L, F, dtype=torch.float64, device=self.device)

        # reshaep main atoms and fill atom tensor
        atom_tensor[:, :, :-1] = main_atoms.reshape(B, L, -1)

        # fill cutpoints
        atom_tensor[:,  0, -1] = cut_points[:, 0]
        atom_tensor[:, -1, -1] = cut_points[:,-1]

        return atom_tensor

