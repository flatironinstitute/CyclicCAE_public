import torch
import torch.nn as nn

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
        for layer in self.encoder_layers:
                x = layer(x) #, True)
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

        # encoder first set
        self.encoder_one = EncoderBlock64(
            num_layers = 1,
            start_size = 116,
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

        # This was a test to see if the pyramid was necessary
        # self.encoder2latent = nn.Sequential(
        #     nn.Linear(64, latent_space_dim),
        #     nn.Tanh(),
        # )

        self.final_layer = nn.Sequential(
            nn.Linear(64, 116),
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
        x = self.encoder_n2(torch.add(x, skip_one))
        skip_n2 = x.clone() 

        # Third Encoder Block 64 -> 64
        x = self.encoder_n3(torch.add(x, skip_n1))

        # Pyramid Encoder 64 -> 32 -> 16 -> 8 -> 3
        x = self.encoder_pyramid(torch.add(x, skip_n2))
        ##### ENCODER END

        # Copy and compute distance for latent space
        latent_space = x.clone()
        latent_euclid_dist = self.compute_euclidean_distance(latent_space)

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

        # distance matrix
        # distance_matrix = torch.sqrt(squared_distances)
        return distances




        



