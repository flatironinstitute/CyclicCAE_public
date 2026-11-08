#!/usr/bin/env python

# Copyright 2026 Vikram K. Mulligan and Andrew Powers.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS”
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

# @author: Andrew Powers (apowers@flatironinstitute.org)
# @brief: Classes and functions helpfulf for inference task
#

######## IMPORT 
import torch
import torch.nn.functional as F
import torch.nn as nn
from collections import defaultdict
import re
import os
import ast
import matplotlib.pyplot as plt
from typing import Union
import numpy as np
from pathlib import Path
# Work on updating the Path, so that it is fixed and doesnt need to be setup manually.

# xyz coordinates
from terminal_error import TerminalPeptide
import biotite.structure as struc
import biotite.structure.io as strucio

####### Class

class HBondCount:
    def __init__(
        self,
        min_hbond: int,
        ):
        """Class is for specifying how many hbonds are within
        a structure at some t step during inference. WHere t in [0, N-1]

        PARAMS
        ------
        min_hbond: int
            The number of internal backbone hydrogen bonds needed to be present
            to count as successful/checkpoint
        """
        self.min_hbond = min_hbond
        self.min_distance = 1.5 # Angstrom
        self.max_distance = 3.2 # Angstrom 

    def measure_internal_distances(self, H_coords: torch.Tensor, O_coords: torch.Tensor, ahd_bool: torch.Tensor, bah_bool: torch.Tensor):
        """Compute Distances between H and O Atoms to check if a hydrogen bond is present

        PARAMS
        ------
        H_coords: Tensor (B, N, 3)
            xyz coordinates of Amide Hydrogen
        O_coords: Tensor (B, N, 3)
            xyz coordinates of Backbone Oxygen
        ahd_bool: Tensor (B, 8, 8)
            Angel for AHD atoms
        bah_bool: Tensor (B, 8, 8)
            Angel for BAH atoms

        RETURNS
        -------
        hbond_count: Tensor (B)
            Number of internal hydrogen bonds present
        """
        # Compute Distance between all points
        dists = torch.norm(O_coords.unsqueeze(2) - H_coords.unsqueeze(1), dim=-1)

        # col indices for selection
        indices = torch.arange(1, dists.size(-1), dtype=torch.long)
        indices = torch.cat((indices, torch.tensor([0], dtype=torch.long)))
        # Next residue also fix
        indices_nextres = torch.arange(0, dists.size(-1)-1, dtype=torch.long)
        indices_nextres = torch.cat((torch.tensor([dists.size(-1)-1], dtype=torch.long), indices_nextres))
        # Rest of indices for selecting the offset diagonals
        batch_indices = torch.arange(dists.size(0))
        row_indices = torch.arange(dists.size(1))
        dists[batch_indices[:, None], row_indices, indices] = torch.tensor([0.0], dtype=dists.dtype)
        dists[batch_indices[:, None], row_indices, indices_nextres] = torch.tensor([0.0], dtype=dists.dtype)
        print(dists)
        # Make a ones matrix
        eye = torch.eye(dists.size(-1), dtype=dists.dtype)
        ones = torch.ones(dists.size(-1), dists.size(-1), dtype=dists.dtype)
        mask = ones - eye
        dists = dists * mask
        print(dists)

        # Check where the distances are within the correct spot
        dists_bool = (dists >= self.min_distance) & (dists <= self.max_distance)
        full_bool = dists_bool & ahd_bool # & bah_bool
        print(full_bool[0])
        print("DISTS:", dists_bool[0])
        print("AHD:", ahd_bool[0])
        print("BAH:", bah_bool[0])

        # Count hbonds per B_i
        # hbond_count = dists_bool.sum(-1) # (B)
        col_count = dists_bool.sum(-1) # Col Sum
        # print("COL")
        # print(col_count)
        col_count[col_count > 1] -= (col_count[col_count > 1] * 2)-1
        # print(col_count)
        # print(col_count.sum(-1))
        # print("ROW")
        row_count = dists_bool.sum(-2) # Row sum
        # print(row_count)
        row_count[row_count > 2] -= (row_count[row_count > 2] * 2) - 2
        # print(row_count)
        # print(row_count.sum(-1))
        # print("CHECK SUM COMBINE")
        # print(col_count.sum(-1) + row_count.sum(-1))
        combo_count = row_count + col_count
        # print(combo_count)
        # print(combo_count.sum(-1))
        # print(dists_bool.sum((1, -1)))
        # hbond_count[hbond_count > 2] = 2
        hbond_count = dists_bool.sum((1,-1)) - combo_count.sum(-1)
        # hbond_count = dists_bool.sum((1, -1))
        return hbond_count

    def calc_ahd_bah_angles(
        self,
        o_coords: torch.Tensor,
        h_coords: torch.Tensor,
        n_coords: torch.Tensor,
        c_coords: torch.Tensor,
        ):
        """Generate polar bond angles for hbond count

        PARAMS
        ------
        o_coords: Tensor (B, L, 3)
            Oxygen generated coords
        h_coords: Tensor (B, L, 3)
            Hydrogen amide generated coords
        n_coords: Tensor (B, L, 3)
            Nitrogen generated coords
        c_coords: Tensor (B, L, 3)
            Carbon generated coords
        """
        # First we calculate vectors for both AHD and BAH
        AH1 = o_coords.unsqueeze(2) - h_coords.unsqueeze(1)
        HD = n_coords.unsqueeze(2) - h_coords.unsqueeze(1)
        BA = c_coords.unsqueeze(2) - o_coords.unsqueeze(1)
        AH2 = h_coords.unsqueeze(2) - o_coords.unsqueeze(1)

        # Now compute dot products and magnitudes
        # ah_mag = torch.sqrt(AH[..., 0] * AH[..., 0] + AH[..., 1] * AH[..., 1] * AH[..., 1] + AH[..., 2] * AH[..., 2])
        # hd_mag = torch.sqrt(HD[..., 0] * HD[..., 0] + HD[..., 1] * HD[..., 1] * HD[..., 1] + HD[..., 2] * HD[..., 2])
        # print(ah_mag.shape)
        # print(AH.shape)
        # ah_norm = AH / ah_mag.unsqueeze(-1)
        # hd_norm = HD / hd_mag.unsqueeze(-1)
        # res = ah_norm[..., 0] * hd_norm[..., 0] + ah_norm[..., 1] * hd_norm[..., 1] + ah_norm[..., 2] * hd_norm[...,2]
        # angle = torch.acos(res)
        # print("ANGLE TEST")
        # print(torch.rad2deg(angle))


        dot_AHD = torch.einsum('blrd,blrd->blr', AH1, HD) #(AH * HD, dim=-1)
        dot_BAH = torch.einsum('blrd,blrd->blr', BA, AH2) #, dim=-1)
        # dot_AHD = torch.einsum('blrd,blrd->blr', AH1/mag_AH1, HD/mag_HD) #(AH * HD, dim=-1)
        # dot_BAH = torch.einsum('blrd,blrd->blr', BA/mag_BA, AH2/mag_AH2) #, dim=-1)
        mag_AH1 = torch.norm(AH1, dim=-1)
        mag_HD = torch.norm(HD, dim=-1)
        mag_BA = torch.norm(BA, dim=-1)
        mag_AH2 = torch.norm(AH2, dim=-1)
        # print("MAG NORM", mag_AH1.shape)
        # print("AH1 shape", AH1.shape)

        # cosine of angles
        cos_AHD = dot_AHD / (mag_AH1 * mag_HD)
        cos_BAH = dot_BAH / (mag_BA * mag_AH2)

        # Clamp output, so that it isn't outside of [-1, 1]
        cos_AHD = torch.clamp(cos_AHD, -1.0, 1.0)
        cos_BAH = torch.clamp(cos_BAH, -1.0, 1.0)
        # cos_AHD = torch.clamp(dot_AHD, -1.0, 1.0)
        # cos_BAH = torch.clamp(dot_BAH, -1.0, 1.0)

        # compute radians
        rad_AHD = torch.acos(cos_AHD)
        rad_BAH = torch.acos(cos_BAH)

        # print("rad_AHD degree", torch.rad2deg(rad_AHD))
        # print("rad_BAH", rad_BAH)
        # print(torch.deg2rad(torch.tensor(90)))
        # print(torch.deg2rad(torch.tensor(180)))

        # compute anlge range bool
        AHD_bool = (torch.deg2rad(torch.tensor(90)) <= rad_AHD) & (rad_AHD <= torch.deg2rad(torch.tensor(180)))
        BAH_bool = (torch.deg2rad(torch.tensor(90)) <= rad_BAH) & (rad_BAH <= torch.deg2rad(torch.tensor(180)))

        # print('AHD:', AHD_bool)
        # print('BAH:', BAH_bool)

        return AHD_bool, BAH_bool


    def checkpoint(self, hbond_count: torch.Tensor, terminal_length: torch.Tensor, hbond_min: int = 3):
        """Checkpoint True/False Predictor for Energy Path outputting

        PARAMS
        ------
        hbond_count: Tensor (B)
            Number of internal hydrogen bonds present
        terminal_length: Tensor (B)
            Terminal bond length 
        hbond_min: int
            Minimum number of internal hydrogen bonds found within the 
            structure that should be allowed

        RETURNS
        -------
        checkpoint_bool: Tensor (B)
            Bool values for if a checkpoint should be saved or not
        """
        # terminal bond error
        lower_error = 1.4 - 0.4
        upper_error = 1.4 + 0.4

        # Get bool if terminal is within error limits
        terminal_bool = (terminal_length >= lower_error) & (terminal_length <= upper_error)

        # Get hbond count bool over min
        hbond_bool = hbond_count >= hbond_min

        # checkpoint bools
        checkpoint_bool = terminal_bool & hbond_bool
        return checkpoint_bool

def extract_pdb_xyz(input_file: str, 
                    full: bool = False, 
                    peptide_chain: int|str = "-1",
                    ):
    """Extract out Target xyz coordinates

    PARAMS
    ------
    input: str
        The pdb input target file
    full: bool
        If you want the full atom side chain information if not then we
        use bb-heavy + Cb
    peptide_chain: int|str
        This is either the char or int representation of where the peptide is
        that we want to use as our native input

    RETURNS
    ------
    atomdict: Dict
        [Atom Type, Residue 3 Letter, Chain, Residue Number, X, Y, Z]
        dict that holds every atom and its xyz coordinates
    """
    # chain string map
    chainint2str = dict(
            zip(
                list(range(1, 27)),
                "A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z".split(",")
                )
            )
    if isinstance(peptide_chain, int):
        pep = chainint2str[peptide_chain]
    else:
        pep = peptide_chain

    # Separate out the relevant information
    atom_array = strucio.load_structure(input_file)
    aa_values = atom_array[struc.filter_amino_acids(atom_array)]

    # extract atom number
    epsilon_key = {
            "CA": 0.062642, "N": 0.161725, "C": 0.141799,
            "O": 0.142417, "H": 0.901681,
            }#0.901681}
    sigma_key   = {
            "CA": 2.011760, "N": 1.802452, "C": 1.916661,
            "O": 1.540580, "H": 1.540580,
            }#0.005000}

    peptide_chain_mask = aa_values.chain_id == pep
    target_chain_mask = ~peptide_chain_mask

    peptide_atom_names = ("N", "CA", "C", "O")
    reduced_target_atom_names = ("N", "CA", "C", "O", "CB")

    peptide_mask = (
            peptide_chain_mask
            & np.isin(aa_values.atom_name, peptide_atom_names)
            )

    if full:
        target_mask = (
                target_chain_mask
                & (np.char.upper(aa_values.element.astype(str)) != "H")
                )
    else:
        target_mask = (
                target_chain_mask
                & np.isin(aa_values.atom_name, reduced_target_atom_names)
                )

    peptide = aa_values[peptide_mask]
    non_peptide = aa_values[target_mask]

    if len(peptide) == 0:
        raise ValueError(f"No peptide N/CA/C/O atoms found for chain {pep}"
                )
#     if len(non_peptide) == 0:
#         target_mode = (
#                 "all target heavy atoms"
#                 if full
#                 else "target N/CA/C/O/CB atoms"
#                 )
#         raise ValueError(f"No {target_mode} found outside peptide chain {pep}")

    selected_mask = peptide_mask | target_mask

    full_xyz = torch.tensor(
            aa_values[selected_mask].coord,
            dtype=torch.float,
            )
    target_xyz = torch.tensor(
            non_peptide.coord,
            dtype=torch.float,
            )
    peptide_xyz = torch.tensor(
            peptide.coord,
            dtype=torch.float,
            )

    peptide_sigma = torch.zeros(len(peptide), dtype=torch.float)
    peptide_epsilon = torch.zeros(len(peptide), dtype=torch.float)
    target_sigma = torch.zeros(len(non_peptide), dtype=torch.float)
    target_epsilon = torch.zeros(len(non_peptide), dtype=torch.float)

    def lj_atom_type(atom):
        """
        Map an atom to the existing reduced LJ parameter classes.
        """
        atom_name = str(atom.atom_name).strip().upper()
        element = str(atom.element).strip().upper()

        if atom_name == "CA":
            return "CA"

        if element in ("N", "C", "O", "H"):
            return element

        # fallback output
        return "C"

    # Fill sigma / epsilon mats
    for k, a in enumerate(peptide):
        atom_type = lj_atom_type(a)
        peptide_epsilon[k] = epsilon_key[atom_type]
        peptide_sigma[k] = sigma_key[atom_type]

    for k, a in enumerate(non_peptide):
        atom_type = lj_atom_type(a)
        target_epsilon[k] = epsilon_key[atom_type]
        target_sigma[k] = sigma_key[atom_type]

    assert target_xyz.size(0) == target_sigma.numel()
    assert target_xyz.size(0) == target_epsilon.numel()
    assert peptide_xyz.size(0) == peptide_sigma.numel()
    assert peptide_xyz.size(0) == peptide_epsilon.numel()

    return full_xyz, target_xyz, peptide_xyz, target_sigma, target_epsilon, peptide_sigma, peptide_epsilon


# def extract_pdb_xyz(input: str, full: bool = False, peptide_chain: int|str = "-1"):
#     """Extract out Target xyz coordinates
# 
#     PARAMS
#     ------
#     input: str
#         The pdb input target file
#     full: bool
#         If you want the full atom side chain information if not then we
#         use bb-heavy + Cb
#     peptide_chain: int|str
#         This is either the char or int representation of where the peptide is
#         that we want to use as our native input
# 
#     RETURNS
#     ------
#     atomdict: Dict
#         [Atom Type, Residue 3 Letter, Chain, Residue Number, X, Y, Z]
#         dict that holds every atom and its xyz coordinates
#     """
#     # chain string map
#     chainint2str = dict(zip([1,2,3,4,5,6,7], ["A", "B", "C", "D", "E", "F", "G"]))
#     if isinstance(peptide_chain, int):
#         pep = chainint2str[peptide_chain]
#     else:
#         pep = peptide_chain
# 
#     # Load in the file
#     data = open(input, 'r')
# 
#     # Init atom dictionary
#     atomdict = defaultdict(list)
#     index = 0
# 
#     # grab ATOM lines
#     for line in data:
#         # Do if atom in line
#         if "ATOM" in line or "HETATM" in line:
#             # Split the line to be readable
#             line_split = re.sub("\s+", ",", line).strip("\n").split(",")
#             # Most likely this is because we have something happening in split section 4 where we get something like B1000
#             # We need to separate this into "B 1000" which are two distinct elements in our line_split list
#             if len(line_split) < 14:
#                 if len(line_split[2]) > 4:
#                     data_split = re.split(r'(\d+)', line_split[2])
#                     if len(data_split[2]) >= 4:
#                         atom = data_split[0] + data_split[1] + data_split[2][0]
#                         res = data_split[2][1:]
#                     else:
#                         atom = data_split[0] + data_split[1] 
#                         res = data_split[2]
#                     line_split = line_split[:2] + [atom] + [res] + line_split[3:]
#                 if len(line_split[4]) > 2:
#                     line_split = line_split[:4] + [line_split[4][0]] + [line_split[4][1:]] + line_split[5:]
#                 if line_split[6].count(".") > 1:
#                     split_6 = line_split[6].split("-")
#                     line_split = line_split[:6] + ["-"+split_6[1]] + ["-"+split_6[-1]] + line_split[7:]
#                 if len(line_split[9]) > 4:
#                     line_split = line_split[:9] + [line_split[9][:4]] + [line_split[9][4:]] + line_split[10:]
#             if full:
#                 if (line_split[4] != pep) & (line_split[11] != "H"):
#                     # [Atom Type, Residue 3 Letter, Chain, Residue Number, X, Y, Z]
#                     if line_split[2] in ["N", "CA", "C", "O"]:
#                         atom_specific = line_split[2]
#                     elif (line_split[2] not in ["N", "CA", "C", "O"]) & (line_split[11] not in ["X", "S"]):
#                         if (len(line_split[2]) > 1) & (line_split[2][0] in ["N", "CA", "C", "O"]):
#                             atom_specific = line_split[2][0]
#                         elif line_split[2][0] == "S":
#                             atom_specific = "C"
#                         elif line_split[12] in ["N", "CA", "C", "O"]:
#                             atom_specific = line_split[12] 
#                         else: atom_specific = "C"
#                     else: 
#                         # atom_specific =  "H"
#                         atom_specific =  "C"
#                     atomdict[index] = [atom_specific, line_split[3], line_split[4], int(line_split[5]), float(line_split[6]), float(line_split[7]), float(line_split[8])]
#                     index += 1
#                 elif (line_split[4] == pep) & (line_split[2] in ["N", "CA", "C", "O"]):
#                     # [Atom Type, Residue 3 Letter, Chain, Residue Number, X, Y, Z]
#                     atomdict[index] = [line_split[2], line_split[3], line_split[4], int(line_split[5]), float(line_split[6]), float(line_split[7]), float(line_split[8])]
#                     index += 1
#                 else:
#                     pass
# 
#             #### full = False
#             #### We use up to Cb for target information
#             else:
#                 if (line_split[4] != pep) & (line_split[11] != "H"):
#                     # [Atom Type, Residue 3 Letter, Chain, Residue Number, X, Y, Z]
#                     if line_split[2] in ["N", "CA", "C", "O"]:
#                         atom_specific = line_split[2]
#                     elif (line_split[2] not in ["N", "CA", "C", "O"]) & (line_split[11] not in ["X", "S"]):
#                         if line_split[2][0] == "CB":
#                             atom_specific = "C"
#                         else:
#                             continue
#                     atomdict[index] = [atom_specific, line_split[3], line_split[4], int(line_split[5]), float(line_split[6]), float(line_split[7]), float(line_split[8])]
#                     index += 1
#                 elif (line_split[4] == pep) & (line_split[2] in ["N", "CA", "C", "O"]):
#                     # [Atom Type, Residue 3 Letter, Chain, Residue Number, X, Y, Z]
#                     atomdict[index] = [line_split[2], line_split[3], line_split[4], int(line_split[5]), float(line_split[6]), float(line_split[7]), float(line_split[8])]
#                     index += 1
#                 else:
#                     pass
# 
# 
#     return atomdict
# 
# def gen_xyz_tensor(atomdict: dict, peptide_chain: int|str):
#     """Convert the atomic dictionary to tensors
# 
#     PARAMS
#     ------
#     atomdict: Dict
#         [Atom Type, Residue 3 Letter, Chain, Residue Number, X, Y, Z]
#         dict that holds every atom and its xyz coordinates
# 
#     RETURNS
#     -------
#     full_xyz: Tensor (A+B, 3)
#         All atoms and xyz
#     target: Tensor (A, 3)
#         target protein chain A with atom numbers and xyz
#     peptide: Tensor(B, 3)
#         Peptide protein chain B with atom numbers and xyz
#     """
#     # chain string map
#     chainint2str = dict(zip([1,2,3,4,5,6,7], ["A", "B", "C", "D", "E", "F", "G"]))
#     if isinstance(peptide_chain, int):
#         pep = chainint2str[peptide_chain]
#     else:
#         pep = peptide_chain
# 
#     # extract atom number
#     atom_max = max([int(x) for x in atomdict.keys()])+1
#     # Key for epsilon and sigma matrixes
#     epsilon_key = {"CA": 0.062642, "N": 0.161725, "C": 0.141799, "O": 0.142417, "H": 0.901681}#0.901681}
#     sigma_key   = {"CA": 2.011760, "N": 1.802452, "C": 1.916661, "O": 1.540580, "H": 1.540580}#0.005000}
#     color_key   = {"CA": 0, "N": 1, "C": 2, "O": 3, "H": 4}
# 
#     # generate tensor 
#     full_xyz = torch.zeros(atom_max, 3, dtype=torch.float)
#     sigma_mat = torch.zeros(atom_max, dtype=torch.float)
#     epsilon_mat = torch.zeros(atom_max, dtype=torch.float)
#     color_tensor = torch.zeros(atom_max, dtype=torch.long)
# 
#     split_tensor_idx = list()
# 
#     # extract and add to tensor
#     for k in atomdict.keys():
# 
#         # Tensor xyz
#         full_xyz[k, :] = torch.tensor(
#             [
#                 atomdict[k][-3],
#                 atomdict[k][-2],
#                 atomdict[k][-1]
#             ]
#         )
#         sigma_mat[k] = sigma_key[atomdict[k][0]]
#         epsilon_mat[k] = epsilon_key[atomdict[k][0]]
#         color_tensor[k] = color_key[atomdict[k][0]]
# 
#         # Extract pep_chain
#         if atomdict[k][2] == pep:
#             split_tensor_idx.append(k)
# 
#     # Grab out the peptide xyz indexes
#     peptide_mask = torch.zeros(full_xyz.size(0), dtype=torch.bool)
#     nonpeptide_mask = torch.ones(full_xyz.size(0), dtype=torch.bool)
#     peptide_mask[split_tensor_idx] = True
#     nonpeptide_mask[split_tensor_idx] = False
# 
#     # Grab the peptide xyz out and everything else is target
#     # peptide
#     peptide = full_xyz[peptide_mask]
#     sigma_peptide = sigma_mat[peptide_mask]
#     epsilon_peptide = epsilon_mat[peptide_mask]
#     # target
#     target = full_xyz[nonpeptide_mask]
#     sigma_target = sigma_mat[nonpeptide_mask]
#     epsilon_target = epsilon_mat[nonpeptide_mask]
#     color_target = color_tensor[nonpeptide_mask]
# 
#     return full_xyz, target, peptide, sigma_target, epsilon_target, sigma_peptide, epsilon_peptide, color_target

def geometric_center(input_tensor: torch.Tensor):
    """Take a xyz structure and compute the centroid node
    geometric center formula: (1/n * sum_i x_i, 1/n * sum_i y_i, 1/n * sum_i z_i)

    PARAMS
    ------
    input_tensor: Tensor (A, 3)
        Some input tensor of shape Atoms by xyz

    RETURNS
    -------
    center_xyz: Tensor (3)
        XYZ point that is the centroid of the structure
    """
    return torch.sum(input_tensor, 0) / input_tensor.size(0)

def grab_interface(
    target_xyz: torch.Tensor,
    peptide_xyz: torch.Tensor,
    dist_cutoff: float = 8.0,
    ):
    """Grab atom xyz coordinates that are associated with the inferace

    PARAMS
    ------
    target: Tensor (A, 3)
        target protein chain A with atom numbers and xyz
    peptide: Tensor(B, 3)
        Peptide protein chain B with atom numbers and xyz
    dist_cutoff: float (Angstrom)
        Anything within this distance from the geometric center, is considered
        the interface

    RETURNS
    -------
    interface_xyz: Tensor (N, 3)
        Target atoms that are involved in the interface; close to the peptide
    mask: Tensor (A)
        True/False for if it is an interface or not; referring to target tensor
    """
    # generate centroid of peptide
    pep_geometric_center = geometric_center(peptide_xyz)

    # generate difference from point
    # distances = torch.sum((target_xyz - pep_geometric_center).pow(2), dim=-1).sqrt()
    distances = torch.cdist(target_xyz, pep_geometric_center.unsqueeze(0), p=2).squeeze(-1)

    # check if within distance cutoff
    mask = distances <= dist_cutoff

    return target_xyz[mask, :], mask


def harmonic_loss_func(xyz_peptide, xyz_target, cutoff_distance=4.0, d_target=2.5, k=10.0):
    print(xyz_peptide.shape)
    print(xyz_target.shape)
    # Compute all pairwise distances between peptide and target atoms
    diff = xyz_peptide - xyz_target[:, None, :]
    distances = torch.norm(diff, dim=2).T  # Shape: (num_peptide_atoms, num_target_atoms)
    print(distances.min())
    print(distances)
    
    # Mask distances greater than cutoff
    within_cutoff = distances < cutoff_distance
    
    # Find the minimum distance for each peptide atom within the cutoff
    # min_distances, _ = torch.min(torch.where(within_cutoff, distances, torch.tensor(float('inf'))), dim=-1)
    min_distances, _ = torch.min(distances, dim=-1)
    print(min_distances)
    
    # Compute the harmonic loss
    loss = k * torch.sum((min_distances - d_target).pow(2))
    print(loss)
    return loss


def build_exclusion_mask(
    num_residues: int,
    atoms_per_res: int,
    # bonded_local_pairs: list[tuple[int,int]],
    # inter_residue_bonds: list[tuple[int,int]],
):
    """
    Returns a boolean adjacency mask of shape (N, N) where
    N = num_residues * atoms_per_res.
    
    We set mask[i,j] = True if (i,j) is bonded or otherwise "excluded"
    from the nonbonded repulsion. Typically you'll invert it
    (mask = ~adj) if you want a "repulsion mask".
    
    Parameters
    ----------
    num_residues: int
    atoms_per_res: int
    bonded_local_pairs: list of (a1, a2) local indices that are bonded in 1 residue
    inter_residue_bonds: list of (local_atom_in_res_r, local_atom_in_res_(r+1))
                         e.g. (2, 0) if 2= C in one residue
                         and 0= N in the next residue.
    """
    N = num_residues * atoms_per_res
    # Start a full "no adjacency" matrix
    adj = torch.zeros(N, N, dtype=torch.bool)
    
    for r in range(0, N, 4):
        offset = torch.arange(r-4, r)
        # local bonded pairs in the same residue
        for i in offset:
            adj[i, offset] = True

    # If you want a ring closure for a cyclic peptide, also connect last->first:
    # For example, (C_{L-1}, N_0)
    # you can do that explicitly outside or pass it in if you prefer.
    
    return adj

def homochiral_loss(
    torsion_dofs: torch.Tensor,
    space: int,
    s: float = 0.05,
    a: float = 0.1, # 0.05
    b: float = 0.0,
    device: str = 'cpu',
    ) -> torch.Tensor:
    """
    Simple Function for computing a torsional degrees of freedom based loss term
    that is linear when the correct chirality and becomes quadratic when it crosses
    over zero to the other chirality of the ramachandran (phi).

    PARAMS
    ------
    :torsional_dofs: A tensor of size (B, L*2) where B is the batch size and L is
        the number of residues
    :space: The homochiral space (1: L-chiral, 2: D-chiral)
    :s: slope float value for our linear term
    :a: Our scaling term for the quadratic term
    :b: linear intercept
    :device: Location of tensors generated
    
    RETURNS
    -------
    :loss: The computed loss term based on DoFs radians
    """
    assert space in [1,2], f"A space of {space} is not recognized"

    # center term
    # center = torch.deg2rad(torch.tensor(90 if space == 1 else -90, dtype=torch.float64))
    phi_target = torch.tensor(-60, dtype=torch.float64)
    psi_target = torch.tensor(-50, dtype=torch.float64)
    # center = torch.tensor(90 if space == 1 else -90, dtype=torch.float64)

    # grab the phi angle
    # phi = torch.deg2rad(torsion_dofs[:, ::3])
    # phi, psi = torch.deg2rad(torsion_dofs[:, ::3]), torch.deg2rad(torsion_dofs[:, 1::3])
    phi, psi = torch.deg2rad(torsion_dofs[:, ::2]), torch.deg2rad(torsion_dofs[:, 1::2])
    print('PHI SHAPE:', phi.shape)
    mask = (phi <= 0).sum(-1) >= phi.shape[-1]-1
    print((phi <= 0).sum(-1))
    # print("MASK:", mask)

    # first define our function
    # linear_term = s * phi + s * center + b
    # quadratic_term = a * phi.pow(2) + s * phi + (center * s + b)

    # Apply piecewise
    # loss = torch.where(phi <= 0, linear_term, quadratic_term)
    # transition = torch.sigmoid(10.0 * phi)
    # loss = (1 - transition) * linear_term + transition * quadratic_term
    # loss = (torch.maximum(torch.zeros_like(phi), phi).pow(2) + torch.maximum(torch.zeros_like(phi), (-1 * phi) - 180).pow(2)).sum(-1)
    # loss = (F.softplus(phi).pow(2) + F.softplus(180.0 - phi).pow(2)).sum(-1)
    # loss = torch.pow(torch.sin(torch.deg2rad(phi)) - torch.sin(torch.deg2rad(torch.tensor(-90.))), 2) + \
    #     torch.pow(torch.cos(torch.deg2rad(phi)) - torch.cos(torch.deg2rad(torch.tensor(-90.))), 2)


    # local_loss = F.softplus(phi).pow(2)
    mu = torch.deg2rad(torch.tensor(90.))
    sigma = torch.deg2rad(torch.tensor(30.))
    diff = torch.atan2(torch.sin(phi - mu), torch.cos(phi - mu))
    local_loss = torch.exp(-0.5 * (diff.pow(2) / sigma**2)) / (sigma * torch.sqrt(torch.tensor(2)*torch.pi)) * 100
    # print("LOCAL:", local_loss)
    # global_loss = (phi.mean(-1).unsqueeze(1) - torch.ones_like(phi, device=device, dtype=torch.float64)*-60).pow(2)
    # global_loss = (phi.mean(-1) + 60)
    # print("GLOBAL:", global_loss)
    # print(local_loss.shape, global_loss.shape)
    # loss = (local_loss + global_loss.unsqueeze(-1)).sum(-1)
    loss = local_loss.sum(-1)
    # loss = torch.pow(torch.exp(-0.5 * ((phi - 90).pow(2) / 90)) / (90**2 * torch.sqrt(torch.tensor(2)*torch.pi)), 1/12) * 1000
    # loss = torch.where(phi <= 0, loss.pow(2), loss).sum(-1)
    # delta_phi = torch.minimum(torch.abs(phi - phi_target), 360 - torch.abs(phi - phi_target))
    # delta_psi = torch.minimum(torch.abs(psi - psi_target), 360 - torch.abs(psi - psi_target))
    #
    # loss = delta_phi.pow(2) + delta_psi.pow(2)
    # print("LOSS MIN:", loss.min())

    # return loss.mean(dim=-1).to(device) * 800, mask
    return loss, mask


def interface_loss(
    interface_atoms: torch.Tensor,
    recomputed_xyz: torch.Tensor,
    sigma_peptide: torch.Tensor,
    epsilon_peptide: torch.Tensor, 
    sigma_interface: torch.Tensor,
    epsilon_interface: torch.Tensor, 
    self_peptide: bool = False,
    ):
    """Compute smooth differentiable distance inference loss metric

    PARAMS
    ------
    interface_atoms: Tensor (N, 3)
        Target atoms that are involved in the interface; close to the peptide
    recomputed_xyz: Tensor (B, Atoms, 3)
        Recomputed Heavy Atoms XYZ that are from the DoFs outputted from the model
    sigma_peptide: Tensor (Atoms)
        Sigma (Atomic Radii) values for leonard jones calculation within the peptide
    epsilon_peptide: Tensor (Atoms)
        Epsilon (well depth) values for leonard jones calculation within the peptide
    sigma_interface: Tensor (N)
        Sigma (Atomic Radii) values for leonard jones calculation within the interface
    epsilon_interface: torch.Tensor, 
        Epsilon (well depth) values for leonard jones calculation within the interface

    RETURNS
    -------
    loss: (B) float
        Computed loss given interface and recomputed_xyz using a smooth distance cutoff
    """
    # Grab sizes
    B, Atoms, _ = recomputed_xyz.shape
    N = interface_atoms.shape[-2]

    # Generate our metrics
    sigma_peptide = torch.tensor([1.8025, 2.0118, 1.9167, 1.5406] * (Atoms // 4), dtype=torch.double, device=recomputed_xyz.device)
    epsilon_peptide = torch.tensor([0.1617, 0.0626, 0.1418, 0.1424] * (Atoms // 4), dtype=torch.double, device=recomputed_xyz.device)

    # check for size
    if interface_atoms.dim() == 2:
        interface_atoms = interface_atoms.unsqueeze(0).expand(B, -1, -1)
    # get pairwise distances
    pairwise_distances = torch.cdist(recomputed_xyz, interface_atoms, p=2)

    # Compute LJ Terms
    S = sigma_peptide.view(1, Atoms, 1) + sigma_interface.view(1, 1, N)
    E = torch.sqrt(epsilon_peptide.view(1, Atoms, 1) * epsilon_interface.view(1, 1, N))
    S = S.expand(B, -1, -1)
    E = E.expand(B, -1, -1)

    # Linear Repulse Term
    m = 20 * E / S * (-(5/3)**12 + (5/3)**6)
    b = E * (13*(5/3)**12 - 14*(5/3)**6 + 1)
    linear_term = m*pairwise_distances + b

    # normal LJ term
    lj_term = E*((S/pairwise_distances).pow(12) - 2*(S/pairwise_distances).pow(6) + 1)

    # Now conditional settings to compute loss (B, Atoms, N)
    loss = torch.where(
        pairwise_distances <= 0.6*S,
        linear_term,
        torch.where(
            pairwise_distances <= S,
            lj_term,
            torch.zeros_like(pairwise_distances, device=pairwise_distances.device)
        )
    ) 

    return loss.sum((-2, -1)) #, pairwise_distances.min(-1)[0].min(-1).values < 1.0

# def interface_loss(
#     interface_atoms: torch.Tensor,
#     recomputed_xyz: torch.Tensor,
#     previous_distances: torch.Tensor = None,
#     d_min: float = 1.4,
#     d_max: float = 10.0,
#     alpha: float = 1.0,
#     beta: float = 1.0,
#     sigma: float = 1.0,
#     ):
#     """Compute smooth differentiable distance inference loss metric
#
#     PARAMS
#     ------
#     interface_atoms: Tensor (N, 3)
#         Target atoms that are involved in the interface; close to the peptide
#     recomputed_xyz: Tensor (B, Atoms, 3)
#         Recomputed Heavy Atoms XYZ that are from the DoFs outputted from the model
#     previous_distances: Tensor (B, Atoms, N)
#     d_min: float,
#         Minimum distance that an atom in peptide can come distance wise to an interface atom and loss explodes
#     d_max: float,
#         Maximum distance that an atom in peptide can go away from an interface atom loss explodes
#     alpha: float,
#         Scale repulsion loss term
#     beta: float,
#         Scale attraction loss term
#     sigma: float,
#         Controls the steepness of the exponential function growth
#
#     RETURNS
#     -------
#     loss: (B) float
#         Computed loss given interface and recomputed_xyz using a smooth distance cutoff
#     """
#     # get pairwise distances
#     pairwise_distances = torch.cdist(recomputed_xyz, interface_atoms, p=2)
#
#     print(pairwise_distances)
#     print("SHAPE:", pairwise_distances.shape)
#     print('min:', pairwise_distances.min().item())
#     print('max:', pairwise_distances.max().item())
#     print('mean:', pairwise_distances.mean().item())
#
#     # attract = (1/100) * torch.exp(pairwise_distances / 1.5)
#     # min_distances, _ = torch.min(pairwise_distances, dim=-1)
#     # print("MIN DISTANCES:", min_distances)
#     # print("MIN DISTANCES MIN:", min_distances.min(dim=-1))
#     repulse = 10 * torch.exp(- pairwise_distances / 2)
#     print("REPULSE:", repulse)
#     print("REPULSE MAX:", repulse.max(dim=-1))
#     print("REPULSE:", repulse.shape)
#
#
#     return repulse.sum((-2, -1)), pairwise_distances.clone().detach()
    # return repulse.sum(-1), pairwise_distances.clone().detach()

# def interface_loss(
#     interface_atoms: torch.Tensor,
#     recomputed_xyz: torch.Tensor,
#     previous_distances: torch.Tensor = None,
#     d_min: float = 1.4,
#     d_max: float = 10.0,
#     alpha: float = 1.0,
#     beta: float = 1.0,
#     sigma: float = 1.0,
#     ):
#     """Compute smooth differentiable distance inference loss metric
#
#     PARAMS
#     ------
#     interface_atoms: Tensor (N, 3)
#         Target atoms that are involved in the interface; close to the peptide
#     recomputed_xyz: Tensor (B, Atoms, 3)
#         Recomputed Heavy Atoms XYZ that are from the DoFs outputted from the model
#     previous_distances: Tensor (B, Atoms, N)
#     d_min: float,
#         Minimum distance that an atom in peptide can come distance wise to an interface atom and loss explodes
#     d_max: float,
#         Maximum distance that an atom in peptide can go away from an interface atom loss explodes
#     alpha: float,
#         Scale repulsion loss term
#     beta: float,
#         Scale attraction loss term
#     sigma: float,
#         Controls the steepness of the exponential function growth
#
#     RETURNS
#     -------
#     loss: (B) float
#         Computed loss given interface and recomputed_xyz using a smooth distance cutoff
#     """
#     # get pairwise distances
#     pairwise_distances = torch.cdist(recomputed_xyz, interface_atoms, p=2)
#     # print(interface_atoms.shape, recomputed_xyz.shape)
#     # interface_atoms_exp = interface_atoms.unsqueeze(0).unsqueeze(0) # (N, 3) -> (1, 1, N, 3)
#     # recomputed_xyz_exp = recomputed_xyz.unsqueeze(2) # (B, A, 3) -> (B, A, 1, 3)
#     # print('Interface:', interface_atoms_exp.shape)
#     # print('recomputed_xyz', recomputed_xyz_exp.shape)
#     # pairwise_distances = torch.norm(recomputed_xyz_exp - interface_atoms_exp, dim=-1)
#     print(pairwise_distances)
#     print(pairwise_distances.shape)
#     print('min:', pairwise_distances.min().item())
#     print('max:', pairwise_distances.max().item())
#     print('mean:', pairwise_distances.mean().item())
#
#     # repulsion and attract function
#     # repulse = torch.exp( - sigma * (pairwise_distances - d_min))
#     # attract = torch.exp( - sigma * (d_max - pairwise_distances))
#
#     # normalizing term
#     # z = torch.sum(torch.exp( - pairwise_distances / sigma), dim=-1, keepdim=True)
#     # print('z', z.shape)
#
#     # Attraction term
#     # attract = torch.exp( - pairwise_distances / sigma) #/ pairwise_distances
#     # repulse = torch.exp( pairwise_distances / sigma) #/ pairwise_distances
#     # repulse = (1 / pairwise_distances).pow(12)
#
#     # print('attract', attract.shape)
#     # print('repulse', repulse.shape)
#
#     attract = 0.5 * torch.exp(pairwise_distances / 4)
#     repulse = (1/pairwise_distances) * torch.exp(- pairwise_distances / 10)
#
#     # repulse loss
#     # repulse_loss = torch.where(pairwise_distances <= d_min, repulse, torch.zeros_like(repulse))
#     # repulse_loss = repulse * torch.sigmoid(sigma * (d_min - pairwise_distances))
#     # attract loss
#     # attract_loss = torch.where(pairwise_distances >= d_max, attract, torch.zeros_like(attract))
#     # attract_loss = attract * torch.sigmoid(sigma * (pairwise_distances - d_max))
#
#     # loss = (alpha * 0*repulse + (1/z) * attract) #* pairwise_distances.pow(2)
#     loss = 0*repulse + attract #* pairwise_distances.pow(2)
#     # loss = alpha * repulse.pow(2) + beta * attract.pow(2)
#     print(loss.shape)
#
#     return loss.sum((-2, -1)), pairwise_distances

def update_move(P: torch.Tensor, Q: torch.Tensor):
    """Perform the kabsch algorithm to find the optimal rotation matrix
    that aligns P to points Q

    PARAMS
    ------
    P: Tensor (B, N, 3)
        Predicted DoF XYZ points
    Q: Tensor (B, N, 3)
        Previous model output xyz points 

    RETURNS
    -------
    P_updated: Tensor (B, N, 3)
        Updated XYZ points
    """
    #  get the centroid and center Q
    print(Q.shape)
    centroid_Q = Q.mean(dim=1)
    print("centroid_Q", centroid_Q.shape)
    centered_Q = Q - centroid_Q.unsqueeze(1)
    print("Centered", centered_Q.shape)

    move = P - centered_Q

    return P + move

# def kabsch_rootatom(P: torch.Tensor, Q: torch.Tensor, motif: torch.Tensor):
#     """Perform the kabsch algorithm to find the optimal rotation matrix
#     that aligns P to points Q
#
#     PARAMS
#     ------
#     P: Tensor (B, N, 3)
#         Predicted DoF XYZ points
#     Q: Tensor (N, 3)
#         PDB Peptide Heavy atom XYZ Points
#     motif: Tensor (2)
#         Residues involved in motif
#
#     RETURNS
#     -------
#     P_aligned: Tensor (B, N, 3)
#         Aligned DoF XYZ points
#     """
#     # motif_atoms
#     # Calculate on my own # THIS NEEDS TO BE FIXED
#     motif_mask = torch.arange(motif*4)
#
#     # Grab the residues you need
#     P_motif = P[:, motif_mask, :]
#     Q_motif = Q[:, motif_mask, :]
#
#     # Calc centroids
#     centroid_P = P.mean(dim=1) # (B, N, 3) -> (B, 3)
#     centroid_Q = Q.mean(dim=0) if len(Q.shape) == 2 else Q.mean(dim=1)# (N, 3) -> (3)
#
#     # Center the points base don centroid
#     P_centered = P - centroid_P.unsqueeze(1) # (B, N, 3) - (B, 1, 3) = (B, N, 3)
#     Q_centered = Q - centroid_Q if len(Q.shape) == 2 else Q - centroid_Q.unsqueeze(1)# (N, 3) - (3) = (N, 3)
#
#     # (B, N, 3) -> (B, 3) -> (B, N, 3) - (B, 1, 3) = (B, N, 3)
#
#     # Adjust Q_centered to match
#     if len(Q.shape) == 2:
#         Q_centered = Q_centered.unsqueeze(0).repeat(P_centered.size(0), 1, 1) # (1, N, 3) -> (B, N, 3)
#
#     # Compute the Covariance matrix
#     H = torch.matmul(P_centered.transpose(-2, -1), Q_centered) # (B, 3, N) x (B, N, 3) = (B, 3, 3)
#
#     # Compute SVD for optimal rotation
#     U, S, Vt = torch.linalg.svd(H) 
#
#     # Compute Rotation Matrix Now
#     mask = (torch.linalg.det(U) * torch.linalg.det(Vt)) < 0.0
#     # R = torch.matmul(Vt.transpose(-2, -1), U.transpose(-2, -1))
#
#     # Special Reflection case
#     # det = torch.linalg.det(R)
#     # mask = det < 0
#     if mask.sum() >= 1:
#         # Vt_copy = Vt.clone()
#         U_copy = U.clone()
#         U_copy[mask, :, -1] *= -1
#         # Vt_copy[mask, -1, :] *= -1
#         # R = torch.matmul(Vt_copy.transpose(-2, -1), U.transpose(-2, -1))
#         R = torch.matmul(U_copy.transpose(-2, -1), Vt.transpose(-2, -1))
#     else:
#         R = torch.matmul(U.transpose(-2, -1), Vt.transpose(-2, -1))
#
#     # print(torch.matmul(P_centered, R).shape)
#     # Apply Rotation
#     if len(Q.shape) == 2:
#         P_aligned = torch.matmul(P_centered, R) + centroid_Q 
#     else:
#         P_aligned = torch.matmul(P_centered, R) + centroid_Q.unsqueeze(1)
#
#     diff = P_aligned - Q
#     rmsd = torch.sqrt((diff * diff).sum(1) / P.size(1))
#     print("RSMD:", rmsd.mean())
#
#     torch.save({"paligned": P_aligned, "control": Q}, "paligned.pt")
#
#     return P_aligned

def kabsch_rigid_transform(P_point: torch.Tensor, Q_point: torch.Tensor):
    """Perform the kabsch algorithm to find the optimal rotation matrix
    that aligns P to points Q

    PARAMS
    ------
    P: Tensor (B, N, 3)
        Predicted DoF XYZ points
    Q: Tensor (B, N, 3)
        PDB Peptide Heavy atom XYZ Points

    RETURNS
    -------
    P_aligned: Tensor (B, N, 3)
        Aligned DoF XYZ points
    t:
    R:
    """
    # Compute centroids
    centroid_P = P_point.mean(dim=1) # (B, 3)
    centroid_Q = Q_point.mean(dim=1) # (3)

    # Centered Points
    P_centered = P_point - centroid_P.unsqueeze(1) # (B, N, 3) - (B, 1, 3) = (B, N 3)
    Q_centered = Q_point - centroid_Q.unsqueeze(1) # (N, 3) - (1, 3) = (N, 3)

    # Calculate covariance matrix
    H = torch.matmul(P_centered.transpose(2, 1), Q_centered) # (B, 3, N) * (1, N, 3) = (B, N, N)

    # Compute SVD for optimal rotation on the covarinace matrix
    U, S, Vt = torch.linalg.svd(H) 

    # Compute Rotation Matrix Now
    R = torch.matmul(Vt.transpose(2, 1), U.transpose(2, 1))

    # Special Reflection case
    det = torch.linalg.det(R)
    mask = det < 0.0
    if mask.sum() >= 1:
        Vt_copy = Vt.clone()
        Vt_copy[mask, -1, :] *= -1
        R = torch.matmul(Vt_copy.transpose(2, 1), U.transpose(2, 1))

    # Translate vector: I want (B, 1, 3) or (B, 3)
    # (1, 1, 3) - ( (B, 1, 3) * (B, 3, 3) ) =  (1, 1, 3) - (B, 3, 3) =  (B, 3, 3)
    # t = centroid_Q - torch.matmul(centroid_P.unsqueeze(1), R.transpose(2, 1)).squeeze(1)

    t = centroid_Q.squeeze(1) - torch.matmul(R, centroid_P.squeeze(1).unsqueeze(2)).squeeze(2)
    
    # Rotate points accordingly
    # P_aligned = torch.matmul(R, P_centered.transpose(2, 1)).transpose(2, 1) + t.unsqueeze(1)
    # P_aligned = torch.matmul(R, P_point.transpose(2, 1)).transpose(2, 1) + t.unsqueeze(1)
    P_aligned = torch.matmul(P_centered, R.transpose(2, 1)) + centroid_Q.unsqueeze(1)
    # print(P_aligned.shape)
    
    # diff = P_aligned - Q
    # rmsd = torch.sqrt((diff * diff).sum(1) / P.size(1))
    # print("RSMD Whole:", rmsd.mean().item())
    
    # diff_motif = P_aligned - Q_point
    # rmsd_motif = torch.sqrt((diff_motif * diff_motif).sum(1) / P_point.size(1))
    # print("RSMD Motif:", rmsd_motif.mean().item())

    # torch.save({"paligned": P_aligned, "control": Q_point}, "paligned.pt")

    return P_aligned, t, R

def kabsch_rootatom(P: torch.Tensor, Q: torch.Tensor, 
                    motif_output: torch.Tensor = slice(None),
                    motif_input: torch.Tensor = slice(None)):
    """Perform the kabsch algorithm to find the optimal rotation matrix
    that aligns P to points Q

    PARAMS
    ------
    P: Tensor (B, N, 3)
        Predicted DoF XYZ points
    Q: Tensor (N, 3)
        PDB Peptide Heavy atom XYZ Points
    motif_output: Tensor (N, A)
        atom index for model outputs
    motif_input: Tensor (M, A)
        atom index for input PDB

    RETURNS
    -------
    P_aligned: Tensor (B, N, 3)
        Aligned DoF XYZ points
    """
    # Calc Translation
    if motif_input.size(0) == Q.size(0):
        Q_point = Q
    else:
        Q_point = Q[motif_input, :] # Same N atom
    P_point = torch.gather(P, 1, motif_output.unsqueeze(-1).expand(-1, -1, 3))

    assert len(P_point.shape) == 3, f"Number of dimensions in our P_point is incorrect. P_point dim size: {P_point.shape}"
    assert len(Q_point.shape) == 2, f"Number of dimensions in our Q_point is incorrect. Q_point dim size: {Q_point.shape}"

    # Compute centroids
    centroid_P = P_point.mean(dim=1) # (B, N, 3) -> (B, 3)
    centroid_Q = Q_point.mean(dim=0,keepdim=True) # (N, 3) -> (1, 3)

    # Centered Points
    P_centered = P_point - centroid_P.unsqueeze(1) # (B, N, 3) - (B, 1, 3) = (B, N 3)
    Q_centered = Q_point - centroid_Q # (N, 3) - (1, 3) = (N, 3)

    # Calculate covariance matrix
    # H = torch.matmul(P_centered.transpose(2, 1), Q_centered.unsqueeze(0)) # (B, 3, N) * (1, N, 3) = (B, N, N)
    H = torch.matmul(P_centered.transpose(2, 1), Q_centered) # (B, 3, N) * (1, N, 3) = (B, N, N)

    # Compute SVD for optimal rotation on the covarinace matrix
    U, S, Vt = torch.linalg.svd(H) 

    # Compute Rotation Matrix Now
    R = torch.matmul(Vt.transpose(2, 1), U.transpose(2, 1))

    # Special Reflection case
    det = torch.linalg.det(R)
    mask = det < 0.0
    if mask.sum() >= 1:
        Vt_copy = Vt.clone()
        Vt_copy[mask, -1, :] *= -1
        R = torch.matmul(Vt_copy.transpose(2, 1), U.transpose(2, 1))

    # Translate vector: I want (B, 1, 3) or (B, 3)
    # (1, 1, 3) - ( (B, 1, 3) * (B, 3, 3) ) =  (1, 1, 3) - (B, 3, 3) =  (B, 3, 3)
    # t = centroid_Q - torch.matmul(centroid_P.unsqueeze(1), R.transpose(2, 1))
    t = centroid_Q.squeeze(1) - torch.matmul(R, centroid_P.squeeze(1).unsqueeze(2)).squeeze(2)
    
    # Rotate points accordingly
    # P_aligned = torch.matmul(R, P_centered.transpose(2, 1)).transpose(2, 1) + t.unsqueeze(1)
    # P_aligned = torch.matmul(R, P.transpose(2, 1)).transpose(2, 1) + t
    P_aligned = torch.matmul(P_centered, R.transpose(2, 1)) + centroid_Q.unsqueeze(1)
    
    # diff = P_aligned - Q
    # rmsd = torch.sqrt((diff * diff).sum(1) / P.size(1))
    # print("RSMD Whole:", rmsd.mean().item())
    
    if motif_input.size(0) == Q.size(0):
        diff_motif = torch.norm(P_aligned - Q.unsqueeze(0), dim=-1)
    else:
        diff_motif = torch.norm(P_aligned - Q[motif_input, :].unsqueeze(0), dim=-1)
#     rmsd_motif = torch.sqrt(diff_motif.pow(2).sum((1,2)) / P.size(1))
#     rmsd_motif = torch.sqrt(diff_motif.pow(2).sum(-1).mean(-1))
    rmsd_motif = torch.sqrt(diff_motif.pow(2).sum(-1) / P.size(1))
    print("Batch Mean RSMD Motif:", rmsd_motif.mean().item())
#     print("RSMD Motif:", rmsd_motif.mean().item())

    P_total = torch.matmul(P, R.transpose(2, 1)) + t.unsqueeze(1)

    # torch.save({"paligned": P_aligned, "control": Q}, "paligned.pt")

    if motif_input == slice(None):
        return P_aligned, t, R
    else:
        return P_total, rmsd_motif.mean()

## Keep this as the original
# def kabsch_rootatom(P: torch.Tensor, Q: torch.Tensor, motif: torch.Tensor = slice(None)):
#     """Perform the kabsch algorithm to find the optimal rotation matrix
#     that aligns P to points Q
#
#     PARAMS
#     ------
#     P: Tensor (B, N, 3)
#         Predicted DoF XYZ points
#     Q: Tensor (N, 3)
#         PDB Peptide Heavy atom XYZ Points
#     motif: Tensor (A)
#         atom indexs 
#
#     RETURNS
#     -------
#     P_aligned: Tensor (B, N, 3)
#         Aligned DoF XYZ points
#     """
#     # Calc Translation
#     P_point = P[:, motif, :] # N atom of 4th residue root
#     Q_point = Q[motif, :] # Same N atom
#
#
#     # Compute centroids
#     centroid_P = P_point.mean(dim=1) # (B, 3)
#     centroid_Q = Q_point.mean(dim=0) # (3)
#
#     # Centered Points
#     P_centered = P_point - centroid_P.unsqueeze(1) # (B, N, 3) - (B, 1, 3) = (B, N 3)
#     Q_centered = Q_point - centroid_Q.unsqueeze(0) # (N, 3) - (1, 3) = (N, 3)
#
#     # Calculate covariance matrix
#     H = torch.matmul(P_centered.transpose(2, 1), Q_centered.unsqueeze(0)) # (B, 3, N) * (1, N, 3) = (B, N, N)
#
#     # Compute SVD for optimal rotation on the covarinace matrix
#     U, S, Vt = torch.linalg.svd(H) 
#
#     # Compute Rotation Matrix Now
#     R = torch.matmul(Vt.transpose(2, 1), U.transpose(2, 1))
#
#     # Special Reflection case
#     det = torch.linalg.det(R)
#     mask = det < 0.0
#     if mask.sum() >= 1:
#         Vt_copy = Vt.clone()
#         Vt_copy[mask, -1, :] *= -1
#         R = torch.matmul(Vt_copy.transpose(2, 1), U.transpose(2, 1))
#
#     # Translate vector: I want (B, 1, 3) or (B, 3)
#     # (1, 1, 3) - ( (B, 1, 3) * (B, 3, 3) ) =  (1, 1, 3) - (B, 3, 3) =  (B, 3, 3)
#     t = centroid_Q - torch.matmul(centroid_P.unsqueeze(1), R.transpose(2, 1))
#     
#     # Rotate points accordingly
#     # P_aligned = torch.matmul(R, P_centered.transpose(2, 1)).transpose(2, 1) + t.unsqueeze(1)
#     P_aligned = torch.matmul(R, P.transpose(2, 1)).transpose(2, 1) + t
#     # print(P_aligned.shape)
#     
#     # diff = P_aligned - Q
#     # rmsd = torch.sqrt((diff * diff).sum(1) / P.size(1))
#     # print("RSMD Whole:", rmsd.mean().item())
#     
#     diff_motif = P_aligned[:, motif, :] - Q[motif, :]
#     rmsd_motif = torch.sqrt((diff_motif * diff_motif).sum(1) / P.size(1))
#     print("RSMD Motif:", rmsd_motif.mean().item())
#
#     torch.save({"paligned": P_aligned, "control": Q}, "paligned.pt")
#
#     if motif == slice(None):
#         return P_aligned, t, R
#     else:
#         return P_aligned

def kabsch_algorithm(P: torch.Tensor, Q: torch.Tensor):
    """Perform the kabsch algorithm to find the optimal rotation matrix
    that aligns P to points Q

    PARAMS
    ------
    P: Tensor (B, N, 3)
        Predicted DoF XYZ points
    Q: Tensor (N, 3)
        PDB Peptide Heavy atom XYZ Points

    RETURNS
    -------
    P_aligned: Tensor (B, N, 3)
        Aligned DoF XYZ points
    """
    # Calc centroids
    centroid_P = P.mean(dim=1) # (B, N, 3) -> (B, 3)
    centroid_Q = Q.mean(dim=0) if len(Q.shape) == 2 else Q.mean(dim=1)# (N, 3) -> (3)

    # Center the points base don centroid
    P_centered = P - centroid_P.unsqueeze(1) # (B, N, 3) - (B, 1, 3) = (B, N, 3)
    Q_centered = Q - centroid_Q if len(Q.shape) == 2 else Q - centroid_Q.unsqueeze(1)# (N, 3) - (3) = (N, 3)

    # (B, N, 3) -> (B, 3) -> (B, N, 3) - (B, 1, 3) = (B, N, 3)

    # optimal translation
    t = centroid_Q - centroid_P # (B, 1, 3) x (B, 1, 3) = (B, 1, 3)

    # Adjust Q_centered to match
    if len(Q.shape) == 2:
        Q_centered = Q_centered.unsqueeze(0).repeat(P_centered.size(0), 1, 1) # (1, N, 3) -> (B, N, 3)

    # Compute the Covariance matrix
    H = torch.matmul(P_centered.transpose(-2, -1), Q_centered) # (B, 3, N) x (B, N, 3) = (B, 3, 3)

    # Compute SVD for optimal rotation
    U, S, Vt = torch.linalg.svd(H) 

    # Compute Rotation Matrix Now
    # mask = (torch.linalg.det(U) * torch.linalg.det(Vt)) < 0.0
    # R = torch.matmul(Vt.transpose(-2, -1), U.transpose(-2, -1))
    R = torch.matmul(Vt.transpose(-2, -1), U.transpose(-2, -1))

    # Special Reflection case
    det = torch.linalg.det(R)
    mask = det < 0.0
    if mask.sum() >= 1:
        Vt_copy = Vt.clone()
        # U_copy = U.clone()
        # U_copy[mask, :, -1] *= -1
        Vt_copy[mask, -1, :] *= -1
        R = torch.matmul(Vt_copy.transpose(1, 2), U.transpose(1, 2))
        # R = torch.matmul(Vt_copy.transpose(-2, -1), U.transpose(-2, -1))
        # R = torch.matmul(U_copy.transpose(-2, -1), Vt.transpose(-2, -1))
    # else:
    #     R = torch.matmul(U.transpose(-2, -1), Vt.transpose(-2, -1))

    # print(torch.matmul(P_centered, R).shape)
    # Apply Rotation
    # if len(Q.shape) == 2:
    #     P_aligned = torch.matmul(P_centered, R.transpose(1,2)) - centroid_Q
    #     # print((R*centroid_P.unsqueeze(1)).shape)
    #     # translation = centroid_Q.unsqueeze(1) - R*centroid_P.unsqueeze(1)  
    # else:
    #     P_aligned = torch.matmul(P_centered, R.transpose(1,2)) - centroid_Q.unsqueeze(1)
        # translation = centroid_Q.unsqueeze(1) - torch.matmul(R, centroid_P)

    # Now Rotate and translate
    P_aligned = torch.matmul(R, P_centered.transpose(2, 1)).transpose(2, 1) + t.unsqueeze(1)

    diff = P_aligned - Q
    rmsd = torch.sqrt((diff * diff).sum(1) / P.size(1))
    print("RSMD:", rmsd.mean())

    # torch.save({"paligned": P_aligned, "control": Q}, "paligned.pt")

    # Original without Rigid Body Transformation loss terms added
    # return P_aligned
    # After terms added
    return P_aligned, t, R

def pdb_index_grab(split_motifs: list) -> list:
    """
    Extract out the coordinates needed for our particular design of interest

    :peptide_atoms: a tensor of size (N * 4, 3) where N is the number of residues
    :split_motifs: list of torch.Tensor residue indexes (0 index)
    """
    out = list()
    print("SPLIT MOTIFS in pdb_index_grab:", split_motifs)
    for region in split_motifs:
        scaled = (region+1) * 4
        out.append(torch.tensor(
            [i for i in range(scaled-4, scaled)]
        ))
    return out

def split_motif_atoms(residue_index: torch.Tensor) -> list:
    """
    Split our motif atoms into multiple regions if they are discontiguous

    PARAMS
    ------
    :residue_index: Tensor of residue positions (0 index)
    
    RETURNS
    -------
    :regions (List[torch.Tensor]): Each region's index in their own tensor
    """
    # Init variables
    regions = list()
    p = 0
    l = 0
    count = 0 # As we are setup for only 2 discontiguous motifs at the moment
    for i in range(1, residue_index.shape[0]):
        check = residue_index[p]+1 != residue_index[i]
        if check:
            regions.append(residue_index[l:i])
            l = i
        p+=1
        count+=1
        if (i+1 == residue_index.shape[0]) and (check):
            regions.append(residue_index[-1].unsqueeze(0))
            count+=1
        elif (i+1 == residue_index.shape[0]) and (not check):
            regions.append(residue_index[l:])
            count+=1
    # make sure that we are only taking into account 2 regions no more
    assert count <= 2, "The current approach is only built for 2 discontinuous regions no more."
    return pdb_index_grab(regions), regions


def discontig_motif_loss(region1: torch.Tensor, region2: torch.Tensor,
                         # atoms_native: torch.Tensor,
                         t_native: torch.Tensor, R_native: torch.Tensor,
                         t_lambda: int = 1, R_lambda: int = 1):
    """
    Run Kabsch Algo to get the translation and Rotation of the rigid body our motifs.
    Calculate loss based on the R and t difference from ideal

    ***Assuming these are two regions. Where region1 is contiguous and region2 is contiguous***

    PARAMS
    ------
    :region1: A contiguous indexing tensor for atoms associated with this region of the motif
    :region2: A contiguous indexing tensor for atoms associated with this (2nd) region of the motif
    :t_native: The native translation vector
    :R_native: The native Rotation vector 

    RETURNS
    -------
    :motif_loss (Tensor[float]): Combined translation and Rotation loss terms
    """
    # Calculate current t and R 
    _, t_pred, R_pred = kabsch_rigid_transform(region1, region2)

    t_loss = torch.pow(t_pred - t_native, 2).mean(-1)
    r_loss = torch.pow(R_pred - R_native, 2).mean((-2, -1)) #.sum(-1).mean(-1)
    # print(r_loss, r_loss * R_lambda)
    combo = t_loss * t_lambda + r_loss * R_lambda

    return combo

def kabsch_test():
    # set seed
    torch.manual_seed(123)

    P = torch.randn(10, 32, 3)

    alpha = torch.randn(10) * 2 * torch.pi
    R = torch.stack([torch.tensor([[torch.cos(a), -torch.sin(a), 0],
                                   [torch.sin(a), torch.cos(a), 0],
                                   [0, 0, 1]]) for a in alpha], axis=0)
    print(R)
    t = torch.randn(10, 3) * 10

    Q = torch.matmul(P, R.transpose(2, 1)) + t[:, None, :]

    R_opt, t_opt, P_aligned = kabsch_algorithm(P, Q)

    l2_t = torch.norm(t - t_opt, dim=1)
    l2_R = torch.norm(R - R_opt, dim=(1,2))
    print("L2 t: {}".format(l2_t.mean()))
    print("L2 R: {}".format(l2_R.mean()))
    return 0

def find_weights(weights: str, weight_path: str) -> str:
    """
    Find the correct path to the weights file should work for 
    either relative or google colab file path search

    PARAMS
    ------
    :weights: The relative path to our file.
    :weight_path: A custom path to search

    RETURNS
    -------
    :path: The true path to our weights
    """
    # If a weight path is given then load and return that 
    if weight_path.endswith(".pt") and os.path.exists(weight_path):
        return weight_path

    path_prefix = ["../", "/content/conformational_autoencoder/", 
                   "/projects/parisahlab/apowers4/projects/conformational_autoencoder/",
                  weight_path,]
    for prefix in path_prefix:
        p = os.path.join(prefix, weights)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Seems that our weights {weights} cannot be found")
    

def load_weights(size: int, device: str, weight_path: str) -> torch.Tensor:
    """Load in weights for a specific size"""
    if size == 6:
        return torch.load(find_weights("weights/6mer/model_1610_current_weights.pt", weight_path),
                          map_location = device)
    elif size == 7:
        return torch.load(find_weights("weights/7mer/model_8811_best_weights.pt", weight_path),
                          map_location = device)
    elif size == 8:
        return torch.load(find_weights("full_weights/8mer/model_7731_best_weights.pt", weight_path),
                          map_location = device)
    else:
        # return torch.load(find_weights("weights/9mer/model_9890_current_weights.pt"),
        return torch.load(find_weights("weights/9mer/model_6049_best_weights.pt", weight_path),
                          map_location = device)

def size_help(size: int, params: dict) -> Union[dict, int]:
    """Alter the size of the input"""
    if size == 6:
        params["torsion_main_n"]       = 44
        params["torsion_side_n"]       = 0
        params["bond_angle_main_n"]    = 44
        params["bond_angle_side_n"]    = 0
        params["bond_length_main_n"]   = 44
        params["bond_length_side_n"]   = 0
        params["jump_n"]               = 0
        params["energy_n"]             = 3
        params["normalize"]            = True
        params["reference_pose"]       = "foo"
        params["ref_pose_len"]         = 128
        params["output_size"]          = 135
        params["input_size"]           = 135
        params["protein_len"]          = 6
        params["dof_residue_metadata"] = torch.tensor([[1., 1., 1.],
        [1., 2., 1.],
        [1., 3., 1.],
        [1., 4., 1.],
        [1., 5., 1.],
        [1., 6., 1.],
        [1., 7., 1.],
        [1., 8., 1.],
        [2., 1., 1.],
        [2., 2., 1.],
        [2., 3., 1.],
        [2., 4., 1.],
        [2., 5., 1.],
        [2., 6., 1.],
        [2., 7., 1.],
        [3., 1., 1.],
        [3., 3., 1.],
        [3., 4., 1.],
        [3., 5., 1.],
        [3., 6., 1.],
        [3., 7., 1.],
        [4., 1., 1.],
        [4., 2., 1.],
        [4., 3., 1.],
        [4., 4., 1.],
        [4., 5., 1.],
        [4., 6., 1.],
        [4., 7., 1.],
        [5., 1., 1.],
        [5., 2., 1.],
        [5., 3., 1.],
        [5., 4., 1.],
        [5., 5., 1.],
        [5., 6., 1.],
        [5., 7., 1.],
        [6., 1., 1.],
        [6., 2., 1.],
        [6., 3., 1.],
        [6., 4., 1.],
        [6., 5., 1.],
        [6., 6., 1.],
        [6., 7., 1.],
        [6., 8., 1.],
        [6., 9., 1.],
        [1., 1., 2.],
        [1., 2., 2.],
        [1., 3., 2.],
        [1., 4., 2.],
        [1., 5., 2.],
        [1., 6., 2.],
        [1., 7., 2.],
        [1., 8., 2.],
        [2., 1., 2.],
        [2., 2., 2.],
        [2., 3., 2.],
        [2., 4., 2.],
        [2., 5., 2.],
        [2., 6., 2.],
        [2., 7., 2.],
        [3., 1., 2.],
        [3., 3., 2.],
        [3., 4., 2.],
        [3., 5., 2.],
        [3., 6., 2.],
        [3., 7., 2.],
        [4., 1., 2.],
        [4., 2., 2.],
        [4., 3., 2.],
        [4., 4., 2.],
        [4., 5., 2.],
        [4., 6., 2.],
        [4., 7., 2.],
        [5., 1., 2.],
        [5., 2., 2.],
        [5., 3., 2.],
        [5., 4., 2.],
        [5., 5., 2.],
        [5., 6., 2.],
        [5., 7., 2.],
        [6., 1., 2.],
        [6., 2., 2.],
        [6., 3., 2.],
        [6., 4., 2.],
        [6., 5., 2.],
        [6., 6., 2.],
        [6., 7., 2.],
        [6., 8., 2.],
        [6., 9., 2.],
        [1., 1., 3.],
        [1., 2., 3.],
        [1., 3., 3.],
        [1., 4., 3.],
        [1., 5., 3.],
        [1., 6., 3.],
        [1., 7., 3.],
        [1., 8., 3.],
        [2., 1., 3.],
        [2., 2., 3.],
        [2., 3., 3.],
        [2., 4., 3.],
        [2., 5., 3.],
        [2., 6., 3.],
        [2., 7., 3.],
        [3., 1., 3.],
        [3., 3., 3.],
        [3., 4., 3.],
        [3., 5., 3.],
        [3., 6., 3.],
        [3., 7., 3.],
        [4., 1., 3.],
        [4., 2., 3.],
        [4., 3., 3.],
        [4., 4., 3.],
        [4., 5., 3.],
        [4., 6., 3.],
        [4., 7., 3.],
        [5., 1., 3.],
        [5., 2., 3.],
        [5., 3., 3.],
        [5., 4., 3.],
        [5., 5., 3.],
        [5., 6., 3.],
        [5., 7., 3.],
        [6., 1., 3.],
        [6., 2., 3.],
        [6., 3., 3.],
        [6., 4., 3.],
        [6., 5., 3.],
        [6., 6., 3.],
        [6., 7., 3.],
        [6., 8., 3.],
        [6., 9., 3.]], dtype=torch.float64)
    elif size == 7:
        params["normalize"]            = True
        params["reference_pose"]       = "foo"
        params["torsion_main_n"]     = 51 
        params["torsion_side_n"]     = 0
        params["bond_angle_main_n"]  = 51
        params["bond_angle_side_n"]  = 0
        params["bond_length_main_n"] = 51
        params["bond_length_side_n"] = 0
        params["jump_n"]             = 0
        params["energy_n"]           = 3
        params["protein_len"]        = 7 
        params["output_size"]        = 156
        params["input_size"]         = 156
        params['dof_residue_metadata'] = torch.tensor([[1., 1., 1.],
         [1., 2., 1.],
         [1., 3., 1.],
         [1., 4., 1.],
         [1., 5., 1.],
         [1., 6., 1.],
         [1., 7., 1.],
         [1., 8., 1.],
         [2., 1., 1.],
         [2., 2., 1.],
         [2., 3., 1.],
         [2., 4., 1.],
         [2., 5., 1.],
         [2., 6., 1.],
         [2., 7., 1.],
         [3., 1., 1.],
         [3., 3., 1.],
         [3., 4., 1.],
         [3., 5., 1.],
         [3., 6., 1.],
         [3., 7., 1.],
         [4., 1., 1.],
         [4., 2., 1.],
         [4., 3., 1.],
         [4., 4., 1.],
         [4., 5., 1.],
         [4., 6., 1.],
         [4., 7., 1.],
         [5., 1., 1.],
         [5., 2., 1.],
         [5., 3., 1.],
         [5., 4., 1.],
         [5., 5., 1.],
         [5., 6., 1.],
         [5., 7., 1.],
         [6., 1., 1.],
         [6., 2., 1.],
         [6., 3., 1.],
         [6., 4., 1.],
         [6., 5., 1.],
         [6., 6., 1.],
         [6., 7., 1.],
         [7., 1., 1.],
         [7., 2., 1.],
         [7., 3., 1.],
         [7., 4., 1.],
         [7., 5., 1.],
         [7., 6., 1.],
         [7., 7., 1.],
         [7., 8., 1.],
         [7., 9., 1.],
         [1., 1., 2.],
         [1., 2., 2.],
         [1., 3., 2.],
         [1., 4., 2.],
         [1., 5., 2.],
         [1., 6., 2.],
         [1., 7., 2.],
         [1., 8., 2.],
         [2., 1., 2.],
         [2., 2., 2.],
         [2., 3., 2.],
         [2., 4., 2.],
         [2., 5., 2.],
         [2., 6., 2.],
         [2., 7., 2.],
         [3., 1., 2.],
         [3., 3., 2.],
         [3., 4., 2.],
         [3., 5., 2.],
         [3., 6., 2.],
         [3., 7., 2.],
         [4., 1., 2.],
         [4., 2., 2.],
         [4., 3., 2.],
         [4., 4., 2.],
         [4., 5., 2.],
         [4., 6., 2.],
         [4., 7., 2.],
         [5., 1., 2.],
         [5., 2., 2.],
         [5., 3., 2.],
         [5., 4., 2.],
         [5., 5., 2.],
         [5., 6., 2.],
         [5., 7., 2.],
         [6., 1., 2.],
         [6., 2., 2.],
         [6., 3., 2.],
         [6., 4., 2.],
         [6., 5., 2.],
         [6., 6., 2.],
         [6., 7., 2.],
         [7., 1., 2.],
         [7., 2., 2.],
         [7., 3., 2.],
         [7., 4., 2.],
         [7., 5., 2.],
         [7., 6., 2.],
         [7., 7., 2.],
         [7., 8., 2.],
         [7., 9., 2.],
         [1., 1., 3.],
         [1., 2., 3.],
         [1., 3., 3.],
         [1., 4., 3.],
         [1., 5., 3.],
         [1., 6., 3.],
         [1., 7., 3.],
         [1., 8., 3.],
         [2., 1., 3.],
         [2., 2., 3.],
         [2., 3., 3.],
         [2., 4., 3.],
         [2., 5., 3.],
         [2., 6., 3.],
         [2., 7., 3.],
         [3., 1., 3.],
         [3., 3., 3.],
         [3., 4., 3.],
         [3., 5., 3.],
         [3., 6., 3.],
         [3., 7., 3.],
         [4., 1., 3.],
         [4., 2., 3.],
         [4., 3., 3.],
         [4., 4., 3.],
         [4., 5., 3.],
         [4., 6., 3.],
         [4., 7., 3.],
         [5., 1., 3.],
         [5., 2., 3.],
         [5., 3., 3.],
         [5., 4., 3.],
         [5., 5., 3.],
         [5., 6., 3.],
         [5., 7., 3.],
         [6., 1., 3.],
         [6., 2., 3.],
         [6., 3., 3.],
         [6., 4., 3.],
         [6., 5., 3.],
         [6., 6., 3.],
         [6., 7., 3.],
         [7., 1., 3.],
         [7., 2., 3.],
         [7., 3., 3.],
         [7., 4., 3.],
         [7., 5., 3.],
         [7., 6., 3.],
         [7., 7., 3.],
         [7., 8., 3.],
         [7., 9., 3.]], dtype=torch.float64)
    elif size == 8:
        params["output_size"]          = 177
        params["input_size"]           = 177
        params["protein_len"]          = 8
        params["torsion_main_n"]       = 58
        params["torsion_side_n"]       = 0
        params["bond_angle_main_n"]    = 58
        params["bond_angle_side_n"]    = 0
        params["bond_length_main_n"]   = 58
        params["bond_length_side_n"]   = 0
        params["normalize"]            = True
        params["reference_pose"]       = "foo"
        params["ref_pose_len"]         = 128
        params["jump_n"]               = 0
        params["energy_n"]             = 3
        params['dof_residue_metadata'] = torch.tensor([[1., 1., 1.],
        [1., 2., 1.],
        [1., 3., 1.],
        [1., 4., 1.],
        [1., 5., 1.],
        [1., 6., 1.],
        [1., 7., 1.],
        [1., 8., 1.],
        [2., 1., 1.],
        [2., 2., 1.],
        [2., 3., 1.],
        [2., 4., 1.],
        [2., 5., 1.],
        [2., 6., 1.],
        [2., 7., 1.],
        [3., 1., 1.],
        [3., 2., 1.],
        [3., 3., 1.],
        [3., 4., 1.],
        [3., 5., 1.],
        [3., 6., 1.],
        [3., 7., 1.],
        [4., 1., 1.],
        [4., 3., 1.],
        [4., 4., 1.],
        [4., 5., 1.],
        [4., 6., 1.],
        [4., 7., 1.],
        [5., 1., 1.],
        [5., 2., 1.],
        [5., 3., 1.],
        [5., 4., 1.],
        [5., 5., 1.],
        [5., 6., 1.],
        [5., 7., 1.],
        [6., 1., 1.],
        [6., 2., 1.],
        [6., 3., 1.],
        [6., 4., 1.],
        [6., 5., 1.],
        [6., 6., 1.],
        [6., 7., 1.],
        [7., 1., 1.],
        [7., 2., 1.],
        [7., 3., 1.],
        [7., 4., 1.],
        [7., 5., 1.],
        [7., 6., 1.],
        [7., 7., 1.],
        [8., 1., 1.],
        [8., 2., 1.],
        [8., 3., 1.],
        [8., 4., 1.],
        [8., 5., 1.],
        [8., 6., 1.],
        [8., 7., 1.],
        [8., 8., 1.],
        [8., 9., 1.],
        [1., 1., 2.],
        [1., 2., 2.],
        [1., 3., 2.],
        [1., 4., 2.],
        [1., 5., 2.],
        [1., 6., 2.],
        [1., 7., 2.],
        [1., 8., 2.],
        [2., 1., 2.],
        [2., 2., 2.],
        [2., 3., 2.],
        [2., 4., 2.],
        [2., 5., 2.],
        [2., 6., 2.],
        [2., 7., 2.],
        [3., 1., 2.],
        [3., 2., 2.],
        [3., 3., 2.],
        [3., 4., 2.],
        [3., 5., 2.],
        [3., 6., 2.],
        [3., 7., 2.],
        [4., 1., 2.],
        [4., 3., 2.],
        [4., 4., 2.],
        [4., 5., 2.],
        [4., 6., 2.],
        [4., 7., 2.],
        [5., 1., 2.],
        [5., 2., 2.],
        [5., 3., 2.],
        [5., 4., 2.],
        [5., 5., 2.],
        [5., 6., 2.],
        [5., 7., 2.],
        [6., 1., 2.],
        [6., 2., 2.],
        [6., 3., 2.],
        [6., 4., 2.],
        [6., 5., 2.],
        [6., 6., 2.],
        [6., 7., 2.],
        [7., 1., 2.],
        [7., 2., 2.],
        [7., 3., 2.],
        [7., 4., 2.],
        [7., 5., 2.],
        [7., 6., 2.],
        [7., 7., 2.],
        [8., 1., 2.],
        [8., 2., 2.],
        [8., 3., 2.],
        [8., 4., 2.],
        [8., 5., 2.],
        [8., 6., 2.],
        [8., 7., 2.],
        [8., 8., 2.],
        [8., 9., 2.],
        [1., 1., 3.],
        [1., 2., 3.],
        [1., 3., 3.],
        [1., 4., 3.],
        [1., 5., 3.],
        [1., 6., 3.],
        [1., 7., 3.],
        [1., 8., 3.],
        [2., 1., 3.],
        [2., 2., 3.],
        [2., 3., 3.],
        [2., 4., 3.],
        [2., 5., 3.],
        [2., 6., 3.],
        [2., 7., 3.],
        [3., 1., 3.],
        [3., 2., 3.],
        [3., 3., 3.],
        [3., 4., 3.],
        [3., 5., 3.],
        [3., 6., 3.],
        [3., 7., 3.],
        [4., 1., 3.],
        [4., 3., 3.],
        [4., 4., 3.],
        [4., 5., 3.],
        [4., 6., 3.],
        [4., 7., 3.],
        [5., 1., 3.],
        [5., 2., 3.],
        [5., 3., 3.],
        [5., 4., 3.],
        [5., 5., 3.],
        [5., 6., 3.],
        [5., 7., 3.],
        [6., 1., 3.],
        [6., 2., 3.],
        [6., 3., 3.],
        [6., 4., 3.],
        [6., 5., 3.],
        [6., 6., 3.],
        [6., 7., 3.],
        [7., 1., 3.],
        [7., 2., 3.],
        [7., 3., 3.],
        [7., 4., 3.],
        [7., 5., 3.],
        [7., 6., 3.],
        [7., 7., 3.],
        [8., 1., 3.],
        [8., 2., 3.],
        [8., 3., 3.],
        [8., 4., 3.],
        [8., 5., 3.],
        [8., 6., 3.],
        [8., 7., 3.],
        [8., 8., 3.],
        [8., 9., 3.]], dtype=torch.float64)
    elif size == 9:
        params["torsion_main_n"]       = 65
        params["torsion_side_n"]       = 0
        params["bond_angle_main_n"]    = 65
        params["bond_angle_side_n"]    = 0
        params["bond_length_main_n"]   = 65
        params["bond_length_side_n"]   = 0
        params["output_size"]          = 198
        params["input_size"]           = 198
        params["normalize"]            = True
        params["reference_pose"]       = "foo"
        params["ref_pose_len"]         = 128
        params["jump_n"]               = 0
        params["energy_n"]             = 3 
        params["protein_len"]          = 9
        params["dof_residue_metadata"] = torch.tensor([[1., 1., 1.],
        [1., 2., 1.],
        [1., 3., 1.],
        [1., 4., 1.],
        [1., 5., 1.],
        [1., 6., 1.],
        [1., 7., 1.],
        [1., 8., 1.],
        [2., 1., 1.],
        [2., 2., 1.],
        [2., 3., 1.],
        [2., 4., 1.],
        [2., 5., 1.],
        [2., 6., 1.],
        [2., 7., 1.],
        [3., 1., 1.],
        [3., 2., 1.],
        [3., 3., 1.],
        [3., 4., 1.],
        [3., 5., 1.],
        [3., 6., 1.],
        [3., 7., 1.],
        [4., 1., 1.],
        [4., 3., 1.],
        [4., 4., 1.],
        [4., 5., 1.],
        [4., 6., 1.],
        [4., 7., 1.],
        [5., 1., 1.],
        [5., 2., 1.],
        [5., 3., 1.],
        [5., 4., 1.],
        [5., 5., 1.],
        [5., 6., 1.],
        [5., 7., 1.],
        [6., 1., 1.],
        [6., 2., 1.],
        [6., 3., 1.],
        [6., 4., 1.],
        [6., 5., 1.],
        [6., 6., 1.],
        [6., 7., 1.],
        [7., 1., 1.],
        [7., 2., 1.],
        [7., 3., 1.],
        [7., 4., 1.],
        [7., 5., 1.],
        [7., 6., 1.],
        [7., 7., 1.],
        [8., 1., 1.],
        [8., 2., 1.],
        [8., 3., 1.],
        [8., 4., 1.],
        [8., 5., 1.],
        [8., 6., 1.],
        [8., 7., 1.],
        [9., 1., 1.],
        [9., 2., 1.],
        [9., 3., 1.],
        [9., 4., 1.],
        [9., 5., 1.],
        [9., 6., 1.],
        [9., 7., 1.],
        [9., 8., 1.],
        [9., 9., 1.],
        [1., 1., 2.],
        [1., 2., 2.],
        [1., 3., 2.],
        [1., 4., 2.],
        [1., 5., 2.],
        [1., 6., 2.],
        [1., 7., 2.],
        [1., 8., 2.],
        [2., 1., 2.],
        [2., 2., 2.],
        [2., 3., 2.],
        [2., 4., 2.],
        [2., 5., 2.],
        [2., 6., 2.],
        [2., 7., 2.],
        [3., 1., 2.],
        [3., 2., 2.],
        [3., 3., 2.],
        [3., 4., 2.],
        [3., 5., 2.],
        [3., 6., 2.],
        [3., 7., 2.],
        [4., 1., 2.],
        [4., 3., 2.],
        [4., 4., 2.],
        [4., 5., 2.],
        [4., 6., 2.],
        [4., 7., 2.],
        [5., 1., 2.],
        [5., 2., 2.],
        [5., 3., 2.],
        [5., 4., 2.],
        [5., 5., 2.],
        [5., 6., 2.],
        [5., 7., 2.],
        [6., 1., 2.],
        [6., 2., 2.],
        [6., 3., 2.],
        [6., 4., 2.],
        [6., 5., 2.],
        [6., 6., 2.],
        [6., 7., 2.],
        [7., 1., 2.],
        [7., 2., 2.],
        [7., 3., 2.],
        [7., 4., 2.],
        [7., 5., 2.],
        [7., 6., 2.],
        [7., 7., 2.],
        [8., 1., 2.],
        [8., 2., 2.],
        [8., 3., 2.],
        [8., 4., 2.],
        [8., 5., 2.],
        [8., 6., 2.],
        [8., 7., 2.],
        [9., 1., 2.],
        [9., 2., 2.],
        [9., 3., 2.],
        [9., 4., 2.],
        [9., 5., 2.],
        [9., 6., 2.],
        [9., 7., 2.],
        [9., 8., 2.],
        [9., 9., 2.],
        [1., 1., 3.],
        [1., 2., 3.],
        [1., 3., 3.],
        [1., 4., 3.],
        [1., 5., 3.],
        [1., 6., 3.],
        [1., 7., 3.],
        [1., 8., 3.],
        [2., 1., 3.],
        [2., 2., 3.],
        [2., 3., 3.],
        [2., 4., 3.],
        [2., 5., 3.],
        [2., 6., 3.],
        [2., 7., 3.],
        [3., 1., 3.],
        [3., 2., 3.],
        [3., 3., 3.],
        [3., 4., 3.],
        [3., 5., 3.],
        [3., 6., 3.],
        [3., 7., 3.],
        [4., 1., 3.],
        [4., 3., 3.],
        [4., 4., 3.],
        [4., 5., 3.],
        [4., 6., 3.],
        [4., 7., 3.],
        [5., 1., 3.],
        [5., 2., 3.],
        [5., 3., 3.],
        [5., 4., 3.],
        [5., 5., 3.],
        [5., 6., 3.],
        [5., 7., 3.],
        [6., 1., 3.],
        [6., 2., 3.],
        [6., 3., 3.],
        [6., 4., 3.],
        [6., 5., 3.],
        [6., 6., 3.],
        [6., 7., 3.],
        [7., 1., 3.],
        [7., 2., 3.],
        [7., 3., 3.],
        [7., 4., 3.],
        [7., 5., 3.],
        [7., 6., 3.],
        [7., 7., 3.],
        [8., 1., 3.],
        [8., 2., 3.],
        [8., 3., 3.],
        [8., 4., 3.],
        [8., 5., 3.],
        [8., 6., 3.],
        [8., 7., 3.],
        [9., 1., 3.],
        [9., 2., 3.],
        [9., 3., 3.],
        [9., 4., 3.],
        [9., 5., 3.],
        [9., 6., 3.],
        [9., 7., 3.],
        [9., 8., 3.],
        [9., 9., 3.]], dtype=torch.float64)
    return params, params["output_size"]

# def kabsch_algorithm(P: torch.Tensor, Q: torch.Tensor):
#     """Perform the kabsch algorithm to find the optimal rotation matrix
#     that aligns P to points Q
#
#     PARAMS
#     ------
#     P: Tensor (B, N, 3)
#         Predicted DoF XYZ points
#     Q: Tensor (N, 3)
#         PDB Peptide Heavy atom XYZ Points
#
#     RETURNS
#     -------
#     P_aligned: Tensor (B, N, 3)
#         Aligned DoF XYZ points
#     """
#     # Calc centroids
#     centroid_P = P.mean(dim=1) # (B, N, 3) -> (B, 3)
#     centroid_Q = Q.mean(dim=0) if len(Q.shape) == 2 else Q.mean(dim=1)# (N, 3) -> (3)
#
#     # Center the points base don centroid
#     P_centered = P - centroid_P.unsqueeze(1) # (B, N, 3) - (B, 1, 3) = (B, N, 3)
#     Q_centered = Q - centroid_Q if len(Q.shape) == 2 else Q - centroid_Q.unsqueeze(1)# (N, 3) - (3) = (N, 3)
#
#     # (B, N, 3) -> (B, 3) -> (B, N, 3) - (B, 1, 3) = (B, N, 3)
#
#     # Adjust Q_centered to match
#     if len(Q.shape) == 2:
#         Q_centered = Q_centered.unsqueeze(0).repeat(P_centered.size(0), 1, 1) # (1, N, 3) -> (B, N, 3)
#
#     # Compute the Covariance matrix
#     H = torch.matmul(P_centered.transpose(-2, -1), Q_centered) # (B, 3, N) x (B, N, 3) = (B, 3, 3)
#
#     # Compute SVD for optimal rotation
#     U, S, Vt = torch.linalg.svd(H) 
#
#     # Compute Rotation Matrix Now
#     # mask = (torch.linalg.det(U) * torch.linalg.det(Vt)) < 0.0
#     # R = torch.matmul(Vt.transpose(-2, -1), U.transpose(-2, -1))
#     R = torch.matmul(Vt.transpose(-2, -1), U.transpose(-2, -1))
#
#     # Special Reflection case
#     det = torch.linalg.det(R)
#     mask = det < 0
#     if mask.sum() >= 1:
#         Vt_copy = Vt.clone()
#         # U_copy = U.clone()
#         # U_copy[mask, :, -1] *= -1
#         Vt_copy[mask, -1, :] *= -1
#         R = torch.matmul(Vt_copy.transpose(-2, -1), U.transpose(-2, -1))
#         # R = torch.matmul(Vt_copy.transpose(-2, -1), U.transpose(-2, -1))
#         # R = torch.matmul(U_copy.transpose(-2, -1), Vt.transpose(-2, -1))
#     # else:
#     #     R = torch.matmul(U.transpose(-2, -1), Vt.transpose(-2, -1))
#
#     # print(torch.matmul(P_centered, R).shape)
#     # Apply Rotation
#     if len(Q.shape) == 2:
#         # P_aligned = torch.matmul(P_centered, R) + centroid_Q 
#         # print((R*centroid_P.unsqueeze(1)).shape)
#         translation = centroid_Q.unsqueeze(1) - R*centroid_P.unsqueeze(1)  
#     else:
#         # P_aligned = torch.matmul(P_centered, R) + centroid_Q.unsqueeze(1)
#         translation = centroid_Q.unsqueeze(1) - torch.matmul(R, centroid_P)
#
#     # Translate and rotate
#     print("P",P.shape)
#     print("R",R.shape)
#     print("MATMUL R*P.T", torch.matmul(R, P.transpose(-2,-1)).shape)
#     print("Translation", translation.shape)
#     P_aligned = torch.matmul(R, P.transpose(-2,-1)) + translation
#
#     diff = P_aligned - Q
#     rmsd = torch.sqrt((diff * diff).sum(1) / P.size(1))
#     print("RSMD:", rmsd.mean())
#
#     torch.save({"paligned": P_aligned, "control": Q}, "paligned.pt")
#
#     return P_aligned


if __name__ == "__main__":
    kabsch_test()



    # # Terminal peptide
    # peptide_term_class = TerminalPeptide()
    #
    # # Glycine each residue has 7 atoms: N, H, CA, 1HA, 2HA, C, O
    # data = torch.load("./binary_import/dof_vectors2024-05-14torch_tensor.pt")
    # params = data["params"]
    # x = data["x"]
    #
    # # pull out values
    # torsion_full = x[:, :params["torsion_main_n"]]
    # bond_angles = x[:, params["torsion_main_n"]:params["torsion_main_n"]+params["bond_angle_main_n"]]
    # bond_lengths = x[:, params["torsion_main_n"]+params["bond_angle_main_n"]:-params["energy_n"]]
    #
    # phi_psi, omega_idx = peptide_term_class.torsion_extract(torsion_full.size(-1), 4)
    #
    # # index tensors
    # omega = torsion_full[:, omega_idx]
    # torsion = torsion_full[:, phi_psi]
    #
    # # generate coordinates given DoF vector
    # coordinates = peptide_term_class.terminal_peptide(torsion, omega, bond_angles, bond_lengths)
    #
    # # Extract H and O coordinates
    # h_coords = coordinates[:, :, 4, :]
    # o_coords = coordinates[:, :, 3, :]
    # # Extract out N and C coords
    # n_coords = coordinates[:, :, 2, :]
    # n_roll = torch.roll(n_coords, shifts=-1, dims=1)
    # c_coords = coordinates[:, :, 1, :]
    #
    # # Get terminal bond lengths
    # all_bonds = torch.norm(n_roll - c_coords, dim=-1)
    # terminal_length = all_bonds[:, -1]
    #
    # # init minimum hbond amount
    # hbond_class = HBondCount(min_hbond=3)
    #
    # # test angles
    # ahd, bah = hbond_class.calc_ahd_bah_angles(
    # o_coords,
    # h_coords,
    # n_roll,
    # c_coords
    # )
    # print("ANGLE TEST FUNC DONE")
    #
    # # measure distances
    # hbond_count_out = hbond_class.measure_internal_distances(h_coords, o_coords, ahd, bah)
    # print("HBOND COUNT:", hbond_count_out)
    #
    # # Figure out if Backbone passes hbond and terminal bond check
    # out = hbond_class.checkpoint(hbond_count_out, terminal_length)
    # print("OUT:", out)



