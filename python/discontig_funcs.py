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

# @brief: functions that are usefulf or discontinuous motif matching
# @author: Andrew Powers (apowers4@uoregon.edu) & (apowers@flatironinstitute.org)
#

import torch
from torch import Tensor
import numpy as np
from motif_functions import extract_dof_of_interest_minimize

def generate_exclusion_vector( exclude_pos: list[str], freeze_residues: list[int] ):
    """Take our input exclusion list[str] (e.g. ['1:0', '1:3', '2:3'])
        and break it up into a vector of positions.
        Additionally: torsion & bondangle/bondlength are different

        Indexing for bondAngle/Bondlength with Torsion highlighted as well:
        --- Indexing ---
        N  : 0 (phi torsion as well)
        CA : 1
        C  : 2 (psi torsion as well)
        O  : 3
        NH : 4 
        (Omega : 6)
        --- Example ---
        Torsion (1: phi, 2: psi, 3: omega) [in this example]
        no masking: Torsion [1,2,3,4,5,6], BondAngle/BondLength [1,2,3,4,5,6,7,8,9,10]
        masking: Torsion [2,3,4,6], BondAngle/BondLength [2,4,5,6,7,9,10]

        We would want to make a mask like:
        Torsion -> [0, 1, 1, 1, 1, 1]
        BondAngle -> [0, 1, 1, 0, 1, 1, 1, 1, 0, 1]

    PARAMS
    ------
    :exclude_pos: A list of strings where an element has the str layout
        of '1:3' where the first 1 is the first residue and the 3 after the :
        indicates the atom of residue 1 to exclude. With size of len(exclude_pos)
    :freeze_residues: A list of residues that are a part of the motif we are
        retaining. E is the len(freeze_residues)

    RETURN
    ------
    :torsion_mask: A tensor of residue torsions we want to exclude from
        the fixed-residues with size (E*3)
    :bondangle_mask: A tensor of residue bondangles we want to exclude from
        the fixed-residues with size (E*5)
    """
    # grab out the set of residues beings affected
    residue_set = {int(k.split(":")[0])-1 for k in exclude_pos}
#     E = len(residue_set)
    E = len(freeze_residues)
    freeze_set = {e.item() for e in freeze_residues}
    torsion_map = {0: 0, 2: 1, 6:2}
    # Create our tensors
    # 3 for torsions (phi, psi, omega)
    torsion_mask = torch.ones(E*3, dtype=torch.int32).reshape(E, 3)
    # 5 for bondangle/length (N, C, CA, C, O, NH)
    bondangle_mask = torch.ones(E*5, dtype=torch.int32).reshape(E, 5)

    # Error out if the masking residues are not equivalent or a subset of the freeze_residues
    assert residue_set.issubset(freeze_set), "Our masked residues are not in our Motif please fix --exclude-atom-dof or --freeze-residues"
    freeze_list = list(freeze_set)

    # iter through
    for i, s in enumerate(exclude_pos):
        s_split = s.split(':')
        resi, atomi = int(s_split[0])-1, int(s_split[1])

        # Grab the resi position, so that it matches our masking scheme.
        row = freeze_list.index(resi)

        if atomi in [0, 2, 6]:
            torsion_mask[row, torsion_map[atomi]] = 0
            if atomi != 6:
                bondangle_mask[row, atomi] = 0
        else:
            bondangle_mask[row, atomi] = 0
        
    return torsion_mask.flatten(), bondangle_mask.flatten()

def inverse_selection(mask: torch.Tensor, D: int) -> torch.Tensor:
    """Invert a selection, so that you grab everything but the elements passed in
        mask

    PARAMS
    ------
    :exclusion_vector: A vector of atoms we want to exclude from
        our fixed-residues (E)
    :D: The length of our motif 

    RETURN
    ------
    :mask_inverse: Return a tensor of size D - E with 
    """
    index_tensor = torch.arange(D, dtype=torch.long)
    mask_inverse = torch.ones(D, dtype=torch.bool)
    mask_inverse[mask] = False
    return index_tensor[mask_inverse]

def harmonic_diff_inpaint(
    real_t_sin: Tensor, 
    real_t_cos: Tensor,
    real_ba_sin: Tensor,
    real_ba_cos: Tensor,
    real_bond_length: Tensor,
    torsion_sin: Tensor,
    torsion_cos: Tensor,
    bond_angle_sin: Tensor,
    bond_angle_cos: Tensor,
    bond_length: Tensor,
    torsion_index: Tensor,
    bond_angle_dofs: Tensor,
    bond_length_dofs: Tensor,
    PARAMS: dict[str, int|Tensor|str],
    freeze_residues: Tensor,
    freeze_torsion: Tensor,
    freeze_bond_angle: Tensor,
    freeze_bond_length: Tensor,
    match_res_mask: dict[int, Tensor],
    inpaint_dist: float,
    device: str,
    dof_minimize: bool = False,
    exclude_positions: list[str] = [],
    minimze_whole_peptide: bool = False,
    ):
    """Harmonic Difference"""
    # Index and reference tensors originate from CPU-side motif extraction.
    # Move them alongside model outputs so GPU inference can gather and compare
    # without mixed-device errors.
    target_device = torsion_sin.device
    real_t_sin = real_t_sin.to(target_device)
    real_t_cos = real_t_cos.to(target_device)
    real_ba_sin = real_ba_sin.to(target_device)
    real_ba_cos = real_ba_cos.to(target_device)
    real_bond_length = real_bond_length.to(target_device)
    torsion_index = torsion_index.to(target_device)
    bond_angle_dofs = bond_angle_dofs.to(target_device)
    bond_length_dofs = bond_length_dofs.to(target_device)

    # Grab DoF values of interest using our values
    if not dof_minimize:
        # Grab our residue's of interest DoFs
        torsion_sin = torch.gather(
            torsion_sin.unsqueeze(1).expand(-1, PARAMS["protein_len"], -1),
            -1,
            torsion_index.expand(torsion_sin.shape[0], -1, -1),
        )
        torsion_cos = torch.gather(
            torsion_cos.unsqueeze(1).expand(-1, PARAMS["protein_len"], -1),
            -1,
            torsion_index.expand(torsion_cos.shape[0], -1, -1),
        )
        bond_angle_sin = torch.gather(
            bond_angle_sin.unsqueeze(1).expand(-1, PARAMS["protein_len"], -1),
            -1,
            bond_angle_dofs.expand(bond_angle_sin.shape[0], -1, -1),
        )
        bond_angle_cos = torch.gather(
            bond_angle_cos.unsqueeze(1).expand(-1, PARAMS["protein_len"], -1),
            -1,
            bond_angle_dofs.expand(bond_angle_cos.shape[0], -1, -1),
        )
        bond_length = torch.gather(
            bond_length.unsqueeze(1).expand(-1, PARAMS["protein_len"], -1),
            -1,
            bond_length_dofs.expand(bond_length.shape[0], -1, -1),
        )

        print("EXCLUDE POS:", exclude_positions)
        if not exclude_positions:
            mask_torsion = torch.ones_like(torsion_sin)
            mask_bondangle = torch.ones_like(bond_angle_sin)
        else:
            # Generate appropriate vector for exclusion atom DoF selection
            mask_torsion, mask_bondangle = generate_exclusion_vector(exclude_positions,
                                                                     freeze_residues)

    elif minimze_whole_peptide:
        torsion_sin = torsion_sin[..., :-1]
        torsion_cos = torsion_cos[..., :-1]
        mask_torsion = torch.ones_like(torsion_sin)
        mask_bondangle = torch.ones_like(bond_angle_sin)

    else:
        # Grab our residue's of interest DoFs
        torsion_sin, torsion_cos, bond_angle_sin, bond_angle_cos, bond_length = extract_dof_of_interest_minimize(
            freeze_residues.shape[0],
            torsion_sin,
            torsion_cos,
            bond_angle_sin,
            bond_angle_cos,
            bond_length,
            torsion_index,
            bond_angle_dofs,
            bond_length_dofs,
            match_res_mask,
        )

        if not exclude_positions:
            mask_torsion = torch.ones_like(torsion_sin)
            mask_bondangle = torch.ones_like(bond_angle_sin)
        else:
            # Generate appropriate vector for exclusion atom DoF selection
            mask_torsion, mask_bondangle = generate_exclusion_vector(exclude_positions,
                                                                     freeze_residues)

    # Compute the harmoinc difference
    harmonic_penalty_sin_torsion = torch.tensor(0.0, device=device)    if not freeze_torsion     else torch.mean(torch.pow(real_t_sin - torsion_sin,2) * mask_torsion, dim=-1).to(device=device)
    harmonic_penalty_cos_torsion = torch.tensor(0.0, device=device)    if not freeze_torsion     else torch.mean(torch.pow(real_t_cos - torsion_cos,2) * mask_torsion, dim=-1).to(device=device)
    harmonic_penalty_sin_bond_angle = torch.tensor(0.0, device=device) if not freeze_bond_angle  else (torch.sum(torch.pow(real_ba_sin - bond_angle_sin,2) * mask_bondangle, dim=-1)/ (real_ba_sin != 0).sum(dim=-1)).to(device=device)
    harmonic_penalty_cos_bond_angle = torch.tensor(0.0, device=device) if not freeze_bond_angle  else (torch.sum(torch.pow(real_ba_cos - bond_angle_cos,2) * mask_bondangle, dim=-1)/ (real_ba_cos != 0).sum(dim=-1)).to(device=device)
    harmonic_penalty_bond_length = torch.tensor(0.0, device=device)    if not freeze_bond_length else (torch.sum(torch.pow(real_bond_length - bond_length,2) * mask_bondangle, dim=-1)/ (real_bond_length != 0).sum(dim=-1)).to(device=device)
    harmonic_loss = harmonic_penalty_sin_torsion + harmonic_penalty_cos_torsion + harmonic_penalty_cos_bond_angle + harmonic_penalty_sin_bond_angle + harmonic_penalty_bond_length

    if not dof_minimize:
        # Compare and pick residues based on output
        valid = harmonic_loss <= inpaint_dist
        eligible_loss = harmonic_loss.masked_fill(
                ~valid,
                float("inf"),
                )

        best_loss, best_start = eligible_loss.min(dim=1)

        # Samples with no passing motif position have best_loss == inf
        sample_idx = torch.isfinite(best_loss).nonzero(as_tuple=True)[0]

        # Column 0: original sampled-latent row
        # Column 1: best motif starting position
        match_res = torch.stack(
                (sample_idx, best_start[sample_idx]),
                dim=1,
                ).long()

        harmonic_out = best_loss[sample_idx]
        return harmonic_out, match_res
    else:
        return harmonic_loss, None
# 
# def harmonic_diff_inpaint(
#     real_t_sin: Tensor, 
#     real_t_cos: Tensor,
#     real_ba_sin: Tensor,
#     real_ba_cos: Tensor,
#     real_bond_length: Tensor,
#     torsion_sin: Tensor,
#     torsion_cos: Tensor,
#     bond_angle_sin: Tensor,
#     bond_angle_cos: Tensor,
#     bond_length: Tensor,
#     torsion_index: Tensor,
#     bond_angle_dofs: Tensor,
#     bond_length_dofs: Tensor,
#     PARAMS: dict[str, int|Tensor|str],
#     freeze_residues: Tensor,
#     freeze_torsion: Tensor,
#     freeze_bond_angle: Tensor,
#     freeze_bond_length: Tensor,
#     match_res_mask: dict[int, Tensor],
#     inpaint_dist: float,
#     device: str,
#     dof_minimize: bool = False,
#     ):
#     """Harmonic Difference"""
#     # Grab DoF values of interest using our values
#     if not dof_minimize:
#         torsion_sin = torch.gather(
#             torsion_sin.unsqueeze(1).expand(-1, PARAMS["protein_len"], -1),
#             -1,
#             torsion_index.expand(torsion_sin.shape[0], -1, -1),
#         )
#         torsion_cos = torch.gather(
#             torsion_cos.unsqueeze(1).expand(-1, PARAMS["protein_len"], -1),
#             -1,
#             torsion_index.expand(torsion_cos.shape[0], -1, -1),
#         )
#         bond_angle_sin = torch.gather(
#             bond_angle_sin.unsqueeze(1).expand(-1, PARAMS["protein_len"], -1),
#             -1,
#             bond_angle_dofs.expand(bond_angle_sin.shape[0], -1, -1),
#         )
#         bond_angle_cos = torch.gather(
#             bond_angle_cos.unsqueeze(1).expand(-1, PARAMS["protein_len"], -1),
#             -1,
#             bond_angle_dofs.expand(bond_angle_cos.shape[0], -1, -1),
#         )
#         bond_length = torch.gather(
#             bond_length.unsqueeze(1).expand(-1, PARAMS["protein_len"], -1),
#             -1,
#             bond_length_dofs.expand(bond_length.shape[0], -1, -1),
#         )
#     else:
#         torsion_sin, torsion_cos, bond_angle_sin, bond_angle_cos, bond_length = extract_dof_of_interest_minimize(
#             freeze_residues.shape[0],
#             torsion_sin,
#             torsion_cos,
#             bond_angle_sin,
#             bond_angle_cos,
#             bond_length,
#             torsion_index,
#             bond_angle_dofs,
#             bond_length_dofs,
#             match_res_mask,
#         )
# 
#     # Compute the harmoinc difference
#     harmonic_penalty_sin_torsion = torch.tensor(0.0, device=device)    if not freeze_torsion     else torch.pow(real_t_sin - torsion_sin,2).mean(dim=-1).to(device=device)
#     harmonic_penalty_cos_torsion = torch.tensor(0.0, device=device)    if not freeze_torsion     else torch.pow(real_t_cos - torsion_cos,2).mean(dim=-1).to(device=device)
#     harmonic_penalty_sin_bond_angle = torch.tensor(0.0, device=device) if not freeze_bond_angle  else (torch.pow(real_ba_sin - bond_angle_sin,2).sum(dim=-1)/ (real_ba_sin != 0).sum(dim=-1)).to(device=device)
#     harmonic_penalty_cos_bond_angle = torch.tensor(0.0, device=device) if not freeze_bond_angle  else (torch.pow(real_ba_cos - bond_angle_cos,2).sum(dim=-1)/ (real_ba_cos != 0).sum(dim=-1)).to(device=device)
# 
#     harmonic_penalty_bond_length = torch.tensor(0.0, device=device)    if not freeze_bond_length else (torch.pow(real_bond_length - bond_length,2).sum(dim=-1)/ (real_bond_length != 0).sum(dim=-1)).to(device=device)
#     harmonic_loss = harmonic_penalty_sin_torsion + harmonic_penalty_cos_torsion + harmonic_penalty_cos_bond_angle + harmonic_penalty_sin_bond_angle + harmonic_penalty_bond_length
# 
#     if not dof_minimize:
#         # Compare and pick residues based on output
#         mask = harmonic_loss <= inpaint_dist
#         match_res = torch.nonzero(mask).to(torch.int)
# 
#         # Now grab non unique non redudent values
#         unique_values, inv_idx = torch.unique_consecutive(match_res[:, 0], return_inverse=True)
# 
#         # Now remove redudent non-first occuring values
#         mask_unique = torch.zeros_like(inv_idx, dtype=torch.bool)
#         mask_unique[torch.unique(inv_idx, return_inverse=True)[1]] = True
# 
#         # Reindex
#         match_res = match_res[mask_unique]
#         harmonic_out = harmonic_loss[match_res[:, 0], match_res[:, 1]]
#         return harmonic_out, match_res
#     else:
#         return harmonic_loss, None
