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

# @author: Andrew Powers (apowers4@uoregon.edu or apowers@flatironinstitue.org)
# @brief: Helper functions for allowing cyclic permutation motif matching
#

#### Import
import torch
import os
from typing import Tuple

#### Functions
def compute_motif_indices(
    pep_len: int,
    dof_len: int,
    motif_len: int,
    root_pos: int,
    phi_psi_idx: torch.Tensor,
    omega_idx: torch.Tensor,
    bond_angle_idx: torch.Tensor,
    bond_length_idx: torch.Tensor,
    ) -> torch.Tensor:
    """
    Calculate the cyclic permutation positional information for all positions. This
    allows us to get an idx for each residue/residue pair that could match our motif.

    PARAMS
    ------
    :pep_len: The length of our input peptide
    :dof_len: The length of the DoF
    :motif_len: Number of residues within our dof.
    :root_pos: The residue index of the root residue
    :phi_psi_idx: DoF Index positions for relevant information
    :omega_idx: DoF Index positions for relevant information
    :bond_angle_idx: DoF Index positions for relevant information
    :bond_length_idx: DoF Index positions for relevant information

    RETURNS
    -------
    :torsions_true: A matrix of size (pep_len, dof_len) where our dof indices are stored
        for torsional DoFs
    :bond_angle_start: The start positions for our motif DoFs that correspond
        for both bond angle and bond length dofs
    :atoms_total: The number of atoms/DoFs that are attributed, in total,
        to that motif. Add this to bond_angle_start to get the last position
    :residue_idx: A matrix of residue index storage for reference
    """
    # Generate our holding matrices
    residue_idx = torch.arange(pep_len, dtype=torch.long).view(-1, 1) + torch.arange(motif_len, dtype=torch.long)
    residue_idx = cyclize_tensor(residue_idx, pep_len, double=False)

    # Check which things are a root, first residue, or last residue
    freeze_root = residue_idx == root_pos
    freeze_first = residue_idx == 1
    freeze_last = residue_idx == pep_len

    # Compute the number of atoms that are within this residue index
    atoms_per_residue = torch.ones(pep_len, motif_len, dtype=torch.long) * 7
    atoms_per_residue[freeze_first] += 1
    atoms_per_residue[freeze_last] += 2
    atoms_total = atoms_per_residue.sum(-1)

    # Extract relevant dof information for torsions
    phi_psi_idx = phi_psi_idx.unsqueeze(0).repeat(pep_len, 1) # (B, M*2)
    phi_psi_selection = ((residue_idx * 2).view(pep_len, -1, 1) + torch.tensor([0, 1])).view(pep_len, -1) # (B, M*2)
    phi_psi_selection_cyclic = cyclize_tensor(
        phi_psi_selection,
        pep_len,
        True,
    )
    phi_psi_true = torch.gather(phi_psi_idx, -1, phi_psi_selection_cyclic) # (B, M*2)
    # Extract out Omega
    #### NOTE: OMEGA is not provided for residue 8, this needs to be grabbed from
    ####    the coordinates extraction script. Therefore, I am going to expand the
    ####    Omega selection tensor to include an additional index dof_len + 1
    omega_idx = torch.cat(
        (
            omega_idx,
            torch.tensor([dof_len], dtype=torch.long),
        ), dim=-1,
    )
    omega_idx[-1] -= 1
    omega_idx = omega_idx.unsqueeze(0).repeat(pep_len, 1) # (B, pep_len)
    omega_true = torch.gather(omega_idx, -1, residue_idx) #.view(pep_len, -1, 1) # (B, M, 1)
    torsions_true = torch.cat(
        (
            phi_psi_true.view(pep_len, motif_len, 2), #(B, P) -> #(B, M, 2)
            omega_true.view(pep_len, motif_len, 1), #(B, M*O, 1)
        ), dim=-1,
    ).view(pep_len, -1)

    # Extract relevant dof information for bond angles/lengths
    bond_angle_dofs, bond_length_dofs = find_atom_indices(residue_idx, 
                                                          pep_len, 
                                                          root_pos, 
                                                          dof_len,
                                                          bond_angle_idx,
                                                          bond_length_idx)

    return torsions_true, bond_angle_dofs, bond_length_dofs, residue_idx

def find_atom_indices(
    residue_tensor: torch.Tensor,
    pep_len: int,
    root_pos: int,
    dof_len: int,
    dof_angle_idx: torch.Tensor,
    dof_length_idx: torch.Tensor,
    ) -> torch.Tensor:
    """
    Fill a tensor with bond angle/length indices

    PARAMS
    ------
    :residue_tensor: A tensor of motif indexes (pep_len, motif_len)
    :pep_len: The length of our input peptide
    :root_pos: Root residue of our pose
    :dof_len: Length of our dof vector
    :dof_angle_idx: The locations of specific bond angles of interest
    :dof_length_idx: The specific locations of bond lengths of interest

    RETURNS
    -------
    :dof_bond_angle: A tensor filled with atom indices and -1 are kept for non-used
        values
    :dof_bond_length: A tensor of size 
    """
    # generate our holding tensor
    dof_angle_reshape, dof_length_reshape = dof_angle_idx.view(-1, 5), dof_length_idx.view(-1, 5)

    # Extract motif len
    M = residue_tensor.shape[-1]

    # Reshape our residue_tensor, so that it can index our reshaped tensors
    residue_reshape = residue_tensor.view(-1)

    # Now generate our output tensors
    dof_bond_angle = dof_angle_reshape[residue_reshape]
    dof_bond_length = dof_length_reshape[residue_reshape]

    return dof_bond_angle.view(-1, M*5), dof_bond_length.view(-1, M*5)

def calculate_motif_numbering_for_kabsch(
    motif_match: torch.Tensor,
    pep_len: int,
    motif_length: int,
    motif_positions: torch.Tensor,
    dof_minimize: bool = False,
    ) -> torch.Tensor:
    """
    Compute the motif atom indexing per input that passes our 
    harmonic function

    PARAMS
    ------
    :motif_match: An index tensor of size (M, 2). Where M is all of the
        matches for our DoFs. The second dimensions:
            - first number is the row/sample 
            - last number is the residue number that matches (0-index).
    :pep_len: The length of our output peptide
    :motif_length: The length of our continuous motif (number of residues)
    :motif_positions: The tensor of residue indexes from our pass PDB
    :dof_minimize: True we are performing regression and motif_match only has 1
        column and not 2.

    RETURN
    ------
    :motif_kabsch: A tensor of size (M, 4) all of our motifs, but the atom indexing for
        selecting the correct one.
    :motif_input: A tensor of size (motif_len*4) which is our index for the atoms form
        our input
    """
    # index generate
    match_device = motif_match.device
    motif_device = motif_positions.device
    # DoFSummary returns peptide-relative residue numbers in Rosetta's 1-based
    # convention. Coordinate tensors below are 0-based.
    motif_positions_zero = motif_positions.to(dtype=torch.long) - 1
    if motif_positions_zero.min().item() < 0 or motif_positions_zero.max().item() >= pep_len:
        raise IndexError(
            f"motif positions {motif_positions.tolist()} fall outside a {pep_len}-residue peptide"
        )
    positions = torch.arange(pep_len * 4, device=match_device).view(pep_len, 4)
    input_positions = torch.arange(
        pep_len * 4, device=motif_device
    ).view(pep_len, 4)
    
    # motifs
    if dof_minimize:
        motif_residues = (motif_match.view(-1, 1) + torch.arange(motif_length, dtype=torch.long, device=match_device).unsqueeze(0)).view(-1, 1)
    else:
        motif_residues = (motif_match[:, -1].view(-1, 1) + torch.arange(motif_length, dtype=torch.long, device=match_device).unsqueeze(0)).view(-1, 1)
    motif_residues_cyclic = cyclize_tensor(motif_residues, pep_len, False)

    # Index our tensor correctly
    motif_kabsch = positions[motif_residues_cyclic]

    # Motif residues of passed PDB
    motif_input = input_positions[motif_positions_zero.view(-1, 1)]

    return motif_kabsch.view(motif_match.shape[0], 4*motif_length), motif_input.view(-1)

def extract_dof_of_interest_minimize(
    motif_length: int,
    torsion_sin: torch.Tensor,
    torsion_cos: torch.Tensor,
    bond_angle_sin: torch.Tensor,
    bond_angle_cos: torch.Tensor,
    bond_length: torch.Tensor,
    torsion_index: torch.Tensor,
    bond_angle_index: torch.Tensor,
    bond_length_index: torch.Tensor,
    match_res_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Take in our model outputs and correctly index the residues that we determined
    match our motif from matched_res_mask. These DoFs will only be returned and not
    the cyclic permutation based approach from `inpaint`

    PARAMS
    ------
    :motif_length: Number of residues within our motif
    :torsion_sin: The entire phi/psi/omega sin DoFs for our outputs (B, DoFSize)
    :torsion_cos: The entire phi/psi/omega cos DoFs for our outputs (B, DoFSize)
    :bond_angle_sin: The entire bond_angle sin DoFs for our outputs (B, DoFSize)
    :bond_angle_cos: The entire bond_angle sin DoFs for our outputs (B, DoFSize)
    :bond_length: The entire bond lenght DoFs for our outputs (B, DoFSize)
    :torsion_index: The indexing for phi/psi/omega values (L, 3*motif_len)
    :bond_angle_index: The indexing for bond angle values (L, 5*motif_len)
    :bond_length_index: The indexing for bond length values (L, 5*motif_len)
    :match_res_mask: The matrix associated with which beginning residue we are
        tracking throughout this minimize process

    RETURNS
    -------
    torsion_sin, torsion_cos, bond_angle_sin, bond_angle_cos, bond_length
    However, the specific DoFs of the residues of interest for that particular
    output are returned
    """
    # generate our indexing tensors
    torsion_slice = torsion_index[match_res_mask]
    bond_angle_slice = bond_angle_index[match_res_mask]
    bond_length_slice = bond_length_index[match_res_mask]

    torsion_sin = torch.gather(torsion_sin, -1, torsion_slice)
    torsion_cos = torch.gather(torsion_cos, -1, torsion_slice)
    bond_angle_sin = torch.gather(bond_angle_sin, -1, bond_angle_slice)
    bond_angle_cos = torch.gather(bond_angle_cos, -1, bond_angle_slice)
    bond_length = torch.gather(bond_length, -1, bond_length_slice)

    return torsion_sin, torsion_cos, bond_angle_sin, bond_angle_cos, bond_length


def cyclize_tensor(
    non_cyclic_tensor: torch.Tensor,
    pep_len: int,
    double: bool,
    ) -> torch.Tensor:
    """
    Take a tensor and cyclize the values

    PARAMS
    ------
    :non_cyclic_tensor: Straight linear tensor, that has values that
        need to be wrapped
    :pep_len: Length of the max non doubled value
    :Whether our pep_len should be doubled or not

    RETURNS
    -------
    :cyclic_tensor: Appropriately cyclized tensor
    """
    # Double our pep_len
    if double:
        pep_len = pep_len*2

    # generate a mask for particular values that need to be adjusted
    mask = non_cyclic_tensor >= pep_len

    # select values and adjust
    non_cyclic_tensor[mask] -= pep_len
    return non_cyclic_tensor

def main() -> int:
    return 0

##### Main
if __name__ == "__main__":
    main()
