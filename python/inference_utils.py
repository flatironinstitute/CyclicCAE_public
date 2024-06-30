#!/usr/bin/env python 
# @author: Andrew Powers (apowers@flatironinstitute.org)
# @brief: Classes and functions helpfulf for inference task
#

######## IMPORT 
import torch
import torch.nn as nn

# xyz coordinates
from terminal_error import TerminalPeptide

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

    def measure_internal_distances(self, H_coords: torch.Tensor, O_coords: torch.Tensor):
        """Compute Distances between H and O Atoms to check if a hydrogen bond is present

        PARAMS
        ------
        H_coords: Tensor (B, N, 3)
            xyz coordinates of Amide Hydrogen
        O_coords: Tensor (B, N, 3)
            xyz coordinates of Backbone Oxygen

        RETURNS
        -------
        hbond_count: Tensor (B)
            Number of internal hydrogen bonds present
        """
        # Compute Distance between all points
        dists = torch.norm(O_coords.unsqueeze(2) - H_coords.unsqueeze(1), dim=-1)
        mask = torch.ones(dists.size(-2), dists.size(-1))
        indices = torch.arange(1, dists.size(-1), dtype=torch.long)
        indices = torch.cat((indices, torch.tensor([0], dtype=torch.long)))
        batch_indices = torch.arange(dists.size(0))
        row_indices = torch.arange(dists.size(1))
        # col_indices = torch.arange(1, dists.size(-2))
        # print(dists[batch_indices[:, None], row_indices, indices])
        # print(dists[batch_indices[:, None], row_indices, indices].shape)
        dists[batch_indices[:, None], row_indices, indices] = torch.tensor([0.0], dtype=dists.dtype)
        print(dists)
        print(indices.shape)
        print(dists.shape)


        # print("OFFSET DIAG:", torch.diagonal(dists, offset=1, dim1=1, dim2=2))

        # Check where the distances are within the correct spot
        dists_bool = (dists >= self.min_distance) & (dists <= self.max_distance)
        # print(dists_bool)

        # a more contained smaller output
        smaller_dists = dists[dists_bool]
        # print(smaller_dists)

        # Count hbonds per B_i
        hbond_count = dists_bool.sum((1, -1)) # (B)
        return hbond_count


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

        

if __name__ == "__main__":
    # Terminal peptide
    peptide_term_class = TerminalPeptide()

    # Glycine each residue has 7 atoms: N, H, CA, 1HA, 2HA, C, O
    data = torch.load("./binary_import/dof_vectors2024-05-14torch_tensor.pt")
    params = data["params"]
    x = data["x"]

    # pull out values
    torsion_full = x[:, :params["torsion_main_n"]]
    bond_angles = x[:, params["torsion_main_n"]:params["torsion_main_n"]+params["bond_angle_main_n"]]
    bond_lengths = x[:, params["torsion_main_n"]+params["bond_angle_main_n"]:-params["energy_n"]]

    phi_psi, omega_idx = peptide_term_class.torsion_extract(torsion_full.size(-1), 4)

    # index tensors
    omega = torsion_full[:, omega_idx]
    torsion = torsion_full[:, phi_psi]

    # generate coordinates given DoF vector
    coordinates = peptide_term_class.terminal_peptide(torsion, omega, bond_angles, bond_lengths)

    # Extract H and O coordinates
    h_coords = coordinates[:, :, 4, :]
    o_coords = coordinates[:, :, 3, :]
    # Extract out N and C coords
    n_coords = coordinates[:, :, 2, :]
    n_roll = torch.roll(n_coords, shifts=-1, dims=1)
    c_coords = coordinates[:, :, 1, :]

    # Get terminal bond lengths
    all_bonds = torch.norm(n_roll - c_coords, dim=-1)
    terminal_length = all_bonds[:, -1]

    # init minimum hbond amount
    hbond_class = HBondCount(min_hbond=3)

    # measure distances
    hbond_count_out = hbond_class.measure_internal_distances(h_coords, o_coords)
    print("HBOND COUNT:", hbond_count_out)

    # Figure out if Backbone passes hbond and terminal bond check
    out = hbond_class.checkpoint(hbond_count_out, terminal_length)
    print("OUT:", out)


