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
        print("MAG NORM", mag_AH1.shape)
        print("AH1 shape", AH1.shape)

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

        print("rad_AHD degree", torch.rad2deg(rad_AHD))
        print("rad_BAH", rad_BAH)
        print(torch.deg2rad(torch.tensor(90)))
        print(torch.deg2rad(torch.tensor(180)))

        # compute anlge range bool
        AHD_bool = (torch.deg2rad(torch.tensor(90)) <= rad_AHD) & (rad_AHD <= torch.deg2rad(torch.tensor(180)))
        BAH_bool = (torch.deg2rad(torch.tensor(90)) <= rad_BAH) & (rad_BAH <= torch.deg2rad(torch.tensor(180)))

        print('AHD:', AHD_bool)
        print('BAH:', BAH_bool)

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

    # test angles
    ahd, bah = hbond_class.calc_ahd_bah_angles(
    o_coords,
    h_coords,
    n_roll,
    c_coords
    )
    print("ANGLE TEST FUNC DONE")

    # measure distances
    hbond_count_out = hbond_class.measure_internal_distances(h_coords, o_coords, ahd, bah)
    print("HBOND COUNT:", hbond_count_out)

    # Figure out if Backbone passes hbond and terminal bond check
    out = hbond_class.checkpoint(hbond_count_out, terminal_length)
    print("OUT:", out)


