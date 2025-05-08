#!/usr/bin/env python
#
# @author: Andrew Powers (apowers4@uoregon.edu) & (apowers@flatironinstitute.org)
# @brief: Helper functions for the inpainting process within `energy_gradient.py`
#

# Packages
import torch
from torch import Tensor

from inference_utils import discontig_motif_loss
from motif_functions import calculate_motif_numbering_for_kabsch
# This needs to be moved to another script, as it is a method of the class currently
from discontig_funcs import harmonic_diff_inpaint



#### Functions
def extract_loss_and_matchres(
        freeze_residues: Tensor|None,
        contiguous: bool,
        regions_atoms: Tensor|None,
        t_native: Tensor|None,
        R_native: Tensor|None,
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
        freeze_torsion: Tensor,           
        freeze_bond_angle: Tensor,
        freeze_bond_length: Tensor,
        inpaint_dist: float,
        match_res_mask: Tensor,
        HNCO: Tensor,
        coords: Tensor,
        device: str,
        dof_minimize: bool = False,
        ) -> tuple[Tensor, Tensor, Tensor, Tensor, bool]:
    """Extract out the motif loss and match_res_mask against our scaffolds from the latent space

    PARAMS
    ------
    :freeze_residues: The residues of interest that we want to freeze (or None if we have no motif)
    :contiguous: Whether the residues are sequentially in order or not
    :regions_atoms: The atoms which are associated with the two discontig regions
    :t_native: The native translation vector for our pdb-input
    :R_native: The native rotation matrix for our pdb-input
    :real_t_sin: The real torsion sin value
    :real_t_cos: The real torsion cos value
    :real_ba_sin: The real bondangle sin value
    :real_ba_cos: The real bondangle cos value
    :real_bond_length: The real bondlength sin value
    :torsion_sin: The scaffold torsion sin value
    :torsion_cos: The scaffold torsion cos value
    :bond_angle_sin: The scaffold bondangle sin value
    :bond_angle_cos: The scaffold bondangle cos value
    :bond_length: The scaffold bondlength value
    :match_res_mask: Either None or the current Tensor of match_res_mask from a previous round

    RETURNS
    -------
    :loss: Loss calculated either from discontig or harmonic diff inpaint
    :match_res_mask: The particular residues that meet our criteria
    :matches: True if we need to skip the rest and go to the next loop, False if we have matches
        and can continue on
    """
    # If we have no motif then return empty variables, so we can continue
    if freeze_residues == None:
        return torch.empty([], device=device), torch.empty([], device=device), torch.empty([], device=device), torch.empty([], device=device), False

    # If we have a motif that we want to inpaint with then we need to extract some information
    # Therefore, we traverse this logic
    else:
        # Add in the terminal peptide bond HNCO radian, as it isn't a part
        # of the DoFs
        HNCO_sin, HNCO_cos = torch.sin(HNCO), torch.cos(HNCO)
        torsion_sin = torch.cat(
            (
                torsion_sin,
                HNCO_sin,
            ), dim=-1,
        )
        torsion_cos = torch.cat(
            (
                torsion_cos,
                HNCO_cos,
            ), dim=-1,
        )

        if not contiguous:
            # If our input is discontinuous then we need to use our discontig loss term
            loss = discontig_motif_loss(
                region1  = coords[:, regions_atoms[0], :].view(coords.shape[0], -1, 3),
                region2  = coords[:, regions_atoms[1], :].view(coords.shape[0], -1, 3),
                t_native = t_native,
                R_native = R_native,
            )
            # For testing purposes
            match_res_mask = torch.empty([], device=device)
        else:
            # If our input is continuous (allow residues are next to each other in sequence and space) then we use normal
            # Harmonic diff on our dofs
            loss, match_res_mask = harmonic_diff_inpaint(
                real_t_sin,
                real_t_cos,
                real_ba_sin,
                real_ba_cos,
                real_bond_length,
                torsion_sin,
                torsion_cos,
                bond_angle_sin,
                bond_angle_cos,
                bond_length,
                torsion_index,
                bond_angle_dofs,
                bond_length_dofs,
                PARAMS,
                freeze_residues,
                freeze_torsion,
                freeze_bond_angle,
                freeze_bond_length,
                match_res_mask,
                inpaint_dist,
                device,
                dof_minimize,
            )

        # If we have no matches then we sound continue
        if match_res_mask.numel() == 0:
            print("No Scaffolds Match Your inpaint-dist. If this happens manytimes, \
            you may want to relax the threshold slightly. It is too strict.")
            return torch.empty([], device=device), torch.empty([], device=device), torch.empty([], device=device), torch.empty([], device=device), True
        # Calculate motif numbering
        # HERE
        motif, motif_input = calculate_motif_numbering_for_kabsch(
            match_res_mask,
            PARAMS["protein_len"],
            freeze_residues.shape[0],
            freeze_residues,
        )

        return loss, match_res_mask, motif, motif_input, False
