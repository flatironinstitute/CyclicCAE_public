#!/usr/bin/env python 
# @author: Andrew Powers (apowers@flatironinstitute.org)
# @brief: Classes and functions helpfulf for inference task
#

######## IMPORT 
import torch
import torch.nn as nn

# xyz coordinates
from terminal_error import TerminalPeptide

######## Class

class HBondCount:
    def __init__(
        self,
        min_hbond: int,
        ):
        """Class is for specifying how many hbonds are within
        a structure at some t step during inference. WHere t \in [0, N-1]

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
        dists = torch.norm(
            H_coords.unsqueeze(-2), # (B, N, 1, 3)
            O_coords.unsqueeze(-2).repeat(1, 1, O_coords.size(1), 1), # (B, N, N, 3)
            dim=-1)

        # Check where the distances are within the correct spot
        dists_bool = (dists >= self.min_hbond) & (dists <= self.max_hbond)

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

        




