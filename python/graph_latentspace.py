#!/usr/bin/env python

#
# @author: Andrew Powers (apowers@flatironinstitute.org) 
# @brief: Heuristic approach to distinguishing between Latent and Real Spacing Representation
#
#

###############################################################################################
#                                   Packages
###############################################################################################

# base
import os
from collections import defaultdict

# tensor dataset
import torch

# graph
import matplotlib.pyplot as plt
import seaborn as sns


###############################################################################################
#                                   Classes
###############################################################################################

class GraphLatent:
    def __init__(
        self,
        latent: torch.Tensor,
        distances: torch.Tensor,
        )
    """Heuristically Interrogate the Latent Space (Color RMSD Close and Far distances differently)

    PARAMS
    ------
        latent: torch.Tensor
            Extracted Tensor of Latent Space R^3

        distances: torch.Tensor
            RMSD distances taken from the real world space
    """


###############################################################################################
#                                   Functions
###############################################################################################

def main():
    """Main function for testing Script"""
    pass

###############################################################################################
#                                   Main
###############################################################################################

if __name__ == "__main__":
    main()

