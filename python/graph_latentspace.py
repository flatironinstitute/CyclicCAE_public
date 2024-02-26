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
import datetime

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
        ):

        """Heuristically Interrogate the Latent Space (Color RMSD Close and Far distances differently)

        PARAMS
        ------
            latent: torch.Tensor
                Extracted Tensor of Latent Space R^3

            distances: torch.Tensor
                RMSD distances taken from the real world space
        """
        # store passed values
        self.latent = latent
        self.distances = distances

    def plot():
        """Plot latent space"""
        # generate space
        fig = plt.figure()
        ax = plt.axes(projection='3d')

        # plot 3d points
        ax.scatter(self.latent[:, 0], self.latent[:, 1], self.latent[:, 2])

        # convert distances to True False for coloring
        color_mat = self.distances < 1

        # new loop through pairs and generate colored lines
        for i in range(self.latent.size(0)):
            for j in range(i+1, self.latent.size(0)):
                if i != j:
                    ax.plot(
                        [self.latent[i,0], self.latent[j, 0]],
                        [self.latent[i,1], self.latent[j, 1]],
                        [self.latent[i,2], self.latent[j, 2]],
                        color="blue" if color_mat[i,j] else "red",
                        alpha=0.5,
                    )
        plt.savefig(f"latentspace_graph_heuristic_{datetime.date.today()}.png")
        plt.show()



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

