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
import argparse

# tensor dataset
import torch

# math
import numpy as np

# graph
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.io as pio


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

    def plot(self):
        """Plot latent space"""
        # generate space
        fig = plt.figure()
        ax = plt.axes(projection='3d')

        # plot 3d points
        ax.scatter(self.latent[:, -1, 0], self.latent[:, -1, 1], self.latent[:, -1, 2], c=self.latent[:, -1, 2], cmap="viridis", s=3)

        # convert distances to True False for coloring
        color_mat = self.distances < 1

        # pick random indices
        indices = np.random.choice(np.arange(0, self.latent.size(0)), size=1, replace=False)

        # new loop through pairs and generate colored lines
        # for i in range(10):
        for j in range(self.latent.size(0)):
            if (indices[0] != j) and (self.distances[indices[0], j] < 1):
                ax.plot(
                    [self.latent[indices[0], -1, 0], self.latent[j, -1, 0]],
                    [self.latent[indices[0], -1, 1], self.latent[j, -1, 1]],
                    [self.latent[indices[0], -1, 2], self.latent[j, -1, 2]],
                    color="blue", #if color_mat[indices[0],j] else "red",
                    alpha=0.5,
                    linewidth=2,
                )
        plt.title(f"Latent Space Representation; Node:{indices[0]} Distance Check")
        
        # legend
        line = Line2D([0], [0], color='blue', alpha=0.5, linewidth=2)
        plt.legend([line], ['RMSD < 1 (Real Space)'])

        plt.savefig(f"latentspace_graph_heuristic_{datetime.date.today()}.png")
        plt.show()

    def animate(self):
        """Animate the latent space"""
        # Create figure with scatter plot
        fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter3d'}]])

        # Add scatter traces for each sample at time point 0
        scatter_trace = go.Scatter3d(
            x=self.latent[:, 0, 0],
            y=self.latent[:, 0, 1],
            z=self.latent[:, 0, 2],
            mode='markers',
            marker=dict(size=1,
                        color=self.latent[:, 0, 2],
                        colorscale='Viridis',
                        colorbar=dict(title='Z axis'),
                        opacity=0.5,
                        ),
            name='Epoch: 0'
        )
        fig.add_trace(scatter_trace)

        # Add frames for animation
        frames = []
        for i in range(1, self.latent.size(1)):
            frame = go.Frame(
                data=[go.Scatter3d(
                    x=self.latent[:, i, 0],
                    y=self.latent[:, i, 1],
                    z=self.latent[:, i, 2],
                    mode='markers',
                    marker=dict(size=1,
                                color=self.latent[:, i, 2],
                                colorscale='Viridis',
                                colorbar=dict(title='Z axis'),
                                opacity=0.5,
                                ),
                    name=f'Epoch: {i*20}'
                )],
                name=f'Epoch: {i*20}'
            )
            frames.append(frame)

        # Update layout with frames
        fig.frames = frames

        # Set animation options
        fig.update_layout(updatemenus=[dict(type='buttons', buttons=[dict(label='Play',
                                                                          method='animate',
                                                                          args=[None, dict(frame=dict(duration=500, redraw=True), fromcurrent=True)])])])

        # Save the animation to an HTML file
        pio.write_html(fig, 'animated_3d_plot.html', auto_open=True)


###############################################################################################
#                                   Functions
###############################################################################################

def main():
    """Main function for testing Script"""
    p = argparse.ArgumentParser()
    p.add_argument("--latent_dataset", type=str, required=True,
                   help="Path to latent dataset extracted from model. (Default: None)")
    p.add_argument("--rmsd_dataset", type=str, default="./binary_import/dof_vectors2024-02-20torch_tensor.pt",
                   help="RMSD Dataset (Default: ./binary_import/dof_vectors2024-02-20torch_tensor.pt)")
    p.add_argument("--animate", action="store_true", help="Create animated Epoch Image of Latent Space")
    args = p.parse_args()

    # load in latent space
    latent = torch.load(args.latent_dataset)

    # rmsd
    data = torch.load(args.rmsd_dataset)
    rmsd = data["rmsd"]

    # init the graph class
    G = GraphLatent(
        latent=latent,
        distances=rmsd,
    )

    # Plot 
    G.plot()

    if args.animate:
        # Create the animation
        G.animate()

    return 0



###############################################################################################
#                                   Main
###############################################################################################

if __name__ == "__main__":
    main()

