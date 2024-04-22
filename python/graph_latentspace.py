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
from scipy.interpolate import griddata
from matplotlib.colors import Normalize
from matplotlib.cm import viridis
from plotly.subplots import make_subplots


###############################################################################################
#                                   Classes
###############################################################################################

class GraphLatent:
    def __init__(
        self,
        latent: torch.Tensor,
        distances: torch.Tensor,
        X: torch.Tensor,
        rmsd_cutoff: float,
        marker_size: int,
        file: str,
        latent_dim: int,
        ):

        """Heuristically Interrogate the Latent Space (Color RMSD Close and Far distances differently)

        PARAMS
        ------
            latent: torch.Tensor
                Extracted Tensor of Latent Space R^3

            distances: torch.Tensor
                RMSD distances taken from the real world space

            X: torch.Tensor
                Full training dataset tensor, we need this for
                coloring by energy

            rmsd_cutoff: float
                Cutoff value for graphing lines.

            marker_size: int
                Size of the points in the plot
        """
        # store passed values
        self.latent = latent
        self.distances = distances
        self.X = X
        self.rmsd_cutoff = rmsd_cutoff
        self.marker_size = marker_size
        self.file = file
        self.latent_dim = latent_dim

    def plot(self):
        """Plot latent space"""
        # norm original scores
        # norm = Normalize(vmin=self.X[:, -1].min(), vmax=self.X[:, -1].max())
        # colors = viridis(norm(self.X[:, -1]))
        #
        # # spline infromation
        # xi, yi = np.linspace(self.latent[:, -1, 0].min(),self.latent[:, -1, 0].max(), self.latent.size(0)), np.linspace(self.latent[:, -1, 1].min(),self.latent[:, -1, 1].max(), self.latent.size(0))
        # xi, yi = np.meshgrid(xi, yi)
        #
        # # griddata for interpolation
        # zi = griddata((self.latent[:, -1, 0], self.latent[:, -1, 1]), self.latent[:, -1, 2], (xi, yi), method='cubic')
        # score_values = griddata((self.latent[:, -1, 0], self.latent[:, -1, 1]), self.X[:, -1], (xi, yi), method='cubic')
        #
        # normalize_score_values = norm(score_values)

        # generate space
        fig = plt.figure(figsize=(12,8))
        # ax = plt.axes(projection='3d')
        ax = fig.add_subplot(111, projection='3d')


        # plot 3d points
        scatter_plot = ax.scatter(self.latent[:, self.latent_dim, 0], self.latent[:, self.latent_dim, 1], self.latent[:, self.latent_dim, 2], c=self.X[:, -1], cmap="viridis", marker="o", s=self.marker_size)
        # scatter_plot = ax.scatter(self.latent[:, 1, 0], self.latent[:, 1, 1], self.latent[:, 1, 2], c=self.X[:, -1], cmap="viridis", marker="o", s=self.marker_size)
        # scatter_plot = ax.scatter(self.latent[:, -1, 0], self.latent[:, -1, 1], self.latent[:, -1, 2], c="red", marker="x", s=self.marker_size)

        # plot interpolated surface
        # surf = ax.plot_surface(xi, yi, zi, facecolors=viridis(normalize_score_values), edgecolor="none", alpha=0.7)

        # convert distances to True False for coloring
        color_mat = self.distances < self.rmsd_cutoff

        # pick random indices
        indices = np.random.choice(np.arange(0, self.X.size(0)), size=1, replace=False)
        ax.scatter(self.latent[indices[0], self.latent_dim, 0], self.latent[indices[0], self.latent_dim, 1], self.latent[indices[0], self.latent_dim, 2], marker="o", c="red", s=self.marker_size*2)

        # new loop through pairs and generate colored lines
        # for i in range(10):
        for j in range(self.X.size(0)):
            if (indices[0] != j) and (self.distances[indices[0], j] < self.rmsd_cutoff):
                ax.plot(
                    [self.latent[indices[0], self.latent_dim, 0], self.latent[j, self.latent_dim, 0]],
                    [self.latent[indices[0], self.latent_dim, 1], self.latent[j, self.latent_dim, 1]],
                    [self.latent[indices[0], self.latent_dim, 2], self.latent[j, self.latent_dim, 2]],
                    color="blue", #if color_mat[indices[0],j] else "red",
                    alpha=0.5,
                    linewidth=2,
                )
        plt.title(f"Latent Space Representation; Node:{indices[0]} Distance Check")
        
        # legend
        line = Line2D([0], [0], color='blue', alpha=0.5, linewidth=2)
        plt.legend([line], [f'RMSD < {self.rmsd_cutoff} (Real Space)'])

        plt.colorbar(scatter_plot, label="Scatter REU")
        # plt.colorbar(surf, label="REU")

        plt.savefig(f"latentspace_graph_heuristic_{self.file}.png")
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
            marker=dict(size=self.marker_size//4,
                        color=self.X[:, -1],
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
                    marker=dict(size=self.marker_size//4,
                                color=self.X[:, -1],
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
        pio.write_html(fig, f'animated_3d_plot_model_{self.file}.html', auto_open=True)


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
    p.add_argument("--rmsd-cutoff", default=1.0, type=float, help="Cutoff point to graph lines drawn on \
    a random sample for the graph (default: 1.0)")
    p.add_argument("--marker-size", default=10, type=int, help="Maker Size in Plots (default: 10)")
    p.add_argument("--latent-dim", default=-1, type=int, help="Specify latent graph (default: -1)")
    args = p.parse_args()

    # load in latent space
    latent = torch.load(args.latent_dataset)
    specific_file = args.latent_dataset.split("_")[4]
    print(specific_file)

    # rmsd
    data = torch.load(args.rmsd_dataset)
    rmsd = data["rmsd"]
    X = data["x"]

    # init the graph class
    G = GraphLatent(
        latent=latent,
        distances=rmsd,
        X=X,
        rmsd_cutoff=args.rmsd_cutoff,
        marker_size=args.marker_size,
        file = specific_file,
        latent_dim = args.latent_dim,
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

