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

# @brief: Class/Funcs to generate presentation graphs
# @author: Andrew Powers (apowers@flatironinstitue.org)
#

#########
# Imports
#########

import torch
import os, sys
import datetime
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import argparse
import numpy as np
from scipy.stats import gaussian_kde

from terminal_error import TerminalPeptide


#########
# Class
#########

class RamaCheck:
    def __init__(self):
        """Ramachandran Graph Dataset (GenKIC) vs Outupts (Model)"""
        

    def graph_dataset(self, torsion_angles: torch.Tensor, file: str = "genkic_rama", label: str = 'GenKIC'):
        """Graph only the one passed in dataset in a Rama Plot

        PARAMS
        ------
        torsion_angles: Tensor (B, D)
            Torsion angles only vector
        file: str
            output file name (default: genkic_rama)
        label: str
            Legend Point Label (default: GenKIC)
        """
        # Reshape torsion
        torsion = torsion_angles.reshape(-1, 2)

        # Set up plot
        fig, ax = plt.subplots(1, 1, figsize=(8,8))

        # plot points
        ax.scatter(torsion[:, 0], torsion[:, 1], c="blue", alpha=0.5, label=label)
        ax.axhline(y=0, color="black", linewidth=2)
        ax.axvline(x=0, color="black", linewidth=2)

        # Set x and y labels and axis
        ax.set_title(f"{label} Rama")
        ax.set_xlabel("φ")
        ax.set_ylabel("ψ")
        ax.set_ylim(-185, 185)
        ax.set_xlim(-185, 185)

        # Set legened
        plt.legend()
        plt.savefig(f'{file}.png')
        plt.show()

    def contour(self, torsion_angles: torch.Tensor, torsion_angles2: torch.Tensor, file: str = "genkic_rama", label: str = 'GenKIC'):
        """Graph only the one passed in dataset in a Rama Plot

        PARAMS
        ------
        torsion_angles: Tensor (B, D)
            Torsion angles only vector
        file: str
            output file name (default: genkic_rama)
        label: str
            Legend Point Label (default: GenKIC)
        """
        # Reshape torsion
        torsion = torsion_angles.reshape(-1, 2).T
        torsion2 = torsion_angles2.reshape(-1, 2).T

        # kenerl density 
        kde = gaussian_kde(torsion)
        kde2 = gaussian_kde(torsion2)

        # create a grid
        x_grid = np.linspace(-180, 180, 200)
        y_grid = np.linspace(-180, 180, 200)
        x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
        grid_coords = np.vstack([x_mesh.ravel(), y_mesh.ravel()])

        # desnity estimate
        z = kde(grid_coords).reshape(x_mesh.shape)
        z2 = kde2(grid_coords).reshape(x_mesh.shape)

        # Set up plot
        fig, ax = plt.subplots(1, 2, figsize=(24, 12))

        # plot points
        ax[0].contourf(x_mesh, y_mesh, z, cmap="Blues")
        ax[0].axhline(y=0, color="black", linewidth=2)
        ax[0].axvline(x=0, color="black", linewidth=2)

        ax[1].contourf(x_mesh, y_mesh, z2, cmap="Oranges")
        ax[1].axhline(y=0, color="black", linewidth=2)
        ax[1].axvline(x=0, color="black", linewidth=2)

        # Set x and y labels and axis
        ax[0].set_title("Decode Rama Contour", fontsize=30, weight="bold")
        ax[0].set_xlabel("φ", fontsize=30)
        ax[0].set_ylabel("ψ", fontsize=30)
        ax[0].set_ylim(-180, 180)
        ax[0].set_xlim(-180, 180)

        ax[1].set_title(f"Decode + Minimize Rama Contour", fontsize=30, weight="bold")
        ax[1].set_xlabel("φ", fontsize=30)
        ax[1].set_ylabel("ψ", fontsize=30)
        ax[1].set_ylim(-180, 180)
        ax[1].set_xlim(-180, 180)

        # Set legened
        fig.suptitle(f"Decode vs. Decode + Minimize Torsion Angle Comparison", fontsize=40, weight="bold")
        plt.tight_layout()
        plt.savefig(f'{file}_{datetime.date.today()}.png')
        plt.show()
        # # Set up plot
        # fig, ax = plt.subplots(1, 1, figsize=(8,8))

        # # plot points
        # ax.contour(x_mesh, y_mesh, z, cmap="Blues")
        # ax.contour(x_mesh, y_mesh, z2, cmap="Oranges")
        # # ax.scatter(torsion[:, 0], torsion[:, 1], color="blue", alpha=0.5, label="Decode")
        # # ax.scatter(torsion2[:, 0], torsion2[:, 1], color="red", alpha=0.5, label="Decode + Minimize")
        # ax.axhline(y=0, color="black", linewidth=2)
        # ax.axvline(x=0, color="black", linewidth=2)
        #
        # # Set x and y labels and axis
        # ax.set_title("Decode vs Decode + Minimize", fontsize=20)
        # ax.set_xlabel("φ", fontsize=15)
        # ax.set_ylabel("ψ", fontsize=15)
        # ax.set_ylim(-185, 185)
        # ax.set_xlim(-185, 185)
        #
        # # Set legened
        # plt.legend()
        # plt.savefig(f'{file}.png')
        # plt.show()

    def graph_both(self, torsion_genkic: torch.Tensor, torsion_model: torch.Tensor,
                   file: str = "comparison_rama"):
        """Graph only the one passed in dataset in a Rama Plot

        PARAMS
        ------
        torsion_genkic: Tensor (B, D)
            Torsion angles only matrix for GenKIC
        torsion_model: Tensor (B, D)
            Torsion angles only matrix for Model Data
        file: str
            output file name (default: comparison_rama)
        """
        # Reshape torsion
        torsion_genkic = torsion_genkic.reshape(-1, 2)
        torsion_model = torsion_model.reshape(-1, 2)
        # Grab dim for title
        B, _ = torsion_model.shape

        # Set up plot
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        # plot points
        ax.scatter(torsion_genkic[:, 0], torsion_genkic[:, 1], c="blue", alpha=0.5, label="GenKIC")
        ax.scatter(torsion_model[:, 0], torsion_model[:, 1], c="red", alpha=0.5, label="Model")
        ax.axhline(y=0, color="black", linewidth=2)
        ax.axvline(x=0, color="black", linewidth=2)

        # Set x and y labels and axis
        ax.set_title(f"GenKIC vs. Model ({int(B/8)} Samples) Rama")
        ax.set_xlabel("φ")
        ax.set_ylabel("ψ")
        ax.set_ylim(-180, 180)
        ax.set_xlim(-180, 180)

        # Set legened
        plt.legend()
        plt.savefig(f'{file}.png')
        plt.show()

    def graph_sidebyside(
        self, 
        torsion_genkic: torch.Tensor, 
        torsion_model: torch.Tensor,
        file: str = "sidebyside_rama",
        genkic_data_title: str = "GenKIC",
        model_data_title: str = "Model",
        ):
        """Graph only the one passed in dataset in a Rama Plot

        PARAMS
        ------
        torsion_genkic: Tensor (B, D)
            Torsion angles only matrix for GenKIC
        torsion_model: Tensor (B, D)
            Torsion angles only matrix for Model Data
        file: str
            output file name (default: sidebyside_rama)
        """
        # Reshape torsion
        torsion_genkic = torsion_genkic.reshape(-1, 2)
        torsion_model = torsion_model.reshape(-1, 2)
        # Grab dim for title
        B, _ = torsion_model.shape

        # Set up plot
        fig, ax = plt.subplots(1, 2, figsize=(24, 12))

        # plot points
        ax[0].scatter(torsion_genkic[:, 0], torsion_genkic[:, 1], c="blue", alpha=0.5, label=genkic_data_title)
        ax[0].axhline(y=0, color="black", linewidth=2)
        ax[0].axvline(x=0, color="black", linewidth=2)

        ax[1].scatter(torsion_model[:, 0], torsion_model[:, 1], c="red", alpha=0.5, label=model_data_title)
        ax[1].axhline(y=0, color="black", linewidth=2)
        ax[1].axvline(x=0, color="black", linewidth=2)

        # Set x and y labels and axis
        ax[0].set_title(f"{genkic_data_title} Rama", fontsize=30, weight="bold")
        ax[0].set_xlabel("φ", fontsize=30)
        ax[0].set_ylabel("ψ", fontsize=30)
        ax[0].set_ylim(-180, 180)
        ax[0].set_xlim(-180, 180)

        ax[1].set_title(f"{model_data_title} Rama ({int(B/8)} Samples)", fontsize=30, weight="bold")
        ax[1].set_xlabel("φ", fontsize=30)
        ax[1].set_ylabel("ψ", fontsize=30)
        ax[1].set_ylim(-180, 180)
        ax[1].set_xlim(-180, 180)

        # Set legened
        fig.suptitle(f"{genkic_data_title} vs. {model_data_title} Torsion Angle Comparison", fontsize=40, weight="bold")
        plt.tight_layout()
        plt.savefig(f'{file}_{datetime.date.today()}.png')
        plt.show()

class TermBondCheck:
    def __init__(self, params, foldtree_root):
        self.term_peptide = TerminalPeptide()
        self.params = params
        self.foldtree_root = foldtree_root

    def plot_histogram(self, non_termconstraint: torch.Tensor, with_termconstraint: torch.Tensor):
        """Plot Histogram of model outputs with/without terminal bond constraint outputs

        PARAMS
        ------
        non_termconstraint: Tensor (B,M)
            Non Terminal Bond Constraint lengths
        with_termconstraint: Tensor (B,M)
            Terminal Bond Constraint lengths
        """
        # generate term lengths
        non_termconstraint_len = self.term_peptide.terminal_peptide_lengths(non_termconstraint,
                                                                            params=self.params,
                                                                            foldtree_root=self.foldtree_root,
                                                                            device="cpu")
        with_termconstraint_len = self.term_peptide.terminal_peptide_lengths(with_termconstraint,
                                                                            params=self.params,
                                                                            foldtree_root=self.foldtree_root,
                                                                            device="cpu")

        bins = np.linspace(0, non_termconstraint_len.max().item(), 50)
        # plot
        fig, ax = plt.subplots(1,1, figsize=(12,8))
        ax.hist(non_termconstraint_len, bins, alpha=0.5, label="No TermBond Constraint")
        ax.hist(with_termconstraint_len, bins, alpha=0.5, label="TermBond Constraint")
        ax.set_title("Terminal Bond Length (With Constraint vs Without)", fontsize=20)
        ax.set_xlabel("Terminal Bond Length (Angstrom)", fontsize=15)
        ax.set_ylabel("Frequency", fontsize=15)
        plt.legend(loc="upper right", fontsize=12)
        plt.savefig(f"termbondHistrogram_{datetime.date.today()}.png")
        plt.show()

def compute_time():
    """Generate a CPU/GPU Model vs GenKIC compute time graph"""
    # Define values
    cpu_terminal = [24.722, 44.538, 172.02, 1298.58]
    cpu_nonterminal = [7.232, 20.464, 126.592, 1084.223]
    gpu_terminal = [58.028, 67.747, 82.773, 386.93]
    gpu_nonterminal = [17.456, 17.314, 37.069, 308.338]
    genkic = [8.995, 17.391, 236.189, 13800.]

    # Labels for the x-axis
    labels = ['10', '100', '1,000', '10,000']

    # Number of groups
    num_groups = len(labels)

    # Number of bars per group
    num_bars = 5

    # Bar width
    bar_width = 0.15

    # Positions of the bars on the x-axis
    r1 = np.arange(num_groups)
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    r4 = [x + bar_width for x in r3]
    r5 = [x + bar_width for x in r4]

    # Create the plot
    fig = plt.figure(figsize=(12,8))
    gs = GridSpec(2, 1, height_ratios=[1, 3])
    # fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12,8))

    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])

    # Make space for break in cut
    # fig.subplots_adjust(hspace=0.1)

    # Make the bar plots
    ax2.bar(r1, cpu_terminal, width=bar_width, label='CPU Terminal')
    ax2.bar(r2, cpu_nonterminal, width=bar_width, label='CPU Nonterminal')
    ax2.bar(r3, gpu_terminal, width=bar_width, label='GPU Terminal')
    ax2.bar(r4, gpu_nonterminal, width=bar_width, label='GPU Nonterminal')
    ax2.bar(r5, genkic, width=bar_width, label='CPU GenKIC')
    ax2.set_ylim(0, 1380)

    # Second higher value plot
    ax1.bar(r1, cpu_terminal, width=bar_width, label='CPU Terminal')
    ax1.bar(r2, cpu_nonterminal, width=bar_width, label='CPU Nonterminal')
    ax1.bar(r3, gpu_terminal, width=bar_width, label='GPU Terminal')
    ax1.bar(r4, gpu_nonterminal, width=bar_width, label='GPU Nonterminal')
    ax1.bar(r5, genkic, width=bar_width, label='CPU GenKIC')
    ax1.set_ylim(13500, max(genkic)+10)

    # Add xticks on the middle of the group bars
    ax2.set_xlabel('Nstruct', fontsize=20)
    ax2.set_xticks([r + bar_width*2 for r in range(num_groups)])
    ax2.set_xticklabels(labels)
    ax1.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off


    # Add labels and title
    ax2.set_ylabel('Time (Seconds)', fontsize=20)
    fig.suptitle('Compute Time of GenKIC vs Model (CPU (single Node)/GPU)', fontsize=30)
    ax2.legend(fontsize=15)

    # add break
    # d = .015
    # kwargs = dict(transform=ax1.transAxes, color="k", clip_on=False)
    # ax1.plot((-d, +d), (-d, +d), **kwargs)
    # ax1.plot((1-d, 1+d), (-d, +d), **kwargs)
    #
    # kwargs.update(transform=ax2.transAxes)
    # ax2.plot((-d, +d), (1-d, 1+d), **kwargs)
    # ax2.plot((1-d, 1+d), (1-d, 1+d), **kwargs)

    # Display the plot
    plt.show()


# Main
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--genkic-data", help="GenKIC Data")
    p.add_argument("--model-data", help="Model Data")
    p.add_argument("--terminalbond-data", help="Dataset with Terminal Bond Constraint")
    p.add_argument("--terminalbond", action="store_true", help="Graph Terminal Bond")
    p.add_argument("--rama-graph", action="store_true", help="Graph Ramachandran")
    p.add_argument("--compute-time", action="store_true", help="Graph CPU/GPU for Model vs GenKIC Graph")
    p.add_argument("--contour-plot", action="store_true", help="Contour Plot")
    args = p.parse_args()

    if args.compute_time:
        compute_time()
    else:
        # Terminal Peptide init
        peptide_term_class = TerminalPeptide()

        # Generate data phi_psi selection and pull out torsion
        data = torch.load(args.genkic_data)
        if args.rama_graph:
            data_model = torch.load(args.model_data)
            data_model = data_model.detach()
        params = data["params"]
        x = data["x"]

        # pull out values
        torsion_full = x[:, :params["torsion_main_n"]]
        if args.rama_graph:
            torsion_model = data_model[:, :params["torsion_main_n"]]

        phi_psi, _ = peptide_term_class.torsion_extract(torsion_full.size(-1), 4)

        genkic_data = torsion_full[:, phi_psi]
        if args.rama_graph:
            model_data = torsion_model[:, phi_psi]

        # Init Class
        graph_rama = RamaCheck()
        graph_termbond = TermBondCheck(params=params, foldtree_root=4)

        # Test Graphs
        # graph_rama.graph_dataset(genkic_data)
        # graph_rama.graph_dataset(model_data, label="model")
        # graph_rama.graph_both(genkic_data, model_data)
        if args.rama_graph:
            graph_rama.graph_sidebyside(genkic_data, model_data)
        elif args.contour_plot:
            model_data = torch.load(args.model_data, map_location="cpu")

            # grab my original data
            original_data = model_data["original_decode_dofs"].cpu().detach()

            decomp_torsion = params["torsion_main_n"]*2
            decomp_bondang = params["bond_angle_main_n"]*2

            # grab out terms of interest
            torsion_orig_full = original_data[:, :decomp_torsion]

            # torsion angles
            torsion_degree = torch.rad2deg( 
                torch.atan2( 
                    torsion_orig_full[:, :torsion_orig_full.size(-1)//2],
                    torsion_orig_full[:, torsion_orig_full.size(-1)//2:],
                )
            )
            minimized_data = model_data["minimized_dofs"][:, phi_psi].cpu().detach()
            graph_rama.contour(torsion_degree[:, phi_psi], minimized_data, "minimized_contour.png")
        elif args.terminalbond:
            # terminal load
            terminal_data = torch.load(args.terminalbond_data)
            # grab data
            non_term = terminal_data["without_constraint"]
            with_term = terminal_data["with_constraint"]
            graph_termbond.plot_histogram(non_term, with_term)
            graph_rama.graph_sidebyside(non_term[:, phi_psi], with_term[:, phi_psi],
                                        file="constraintWithvsWithout_RAMA",
                                        genkic_data_title="Without Constraint",
                                    model_data_title="With Constraint")
