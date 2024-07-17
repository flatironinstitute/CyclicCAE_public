#!/usr/bin/env python
#
# @brief: Class/Funcs to generate presentation graphs
# @author: Andrew Powers (apowers@flatironinstitue.org)
#

#########
# Imports
#########

import torch
import os, sys
import matplotlib.pyplot as plt
import argparse

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
        ax.set_title(f"GenKIC vs. Model ({B} Samples) Rama")
        ax.set_xlabel("φ")
        ax.set_ylabel("ψ")
        ax.set_ylim(-180, 180)
        ax.set_xlim(-180, 180)

        # Set legened
        plt.legend()
        plt.savefig(f'{file}.png')
        plt.show()

    def graph_sidebyside(self, torsion_genkic: torch.Tensor, torsion_model: torch.Tensor,
                   file: str = "sidebyside_rama"):
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
        fig, ax = plt.subplots(1, 2, figsize=(12,8))

        # plot points
        ax[0].scatter(torsion_genkic[:, 0], torsion_genkic[:, 1], c="blue", alpha=0.5, label="GenKIC")
        ax[0].axhline(y=0, color="black", linewidth=2)
        ax[0].axvline(x=0, color="black", linewidth=2)

        ax[1].scatter(torsion_model[:, 0], torsion_model[:, 1], c="red", alpha=0.5, label="Model")
        ax[1].axhline(y=0, color="black", linewidth=2)
        ax[1].axvline(x=0, color="black", linewidth=2)

        # Set x and y labels and axis
        ax[0].set_title("GenKIC Rama")
        ax[0].set_xlabel("φ")
        ax[0].set_ylabel("ψ")
        ax[0].set_ylim(-180, 180)
        ax[0].set_xlim(-180, 180)

        ax[1].set_title(f"Model Rama {B}")
        ax[1].set_xlabel("φ")
        ax[1].set_ylabel("ψ")
        ax[1].set_ylim(-180, 180)
        ax[1].set_xlim(-180, 180)

        # Set legened
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{file}.png')
        plt.show()

# Main
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--genkic-data", help="GenKIC Data")
    p.add_argument("--model-data", help="Model Data")
    args = p.parse_args()

    # Terminal Peptide init
    peptide_term_class = TerminalPeptide()

    # Generate data phi_psi selection and pull out torsion
    data = torch.load(args.genkic_data)
    data_model = torch.load(args.model_data)
    params = data["params"]
    x = data["x"]

    # pull out values
    torsion_full = x[:, :params["torsion_main_n"]]
    torsion_model = data_model[:, :params["torsion_main_n"]]

    genkic_data = torsion_full[:, phi_psi]
    model_data = torsion_model[:, phi_psi]

    # Init Class
    graph_rama = RamaCheck()

    # Test Graphs
    # graph_rama.graph_dataset(genkic_data)
    # graph_rama.graph_dataset(model_data, label="model")
    # graph_rama.graph_both(genkic_data, model_data)
    graph_rama.graph_sidebyside(genkic_data, model_data)
