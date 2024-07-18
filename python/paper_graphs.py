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
        ax.set_title(f"GenKIC vs. Model ({int(B/8)} Samples) Rama")
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
        fig, ax = plt.subplots(1, 2, figsize=(24, 12))

        # plot points
        ax[0].scatter(torsion_genkic[:, 0], torsion_genkic[:, 1], c="blue", alpha=0.5, label="GenKIC")
        ax[0].axhline(y=0, color="black", linewidth=2)
        ax[0].axvline(x=0, color="black", linewidth=2)

        ax[1].scatter(torsion_model[:, 0], torsion_model[:, 1], c="red", alpha=0.5, label="Model")
        ax[1].axhline(y=0, color="black", linewidth=2)
        ax[1].axvline(x=0, color="black", linewidth=2)

        # Set x and y labels and axis
        ax[0].set_title("GenKIC Rama", fontsize=30, weight="bold")
        ax[0].set_xlabel("φ", fontsize=30)
        ax[0].set_ylabel("ψ", fontsize=30)
        ax[0].set_ylim(-180, 180)
        ax[0].set_xlim(-180, 180)

        ax[1].set_title(f"Model Rama ({int(B/8)} Samples)", fontsize=30, weight="bold")
        ax[1].set_xlabel("φ", fontsize=30)
        ax[1].set_ylabel("ψ", fontsize=30)
        ax[1].set_ylim(-180, 180)
        ax[1].set_xlim(-180, 180)

        # Set legened
        fig.suptitle("GenKIC vs. Model Torsion Angle Comparison", fontsize=40, weight="bold")
        plt.tight_layout()
        plt.savefig(f'{file}.png')
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


        # plot
        plt.hist(non_termconstraint_len, alpha=0.5, label="No TermBond Constraint")
        plt.hist(with_termconstraint_len, alpha=0.5, label="TermBond Constraint")
        plt.legend(loc="upper right")
        plt.savefig("termbondHistrogram.png")
        plt.show()


# Main
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--genkic-data", help="GenKIC Data")
    p.add_argument("--model-data", help="Model Data")
    p.add_argument("--terminalbond-data", help="Dataset with Terminal Bond Constraint")
    p.add_argument("--no-terminalbond-data", help="Dataset without Terminal Bond Constraint")
    args = p.parse_args()

    # Terminal Peptide init
    peptide_term_class = TerminalPeptide()

    # Generate data phi_psi selection and pull out torsion
    data = torch.load(args.genkic_data)
    data_model = torch.load(args.model_data)
    data_model = data_model.detach()
    params = data["params"]
    x = data["x"]

    # pull out values
    torsion_full = x[:, :params["torsion_main_n"]]
    torsion_model = data_model[:, :params["torsion_main_n"]]

    phi_psi, _ = peptide_term_class.torsion_extract(torsion_full.size(-1), 4)

    genkic_data = torsion_full[:, phi_psi]
    model_data = torsion_model[:, phi_psi]

    # Init Class
    graph_rama = RamaCheck()
    graph_termbond = TermBondCheck(params=params, foldtree_root=4)

    # Test Graphs
    # graph_rama.graph_dataset(genkic_data)
    # graph_rama.graph_dataset(model_data, label="model")
    # graph_rama.graph_both(genkic_data, model_data)
    graph_rama.graph_sidebyside(genkic_data, model_data)
    # graph_termbond.plot_histogram(non_term, with_term)
