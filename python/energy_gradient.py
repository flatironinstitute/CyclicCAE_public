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

# @author: Andrew Powers (apowers@flatironinstitute.org)
# @brief: Function to carry out Gradient Descent Marching along energy
#       term, during evaluation mode.
#

# native Packages
import os, sys
import datetime
import argparse
import ast
import time
from datetime import timedelta
from decorators import timeit

# import torch
import torch
import torch.nn as nn
import gc

# graphing
import matplotlib.pyplot as plt

# import model and functions
# from test_case import ConvCase4
from model_house import CyclicCAE
from model_functions import scale, ScaleOutputs
from terminal_error import TerminalPeptide
from inference_utils import (
    extract_pdb_xyz, geometric_center, grab_interface,
    interface_loss, kabsch_algorithm, kabsch_rootatom, size_help,
    split_motif_atoms, discontig_motif_loss, kabsch_rigid_transform,
    homochiral_loss, load_weights,
)
from dofSummary import DoFSummary
from motif_functions import (
    compute_motif_indices, calculate_motif_numbering_for_kabsch,
    extract_dof_of_interest_minimize,
)

import discontig_funcs

from inpaint_funcs import extract_loss_and_matchres

#############
##### Classes
#############

class CustomSinCosLoss(nn.Module):
    def __init__(self, sin_weight=1.0, cos_weight=1.0):
        super().__init__()
        self.sin_weight = sin_weight
        self.cos_weight = cos_weight

    def forward(self, preds, targets):
        """Custom MSE Loss for sin/cos"""
        # Grab sin and cos values
        sin_preds, sin_targets = preds[:, :preds.size(-1)//2], targets[:, :targets.size(-1)//2]
        cos_preds, cos_targets = preds[:, preds.size(-1)//2:], targets[:, targets.size(-1)//2:]

        # calculate the MSE loss
        diff_sin = torch.pow( sin_targets - sin_preds, 2)
        diff_cos = torch.pow( cos_targets - cos_preds, 2)

        # Reweight based on importance
        loss = torch.mean( diff_sin*self.sin_weight + diff_cos*self.cos_weight )

        return loss

class EnergyPath:
    def __init__(
        self,
        steps: int,
        weight_passed: str,
        lr: float,
        cutoff: float,
        cutoff_true: bool,
        partial_optimization: bool,
        freeze_residues: list,
        freeze_torsion: bool,
        freeze_bond_angle: bool,
        freeze_bond_length: bool,
        lambda_partial: float,
        initial_set: int,
        temp: float,
        rampdown: int,
        latent_dim: int,
        retain_structure: bool,
        inpaint_dist: float,
        device: str,
        scaffold_number: int,
        mcmc_paths: int = 10,
        adam_optim: bool = True,
        sgd_optim: bool = False,
        out_size: int = 8,
        interface_cutoff: float = 1.0,
        ramp_constraint: bool = True,
        full_pocket: bool = False,
        random_sample: bool = False,
        exclude_atom_positions: list[str] = [],
        energy_weight: float = 1.0,
        encode_decode: bool = False,
        ):
        """GDS for energy minima structure from starting point

        PARAMS
        ------
            steps: int
                Number of steps for the search to take
            start_point: torch.Tensor
                DoF Vector to use as a starting point
            weight_passed: str
                Path to model weights
            lr: float
                LR for optimizer
            dataset: str
                path to the dataset
            cutoff: float
                Distance in Latent Space cutoff for gradient direction
                control
            partial_optimization: bool
                Whether partial optimization should be used or not
            freeze_residues: list,
                Residues to freeze (zero index)
            freeze_torsion, bond_angle, bond_length : bool, bool, bool
                Specify the DoF to freeze if not specified then all are frozen
            labmda_partial: float,
                How importantly you want to weight partial freezing
            initial_set: int,
                Amount of steps to always accept movement
            temp: float,
                Temperature for increasing or decreasing noise
            rampdown: int,
                Number of steps to go from a low temp to high temp
            latent_dim: int,
                Latent Space Dimension Size Needed for init the Model
            retain_structure: bool,
                While doing MCMC keep structure close to input
            inpaint_dist: float
                The pnorm distance that is close enough to accept
            device: str
                Either CPU or CUDA for GPU access
            scaffold_number: int
                Number of scaffold to make; this is a cutoff
            adam_optim: bool
                Use Adam optimizer for energy search
            sgd_optim: bool
                Use SGD optimizer for energy search
            out_size: int
                The size of the peptide you would liek to produce [6 - 8mer currently 12-08-2024]
            interface_cutoff: int
                The Angstrom distance that should be applied for designating something is not clashing
            ramp_constraint: bool
                Specifies if we are ramping from cb -> full pocket or not during energy minimization.
            full_pocket: bool
                Specify if we are only keeping full pocket or cb if ramp_constraints not set
            random_sample: bool
                Generate random samples from the latent space for whatever passed batch size in params.
            exclude_atom_positions: list[str]
                Atom positions to exclude from our DoF loss calculation (e.g. input [1:0, 1:3]) This would exclude
                the first residues Nitrogen and Oxygen
            energy_weight: float
                Value to scale the energy during energy_steps()
        """
        # Pull up points
        self.steps = steps
        self.outstep = steps
        self.lr = lr
        self.lambda_2 = lambda_partial
        self.cutoff = cutoff
        self.cutoff_true = cutoff_true
        self.partial_optimization = partial_optimization
        self.freeze_residues = freeze_residues
        self.freeze_torsion = freeze_torsion
        self.random_sample = random_sample
        self.freeze_bond_angle = freeze_bond_angle
        self.freeze_bond_length = freeze_bond_length
        if not self.partial_optimization:
            self.freeze_torsion = True
            self.freeze_bond_angle = True
            self.freeze_bond_length = True
        self.contiguous = self.check_contiguous(self.freeze_residues, out_size)
        self.exclude_atom_positions = exclude_atom_positions
        if not self.contiguous:
            self.regions_atoms, self.regions = split_motif_atoms(self.freeze_residues)
        else:
            self.regions = [self.freeze_residues]
        self.initial_set = initial_set
        self.temp = temp
        self.rampdown = rampdown
        self.latent_dim = latent_dim
        self.retain_structure = retain_structure
        # If retain structure, we want to keep it as close as we can
        if self.retain_structure:
            self.freeze_torsion = True
            self.freeze_bond_angle = True
            self.freeze_bond_length = True
        self.inpaint_dist = inpaint_dist
        self.adam_optim = adam_optim
        self.sgd_optim = sgd_optim
        # Terminal Bond N-C class
        self.term_bond = TerminalPeptide()
        self.device = device
        self.scaffold_number = scaffold_number
        self.weight_passed = weight_passed
        self.mcmc_paths = mcmc_paths
        self.interface_cutoff = interface_cutoff
        self.ramp_constraint = ramp_constraint
        self.fullpocket = full_pocket
        self.match_res_mask = torch.empty((0, 2), dtype=torch.int, device=self.device)
        self.energy_weight = energy_weight
        self.extract_minimize_input = encode_decode
        self.ideal_term_length = 1.36 # 1.4

        # Grab values required for model and other parts
        self.PARAMS, self.dof_size = size_help(out_size, dict())
        self.energy_n = self.PARAMS["energy_n"]

        # generate holding tensor and input the beginning traj
        self.dof_traj = torch.zeros((steps, self.dof_size - self.energy_n), dtype=torch.float64, device=device)

        # init the model and eval mode
        # self.model = ConvCase4(
        self.model = CyclicCAE(
            encoder_layers    = 3,
            decoder_layers    = 3,
            latent_space_dim  = self.latent_dim,
            decomposition_n   = 1,
            dropout           = 0.1,
            noise_std         = 3,
            cyclic            = False, #True,
            pe_linear         = True, #False,
            convolution_layer = True,
            device            = self.device, #"cpu",
            **self.PARAMS,
        ).to(device).double()
        weights = load_weights(self.PARAMS["protein_len"],
                               self.device,
                               self.weight_passed)
        self.model.load_state_dict(weights["model_state_dict"])
        self.model.eval()

        # scale start point data
        # X_start_point = torch.cat((X, self.start_point_dof.unsqueeze(0)), dim=0)
        # scaled_X_start_point = scale(X_start_point, self.ORIG_PARAMS)
        # traj_scale_dof = scale(X_start_point[:, :self.dof_size], self.PARAMS)
        # print(scaled_X_start_point.shape, traj_scale_dof.shape)

        #### Since right now we are not inputting a pdb to search we will select a
        #### start point for the beginning of the search
        self.path = torch.zeros(self.steps)
        self.energy_path = torch.zeros(self.steps)

        # scale energy for conversion importance
        score_terms = torch.tensor([2]) if self.energy_n == 1 else torch.arange(self.energy_n)
        self.energy_scale = ScaleOutputs(None, train=False, score_terms=score_terms, protein_len=self.PARAMS["protein_len"])


        ### Gonna try terminal index functions
        root = self.term_bond.calculate_root(self.PARAMS["protein_len"])
        self.phi_psi_idx, self.omega_idx = self.term_bond.torsion_extract(self.PARAMS["torsion_main_n"], self.PARAMS["protein_len"], root)
        bond_angle_idx, bond_length_idx = self.term_bond.extract_bond_angles(self.PARAMS["bond_angle_main_n"], root=root, L=self.PARAMS["protein_len"])
        if self.freeze_residues != None:
            self.torsion_index, self.bond_angle_dofs, self.bond_length_dofs, self.residue_index = compute_motif_indices(
                self.PARAMS["protein_len"],
                self.PARAMS["torsion_main_n"],
                self.freeze_residues.shape[0],
                root,
                self.phi_psi_idx,
                self.omega_idx,
                bond_angle_idx,
                bond_length_idx,
            )
        else:
            self.torsion_index = slice(None)
            self.bond_angle_dof = slice(None)
            self.bond_length_dof = slice(None)
            self.residue_index = slice(None)

    def check_contiguous(self, motif: list[int], peptide_length: int | None = None):
        """Check to see if the motif passed is contiguous or not

        PARAMS
        ------
        motif: list[int]
            Motifs passed as integers non zero index
        RETURNS
        ------
        out: True/False
            If the motif is contiguous or not
        """
        # If it is only 1 residue, no need to check just pass true
        if motif == None:
            return True
        elif len(motif) == 1:
            return True

        values = [int(value) for value in motif]
        for previous, current in zip(values, values[1:]):
            if peptide_length is None:
                adjacent = previous + 1 == current
            else:
                adjacent = (current - previous) % peptide_length == 1
            if not adjacent:
                return False
        return True

    def extract_latent_information( self, dataset: torch.Tensor ) -> torch.Tensor:
        """Extract out latent space information from a dataset

        PARAMS
        ------
        :dataset: This ideally should be a full dataset used to train the model, so that the bounds
            of the latentspace can be assessed.
        """
        out, _, latent_points, _ = self.model.forward( dataset, noise=False )
        return out, latent_points


    def encode_decode(self, pdb_input: str, dofsummary: DoFSummary, chain: str = 'A'):
        """Pass a pdb structure, extract the DoFs from the pose, encode and then decode them

        PARAMS
        ------
        :pdb_input: Path to our pdb chain
        :chain: The cyclic peptide chain
        """
        extracted_dofs, motif_dofs = dofsummary.extract_all_dofs_from_pdb(pdb_input, chain)
        out, _, latent_points, _ = self.model.forward( extracted_dofs, noise=False )
        out = out.detach().clone()
        torsion = self.calc_angles(out[:, :self.PARAMS["torsion_main_n"]*2])
        bond_angle = self.calc_angles(out[:, self.PARAMS["torsion_main_n"]*2:self.PARAMS["bond_angle_main_n"]*2*2])
        preds = torch.cat((torsion, bond_angle, out[:, self.PARAMS["torsion_main_n"]*2*2:-self.energy_n]), dim=-1)
        self.dof_traj = preds.unsqueeze(0)
        return motif_dofs, latent_points.detach().clone()

    def calc_angles(self, preds: torch.Tensor):
        """Display the Computed Angles"""
        sin_pred = preds[:, :preds.size(-1)//2]
        cos_pred = preds[:, preds.size(-1)//2:]

        return torch.rad2deg(torch.atan2(sin_pred, cos_pred))

    def euclidean_measure(self, cutoff: float, latent_space: torch.Tensor, original_point: torch.Tensor):
        """Term to keep energy refinement to a minimum if wanted.

        PARAMS
        ------
            cutoff: float
                The Euclidean distance cutoff you want to enforce to stay within

            latent_space: torch.Tensor
                Current Latent Space position (N, 3)

            original_point: torch.Tensor
                Origional Startin position within the Latent Space (N, 3)
        """
        # get the distance
        dist = torch.cdist(original_point, latent_space, p=2)

        return torch.pow(torch.maximum(dist - cutoff, torch.tensor(0)), 2), dist

    def linear_loss(self, real:torch.Tensor, pred: torch.Tensor):
        return torch.sum( torch.pow( real - pred, 2) )

    def dist_energy_plot(self):
        """Plot Distance and Energy Path through SGD"""
        # set up plot
        fig, ax = plt.subplots(2, 1, figsize=(12, 8))

        # set up first plot
        ax[0].plot(torch.arange(self.outstep), self.path)
        ax[0].set_title("Movement In Latent Space Away During SGD")
        ax[0].set_xlabel("Gradient Step")
        ax[0].set_ylabel("Distance from initial point (Euclidean Distance)")

        # set up energy plot
        ax[1].plot(torch.arange(self.outstep), self.energy_path, c="orange")
        ax[1].set_title("Energy Change During SGD")
        ax[1].set_xlabel("Gradient Step")
        ax[1].set_ylabel("Energy Change (REU Pred)")

        # savefig
        plt.tight_layout()
        plt.savefig(f"plotstep{self.outstep}-{datetime.date.today()}-energy-dist.png")

    def generate_mask(self, a: torch.Tensor, idx: torch.Tensor):
        """Grab all values that are not the idx values passed

        PARAMS
        ------
            a: torch.Tensor
                Tensor of all values
            idx: torch.Tenosr
                Indexes of values you do not want to keep
        RETURN
        ------
            torch.Tensor
                a , but the all of the values in idx have been removed from it.
        """
        mask = torch.ones(a.numel(), dtype=torch.bool)
        mask[idx] = False
        return a[mask.unsqueeze(0)]

    def harmonic_diff_inpaint(
        self,
        real_t_sin: torch.Tensor,
        real_t_cos: torch.Tensor,
        real_ba_sin: torch.Tensor,
        real_ba_cos: torch.Tensor,
        real_bond_length: torch.Tensor,
        torsion_sin: torch.Tensor,
        torsion_cos: torch.Tensor,
        bond_angle_sin: torch.Tensor,
        bond_angle_cos: torch.Tensor,
        bond_length: torch.Tensor,
        dof_minimize: bool = False,
        ):
        """Harmonic Difference"""
        # Grab DoF values of interest using our values
        if not dof_minimize:
            torsion_sin = torch.gather(
                torsion_sin.unsqueeze(1).expand(-1, self.PARAMS["protein_len"], -1),
                -1,
                self.torsion_index.expand(torsion_sin.shape[0], -1, -1),
            )
            torsion_cos = torch.gather(
                torsion_cos.unsqueeze(1).expand(-1, self.PARAMS["protein_len"], -1),
                -1,
                self.torsion_index.expand(torsion_cos.shape[0], -1, -1),
            )
            bond_angle_sin = torch.gather(
                bond_angle_sin.unsqueeze(1).expand(-1, self.PARAMS["protein_len"], -1),
                -1,
                self.bond_angle_dofs.expand(bond_angle_sin.shape[0], -1, -1),
            )
            bond_angle_cos = torch.gather(
                bond_angle_cos.unsqueeze(1).expand(-1, self.PARAMS["protein_len"], -1),
                -1,
                self.bond_angle_dofs.expand(bond_angle_cos.shape[0], -1, -1),
            )
            bond_length = torch.gather(
                bond_length.unsqueeze(1).expand(-1, self.PARAMS["protein_len"], -1),
                -1,
                self.bond_length_dofs.expand(bond_length.shape[0], -1, -1),
            )
        else:
            torsion_sin, torsion_cos, bond_angle_sin, bond_angle_cos, bond_length = extract_dof_of_interest_minimize(
                self.freeze_residues.shape[0],
                torsion_sin,
                torsion_cos,
                bond_angle_sin,
                bond_angle_cos,
                bond_length,
                self.torsion_index,
                self.bond_angle_dofs,
                self.bond_length_dofs,
                self.match_res_mask,
            )

        # Compute the harmoinc difference
        harmonic_penalty_sin_torsion = torch.tensor(0.0, device=self.device)    if not self.freeze_torsion     else torch.pow(real_t_sin - torsion_sin,2).mean(dim=-1).to(device=self.device)
        harmonic_penalty_cos_torsion = torch.tensor(0.0, device=self.device)    if not self.freeze_torsion     else torch.pow(real_t_cos - torsion_cos,2).mean(dim=-1).to(device=self.device)
        harmonic_penalty_sin_bond_angle = torch.tensor(0.0, device=self.device) if not self.freeze_bond_angle  else (torch.pow(real_ba_sin - bond_angle_sin,2).sum(dim=-1)/ (real_ba_sin != 0).sum(dim=-1)).to(device=self.device)
        harmonic_penalty_cos_bond_angle = torch.tensor(0.0, device=self.device) if not self.freeze_bond_angle  else (torch.pow(real_ba_cos - bond_angle_cos,2).sum(dim=-1)/ (real_ba_cos != 0).sum(dim=-1)).to(device=self.device)
        harmonic_penalty_bond_length = torch.tensor(0.0, device=self.device)    if not self.freeze_bond_length else (torch.pow(real_bond_length - bond_length,2).sum(dim=-1)/ (real_bond_length != 0).sum(dim=-1)).to(device=self.device)
        harmonic_loss = harmonic_penalty_sin_torsion + harmonic_penalty_cos_torsion + harmonic_penalty_cos_bond_angle + harmonic_penalty_sin_bond_angle + harmonic_penalty_bond_length

        if not dof_minimize:
            # Compare and pick residues based on output
            mask = harmonic_loss <= self.inpaint_dist
            match_res = torch.nonzero(mask).to(torch.int)

            # Now grab non unique non redudent values
            unique_values, inv_idx = torch.unique_consecutive(match_res[:, 0], return_inverse=True)

            # Now remove redudent non-first occuring values
            mask_unique = torch.zeros_like(inv_idx, dtype=torch.bool)
            mask_unique[torch.unique(inv_idx, return_inverse=True)[1]] = True

            # Reindex
            match_res = match_res[mask_unique]
            harmonic_out = harmonic_loss[match_res[:, 0], match_res[:, 1]]
            return harmonic_out, match_res
        else:
            return harmonic_loss, None



    # @timeit
    def inpainting(self, B: int, random_no_reference: bool = False, interface_dict = None,
                   terminal = True, motif_dofs = None, l_homochiral: bool = False,
                   updated_freeze_residues: torch.Tensor = None):
        """given frozen residues search for scaffolds to hold motif residues

        PARAMS
        ------
            B: int
                Batch size. Number of places to search within the latent space
            random_no_reference: bool = False
                If you want to generate completely random samples you can do, so by setting
                this to true. Meaning you want need to pass in a reference to randomly sample
                the latent space
            interface_dict: dict
                All PDB atoms that we are concerned with incorporating into our loss terms
            terminal: bool = True
                Determines if you want the terminal loss terms to be present while searching
                the latent space
            motif_dofs: tuple (torsion, bondangle, bondlength)
                DoFs of the Motif provided that were extracted from the passed PDB.
            :l_homochiral: True a homochiral loss term is added/ if not then it always zero.

        RETURNS
        -------
            scaffolds: torch.Tensor
                DoF Structures that are close in proximity to the motif residues
            latent_points: torch.Tensor
                The latent point representation of the selected outputs
        """
        no_scaffolds = True
        run=0
        scaffolds = 0
        points = 0
        if updated_freeze_residues is not None:
            pyrosetta_freeze_residues = torch.as_tensor(
                    updated_freeze_residues,
                    dtype=torch.long,
                    )
            self.freeze_residues = pyrosetta_freeze_residues - 1

            assert torch.all(self.freeze_residues >= 0), (
                    f"Converted motif indices {self.freeze_residues.tolist()} must be non-negative"
                    )
            self.contiguous = self.check_contiguous(
                self.freeze_residues,
                self.PARAMS["protein_len"],
            )
            if self.contiguous:
                self.regions = [self.freeze_residues]
            else:
                self.regions_atoms, self.regions = split_motif_atoms(
                    self.freeze_residues
                )

        # Need the coordinates calculated here, for discontinuous motifs
        if not self.contiguous:
            self.atoms, self.t_native, self.R_native = kabsch_rigid_transform(
                interface_dict["peptide"][self.regions_atoms[0]].unsqueeze(0),
                interface_dict["peptide"][self.regions_atoms[1]].unsqueeze(0),
            )
        else:
            self.regions_atoms = None
            self.t_native = None
            self.R_native = None

        # input information
        if args.freeze_residues != None:
            real_t_sin = torch.sin(motif_dofs[0])
            real_t_cos = torch.cos(motif_dofs[0])
            real_ba_sin = torch.sin(motif_dofs[1])
            real_ba_cos = torch.cos(motif_dofs[1])
            real_bond_length = motif_dofs[2]

        while no_scaffolds:
#             scatter_latent = torch.zeros((B, self.latent_dim), dtype=torch.float64, device=self.device)
            # set up tensor of random init values
            latent_dim1 = torch.zeros((B, 1), dtype=torch.float64, device=self.device)
            latent_dim2 = torch.zeros((B, 1), dtype=torch.float64, device=self.device)
            latent_dim3 = torch.zeros((B, 1), dtype=torch.float64, device=self.device)
            latent_dim4 = torch.zeros((B, 1), dtype=torch.float64, device=self.device)
            latent_dim5 = torch.zeros((B, 1), dtype=torch.float64, device=self.device)
            latent_dim6 = torch.zeros((B, 1), dtype=torch.float64, device=self.device)

            if self.PARAMS["protein_len"] == 6:
                latent_dim1.uniform_(-0.5391, 0.5914)
                latent_dim2.uniform_(-0.6257, 0.6309)
                latent_dim3.uniform_(-0.7465, 0.7524)
                latent_dim4.uniform_(-0.6096, 0.6380)
                latent_dim5.uniform_(-0.6401, 0.6702)
                latent_dim6.uniform_(-0.6841, 0.7061)
            elif self.PARAMS["protein_len"] == 7:
                latent_dim1.uniform_(-0.5453, 0.5598)
                latent_dim2.uniform_(-0.6588, 0.6565)
                latent_dim3.uniform_(-0.6049, 0.6327)
                latent_dim4.uniform_(-0.6324, 0.6509)
                latent_dim5.uniform_(-0.6679, 0.6894)
                latent_dim6.uniform_(-0.4654, 0.4858)
            elif self.PARAMS["protein_len"] == 8:
                latent_dim1.uniform_(-0.3712, 0.3747)
                latent_dim2.uniform_(-0.3114, 0.3205)
                latent_dim3.uniform_(-0.3781, 0.3776)
                latent_dim4.uniform_(-0.2715, 0.2808)
                latent_dim5.uniform_(-0.2979, 0.3047)
                latent_dim6.uniform_(-0.3525, 0.3567)
            elif self.PARAMS["protein_len"] == 9:
#                 # This is for the new weights 6049
                latent_dim1.uniform_(-0.2128, 0.2093)
                latent_dim2.uniform_(-0.2182, 0.2157)
                latent_dim3.uniform_(-0.2831, 0.2510)
                latent_dim4.uniform_(-0.1752, 0.1900)
                latent_dim5.uniform_(-0.2239, 0.2302)
                latent_dim6.uniform_(-0.2539, 0.2441)

            scatter_latent = torch.cat(
                    (
                     latent_dim1, latent_dim2, latent_dim3,
                     latent_dim4, latent_dim5, latent_dim6,
                     ), dim=-1
                    )

            # run the points through the decoder
            out = self.model.decoder(scatter_latent)

            # Prediction Information
            torsion = out[:, :self.PARAMS["torsion_main_n"]*2]
            torsion_sin = torsion[:, :self.PARAMS["torsion_main_n"]]
            torsion_cos = torsion[:, self.PARAMS["torsion_main_n"]:]
            bond_angle = out[:, self.PARAMS["bond_angle_main_n"]*2:self.PARAMS["bond_angle_main_n"]*2*2]
            bond_angle_sin = bond_angle[:, :self.PARAMS["bond_angle_main_n"]]
            bond_angle_cos = bond_angle[:, self.PARAMS["bond_angle_main_n"]:]
            bond_length = out[:, self.PARAMS["bond_angle_main_n"]*2*2:-self.energy_n]
            preds = torch.cat((torsion, bond_angle, out[:, self.PARAMS["bond_angle_main_n"]*2*2:-self.energy_n]), dim=-1)

            # Grab ref2015 energy from output
            energy_all = out[:, -self.energy_n:]

            # term_bond_loss, bah_loss, ahd_loss, ocnh_loss, ncaco_loss, clash_penalty, coordinates = self.term_bond.terminal_peptide_loss(
            term_bond_loss, nco_loss, hnc_loss, ocnh_loss, _, HNCO, coordinates = self.term_bond.terminal_peptide_loss(
                dof_inputs = out,
                real_term = torch.ones(out.size(0), device=out.device, dtype=out.dtype) * self.ideal_term_length,
                params = self.PARAMS,
                foldtree_root = self.term_bond.calculate_root(self.PARAMS["protein_len"]),
                device = out.device,
            )

            # We trigger this early to remove structures
            if random_no_reference:
                if terminal:
                    # Determine which points are within the latent space cutoff
                    terminal_mask = term_bond_loss < 0.5 #2.0
                    ocnh_mask = ocnh_loss < 0.5
                    hnc_mask = hnc_loss < 0.5
                    nco_mask = nco_loss < 0.5
                    mask = torch.logical_and(
                        ocnh_mask.view(-1, 1), torch.logical_and(
                            hnc_mask.view(-1, 1), torch.logical_and(
                                nco_mask.view(-1, 1), terminal_mask.view(-1, 1)
                            ),
                        )
                    ).view(-1)
                else:
                    mask = torch.ones(term_bond_loss.shape, dtype=bool)

                if not torch.is_tensor(scaffolds):
                    scaffolds = preds[mask, :-3].detach()
                    points = scatter_latent[mask].detach()
                else:
                    # add to set of scaffolds
                    scaffolds = torch.cat(
                        (
                            scaffolds,
                            preds[mask, :-3],
                        ), dim=0,
                    ).detach()
                    points = torch.cat(
                        (
                            points,
                            scatter_latent[mask],
                        ), dim=0,
                    ).detach()

                # If scaffolds found Stop While Loop
                run+=1
                if run > 1000 or scaffolds.size(0) >= self.scaffold_number:
                    no_scaffolds = False

                # Save outputs
                if torch.is_tensor(scaffolds): print(f"Found {scaffolds.size(0)} To Proceed With")
                else: print(f"No Scaffolds Found with residue similarity as {self.inpaint_dist} in {run-1} iterations")
                if scaffolds.size(0) >= self.scaffold_number:
                    return scaffolds[:self.scaffold_number], points[:self.scaffold_number]
                continue


            coords_inter = torch.cat(
                (
                    coordinates[:, :, 2, :].unsqueeze(2), # N
                    coordinates[:, :, 0, :].unsqueeze(2), # CA
                    coordinates[:, :, 1, :].unsqueeze(2), # C
                    coordinates[:, :, 3, :].unsqueeze(2), # O

                ), dim = 2,
            )

            # Align points
            coord_B, coord_L, coord_A, coord_D = coords_inter.shape
            coords = coords_inter.reshape(coord_B, coord_L * coord_A, coord_D).squeeze(0)
            if len(coords.shape) == 2:
                coords = coords.unsqueeze(0)

            # If we have a motif that we want to inpaint with then we need to extract some information
            # Therefore, we traverse this logic
            loss, match_res_mask, self.motif, self.motif_input, no_matches = extract_loss_and_matchres(
                self.freeze_residues,
                self.contiguous,
                self.regions_atoms,
                self.t_native,
                self.R_native,
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
                self.torsion_index,
                self.bond_angle_dofs,
                self.bond_length_dofs,
                self.PARAMS,
                self.freeze_torsion,
                self.freeze_bond_angle,
                self.freeze_bond_length,
                self.inpaint_dist,
                self.match_res_mask,
                HNCO,
                coords,
                self.device,
                False,
                self.exclude_atom_positions,
                )
            if no_matches:
                # Rejected batches must count toward termination.  Otherwise an
                # unreachable motif threshold loops indefinitely.
                run += 1
                if run > 1000:
                    no_scaffolds = False
                continue

            #########
            # Interface calculate
            #########


            if interface_dict != None:
                peptide_nonhydrogen_mask = interface_dict["epsilon_peptide"] != 0.901681
                if not self.contiguous:
                    aligned_coordinates, rmsd_loss = kabsch_rootatom(P = coords, Q = interface_dict["peptide"][peptide_nonhydrogen_mask].double(),
#                                                           motif_input_ = self.regions_atoms[0],
                                                          motif_input = self.regions_atoms[0].unsqueeze(0),
                                                          motif_output = self.regions_atoms[0].unsqueeze(0),
                                                          )
                else:
                    # Non contiguous indexing of our outputs
                    matched_sample_idx = match_res_mask[:, 0].long()
                    coords = coords[matched_sample_idx]
                    aligned_coordinates, rmsd_loss = kabsch_rootatom(P = coords, Q = interface_dict["peptide"][peptide_nonhydrogen_mask].double(),
                                                          motif_output = self.motif,
                                                          motif_input = self.motif_input,
                                                          )

                interface_nonhydrogen_mask = interface_dict["epsilon_interface"] != 0.901681
                interface_nonorm_loss= interface_loss(interface_atoms = interface_dict["interface_atoms"][interface_nonhydrogen_mask].double(),
                                                      recomputed_xyz = aligned_coordinates,
                                                      sigma_peptide = interface_dict["sigma_peptide"][peptide_nonhydrogen_mask],
                                                      epsilon_peptide = interface_dict["epsilon_peptide"][peptide_nonhydrogen_mask],
                                                      sigma_interface = interface_dict["sigma_interface"][interface_nonhydrogen_mask],
                                                      epsilon_interface = interface_dict["epsilon_interface"][interface_nonhydrogen_mask])
                interface_total_loss = interface_nonorm_loss # / aligned_coordinates.size(0)
            else:
                # Every motif match is acceptable when the input has no target
                # chain. Keep this vector aligned to the matched-candidate rows,
                # not the full decoder batch.
                interface_total_loss = torch.zeros(
                    match_res_mask[:B].shape[0], dtype=torch.bool, device=self.device
                )

            if l_homochiral:
                # This is for homochiral
                torsion_angles = torch.cat((torch.rad2deg(torch.atan2(torsion_sin, torsion_cos)), HNCO), dim=-1)
                torsion_angles = torsion_angles[:, self.phi_psi_idx]

                l_homochiral_loss, homochiral_mask = homochiral_loss(
                    torsion_dofs = torsion_angles,
                    space = 1,
                    device = self.device,
                )
            else:
                l_homochiral_loss = torch.zeros(out.shape[0], dtype=torch.float32, device=self.device)
                homochiral_mask = torch.ones(out.size(0),dtype=torch.bool, device=out.device)

            # Determine which points are within the latent space cutoff
            int_mask = interface_total_loss <= self.interface_cutoff
            matched_sample_idx = match_res_mask[:, 0].long()
            terminal_mask = term_bond_loss[matched_sample_idx].view(-1) < 0.5 #2.0
            ocnh_mask = ocnh_loss[matched_sample_idx].view(-1) < 0.5
            hnc_mask = hnc_loss[matched_sample_idx].view(-1) < 0.5
            nco_mask = nco_loss[matched_sample_idx].view(-1) < 0.5
            l_homochiral_mask = homochiral_mask[matched_sample_idx].view(-1, 1) # <= 100#homochiral_mask[match_res_mask[:, 0]]
            # print("MASKS:", int_mask.sum(), terminal_mask.sum(), ocnh_mask.sum(), hnc_mask.sum(), nco_mask.sum(), harm_mask.sum())
            if self.contiguous:
                harm_mask = loss <= self.inpaint_dist
                mask = torch.logical_and(
                    ocnh_mask.view(-1, 1), torch.logical_and(
                        hnc_mask.view(-1, 1), torch.logical_and(
                            nco_mask.view(-1, 1), torch.logical_and(
                                harm_mask.view(-1,1), torch.logical_and(
                                    int_mask.view(-1,1), torch.logical_and(
                                        terminal_mask.view(-1, 1), l_homochiral_mask.view(-1, 1),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ).view(-1) #).view(-1)
            else:
                ca_motif_distance_mask = loss <= 5.0
                mask = torch.logical_and(
                    ocnh_mask.view(-1, 1), torch.logical_and(
                        hnc_mask.view(-1, 1), torch.logical_and(
                            nco_mask.view(-1, 1), torch.logical_and(
#                                     harm_mask.view(-1,1), torch.logical_and(
                                    int_mask.view(-1,1), torch.logical_and(
                                        terminal_mask.view(-1, 1), ca_motif_distance_mask.view(-1, 1)),
#                                     )
                            ),
                        ),
                    )).view(-1)


            # Grab out specific points that only pass
            if self.freeze_residues != None:
                matched_sample_idx = match_res_mask[:, 0].long()

                preds = preds[matched_sample_idx]
                scatter_latent = scatter_latent[matched_sample_idx]
                # concat onto self.match_res
                self.match_res_mask = torch.cat((self.match_res_mask, match_res_mask[mask]), dim=0)
            # add to set of scaffolds
            if not torch.is_tensor(scaffolds):
                scaffolds = preds[mask, :-3].detach()
                points = scatter_latent[mask].detach()
                harm_values = loss[mask].detach()
            else:
                scaffolds = torch.cat(
                    (
                        scaffolds,
                        preds[mask, :-3],
                    ), dim=0,
                ).detach()
                points = torch.cat(
                    (
                        points,
                        scatter_latent[mask],
                    ), dim=0,
                ).detach()
                harm_values = torch.cat(
                    (
                        harm_values,
                        loss[mask],
                    ), dim=0,
                ).detach()

            # Report run statistics
            if random_no_reference:
                print(f'BATCH: {run}, FOUND SCAFFOLD START POINTS: {mask.sum().item()}, SCAFFOLDS FOUND: {points.size(0)}')
            elif match_res_mask.numel() == 0:
                print("None Found. Match Res Mask Number: 0")
            else:
                print(f'BATCH: {run}, FOUND SCAFFOLD START POINTS: {mask.sum().item()}, BATCH LOW: {loss.min().item():.2f}, BATCH MEAN: {loss.mean().item():.2f}, CUTOFF: {self.inpaint_dist}, SCAFFOLDS FOUND: {points.size(0)}')

            # If scaffolds found Stop While Loop
            run+=1
            if run > 1000 or scaffolds.size(0) >= self.scaffold_number: # 100
                no_scaffolds = False

        # Save outputs
        if torch.is_tensor(scaffolds):
            print(f"Found {scaffolds.size(0)} To Proceed With")
            torsion = self.calc_angles(scaffolds[:, :self.PARAMS["torsion_main_n"]*2])
            bond_angle = self.calc_angles(scaffolds[:, self.PARAMS["torsion_main_n"]*2:self.PARAMS["bond_angle_main_n"]*2*2])
            preds = torch.cat((torsion, bond_angle, scaffolds[:, self.PARAMS["torsion_main_n"]*2*2:]), dim=-1)
        else:
            print(f"No Scaffolds Found with residue similarity as {self.inpaint_dist} in {run-1} iterations")
            empty = torch.empty((0,), dtype=torch.float64, device=self.device)
            self.match_res_mask = torch.empty((0,), dtype=torch.int, device=self.device)
            return empty, empty, empty
        if scaffolds.size(0) > self.scaffold_number:
            scaffolds = scaffolds[:self.scaffold_number]
            points = points[:self.scaffold_number]
            harm_values = harm_values[:self.scaffold_number]

            self.match_res_mask = self.match_res_mask[:self.scaffold_number, 1].long()
            assert points.size(0) == self.match_res_mask.numel(), (
                    f"Inpainting produced {points.size(0)} latent points "
                    f"but {self.match_res_mask.numel()} motif starts"
                    )
            return scaffolds, points, harm_values
        else:
            self.match_res_mask = self.match_res_mask[:self.scaffold_number, 1].long()
            return scaffolds, points, harm_values

    def metropolis_hastings(self, B: int, sigma: torch.float,
                            start_points: torch.Tensor|None = None,
                            mcmc_steps: int = 0,
                            motif_dofs: torch.Tensor|None = None,
                            harmonic_start: torch.Tensor|None = None,
                            ):
        """MCMC Search for energy minima structures

        PARAMS
        ------
            B: int,
                Batch Size. Amount of attempts to try in parallel
            sigma: torch.float
                Amount of noise for normal distribution
            start_points: torch.Tensor
                Start latent points from inpainting
            mcmc_steps: int,
                The number of steps to talk within our MCMC walk
            motif_dofs: torch.Tensor,
                The dofs we will use for motif matching if they are supplied and
                partial_optimization is True
            harmonic_start: torch.Tensor
                The harmonic loss values associated with our starting values

        RETURNS
        -------
            latent_trajectories: torch.tensor
                Shape (B, T, 3) Where B is the number of Trajectories,
                T is the time step, and 3 is the xyz coordinates for that
                step
            lowest_predicted: torch.Tensor
                The lowest energy DoF Vector sample trajectory (B_i) out of the search
        """
        if start_points == None:
            F = 1
            latent_trajectories = torch.zeros((B, F, 2, self.latent_dim), dtype=torch.double, device=self.device)
            H = torch.zeros((B*F, 2, 1), dtype=torch.double, device=self.device)
            # get starting point and enable grad
            self.start_latent_point = self.model.encoder_forward(self.start_point_scaled, noise=False).detach()

            # Set this as t_0 for all trajectories
            latent_trajectories[:, :,  0, :] = self.start_latent_point.unsqueeze(1).repeat(B, F, 1)
            latent_trajectories[:, :, -1, :] = self.start_latent_point.unsqueeze(1).repeat(B, F, 1)
            # reshape for model input
            latent_trajectories = latent_trajectories.reshape(B*F, 2, self.latent_dim)
        else:
            F = self.mcmc_paths
            latent_trajectories = torch.zeros((B, F, 2, self.latent_dim), dtype=torch.double, device=self.device)
            H = harmonic_start[..., None, None, None].repeat(1, F, 2, 1).view(B*F, 2, 1)
            latent_trajectories[:, :, 0, :] = start_points.unsqueeze(1).repeat(1, F, 1)
            latent_trajectories[:, :, -1, :] = start_points.unsqueeze(1).repeat(1, F, 1)
            self.start_latent_point = start_points
            # reshape for model input
            latent_trajectories = latent_trajectories.reshape(B*F, 2, self.latent_dim)

        # Decode our points to get the initial output, where we want the energy to compare against
        initial_out = self.model.decoder((self.start_latent_point).to(self.device))
        # Grab out the initial ref2015 scores
        energies_trajectories = initial_out[:, -2].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, F, 2, 1)
        # Reshape energy to (Batch * NUM_PATHS, 2, 1)
        energies_trajectories = energies_trajectories.reshape(B*F, 2, 1)

        # updated temperature
        temp_linspace = torch.linspace(1, self.temp, steps=self.rampdown) if self.temp != 1 else torch.tensor([1])

        # Harmonic Dampening on Structural Similarity Variables
        # Real Information
        real_t_sin = torch.sin(motif_dofs[0])
        real_t_cos = torch.cos(motif_dofs[0])
        real_ba_sin = torch.sin(motif_dofs[1])
        real_ba_cos = torch.cos(motif_dofs[1])
        real_bond_length = motif_dofs[2]

        # Setup our steps to be the same as gradient steps or custom
        steps = self.steps if mcmc_steps == 0 else mcmc_steps

        # Update self match res to be the same shape
        match_res_mask_mcmc = self.match_res_mask.unsqueeze(0).repeat(F, 1).view(-1)

        # Perform MCMC for n steps
        for t in range(1, steps):
            # Generate random walk noise
            epsilon = torch.normal(0, sigma, size=(B*F, self.latent_dim), dtype=torch.double)

            # Apply noise to previous t-1 point
            x_t = latent_trajectories[:, -1, :] + epsilon

            # decode points
            out = self.model.decoder(x_t.to(self.device))

            # Prediction Information
            torsion = out[:, :self.PARAMS["torsion_main_n"]*2]
            torsion_sin = torsion[:, :self.PARAMS["torsion_main_n"]]
            torsion_cos = torsion[:, self.PARAMS["torsion_main_n"]:]
            bond_angle = out[:, self.PARAMS["bond_angle_main_n"]*2:self.PARAMS["bond_angle_main_n"]*2*2]
            bond_angle_sin = bond_angle[:, :self.PARAMS["bond_angle_main_n"]]
            bond_angle_cos = bond_angle[:, self.PARAMS["bond_angle_main_n"]:]
            bond_length = out[:, self.PARAMS["bond_angle_main_n"]*2*2:-3]

            # Grab ref2015 energy from output
#             energy_all = self.energy_scale.inverse_scale(out[:, -self.energy_n:])
            energy_all = out[:, -self.energy_n:]
            if energy_all.size(-1) == 1:
                energy = torch.zeros(energy_all.size(0), 3, dtype=energy_all.dtype)
                energy[:, -2] = energy_all.squeeze(-1)
                energy_all = energy
            energy_t = energy_all[:, -2]

            # Determine our terminal bond loss
            term_bond_loss, nco_loss, hnc_loss, ocnh_loss, _, HNCO, coordinates = self.term_bond.terminal_peptide_loss(
                dof_inputs = out,
                real_term = torch.ones(out.size(0), device=out.device, dtype=out.dtype) * self.ideal_term_length,
                params = self.PARAMS,
                foldtree_root = self.term_bond.calculate_root(self.PARAMS["protein_len"]),
                device = out.device,
            )

            if self.freeze_residues != None:

                coords_inter = torch.cat(
                    (
                        coordinates[:, :, 2, :].unsqueeze(2), # N
                        coordinates[:, :, 0, :].unsqueeze(2), # CA
                        coordinates[:, :, 1, :].unsqueeze(2), # C
                        coordinates[:, :, 3, :].unsqueeze(2), # O

                    ), dim = 2,
                )

                # Align points
                coord_B, coord_L, coord_A, coord_D = coords_inter.shape
                coords = coords_inter.reshape(coord_B, coord_L * coord_A, coord_D).squeeze(0)
                if len(coords.shape) == 2:
                    coords = coords.unsqueeze(0)

                # If we have a motif that we want to inpaint with then we need to extract some information
                # Therefore, we traverse this logic
                harmonic_loss, _ = discontig_funcs.harmonic_diff_inpaint(
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
                        self.torsion_index,
                        self.bond_angle_dofs,
                        self.bond_length_dofs,
                        self.PARAMS,
                        self.freeze_residues,
                        self.freeze_torsion,
                        self.freeze_bond_angle,
                        self.freeze_bond_length,
                        match_res_mask_mcmc,
                        self.inpaint_dist,
                        self.device,
                        True,
                        self.exclude_atom_positions,
                        )
                harmonic_change = H[:, 1, :] - harmonic_loss.unsqueeze(-1).detach() #.cpu()


            # Accept a step or Not
            if t > self.initial_set:
                # We want to play around with the temperature of our MCMC approach,
                # if not then it is just always 1.
                if self.temp != 1:
                    temp_t = t - self.initial_set
                    temp_coeff = temp_linspace[temp_t-1] if temp_t <= self.rampdown else 0.01
                else:
                    temp_coeff = 1

                out_num = (
                    (energies_trajectories[:, 0, :].squeeze(-1) - energy_t ) + self.lambda_2*harmonic_change.squeeze(-1) + term_bond_loss.squeeze(-1).detach().cpu()
                    + nco_loss.squeeze(-1).detach().cpu() + hnc_loss.squeeze(-1).detach().cpu() + ocnh_loss.squeeze(-1).detach().cpu() #+ ncaco_loss.squeeze(-1).detach().cpu()
                )
                temp_coeff = temp_coeff if out_num.mean() > 1 else 1/temp_coeff
                alpha = torch.minimum(torch.ones(energy_t.shape), torch.exp( -( out_num / temp_coeff ) ) )

                # generate bool of T/F
                u = torch.ones((B*F))
                u.uniform_(0, 1)
                condition_u = u <= alpha

                # Place energies or reject
                energies_trajectories[condition_u, -1, :] = energy_t[condition_u].view(-1, 1)
                # Place steps or reject
                latent_trajectories[condition_u, -1, :] = x_t[condition_u]
                # Now replace values if needed
                mask = (energies_trajectories[:, -1, :] < energies_trajectories[:, 0, :]).squeeze(-1)
                energies_trajectories[mask, 0] = energies_trajectories[mask, -1]
                latent_trajectories[mask, 0, :] = latent_trajectories[mask, -1, :]
                H[mask, 0, :] = harmonic_loss[mask].unsqueeze(-1).cpu()

            # Accept all movements that are within the initial set range
            else:
                energies_trajectories[:, -1, :] = energy_t.view(-1, 1)
                latent_trajectories[:, -1, :] = x_t
                H[:, 1, :] = harmonic_loss.unsqueeze(-1) #.cpu()

                if t == steps-1:
                    latent_trajectories[:, 0, :] = latent_trajectories[:, -1, :]

            if t % 5 == 0:
                print(f"Current Time Step: {t} Current Mean Energy: {energies_trajectories[:, 0, :].mean().item()}, Current Mean Harmonic Constraint: {H[:, 0, :].mean().item()}, Start Mean Harmonic Constraint: {harmonic_start.mean().item()}")

        # grab minimum of each B_i from F
        latent_trajectories = latent_trajectories.reshape(B, F, 2, self.latent_dim)
        energies_trajectories = energies_trajectories.reshape(B, F, 2, 1)

        if self.freeze_residues != None or self.retain_structure:
            # Reshape our H, so we can start grabbing best runs
            H = H.reshape(B, F, 2, 1)

            # First we grab the lowest of our the 2 tracks that are recorded
            latent_trajectories =  latent_trajectories[:, :, 0, :]
            H = H[:, :, 0, :]

            # Now grab the lowest MCMC PATH (F, dim-1) that started from the same B_i
            h_path_min = H.argmin(dim=1, keepdim=True)
            H = torch.gather(H, 1, h_path_min).squeeze(1)
            latent_trajectories = torch.gather(latent_trajectories, 1, h_path_min.expand(-1, -1, latent_trajectories.size(-1))).squeeze(1)
        else:
#             energy_min = energies_trajectories.min(dim=1).indices if F != 1 else 0
            energy_min = energies_trajectories.argmin(dim=1, keepdim=True) #.indices if F != 1 else 0
            energy_min_expanded = energy_min.expand(-1, -1, -1, latent_trajectories.size(-1))
            latent_trajectories = torch.gather(latent_trajectories, 1, energy_min_expanded).squeeze(1)


        if start_points == None:
            print(f"Final Lowest Energy Minima: {energies_trajectories.min().item()}, Lowest Harmonic Constraint: {H[:, 0, :].min().item()}, Start Mean Harmonic Constraint: {H[:, 0, :].mean().item()}")
        else:
            print(f"Final Lowest Energy Minima: {energies_trajectories.min().item()}, Average Distance: {torch.norm(latent_trajectories - self.start_latent_point, dim=-1).mean().item()}, Lowest Harmonic Constraint: {H.min().item()}, Start Mean Harmonic Constraint: {harmonic_start.mean().item()}, Mean Finish Harmonic Constraint: {H.mean().item()}")

        # Get lowest energy structures
        out = self.model.decoder(latent_trajectories.to(self.device))

        # generate self
        self.start_latent_point = latent_trajectories.to(self.device)

        return latent_trajectories, energies_trajectories

    # @timeit
    def energy_steps(self, start_points: torch.Tensor = None, mcmc_minimize: bool = False, minimize_steps: int = 1000,
                     interface_full_dict = None, interface_cb_dict = None,
                     motif_dofs = None, l_homochiral: bool = False):
        """Model Eval Loop for GDS"""
        if (self.freeze_residues is not None
            and not self.extract_minimize_input
            ):
            assert start_points.size(0) == self.match_res_mask.numel(), (
                    f"energy_steps received {start_points.size(0)} "
                    "but latent points has"
                    f"{self.match_res_mask.numel()} motif starts"
                    )
        if start_points.size(0) == 1:
            self.start_latent_point = start_points.to(self.device)
            start_range = 0
            self.path = torch.zeros((start_points.size(0),self.steps), dtype=torch.float32, device=self.device)
            self.energy_path = torch.zeros(start_points.size(0), self.steps,  device=self.device)
            self.dof_traj = torch.zeros((start_points.size(0), self.steps, self.dof_size - self.energy_n), dtype=torch.float64, device=self.device)
        elif mcmc_minimize:
            self.steps = minimize_steps
            start_range = 0
            self.path = torch.zeros((start_points.size(0),self.steps), device=self.device)
            self.energy_path = torch.zeros(start_points.size(0), self.steps,  device=self.device)
            self.dof_traj = torch.zeros((start_points.size(0), self.steps, self.dof_size - self.energy_n), dtype=torch.float64, device=self.device)
        else:
            self.start_latent_point = start_points.to(self.device)
            self.path = torch.zeros((start_points.size(0), self.steps), dtype=torch.float32, device=self.device)
            self.energy_path = torch.zeros((start_points.size(0), self.steps), device=self.device)
            # generate holding tensor and input the beginning traj
            self.dof_traj = torch.zeros((start_points.size(0), self.steps, self.dof_size - self.energy_n), dtype=torch.float64, device=self.device)
            start_range = 0
        original_point = self.start_latent_point.clone()
        original_shape = self.start_latent_point.shape[0]
        total_mask = torch.ones(self.start_latent_point.shape[0], dtype=torch.bool, device=self.start_latent_point.device)
        self.start_latent_point.requires_grad_()
        # Set our while minimize term
        minimize_still = True
        redo = 0
        # Calculate motif numbering
        if self.freeze_residues != None:
            self.motif, self.motif_input = calculate_motif_numbering_for_kabsch(
                torch.zeros((1,1), dtype=torch.long) if self.extract_minimize_input else self.match_res_mask,
                self.PARAMS["protein_len"],
                self.freeze_residues.shape[0],
                self.freeze_residues,
                True,
            )
            assert self.motif.size(0) == start_points.size(0), (
                    f"Built {self.motif.size(0)} motif index rows for "
                    f"{start_points.size(0)} starting points."
                    )
#         if self.extract_minimize_input:
#             self.motif, self.motif_input = calculate_motif_numbering_for_kabsch(
#                 torch.zeros((1,1)),
#                 self.PARAMS["protein_len"],
#                 self.freeze_residues.shape[0],
#                 self.freeze_residues,
#                 True,
#             )

        while minimize_still:
            original_point = self.start_latent_point.clone()
            # Define optimizer
            if self.adam_optim:
                optimizer = torch.optim.AdamW([self.start_latent_point], lr=self.lr)
            elif self.sgd_optim:
                optimizer = torch.optim.SGD([self.start_latent_point], lr=self.lr, momentum=0.9)

            for step in range(start_range, self.steps):
                # zero grad
                optimizer.zero_grad()

                # get the structure from the latent point
                out = self.model.decoder(self.start_latent_point)

                # Prediction Information
                torsion = out[:, :self.PARAMS["torsion_main_n"]*2]
                torsion_sin = torsion[:, :self.PARAMS["torsion_main_n"]]
                torsion_cos = torsion[:, self.PARAMS["torsion_main_n"]:]
                bond_angle = out[:, self.PARAMS["bond_angle_main_n"]*2:self.PARAMS["bond_angle_main_n"]*2*2]
                bond_angle_sin = bond_angle[:, :self.PARAMS["bond_angle_main_n"]]
                bond_angle_cos = bond_angle[:, self.PARAMS["bond_angle_main_n"]:]
                bond_length = out[:, self.PARAMS["bond_angle_main_n"]*2*2:-3]
                preds = torch.cat((torsion, bond_angle, out[:, self.PARAMS["bond_angle_main_n"]*2*2:-3]), dim=-1)

                if self.freeze_residues != None:
                    # Real Information
                    real_t_sin = torch.sin(motif_dofs[0])
                    real_t_cos = torch.cos(motif_dofs[0])
                    real_ba_sin = torch.sin(motif_dofs[1])
                    real_ba_cos = torch.cos(motif_dofs[1])
                    real_bond_length = motif_dofs[2]

                # grab the energies
                energies = out[:, -self.energy_n:]
                # energies = self.energy_scale.inverse_scale(out[:, -self.energy_n:]).to(self.device)
                # Get rid of the last fa_sol (this is because of the way I am printing each step down below)
                energies = energies[:, :2]
                # Grab specifically ref2015
                energy = energies[:, -1]


                dist = torch.norm(original_point - self.start_latent_point, dim=-1)

                # Compute coordinates first for HNCO calculation
                term_bond_loss, bah_loss, ahd_loss, ocnh_loss, ncaco_loss, HNCO, coordinates = self.term_bond.terminal_peptide_loss(
                    dof_inputs = out,
                    real_term = torch.ones(out.size(0), device=out.device, dtype=out.dtype) * self.ideal_term_length,
                    params = self.PARAMS,
                    foldtree_root = self.term_bond.calculate_root(self.PARAMS["protein_len"]),
                    device = out.device,
                )

                if self.freeze_residues != None:
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
                    # Now compute which residues match our motif
                    harmonic_loss, _ = discontig_funcs.harmonic_diff_inpaint(
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
                            self.torsion_index,
                            self.bond_angle_dofs,
                            self.bond_length_dofs,
                            self.PARAMS,
                            self.freeze_residues,
                            self.freeze_torsion,
                            self.freeze_bond_angle,
                            self.freeze_bond_length,
                            self.match_res_mask,
                            self.inpaint_dist,
                            self.device,
                            True,
                            self.exclude_atom_positions,
                            self.extract_minimize_input,
                            )
                else:
                    harmonic_loss = torch.tensor([0.0], device=self.device)

                term_bond_coeff = 1


                coords_inter = torch.cat(
                    (
                        coordinates[:, :, 2, :].unsqueeze(2), # N
                        coordinates[:, :, 0, :].unsqueeze(2), # CA
                        coordinates[:, :, 1, :].unsqueeze(2), # C
                        coordinates[:, :, 3, :].unsqueeze(2), # O

                    ), dim = 2,
                )

                # Align points
                coord_B, coord_L, coord_A, coord_D = coords_inter.shape
                coords = coords_inter.reshape(coord_B, coord_L * coord_A, coord_D).squeeze(0)
                if len(coords.shape) == 2:
                    coords = coords.unsqueeze(0)

                #########
                # Interface calculate
                #########

                if self.extract_minimize_input:
                    rmsd_weight = 10.0
                    self.energy_weight = 0.0
                    self.lambda_2 = 0.0
                    term_bond_coeff = 0.0
                    peptide_nonhydrogen_mask = interface_cb_dict["epsilon_peptide"] != 0.901681
                    aligned_coordinates, rmsd_loss = kabsch_rootatom(P = coords, Q = interface_cb_dict["peptide"][peptide_nonhydrogen_mask].double(),
                                                            motif_output = self.motif,
                                                            motif_input = self.motif_input,
                                                            )
#                     aligned_coordinates, rmsd_loss = kabsch_rootatom(
#                             P = coords,
#                             Q = interface_cb_dict["peptide"][peptide_nonhydrogen_mask].double(),
#                             motif_input = torch.arange( interface_cb_dict["peptide"][peptide_nonhydrogen_mask].size(0) ),
#                             motif_output = torch.arange( coords.size(1) ),
#                             )
                else:
                    rmsd_weight = 0.0

                if interface_full_dict != None:
                    # We must specify which interface context we are using whether it be ramp or one consistent view
                    if (step < self.steps//4) and (self.ramp_constraint):
                        interface_dict = interface_cb_dict
                    elif (step > self.steps//4) and (self.ramp_constraint):
                        interface_dict = interface_full_dict
                    elif self.fullpocket:
                        interface_dict = interface_full_dict
                    else:
                        interface_dict = interface_cb_dict

                    peptide_nonhydrogen_mask = interface_dict["epsilon_peptide"] != 0.901681
                    if not self.contiguous:
                        aligned_coordinates, rmsd_loss = kabsch_rootatom(P = coords, Q = interface_dict["peptide"][peptide_nonhydrogen_mask].double(),
                                                                motif = self.regions_atoms[0],
                                                                )
                        discontig_loss = discontig_motif_loss(
                            region1  = aligned_coordinates[:, self.regions_atoms[0], :].view(aligned_coordinates.shape[0], -1, 3),
                            region2  = aligned_coordinates[:, self.regions_atoms[1], :].view(aligned_coordinates.shape[0], -1, 3),
                            t_native = self.t_native,
                            R_native = self.R_native,
                            R_lambda = 1,
                        )
                    else:
                        discontig_loss = torch.zeros(1, device=self.device)
                        aligned_coordinates, rmsd_loss = kabsch_rootatom(P = coords, Q = interface_dict["peptide"][peptide_nonhydrogen_mask].double(),
                                                                motif_output = self.motif,
                                                                motif_input = self.motif_input,
                                                                )


                    interface_nonhydrogen_mask = interface_dict["epsilon_interface"] != 0.901681
                    interface_nonorm_loss = interface_loss(interface_atoms = interface_dict["interface_atoms"][interface_nonhydrogen_mask].double(),
                                                           recomputed_xyz = aligned_coordinates,
                                                           sigma_peptide = interface_dict["sigma_peptide"][peptide_nonhydrogen_mask],
                                                           epsilon_peptide = interface_dict["epsilon_peptide"][peptide_nonhydrogen_mask],
                                                           sigma_interface = interface_dict["sigma_interface"][interface_nonhydrogen_mask],
                                                           epsilon_interface = interface_dict["epsilon_interface"][interface_nonhydrogen_mask])

                    interface_total_loss = interface_nonorm_loss #/ aligned_coordinates.size(0)

                else:
                    interface_total_loss = torch.zeros(1, device=energy.device)
                    discontig_loss = torch.zeros(1, device=self.device)
                    rmsd_loss = torch.zeros(1, device=self.device)

                if l_homochiral:
                    # THis is for homochiral
                    torsion_angles = torch.cat((torch.rad2deg(torch.atan2(torsion_sin, torsion_cos)), HNCO), dim=-1)
                    torsion_angles = torsion_angles[:, self.phi_psi_idx]

                    l_homochiral_loss, homochiral_mask = homochiral_loss(
                        torsion_dofs = torsion_angles,
                        space = 1,
                        device = self.device,
                    )
                else:
                    l_homochiral_loss = torch.zeros(out.shape[0], dtype=torch.float32, device=self.device)

                # Clone terms
                energy_pred = energy.clone()
                ocnh_loss_term = ocnh_loss.clone()
                ahd_loss_term = ahd_loss.clone()
                bah_loss_term = bah_loss.clone()
                ncaco_loss_term = ncaco_loss.clone()
                term_bond_loss_term = term_bond_loss.clone()


                combo_angles = ahd_loss_term + bah_loss_term + ncaco_loss_term + ocnh_loss_term
#                 interface_total_loss = torch.zeros_like(interface_total_loss)

                if redo == 0:
                    loss = (
                        energy_pred*self.energy_weight + \
                        self.lambda_2*harmonic_loss + \
                        term_bond_coeff*discontig_loss + \
                        term_bond_coeff*term_bond_loss_term + \
                        term_bond_coeff*combo_angles + \
                        interface_total_loss + \
                        self.lambda_2*l_homochiral_loss + \
                        rmsd_loss*rmsd_weight
                    ).mean()
                else:
                    loss = (term_bond_coeff*term_bond_loss_term + interface_total_loss*(redo+1)).mean()


                if mcmc_minimize:
                    self.path[:, step] = dist.detach()
                    self.energy_path[:, step] = energy.detach()
                    print(f"Step: {step}, QM: {energies[:, 0].mean().item()}, Ref2015: {energies[:, -1].mean().item()}, Dist: {dist.mean().item()}, Harmonic Penalty: {harmonic_loss.mean().item()}, Terminal Bond: {term_bond_loss.mean().item()}, AHD: {ahd_loss.mean().item()}, BAH: {bah_loss.mean().item()}, Interface Loss: {interface_total_loss.mean().item()}, Lowest Harmonic Penalty: {harmonic_loss.min().item()}, Loss: {loss.item()}")
                elif start_points == None:
                    self.path[step] = dist.detach()
                    self.energy_path[step] = energy.detach()
                    print(f"Step: {step}, QM: {energies[:, 0].item()}, Ref2015: {energy.item()}, Dist: {dist.item()}, Harmonic Penalty: {harmonic_loss.item()}, Terminal Bond: {term_bond_loss.item()}, AHD: {ahd_loss.item()}, BAH: {bah_loss.item()}, LJ penalty: {clash_penalty.item()}, Interface Loss: {interface_total_loss.mean().item()}, Lowest Harmonic Penalty: {harmonic_loss.item()}, Loss: {loss.item()}")
                else:
                    self.path[total_mask, step] = dist.detach().to(torch.float32)
                    self.energy_path[total_mask, step] = energy.detach().to(torch.float32)
                    print(f"Step: {step}, QM: {energies[:, 0].mean().item()}, Ref2015: {energies[:, -1].mean().item()}, L-homochiral: {l_homochiral_loss.mean().item()}, Harmonic Penalty: {harmonic_loss.mean().item()}, Discontig Loss: {discontig_loss.mean().item()} Terminal Bond: {term_bond_loss.mean().item()}, OCNH Plane: {ocnh_loss_term.mean().item()} AHD: {ahd_loss.mean().item()}, BAH: {bah_loss.mean().item()}, Interface Loss: {interface_total_loss.mean().item()}, Lowest Harmonic Penalty: {harmonic_loss.min().item()}, Loss: {loss.item()}")

                # back prop
                loss.backward()
                optimizer.step()

                all_selected_target_atoms = interface_dict["target"].double()

                pairwise_distances = torch.cdist(
                        aligned_coordinates,
                        all_selected_target_atoms.unsqueeze(0).expand(aligned_coordinates.size(0), -1, -1)
                    )
                min_target_distance = pairwise_distances.amin(dim=(1, 2))

                if (step == self.steps-1):
                    minimize_still = False
                    clash_mask = interface_total_loss > self.interface_cutoff
                    distance_pass = min_target_distance >= 1.0
                    inverse_mask = ~clash_mask

                    print(
                            "Minimum target-peptide distances:",
                            min_target_distance.detach().cpu().tolist(),
                            )
                    print(
                            f"Number of successful outputs: {inverse_mask.sum().item()} / "
                            f"{inverse_mask.numel()}"
                            )

                    out = out.detach().clone()
                    kept_out = out[inverse_mask]
                    torsion = self.calc_angles(kept_out[:, :self.PARAMS["torsion_main_n"]*2])
                    bond_angle = self.calc_angles(kept_out[:, self.PARAMS["torsion_main_n"]*2:self.PARAMS["bond_angle_main_n"]*2*2])
                    preds = torch.cat((torsion, bond_angle, kept_out[:, self.PARAMS["torsion_main_n"]*2*2:-self.energy_n]), dim=-1)
                else:
                    out = out.detach().clone()
                    torsion = self.calc_angles(out[:, :self.PARAMS["torsion_main_n"]*2])
                    bond_angle = self.calc_angles(out[:, self.PARAMS["torsion_main_n"]*2:self.PARAMS["bond_angle_main_n"]*2*2])
                    preds = torch.cat((torsion, bond_angle, out[:, self.PARAMS["torsion_main_n"]*2*2:-self.energy_n]), dim=-1)

                if mcmc_minimize:
                    self.dof_traj[:, step] = preds
                elif start_points == None:
                    self.dof_traj[step] = preds
                else:
                    if (interface_total_loss.size(0) != 1) and (step == self.steps-1):
                        self.dof_traj[inverse_mask, step] = preds
                        self.dof_traj = self.dof_traj[inverse_mask]
                        self.match_res_mask = self.match_res_mask[inverse_mask]
                    else:
                        self.dof_traj[:, step] = preds


    # @timeit
    def output_GDS(
        self,
        dofsummary: DoFSummary,
        skip_individual = False,
        silentfile: bool = False,
        score: bool = False,
        output_path: str = "./",
        random_outputs: bool = False,
        minimize_pose: bool = False,
        complex_output: bool = False,
        ) -> int:
        """Output the GDS trajectory"""
        if self.dof_traj.ndim == 3:
            final_dofs = self.dof_traj[:, -1, :]
        elif self.dof_traj.ndim == 2:
            final_dofs = self.dof_traj
        else:
            raise ValueError(
                    f"Unexpected dof_traj shape: {tuple(self.dof_traj.shape)}"
                    )

        if final_dofs.size(0) == 0:
            print("No non-clashing structures survived; no structures will be written.")
            return 0

        thread_data = {'x': final_dofs,
                        'params': self.PARAMS,}
        dofsummary.thread_dofs_to_poses(
            thread_data,
            silentfile = silentfile,
            score_outputs = score,
            output_path = output_path,
            random_outputs = random_outputs,
            minimize_pose = minimize_pose,
            residue_start = self.match_res_mask,
            complex_out = complex_output,
        )

        return 0


if __name__ == "__main__":
    start_time = time.perf_counter()
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--steps", type=int, default=10, help="Gradient Descent Number Of Steps. Energy Minimization always happens, unless you specify this as 1.")
    p.add_argument("--prefix-outfile", type=str, default="test", help="PDB/SilentFile prefix")
    p.add_argument("--silentfile", action="store_true", help="Save Model outputs in a SilentFile format \
       if not specified then individual PDBs are produced.")
    p.add_argument("--score-outputs", action="store_true", help="Score each Model output and save in \
        SilentFile (not implemented yet to save score to separate file when Silent not set).")
    p.add_argument("--apply-minmover-outputs", action="store_true", help="Apply the MinMover with ref2015 score to \
         our threaded poses.")
    p.add_argument("--desired-motif-start-position", default=4, type=int, help="Specifies where the Moitf will start \
          on CyclicCAE outputs. We suggest the middle of the peptide, so for an 8mer output, resi 4 for a singular AA.")
    p.add_argument("--output-path", type=str, default="./", help="Determines location PDBs/SilentFile are saved.")
    p.add_argument("--extract-minimize-input", action="store_true", help="Minimize a given input only")
    p.add_argument("--weights-load", type=str, default='.', help="Specify Model Weights To Use. Only needs to be specified for custom weights or custom default weight locations.")
    p.add_argument("--out-size", type=int, help="Specify the output size")
    p.add_argument("--lr", type=float, default=0.001, help="Optimizer Learning Rate")
    p.add_argument("--cutoff-true", action="store_true", help="Apply a distance cutoff")
    p.add_argument("--cutoff", type=float, default=0.005, help="Specify Euclidean Distance Cutoff in Latent Space to Search")
    p.add_argument("--latent-dim", type=int, default=6, help="Latent Size of the model")
    p.add_argument("--partial-optimization", action="store_true", help="Apply partial optimization to a structure")
    p.add_argument("--freeze-residues", nargs="+", type=int, help="Used wth partial-optimization. If no sub freeze flags specified all are frozen \
           Specify residues in a whitespace manner to freeze (e.g. 1 2 3)")
    p.add_argument("--l-homochiral", action="store_true", help="Generate only L-homochiral outputs")
    p.add_argument("--freeze-torsion", action="store_true", help="Use with partial-optimization")
    p.add_argument("--freeze-bond-angle", action="store_true", help="Use with partial-optimization")
    p.add_argument("--freeze-bond-length", action="store_true", help="Use with partial-optimization")
    p.add_argument("--lambda-partial", type=float, default=1.0, help="Reweight Partial Freezing")
    p.add_argument("--energy-search", action="store_true", help="Gradient Search on dE/d(latent space)")
    p.add_argument("--mcmc", action="store_true", help="Markov Chain Monte Carlo Metropolis Hastings Search, you do not need to set this with --inpaint")
    p.add_argument("--batch-size", type=int, default=10000, help="Used with mcmc. Set number of trajectories")
    p.add_argument("--sigma", type=float, default=0.01, help="Used with mcmc. Set sigma of normal movement to add to latent point")
    p.add_argument("--temperature", type=float, default=1, help="Used with mcmc. Set Temperature")
    p.add_argument("--initial-steps", type=int, default=0, help="Used with mcmc. Set initial steps to accept no matter what")
    p.add_argument("--ramp-down", type=int, default=0, help="Used with mcmc. Steps to go from temperature to 1")
    p.add_argument("--retain-structure", action="store_true", help="Used with mcmc. Walks path closer to input structure")
    p.add_argument("--inpaint", action="store_true", help="Provide Frozen Residues and generate scaffolds")
    p.add_argument("--inpaint-dist", type=float, default=0.1, help="Cutoff distance (p1norm) to consider a scaffold a success")
    p.add_argument("--scaffold-num", type=int, default=100, help="Number of Inpainted Scaffolds to generate")
    p.add_argument("--sgd-optim", action="store_true", help="Use SGD optimizer with momentum 0.9. AdamW is used by default")
    p.add_argument("--gpu", action="store_true", help="Use a GPU for inference")
    p.add_argument("--termbond-test", action="store_true", help="Run Terminal Bond Constraint Generation")
    p.add_argument("--minimize-steps", type=int, default=1000, help="Number of energy minimization steps to take when using --minimize-mcmc")
    p.add_argument("--mcmc-steps", type=int, default=0, help="Number of MCMC steps (t), this needs to be passed and > 0 to work if you want MCMC search after finding scaffolds with --inpaint.")
    p.add_argument("--mcmc-paths", type=int, default=10, help="Number of MCMC Paths that use B_i as a starting point. This generates (B) -> (B, mcmc_paths) number paths. Only the lowest energy path for a B_i is selected at the end of MCMC.")
    p.add_argument("--pdb-input", type=str, default=None, help="PDB input that has our target proteins and our motif reference.")
    p.add_argument("--random-sample", action="store_true", help="Sample latent space randomly, without a reference.")
    p.add_argument("--skip-individual", action="store_true", help="For skipping individual construct trajectory outputting during inpainting")
    p.add_argument("--interface-cutoff", type=float, default=20.0, help="Metric for accepting a not clashing peptide [10.0 for compact pockets, 20.0 for others] ")
    p.add_argument("--treat-input-binder-as-cyclic", action="store_true", help="Apply Rosetta cyclic constraints to our Motif Chain (--peptide-chain) \
            Only specify this if the Motif Chain is a N-C cyclic peptide.")
    p.add_argument("--peptide-chain", type=str, default="B", help="Chain String that contains our motif of interest. The model \
            expects input PDBS to have two chains currently (target & binder/peptide.)")
    p.add_argument("--interface-distance-cutoff", type=float, default=15.0, help="Atoms within this distance cutoff of the target will be considered \
            when gradient-marching peptide")
    p.add_argument("--fullpocket-minimize", action="store_true", help="Specify if pocket context takes into account all non-hydrgoen bb and side-chain atoms \
                        or only bb-heavy + Cb atoms (which is default if you don't specify this flag)")
    p.add_argument("--ramp-interface-constraint", action="store_true", help="This sets the first 0.50 of the energy minimize steps to bb-heavy + Cb pocket context. \
                        Then for the last 0.50 switches to full pocket context (all heavy atoms). If not set then uses passed pocket specification throughout")
    p.add_argument("--complex-output", action="store_true", help="Output scaffolds in complex with target structure")
    p.add_argument("--exclude-atom-from-dof", nargs="+", default=[], type=str, help="Exclude an atom's DoFs from consideration by passing in the residue index number and atom number with (N,phi(0), Ca(1), C,psi(2), O(3), NH(4), Omega: 6) \
            as the numbering. (e.g. --exclude-atom-from-dof 1:0 1:3) (STILL IN DEVELOPMENT)")
    p.add_argument("--energy-weight", default=1.0, type=float, help="Energy weight term to be used during gradient descent energy minimization.")
    p.add_argument("--latent-space-exploration", action="store_true", help="This is only used to extract latent space information, so we can plot and extract metadata")
    p.add_argument("--latent-space-training-dataset", type=str, help="If using --latent-space-exploration, then pass this as well. This should be an already extracted .pt dataset used in training")
    args = p.parse_args()

    use_device = "cuda:0" if args.gpu else "cpu"

    # init the start path with X tensor 14
    energySearch = EnergyPath(
        steps = args.steps,
        weight_passed = args.weights_load,
        lr=args.lr,
        cutoff = args.cutoff,
        cutoff_true = args.cutoff_true,
        partial_optimization = args.partial_optimization,
        freeze_residues = torch.tensor(args.freeze_residues) - 1 if args.freeze_residues else None,
        freeze_torsion = args.freeze_torsion,
        freeze_bond_angle = args.freeze_bond_angle,
        freeze_bond_length = args.freeze_bond_length,
        lambda_partial = args.lambda_partial,
        initial_set = args.initial_steps,
        temp = args.temperature,
        rampdown = args.ramp_down,
        latent_dim = args.latent_dim,
        retain_structure = args.retain_structure,
        inpaint_dist = args.inpaint_dist,
        device = use_device,
        scaffold_number = args.scaffold_num,
        adam_optim = False if args.sgd_optim else True,
        sgd_optim = args.sgd_optim,
        mcmc_paths = args.mcmc_paths,
        out_size = args.out_size,
        interface_cutoff = args.interface_cutoff,
        ramp_constraint = args.ramp_interface_constraint,
        full_pocket = args.fullpocket_minimize,
        random_sample = args.random_sample,
        exclude_atom_positions = args.exclude_atom_from_dof,
        energy_weight = args.energy_weight,
        encode_decode = args.extract_minimize_input
    )

    # This is only for running a dataset through a model and extracting out information from it.
    if args.latent_space_exploration:
        data = torch.load(args.latent_space_training_dataset)
        data_dof = data["x"].double()
        points_left = True
        loop = 0

        while points_left:
            print("Current dof_size:", data_dof.size(0))
            section = data_dof[:10000] if data_dof.size(0) > 10000 else data_dof
            out_dof, latent_points = energySearch.extract_latent_information( section )
            if loop == 0:
                storage = latent_points
                loop+=1
            else:
                # Should be in shape (B, 6)
                storage = torch.cat( ( storage, latent_points ), dim=0 )
            if data_dof.size(0) < 10000: points_left = False
            else: data_dof = data_dof[10000:]

        assert storage.size(0) == data["x"].size(0), "Not all of the points are accounted for!"
        max_min_values = torch.cat(
                (
                    storage.max(dim=0).values.unsqueeze(0),
                    storage.min(dim=0).values.unsqueeze(0),
                    ), dim=0,
                )
        print("Max-Min (Max row 0, Min row 1):", max_min_values)
        outfilename = args.latent_space_training_dataset.split(".pt")[0]
        torch.save({"latent-space-points": storage, "max-min": max_min_values}, f"{outfilename}-latentspace-info.pt")
        sys.exit()

    # Init dofsummary which is used in both the beginning and end of the script
    dofsummary = DoFSummary(
        prefix = args.prefix_outfile,
        extract_minimize_input = args.extract_minimize_input,
        is_cyclic = args.treat_input_binder_as_cyclic,
        pose_motif_location = args.desired_motif_start_position,
    )

    # generate our output path if not yet created
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)


    ## Inpainting Task
    if args.inpaint:
        # Checks before proceeding
        assert args.pdb_input != None, "You did not pass a PDB File for inpainting. \
               Please Provide a path to one with `--pdb-input`"

        #### Here we are using our custom functions to extract Target and PDB
        #### information for inpainting and gradient-marching
        # All atoms are Extracted within cutoff distance
        full, target, peptide, sigma_target, epsilon_target, sigma_peptide, epsilon_peptide = extract_pdb_xyz(args.pdb_input, full=True, peptide_chain=args.peptide_chain)
        if target.numel() == 0:
            interface_full_dict = None
        else:
            interface, mask = grab_interface(target, peptide, dist_cutoff = args.interface_distance_cutoff)
            interface_full_dict = {
                "target": target.to(use_device),
                "peptide": peptide.to(use_device),
                "interface_atoms": interface.to(use_device),
                "sigma_peptide": sigma_peptide.to(use_device),
                "epsilon_peptide": epsilon_peptide.to(use_device),
                "sigma_interface": sigma_target[mask].to(use_device),
                "epsilon_interface": epsilon_target[mask].to(use_device)
            }
        # Cb Pocket Context Extract
        full, target, peptide, sigma_target, epsilon_target, sigma_peptide, epsilon_peptide = extract_pdb_xyz(args.pdb_input, full=False, peptide_chain=args.peptide_chain)
        if target.numel() == 0:
            interface_cb_dict = None
        else:
            interface, mask = grab_interface(target, peptide, dist_cutoff = args.interface_distance_cutoff)
            interface_cb_dict = {
                "target": target.to(use_device),
                "peptide": peptide.to(use_device),
                "interface_atoms": interface.to(use_device),
                "sigma_peptide": sigma_peptide.to(use_device),
                "epsilon_peptide": epsilon_peptide.to(use_device),
                "sigma_interface": sigma_target[mask].to(use_device),
                "epsilon_interface": epsilon_target[mask].to(use_device)
            }

        #### Here we are extracting the DoFs needed for Harmonic DoF Inpainting Regression
        #### these are taken from our passed PDB
        motif_dofs, updated_freeze_residues = dofsummary.extract_pdb_pose_and_dof(
            input_pdb = args.pdb_input,
            peptide_chain = args.peptide_chain,
            residues_of_interest = torch.tensor(args.freeze_residues),
            include_torsion = args.freeze_torsion if args.partial_optimization else True,
            include_bondangle = args.freeze_bond_angle if args.partial_optimization else True,
            include_bondlength = args.freeze_bond_length if args.partial_optimization else True,
        )

        #### Based on Target Information perform inpainting
        if args.fullpocket_minimize:
            scaffolds, latent_points, harm_values = energySearch.inpainting(args.batch_size, args.random_sample, interface_dict = interface_full_dict,
                                                               motif_dofs = motif_dofs, l_homochiral = args.l_homochiral,
                                                               updated_freeze_residues = updated_freeze_residues)
        else:
            scaffolds, latent_points, harm_values = energySearch.inpainting(args.batch_size, args.random_sample, interface_dict = interface_cb_dict,
                                                               motif_dofs = motif_dofs, l_homochiral = args.l_homochiral,
                                                               updated_freeze_residues = updated_freeze_residues)

        if latent_points.numel() == 0:
            print("No valid starting points were found; stopping before energy minimization.")
            raise SystemExit(3)

        #### Take either inpaint output and use for MCMC/Gradient-march or just graident-march
        if args.mcmc_steps >= 1:
            latent_traj, _ = energySearch.metropolis_hastings(scaffolds.size(0), args.sigma, latent_points, mcmc_steps=args.mcmc_steps, motif_dofs = motif_dofs,
                                                              harmonic_start = harm_values)
            energySearch.energy_steps(latent_traj, interface_full_dict = interface_full_dict, interface_cb_dict = interface_cb_dict,
                                      motif_dofs = motif_dofs, l_homochiral = args.l_homochiral)
        elif args.mcmc_steps < 1:
            energySearch.energy_steps(latent_points, interface_full_dict = interface_full_dict, interface_cb_dict = interface_cb_dict,
                                      motif_dofs = motif_dofs, l_homochiral = args.l_homochiral)

    #### Here we are just randomly sampling the latent space to get constructs.
    #### There are no conditions being applied
    elif args.random_sample:
        if args.mcmc_steps < 1:
            scaffolds, latent_points = energySearch.inpainting(args.batch_size, args.random_sample, l_homochiral = args.l_homochiral)
            energySearch.energy_steps(latent_points, l_homochiral = args.l_homochiral)
        else:
            scaffolds, latent_points = energySearch.inpainting(args.batch_size, args.random_sample, l_homochiral = args.l_homochiral)
            latent_traj, _ = energySearch.metropolis_hastings(scaffolds.size(0), args.sigma, latent_points, mcmc_steps=args.mcmc_steps)
            energySearch.energy_steps(latent_points, l_homochiral = args.l_homochiral)

    #### encode-decode, so we want to encode and then decode directly. To check to see if we can have a minimized version of our structure
    elif args.extract_minimize_input:
        # Cb Pocket Context Extract
        full, target, peptide, sigma_target, epsilon_target, sigma_peptide, epsilon_peptide = extract_pdb_xyz(args.pdb_input, full=False, peptide_chain=args.peptide_chain)
        interface, mask = grab_interface(target, peptide, dist_cutoff = args.interface_distance_cutoff)
        interface_cb_dict = {"target": target.to(use_device), "peptide": peptide.to(use_device), "interface_atoms": interface.to(use_device),
                          "sigma_peptide": sigma_peptide.to(use_device), "epsilon_peptide": epsilon_peptide.to(use_device),
                          "sigma_interface": sigma_target[mask].to(use_device), "epsilon_interface": epsilon_target[mask].to(use_device)}
        motif_dofs, latent_points = energySearch.encode_decode(args.pdb_input, dofsummary, args.peptide_chain)
        energySearch.energy_steps(latent_points, interface_cb_dict = interface_cb_dict, motif_dofs = motif_dofs)

    # Write our outputs out to this location
    energySearch.output_GDS(skip_individual = args.skip_individual,
                            dofsummary = dofsummary,
                            silentfile = args.silentfile,
                            score = args.score_outputs,
                            output_path = args.output_path,
                            random_outputs = args.random_sample,
                            minimize_pose = args.apply_minmover_outputs,
                            complex_output= args.complex_output,
                            )
    print("TOTAL CyclicCAE RUNTIME:", timedelta(minutes=time.perf_counter() - start_time))
