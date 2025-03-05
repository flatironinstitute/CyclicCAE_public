#!/usr/bin/env python
#
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

# import torch
import torch
import torch.nn as nn
import gc

# graphing
import matplotlib.pyplot as plt

# import model and functions
from test_case import ConvCase4
from model_functions import scale, ScaleOutputs
from terminal_error import TerminalPeptide
from inference_utils import (
    extract_pdb_xyz, gen_xyz_tensor, geometric_center, grab_interface, 
    interface_loss, kabsch_algorithm, kabsch_rootatom, size_help, 
    split_motif_atoms, discontig_motif_loss, kabsch_rigid_transform,
)
from dofSummary import DoFSummary
from motif_functions import (
    compute_motif_indices, calculate_motif_numbering_for_kabsch,
    extract_dof_of_interest_minimize,
)

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
        minima: float,
        # start_point: torch.Tensor,
        weight_passed: str,
        lr: float,
        # dataset: str,
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
        ):
        """GDS for energy minima structure from starting point

        PARAMS
        ------
            steps: int
                Number of steps for the search to take
            minima: float
                Cutoff for energy minima to descend too
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
        """
        # Pull up points
        self.steps = steps
        self.minima = minima
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
        if not self.partial_optimization: #and (torch.sum(torch.tensor([self.freeze_torsion, self.freeze_bond_angle, self.freeze_bond_length])).item() == 0):
            self.freeze_torsion = True
            self.freeze_bond_angle = True
            self.freeze_bond_length = True
        self.contiguous = self.check_contiguous(self.freeze_residues)
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
        self.match_res_mask = torch.empty((0, 2), dtype=torch.int)

        # Grab values required for model and other parts
        # dataset = torch.load(dataset)
        # self.ORIG_PARAMS = dataset["params"]
        # params = dataset["params"].copy()
        # print("PARAMS AFTER ASSIGN")
        # print(self.ORIG_PARAMS)
        # Added this, but don't know if it is necessary yet
        self.PARAMS, self.dof_size = size_help(out_size, dict())
        # print("PARAMS AFTER Size HELP")
        # print(self.ORIG_PARAMS)
        # X = dataset["x"].double()
        self.energy_n = self.PARAMS["energy_n"]
        # print("grab min and max of energy")
        # print("MIN:", X[:, -self.energy_n:].min(dim=0).values)
        # print("MAX:", X[:, -self.energy_n:].max(dim=0).values)

        # self.minima_energy = X[:, -self.energy_n:].min(dim=0).values

        # start point select without pdb input
        # self.start_point_dof = X[start_point, :]

        # generate holding tensor and input the beginning traj
        # self.dof_traj = torch.zeros((steps, start_point_dof.size(-1) - self.energy_n), dtype=torch.float64, device=device)
        # if self.energy_n == 3:
        self.dof_traj = torch.zeros((steps, self.dof_size - self.energy_n), dtype=torch.float64, device=device)
        # else:
        #     self.dof_traj = torch.zeros((steps, self.dof_size - 3), dtype=torch.float64, device=device)
        # self.dof_traj[0, :] = start_point_dof[:-self.energy_n]
        # Store points 
        # self.start_point = self.start_point_dof[:-self.energy_n]
        # self.start_point = self.start_point_dof[:-self.ORIG_PARAMS["energy_n"]]
        # self.start_point = self.start_point.unsqueeze(0)
        # self.start_point = self.start_point.to(device)
        # # Energy store
        # self.sample_energy = self.start_point_dof[-self.energy_n:]

        # init the model and eval mode
        self.model = ConvCase4(
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
        # ).to("cuda" if torch.cuda.is_available() else "cpu").double()
        ).to(device).double()
        # weights = torch.load(weight_passed, map_location="cuda" if torch.cuda.is_available() else "cpu")
        weights = torch.load(self.weight_passed, map_location=self.device)
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

        # Pull out scaled start point
        # self.start_point_scaled = scaled_X_start_point[-1, :]
        # self.start_point_scaled = self.start_point_scaled.unsqueeze(0)
        # self.start_point = self.start_point.to("cuda" if torch.cuda.is_available() else "cpu")
        # self.start_point_scaled = self.start_point_scaled.to(device)

        # scale energy for conversion importance
        score_terms = torch.tensor([2]) if self.energy_n == 1 else torch.arange(self.energy_n)
        self.energy_scale = ScaleOutputs(None, train=False, score_terms=score_terms, protein_len=self.PARAMS["protein_len"])

        if self.freeze_residues != None:
            # # Set up atoms to start with per residue
            # freeze_one = freeze_residues == 0
            # # Generate for both our reference and our constructs
            # # freeze_eight_ref = freeze_residues == self.ORIG_PARAMS["protein_len"]-1
            # freeze_eight_target = freeze_residues == self.PARAMS["protein_len"]-1
            # # freeze_root_ref = freeze_residues == self.term_bond.calculate_root(self.ORIG_PARAMS["protein_len"])-1
            # freeze_root_target = freeze_residues == self.term_bond.calculate_root(self.PARAMS["protein_len"])-1
            # # atoms_per_residue_ref = torch.ones(freeze_residues.shape, dtype=torch.int8) * 7
            # atoms_per_residue_target = torch.ones(freeze_residues.shape, dtype=torch.int8) * 7
            # # atoms_per_residue_ref[freeze_one] += 1
            # # atoms_per_residue_ref[freeze_eight_ref] += 2
            # atoms_per_residue_target[freeze_one] += 1
            # atoms_per_residue_target[freeze_eight_target] += 2
            # freeze_mas = freeze_residues > 0
            #
            # # For kabsch algo
            # residues_clone = self.freeze_residues.clone() + 1
            # residues_clone[0] = residues_clone[0] - 1
            # motif_res = (residues_clone * 4)
            #
            # # NOTE: This is a quick fix and not suitable for for the overall fix. Needs to be reworked.
            # if self.contiguous:
            #     self.motif = torch.arange(motif_res[0], motif_res[-1]) if len(self.freeze_residues) > 1 else torch.arange(motif_res[0], motif_res[0]+4)
            # else:
            #     self.motif = torch.cat((self.regions_atoms[0], self.regions_atoms[1])) 



            ### Gonna try terminal index functions
            root = self.term_bond.calculate_root(self.PARAMS["protein_len"])
            phi_psi_idx, omega_idx = self.term_bond.torsion_extract(self.PARAMS["torsion_main_n"], self.PARAMS["protein_len"], root)
            bond_angle_idx, bond_length_idx = self.term_bond.extract_bond_angles(self.PARAMS["bond_angle_main_n"], root=root, L=self.PARAMS["protein_len"])
            # phi_psi_true = phi_psi_idx[torch.cat(((self.freeze_residues * 2).view(-1, 1), (self.freeze_residues * 2).view(-1, 1) + 1), dim=-1).view(-1)]
            # phi_psi_true = phi_psi_idx[((self.freeze_residues * 2).view(-1, 1) + torch.tensor([0, 1])).view(-1)]
            #### NOTE to Andrew: OMEGA_idx only has 7 elements, since the terminal omega DoF isn't available in the data
            #### NOTE: This means that I need to either compute it for this one. Though This should be fixed if I extract info
            #### NOTE: from a pdb and calculate dihedrals/angles/lengths from there. Right now last residue doesn't work.
            # torsion_true = torch.cat((phi_psi_true.view(len(self.freeze_residues), -1), omega_idx[self.freeze_residues].view(-1, 1)), dim=-1).view(-1)
            # # bond_angle_true = ((self.freeze_residues * 2).view(-1, 1) + torch.arange(5).unsqueeze(0)).view(-1)
            # append_angles = list()
            # for ir in self.freeze_residues:
            #     if ir.item() == 0:
            #         append_angles.extend((torch.arange(8, dtype=torch.long) + (ir*2)).tolist())
            #     elif ir.item() == root:
            #         append_angles.extend((torch.arange(6, dtype=torch.long) + (ir*2)).tolist())
            #     elif ir.item() == self.PARAMS["protein_len"]-1:
            #         append_angles.extend((torch.arange(9, dtype=torch.long) + (ir*2)).tolist())
            #     else:
            #         append_angles.extend((torch.arange(7, dtype=torch.long) + (ir*2)).tolist())
            # bond_angle_true = torch.tensor(append_angles)


            # self.torsion_index_ref         = torsion_true.to(torch.long)
            # self.torsion_index_target      = torsion_true.to(torch.long)
            # self.bond_angle_index_ref      = bond_angle_true.to(torch.long)
            # self.bond_length_index_ref     = bond_angle_true.to(torch.long)
            # self.bond_angle_index_target   = bond_angle_true.to(torch.long) #- self.PARAMS["torsion_main_n"]
            # self.bond_length_index_target  = bond_angle_true.to(torch.long)# - (self.PARAMS["torsion_main_n"]*2)
            self.torsion_index, self.bond_angle_dofs, self.bond_length_dofs, self.residue_index = compute_motif_indices(
                self.PARAMS["protein_len"],
                self.PARAMS["torsion_main_n"],
                self.freeze_residues.shape[0],
                root,
                phi_psi_idx,
                omega_idx,
                bond_angle_idx,
                bond_length_idx,
            )
        else:
            self.torsion_index = slice(None)
            self.bond_angle_dof = slice(None)
            self.bond_length_dof = slice(None)
            self.residue_index = slice(None)

            # self.torsion_index_ref         = slice(None)
            # self.torsion_index_target      = slice(None)
            # self.bond_angle_index_ref      = slice(None)
            # self.bond_length_index_ref     = slice(None) 
            # self.bond_angle_index_target   = slice(None) 
            # self.bond_length_index_target  = slice(None) 


    def check_contiguous(self, motif: list[int]):
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
        # If it is longer then check
        else:
            j = 0
            for i in range(1, len(motif)):
                # If the previous + 1 does not equal the next one. Then we
                # know that this is non-contiguous
                if motif[j]+1 != motif[i]:
                    return False
                j = i
        return True

    def encoder(self, input: torch.Tensor):
        """Input the DoF Vector and get out the latent space point to optimize

        PARAMS
        ------
            input: torch.Tensor
                DoF Input vector (1, 174) (for glycine 8mer cyclic petpide)
        
        RETURNS
            torch.Tensor
                Latent Space Tensor (1, 3)
        """
        # decompose angles/ return linear terms
        x, bond_angle, bond_length, energy = self.model.input_decompose(input, self.model.training, False)
        init_x = torch.cat((x, bond_angle, bond_length, energy), dim=-1)
        initial_input = init_x.clone().detach()

        # encode energy
        # energy_encode = self.encode_energy(energy)

        # generate linear pe
        pe = self.model.pe()

        ##### ENCODER
        # If convolution is true then reformat and apply convolution
        if self.model.convolution:
            # Torsion reshape
            sin_x = self.model.reformat_convolution2(x[:, :self.model.torsion_main_n], self.model.dof_residue_metadata[:self.model.torsion_main_n,:])
            cos_x = self.model.reformat_convolution2(x[:, self.model.torsion_main_n:], self.model.dof_residue_metadata[:self.model.torsion_main_n,:])
            # Bond Angle Reshape
            sin_x_bond = self.model.reformat_convolution2(bond_angle[:, :self.model.bond_angle_main_n], self.model.dof_residue_metadata[self.model.torsion_main_n:self.model.torsion_main_n + self.model.bond_angle_main_n, :])
            cos_x_bond = self.model.reformat_convolution2(bond_angle[:, self.model.bond_angle_main_n:], self.model.dof_residue_metadata[self.model.torsion_main_n+self.model.bond_angle_main_n : self.model.torsion_main_n+self.model.bond_angle_main_n+self.model.bond_length_main_n, :])
            # non-linear transform bond_length to match input dim for convolution
            bond_length_transform = self.model.bond_length_trans(bond_length)
            B, L, D = sin_x_bond.shape

            # Concat together
            x_combo = torch.cat(
                (
                    sin_x.unsqueeze(-1),
                    cos_x.unsqueeze(-1),
                    sin_x_bond.unsqueeze(-1),
                    cos_x_bond.unsqueeze(-1),
                    bond_length_transform.reshape(B, L, D).unsqueeze(-1),
                ),
                dim=-1,
            )
            # Add in positional encodings
            x_combo = torch.add(x_combo, pe.unsqueeze(0))
            # reshape from (N, L, D, 2) -> (N, 2, D*L)
            B, L, D, C = x_combo.shape
            x = self.model.conv_cyc_encoder(x_combo.reshape(B, C, D*L)) # 1D
            # x = self.model.conv_cyc2d(x_combo.permute(0, 3, 1, 2)) # 2D
        else:
        # Convert 116 -> 64 for encoder block training
            x = self.model.encoder_one(x)
        skip_one = x.clone()

        # First Encoder block 64 -> 64
        x = self.model.encoder(x)
        skip_n1 = x.clone()

        # Second Encoder Block 64 -> 64 
        x = self.model.encoder_n2(torch.add(x, skip_one))
        skip_n2 = x.clone() 

        # Third Encoder Block 64 -> 64
        x = self.model.encoder_n3(torch.add(x, skip_n1))

        # Pyramid Encoder 64 -> 32 -> 16 -> 8 -> 3
        x = self.model.encoder_pyramid(torch.add(x, skip_n2))
        ##### ENCODER END
        return x

    def decoder(self, latent_space: torch.Tensor):
        """Decode straight from the latent space optimization

        PARAMS
        ------
            latent_space: torch.Tensor
                Latnet Space Vector (1, 3)
        
        RETURNS
        -------
            torch.Tensor: Recomputed DoFs
        """
        energy_x = self.model.energy_decoder1(latent_space)
        energy_skip1 = energy_x.clone()
        # Decoder latent to decoder blocks size 3 -> 64
        x = self.model.decoder_pyramid(latent_space)
        decode_skip1 = x.clone()

        # First Decoder Block 64 -> 64
        x = self.model.decoder_n(x)
        dskip_n1 = x.clone()
        # compute first energy block
        energy_x = self.model.energy_decoder2(torch.add(energy_x, decode_skip1))
        energy_skip2 = energy_x.clone()

        # Second Decoder Block 64 -> 64
        x = self.model.decoder_n2(torch.add(x, decode_skip1))
        dskip_n2 = x.clone()
        # compute second block of energies
        energy_x = self.model.energy_decoder3(torch.add(torch.add(energy_x, energy_skip1), dskip_n2))
        energy_skip3 = energy_x.clone()

        x = self.model.decoder_n3(torch.add(x, dskip_n1))
        dskip_n3 = x.clone()
        # final energy compute
        energy_x = self.model.energy_decoder4(torch.add(torch.add(energy_x, energy_skip2), dskip_n3))

        # Third Decoder Block 64 -> 64
        if self.model.convolution:
            B, D = x.shape
            x = self.model.conv_cyc_decoder(x).view(B, -1)

        # FINAL decoder layer 64 -> 116 out size
        torsion_angle_pred = self.model.head_torsion_angle(x)
        bond_angle_pred = self.model.head_bond_angle(x)
        bond_length_pred = self.model.head_bond_length(x)
        energy_pred = self.model.head_energy(energy_x)

        out = torch.cat(
            (
                torsion_angle_pred,
                bond_angle_pred,
                bond_length_pred,
                energy_pred,
            ),
            dim=-1
        )
        ##### DECODER END
        
        return out


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

    # def harmonic_diff_inpaint(
    #     self, 
    #     real_t_sin: torch.Tensor, 
    #     real_t_cos: torch.Tensor,
    #     real_ba_sin: torch.Tensor,
    #     real_ba_cos: torch.Tensor,
    #     real_bond_length: torch.Tensor,
    #     torsion_sin: torch.Tensor,
    #     torsion_cos: torch.Tensor,
    #     bond_angle_sin: torch.Tensor,
    #     bond_angle_cos: torch.Tensor,
    #     bond_length: torch.Tensor,
    #     ):
    #     """Harmonic Difference"""
    #     print(real_t_sin.shape)
    #     print(real_ba_sin.shape)
    #     print(bond_angle_sin.shape)
    #     print(self.bond_angle_index_ref)
    #     harmonic_penalty_sin_torsion = torch.tensor(0.0, device=self.device)    if not self.freeze_torsion     else torch.pow(real_t_sin - torsion_sin[:, self.torsion_index_target],2).mean(dim=-1).to(device=self.device)
    #     harmonic_penalty_cos_torsion = torch.tensor(0.0, device=self.device)    if not self.freeze_torsion     else torch.pow(real_t_cos - torsion_cos[:, self.torsion_index_target],2).mean(dim=-1).to(device=self.device)
    #     # harmonic_penalty_sin_bond_angle = torch.tensor(0.0, device=self.device) if not self.freeze_bond_angle  else torch.pow(real_ba_sin[:, self.bond_angle_index_ref] - bond_angle_sin[:, self.bond_angle_index_target],2).mean(dim=-1).to(device=self.device)
    #     # harmonic_penalty_cos_bond_angle = torch.tensor(0.0, device=self.device) if not self.freeze_bond_angle  else torch.pow(real_ba_cos[:, self.bond_angle_index_ref] - bond_angle_cos[:, self.bond_angle_index_target],2).mean(dim=-1).to(device=self.device)
    #     # harmonic_penalty_bond_length = torch.tensor(0.0, device=self.device)    if not self.freeze_bond_length else torch.pow(real_bond_length[:, self.bond_length_index_ref] - bond_length[:, self.bond_length_index_target],2).mean(dim=-1).to(device=self.device)
    #     harmonic_penalty_sin_bond_angle = torch.tensor(0.0, device=self.device) if not self.freeze_bond_angle  else torch.pow(real_ba_sin - bond_angle_sin[:, self.bond_angle_index_target],2).mean(dim=-1).to(device=self.device)
    #     harmonic_penalty_cos_bond_angle = torch.tensor(0.0, device=self.device) if not self.freeze_bond_angle  else torch.pow(real_ba_cos - bond_angle_cos[:, self.bond_angle_index_target],2).mean(dim=-1).to(device=self.device)
    #     harmonic_penalty_bond_length = torch.tensor(0.0, device=self.device)    if not self.freeze_bond_length else torch.pow(real_bond_length - bond_length[:, self.bond_length_index_target],2).mean(dim=-1).to(device=self.device)
    #     harmonic_loss = harmonic_penalty_sin_torsion + harmonic_penalty_cos_torsion + harmonic_penalty_cos_bond_angle + harmonic_penalty_sin_bond_angle + harmonic_penalty_bond_length
    #     # harmonic_loss /= torch.sum(torch.tensor([self.freeze_torsion, self.freeze_torsion, self.freeze_bond_angle, self.freeze_bond_angle, self.freeze_bond_length]))
    #     print(harmonic_loss)
    #     print(harmonic_loss.min())
    #     return harmonic_loss

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
            # real_t_sin = real_t_sin.unsqueeze(0).expand(torsion_sin.shape[0], torsion_sin.shape[1], -1)
            # real_t_cos = real_t_cos.unsqueeze(0).expand(torsion_sin.shape[0], torsion_sin.shape[1], -1)
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
        # harmonic_loss /= torch.sum(torch.tensor([self.freeze_torsion, self.freeze_torsion, self.freeze_bond_angle, self.freeze_bond_angle, self.freeze_bond_length]))

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
    


    def inpainting(self, B: int, random_no_reference: bool = False, interface_dict = None,
                   terminal = True, motif_dofs = None):
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

        # Need the coordinates calculated here, for discontinuous motifs
        if not self.contiguous:
            self.atoms, self.t_native, self.R_native = kabsch_rigid_transform(
                interface_dict["peptide"][self.regions_atoms[0]].unsqueeze(0),
                interface_dict["peptide"][self.regions_atoms[1]].unsqueeze(0),
            )

        # input information
        if args.freeze_residues != None:
            real_t_sin = torch.sin(motif_dofs[0])
            real_t_cos = torch.cos(motif_dofs[0])
            real_ba_sin = torch.sin(motif_dofs[1])
            real_ba_cos = torch.cos(motif_dofs[1])
            real_bond_length = motif_dofs[2] 
        # real_t_sin = torch.sin(torch.deg2rad(self.start_point[:, :self.ORIG_PARAMS["torsion_main_n"]]))
        # real_t_cos = torch.cos(torch.deg2rad(self.start_point[:, :self.ORIG_PARAMS["torsion_main_n"]]))
        # real_ba_sin = torch.sin(torch.deg2rad(self.start_point[:, self.ORIG_PARAMS["bond_angle_main_n"]:self.ORIG_PARAMS["bond_angle_main_n"]*2]))
        # real_ba_cos = torch.cos(torch.deg2rad(self.start_point[:, self.ORIG_PARAMS["bond_angle_main_n"]:self.ORIG_PARAMS["bond_angle_main_n"]*2]))
        # real_bond_length = self.start_point[:, self.ORIG_PARAMS["bond_angle_main_n"]*2:]

        while no_scaffolds:
            # set up tensor of random init values
            scatter_latent = torch.zeros((B, self.latent_dim), dtype=torch.float64, device=self.device)

            if self.PARAMS["protein_len"] == 6:
                scatter_latent.uniform_(-0.7400, 0.7400) # This is set up for model 7731
            elif self.PARAMS["protein_len"] == 8:
                scatter_latent.uniform_(-0.3824, 0.3802) # This is set up for model 7731
            elif self.PARAMS["protein_len"] == 9:
                scatter_latent.uniform_(-0.20100, 0.20100)
                # scatter_latent.uniform_(-0.9800, 0.9800)
            
            # run the points through the decoder
            out = self.decoder(scatter_latent)

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
            energy_all = self.energy_scale.inverse_scale(out[:, -self.energy_n:])

            ######### REMOVE SECTION LATER
            if energy_all.size(-1) == 1:
                energy = torch.zeros(energy_all.size(0), 3)
                energy[:, -2] = energy_all.squeeze(-1)
                energy_all = energy

            energy_ref = energy_all[:, -2]
            # energy_ref = energy_all[:, -1]
            ######### REMOVE SECTION LATER


            # term_bond_loss, bah_loss, ahd_loss, ocnh_loss, ncaco_loss, clash_penalty, coordinates = self.term_bond.terminal_peptide_loss(
            term_bond_loss, nco_loss, hnc_loss, ocnh_loss, _, HNCO, coordinates = self.term_bond.terminal_peptide_loss(
                dof_inputs = out,
                real_term = torch.ones(out.size(0), device=out.device, dtype=out.dtype) * 1.4,
                params = self.PARAMS,
                foldtree_root = self.term_bond.calculate_root(self.PARAMS["protein_len"]),
                device = out.device,
            )

            if args.freeze_residues != None:
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
                harmonic_loss, match_res_mask = self.harmonic_diff_inpaint(
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
                )
                # Calculate motif numbering
                # HERE
                self.motif, self.motif_input = calculate_motif_numbering_for_kabsch(
                    match_res_mask,
                    self.PARAMS["protein_len"],
                    self.freeze_residues.shape[0],
                    self.freeze_residues,
                )

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

            if interface_dict != None:
                peptide_nonhydrogen_mask = interface_dict["epsilon_peptide"] != 0.901681
                if not self.contiguous:
                    aligned_coordinates = kabsch_rootatom(P = coords, Q = interface_dict["peptide"][peptide_nonhydrogen_mask].double(),
                                                          # motif = torch.arange(12, 16),
                                                          # motif = torch.tensor(12),
                                                          motif = self.regions_atoms[0],
                                                          )
                    discontig_loss = discontig_motif_loss(
                        region1  = aligned_coordinates[:, self.regions_atoms[0], :].view(aligned_coordinates.shape[0], -1, 3),
                        region2  = aligned_coordinates[:, self.regions_atoms[1], :].view(aligned_coordinates.shape[0], -1, 3),
                        t_native = self.t_native,
                        R_native = self.R_native,
                    )
                else:
                    # Non contiguous indexing of our outputs
                    coords = coords[match_res_mask[:, 0]]
                    aligned_coordinates = kabsch_rootatom(P = coords, Q = interface_dict["peptide"][peptide_nonhydrogen_mask].double(),
                                                          motif_output = self.motif,
                                                          motif_input = self.motif_input,
                                                          )

                interface_nonhydrogen_mask = interface_dict["epsilon_interface"] != 0.901681
                interface_nonorm_loss, energy_mask = interface_loss(interface_atoms = interface_dict["interface_atoms"][interface_nonhydrogen_mask].double(), recomputed_xyz = aligned_coordinates,
                                                      sigma_peptide = interface_dict["sigma_peptide"][peptide_nonhydrogen_mask],
                                                      epsilon_peptide = interface_dict["epsilon_peptide"][peptide_nonhydrogen_mask],
                                                      sigma_interface = interface_dict["sigma_interface"][interface_nonhydrogen_mask],
                                                      epsilon_interface = interface_dict["epsilon_interface"][interface_nonhydrogen_mask])
                interface_total_loss = interface_nonorm_loss # / aligned_coordinates.size(0)
            else:
                interface_total_loss = torch.zeros(out.size(0), dtype=torch.bool, device=self.device)

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
                # torsion = self.calc_angles(out[:, :self.PARAMS["torsion_main_n"]*2])
                # bond_angle = self.calc_angles(out[:, self.PARAMS["torsion_main_n"]*2:self.PARAMS["bond_angle_main_n"]*2*2])
                # preds = torch.cat((torsion, bond_angle, out[:, self.PARAMS["torsion_main_n"]*2*2:-self.energy_n]), dim=-1)
                # torch.save({"random_x": preds, "latent_point": scatter_latent}, f"randomPoints_batch{B}_{datetime.date.today()}.pt")
                # print("Random Scatter DONE")
                # return preds, scatter_latent

            else:
                # Determine which points are within the latent space cutoff
                int_mask = interface_total_loss <= self.interface_cutoff
                terminal_mask = term_bond_loss[match_res_mask[:, 0]].view(-1) < 0.5 #2.0
                ocnh_mask = ocnh_loss[match_res_mask[:, 0]].view(-1) < 0.5
                hnc_mask = hnc_loss[match_res_mask[:, 0]].view(-1) < 0.5
                nco_mask = nco_loss[match_res_mask[:, 0]].view(-1) < 0.5
                harm_mask = harmonic_loss <= self.inpaint_dist
                # print("MASKS:", int_mask.sum(), terminal_mask.sum(), ocnh_mask.sum(), hnc_mask.sum(), nco_mask.sum(), harm_mask.sum())
                if self.contiguous:
                    mask = torch.logical_and(
                        ocnh_mask.view(-1, 1), torch.logical_and(
                            hnc_mask.view(-1, 1), torch.logical_and(
                                nco_mask.view(-1, 1), torch.logical_and(
                                    harm_mask.view(-1,1), torch.logical_and(
                                        int_mask.view(-1,1), terminal_mask.view(-1, 1)
                                    )
                                ),
                            ),
                        )).view(-1)
                #### Need to rework this method
                #### Might be beneficial to take the residue spacing for this one and check all (i, j, k) pairs and not just the specific
                #### residue we are starting with
                #### This would be a somewhat lengthy change, so not important for the moment.
                else:
                    # print("CA non-contiguous")
                    # ca_motif_distance_mask = ca_distance_motif_initial <= 0.5
                    ca_motif_distance_mask = discontig_loss <= 5.0
                    mask = torch.logical_and(
                        ocnh_mask.view(-1, 1), torch.logical_and(
                            hnc_mask.view(-1, 1), torch.logical_and(
                                nco_mask.view(-1, 1), torch.logical_and(
                                    harm_mask.view(-1,1), torch.logical_and(
                                        int_mask.view(-1,1), torch.logical_and(
                                            terminal_mask.view(-1, 1), ca_motif_distance_mask.view(-1, 1)),
                                    )
                                ),
                            ),
                        )).view(-1)
                    # mask = torch.logical_and(
                    #                 harm_mask.view(-1,1), torch.logical_and(
                    #                     int_mask.view(-1,1), torch.logical_and(
                    #                         terminal_mask.view(-1, 1), ca_motif_distance_mask.view(-1, 1)),
                    #     )).view(-1)


                # mask = torch.logical_and(torch.logical_and(int_mask.view(-1, 1), harm_mask.view(-1, 1)),terminal_mask.view(-1, 1)).view(-1),
                # ),


            # Grab out specific points that only pass
            preds = preds[match_res_mask[:, 0]]
            scatter_latent = scatter_latent[match_res_mask[:, 0]]
            # concat onto self.match_res
            self.match_res_mask = torch.cat((self.match_res_mask, match_res_mask[mask]), dim=0)
            # add to set of scaffolds
            if not torch.is_tensor(scaffolds):
                scaffolds = preds[mask, :-3].detach()
                points = scatter_latent[mask].detach()
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

            # Report run statistics
            if random_no_reference:
                print(f'BATCH: {run}, FOUND SCAFFOLD START POINTS: {mask.sum().item()}, SCAFFOLDS FOUND: {points.size(0)}')
            else:
                print(f'BATCH: {run}, FOUND SCAFFOLD START POINTS: {mask.sum().item()}, BATCH LOW: {harmonic_loss.min().item():.2f}, BATCH MEAN: {harmonic_loss.mean().item():.2f}, CUTOFF: {self.inpaint_dist}, SCAFFOLDS FOUND: {points.size(0)}')

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
            torch.save({"scaffold": preds, "latent_point": points}, "inpaintStartOutStructures.pt")
        else:
            print(f"No Scaffolds Found with residue similarity as {self.inpaint_dist} in {run-1} iterations")
        if scaffolds.size(0) > self.scaffold_number:
            self.match_res_mask = self.match_res_mask[:self.scaffold_number, 1]
            return scaffolds[:self.scaffold_number], points[:self.scaffold_number]
        else:
            return scaffolds, points

    def metropolis_hastings(self, B: int, sigma: torch.float, start_points: torch.Tensor = None, mcmc_steps = 0):
        """MCMC Search for energy minima structures
        PARAMS
        ------
            B: int,
                Batch Size. Amount of attempts to try in parallel
            sigma: torch.float
                Amount of noise for normal distribution
            start_points: torch.Tensor
                Start latent points from inpainting

        RETURNS
        -------
            latent_trajectories: torch.tensor
                Shape (B, T, 3) Where B is the number of Trajectories,
                T is the time step, and 3 is the xyz coordinates for that
                step
            lowest_predicted: torch.Tensor
                The lowest energy DoF Vector sample trajectory (B_i) out of the search
        """
        # Latent Trajectories
        # latent_trajectories = torch.zeros((B, self.steps, 3), dtype=torch.double)
        # latent_trajectories = torch.zeros((B, 2, self.latent_dim), dtype=torch.double)
        #
        # # Lowest energy holder for the trajectories. This determines which point to return and decode
        # # energies_trajectories = torch.zeros((B, self.steps, 1), dtype=torch.double)
        # energies_trajectories = torch.zeros((B, 2, 1), dtype=torch.double)
        #
        # # Harmonic Constrain Loss
        # H = torch.zeros((B, 2, 1), dtype=torch.double)

        # minima structures
        # self.minima_structures = torch.zeros((B, self.start_point.size(-1)-self.energy_n), dtype=torch.float64)

        if start_points == None:
            F = 1
            latent_trajectories = torch.zeros((B, F, 2, self.latent_dim), dtype=torch.double)
            energies_trajectories = torch.zeros((B, F, 2, 1), dtype=torch.double)
            H = torch.zeros((B*F, 2, 1), dtype=torch.double)
            # get starting point and enable grad
            self.start_latent_point = self.encoder(self.start_point_scaled).detach()

            # Set this as t_0 for all trajectories
            latent_trajectories[:, :,  0, :] = self.start_latent_point.unsqueeze(1).repeat(B, F, 1)
            latent_trajectories[:, :, -1, :] = self.start_latent_point.unsqueeze(1).repeat(B, F, 1)
            # reshape for model input
            latent_trajectories = latent_trajectories.reshape(B*F, 2, self.latent_dim)
        else:
            # F = 30
            F = self.mcmc_paths
            latent_trajectories = torch.zeros((B, F, 2, self.latent_dim), dtype=torch.double)
            energies_trajectories = torch.zeros((B, F, 2, 1), dtype=torch.double)
            H = torch.zeros((B*F, 2, 1), dtype=torch.double)
            latent_trajectories[:, :, 0, :] = start_points.unsqueeze(1).repeat(1, F, 1)
            latent_trajectories[:, :, -1, :] = start_points.unsqueeze(1).repeat(1, F, 1)
            # reshape for model input
            latent_trajectories = latent_trajectories.reshape(B*F, 2, self.latent_dim)


        # set initial energy to everything (specific Ref2015 energies)
        # print(self.start_point_scaled.shape)
        # encode_decode_input = self.model(self.start_point_scaled, False)[0]
        # energy_out = self.energy_scale.inverse_scale(encode_decode_input[:, -self.energy_n:])
        # # energies_trajectories[:, 0, :] = energy_out[:, -1]
        #
        # energies_trajectories[:, :, 0, :] = energy_out[:, -1].unsqueeze(1).repeat(B, F, 1)

        #reshape energy
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
        # real_t_sin = torch.sin(torch.deg2rad(self.start_point[:, :self.ORIG_PARAMS["torsion_main_n"]]))
        # real_t_cos = torch.cos(torch.deg2rad(self.start_point[:, :self.ORIG_PARAMS["torsion_main_n"]]))
        # real_ba_sin = torch.sin(torch.deg2rad(self.start_point[:, self.ORIG_PARAMS["bond_angle_main_n"]:self.ORIG_PARAMS["bond_angle_main_n"]*2]))
        # real_ba_cos = torch.cos(torch.deg2rad(self.start_point[:, self.ORIG_PARAMS["bond_angle_main_n"]:self.ORIG_PARAMS["bond_angle_main_n"]*2]))
        # real_bond_length = self.start_point[:, self.ORIG_PARAMS["bond_angle_main_n"]*2:]

        steps = self.steps if mcmc_steps == 0 else mcmc_steps
        # Perform MCMC for n steps
        for t in range(1, steps):
            
            # Generate random walk noise
            epsilon = torch.normal(0, sigma, size=(B*F, self.latent_dim), dtype=torch.double)
            # Apply noise to previous t-1 point
            # x_t = latent_trajectories[:, t-1, :] + epsilon
            x_t = latent_trajectories[:, -1, :] + epsilon

            # decode points
            out = self.decoder(x_t.to(self.device))

            # Prediction Information
            torsion = out[:, :self.PARAMS["torsion_main_n"]*2]
            torsion_sin = torsion[:, :self.PARAMS["torsion_main_n"]]
            torsion_cos = torsion[:, self.PARAMS["torsion_main_n"]:]
            bond_angle = out[:, self.PARAMS["bond_angle_main_n"]*2:self.PARAMS["bond_angle_main_n"]*2*2]
            bond_angle_sin = bond_angle[:, :self.PARAMS["bond_angle_main_n"]]
            bond_angle_cos = bond_angle[:, self.PARAMS["bond_angle_main_n"]:]
            bond_length = out[:, self.PARAMS["bond_angle_main_n"]*2*2:-3]
            # torsion = out[:, :116]
            # torsion_sin = torsion[:, :58]
            # torsion_cos = torsion[:, 58:]
            # bond_angle = out[:, 116:116*2]
            # bond_angle_sin = bond_angle[:, :58]
            # bond_angle_cos = bond_angle[:, 58:]
            # bond_length = out[:, 116*2:-3]

            # Grab ref2015 energy from output
            energy_all = self.energy_scale.inverse_scale(out[:, -self.energy_n:])
            if energy_all.size(-1) == 1:
                energy = torch.zeros(energy_all.size(0), 3, dtype=energy_all.dtype)
                energy[:, -2] = energy_all.squeeze(-1)
                energy_all = energy

            # energy_ref = energy_all[:, -2]
            energy_t = energy_all[:, -2]
            # energy_t = energy_all[:, -1]

            # Harmonic Change for if none of these are true if they are then it will change
            harmonic_change = torch.zeros((B, 1), dtype=torch.double)
            harmonic_loss = torch.zeros((B), dtype=torch.double)
            if self.partial_optimization:

                harmonic_loss = self.harmonic_diff_inpaint(
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
                )
                if t == 1:
                    H[:, 0, :] = harmonic_loss.unsqueeze(-1)
                    H[:, 1, :] = harmonic_loss.unsqueeze(-1)
                    harmonic_change = harmonic_loss.squeeze(0).unsqueeze(-1)
                else:
                    harmonic_change = H[:, 1, :] - harmonic_loss.squeeze(0).unsqueeze(-1).detach().cpu()
            if self.retain_structure:
                harmonic_loss = self.harmonic_diff_inpaint(
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
                )

                # harmonic_penalty_sin_torsion     = torch.pow(torch.norm(real_t_sin - torsion_sin, dim=-1)*10, 2)   
                # harmonic_penalty_cos_torsion     = torch.pow(torch.norm(real_t_cos - torsion_cos, dim=-1)*10, 2)   
                # harmonic_penalty_sin_bond_angle  = torch.pow(torch.norm(real_ba_sin - bond_angle_sin, dim=-1)*10, 2) 
                # harmonic_penalty_cos_bond_angle  = torch.pow(torch.norm(real_ba_cos - bond_angle_cos, dim=-1)*10, 2) 
                # harmonic_penalty_bond_length     = torch.pow(torch.norm(real_bond_length - bond_length, dim=-1), 2)    
                # harmonic_loss = harmonic_penalty_sin_torsion + harmonic_penalty_cos_torsion + harmonic_penalty_cos_bond_angle + harmonic_penalty_sin_bond_angle + harmonic_penalty_bond_length
                # harmonic_loss /= 5*58
                if t == 1:
                    H[:, 0, :] = harmonic_loss.unsqueeze(-1)
                    H[:, 1, :] = harmonic_loss.unsqueeze(-1)
                    harmonic_change = harmonic_loss.squeeze(0).unsqueeze(-1)
                else:
                    harmonic_change = H[:, 1, :] - harmonic_loss.squeeze(0).unsqueeze(-1)


            # temp_coeff = temp_linspace[(self.initial_set+t)-(self.initial_set+1)] if t <= self.rampdown+self.initial_set else 0.01
            # alpha = torch.minimum(torch.ones(energy_t.shape), torch.exp( -( ( (energies_trajectories[:, -1, :].squeeze(-1) - energy_t ) + self.lambda_2*harmonic_change.squeeze(-1) ) / temp_coeff ) ) )

            # accept or not
            if t > self.initial_set:
                if self.temp != 1:
                    temp_t = t - self.initial_set
                    temp_coeff = temp_linspace[temp_t-1] if temp_t <= self.rampdown else 0.01
                else:
                    temp_coeff = 1
                # print("TEMP:", temp_coeff)
                # print("ENERGY:", energies_trajectories[:, -1, :].squeeze(-1) - energy_t)
                # print('energy', energies_trajectories[:, -1, :].shape)
                # print('harmonic change', harmonic_change.shape)
                # term_bond_loss = self.term_bond.terminal_peptide_loss(
                term_bond_loss, bah_loss, ahd_loss, ocnh_loss, ncaco_loss, _, _ = self.term_bond.terminal_peptide_loss(
                    dof_inputs = out,
                    real_term = torch.ones(out.size(0), device=out.device, dtype=out.dtype) * 1.4,
                    params = self.PARAMS,
                    foldtree_root = self.term_bond.calculate_root(self.PARAMS["protein_len"]),
                    device = out.device,
                )
                # print("TERM BOND LOSS:", term_bond_loss)

                out_num = ( 
                    (energies_trajectories[:, -1, :].squeeze(-1) - energy_t ) + self.lambda_2*harmonic_change.squeeze(-1) + term_bond_loss.squeeze(-1).detach().cpu() 
                    + bah_loss.squeeze(-1).detach().cpu() + ahd_loss.squeeze(-1).detach().cpu() + ocnh_loss.squeeze(-1).detach().cpu() + ncaco_loss.squeeze(-1).detach().cpu()
                )
                temp_coeff = temp_coeff if out_num.mean() > 1 else 1/temp_coeff
                # alpha = torch.minimum(torch.ones(energy_t.shape), torch.exp( -( ( (energies_trajectories[:, -1, :].squeeze(-1) - energy_t ) + self.lambda_2*harmonic_change.squeeze(-1) + term_bond_loss.squeeze(-1)) / temp_coeff ) ) )
                alpha = torch.minimum(torch.ones(energy_t.shape), torch.exp( -( out_num / temp_coeff ) ) )
                # print("ALPHA:", alpha)

                u = torch.ones((B*F))
                u.uniform_(0, 1)
                # generate bool of T/F
                condition_u = u > alpha
                # print("CONDITION:", condition_u)
                # print('CONDITION SUM:', condition_u.sum())
                # print(x_t[:20])

                # Place energies or reject
                # energies_trajectories[condition_u, t, :] = energy_t[condition_u].view(-1, 1)
                # energies_trajectories[~condition_u, t, :] = energies_trajectories[~condition_u, t-1, :]
                energies_trajectories[condition_u, -1, :] = energy_t[condition_u].view(-1, 1)
                # energies_trajectories[~condition_u, -1, :] = energies_trajectories[~condition_u, :]
                # Place steps or reject
                # latent_trajectories[condition_u, t, :] = x_t[condition_u]
                # latent_trajectories[~condition_u, t, :] = latent_trajectories[~condition_u, t-1, :]
                latent_trajectories[condition_u, -1, :] = x_t[condition_u]
                # latent_trajectories[~condition_u, -1, :] = latent_trajectories[~condition_u, :]
                # Now replace values if needed
                mask = (energies_trajectories[:, -1, :] < energies_trajectories[:, 0, :]).squeeze(-1)
                energies_trajectories[mask, 0] = energies_trajectories[mask, -1]
                latent_trajectories[mask, 0, :] = latent_trajectories[mask, -1, :]
                H[mask, 0, :] = harmonic_loss[mask].unsqueeze(-1).cpu()
            # Accept all movements that are within the initial set range
            else:
                energies_trajectories[:, -1, :] = energy_t.view(-1, 1)
                latent_trajectories[:, -1, :] = x_t
                H[:, 1, :] = harmonic_loss.unsqueeze(-1).cpu()

                if t == steps-1:
                    latent_trajectories[:, 0, :] = latent_trajectories[:, -1, :]

            if t % 100 == 0:
                # print(f"Current Time Step: {t} Current Lowest Energy Minima: {energies_trajectories[:, :t, :].min().item()} Starting Point Energy: {energies_trajectories[0, 0, :].item()}")
                # print(f"Current Time Step: {t} Current Lowest Energy Minima: {energies_trajectories.min().item()}, Starting Point Energy: {energy_out[0, -1].item()}, Current Lowest Harmonic Constraint: {H[:, -1, :].min().item()}, Start Harmonic Constraint: {H[0, 0, 0].item()}")
                print(f"Current Time Step: {t} Current Lowest Energy Minima: {energies_trajectories.min().item()}, Current Lowest Harmonic Constraint: {H[:, -1, :].min().item()}, Start Harmonic Constraint: {H[0, 0, 0].item()}")


        # grab minimum of each B_i from F
        latent_trajectories = latent_trajectories.reshape(B, F, 2, self.latent_dim)
        energies_trajectories = energies_trajectories.reshape(B, F, 2, 1)
        
        if self.partial_optimization or self.retain_structure:
            H = H.reshape(B, F, 2, 1)
            # grab min of H
            h_min = H.argmin(dim=1, keepdim=True)
            h_min_expanded = h_min.expand(-1, -1, -1, latent_trajectories.size(-1))
            latent_trajectories = torch.gather(latent_trajectories, 1, h_min_expanded).squeeze(1)
            H = torch.gather(H, 1, h_min).squeeze(1)
        else:
            energy_min = energies_trajectories.min(dim=1).indices if F != 1 else 0
            latent_trajectories = latent_trajectories[:, energy_min, :, :]

        if start_points == None:
            # print(f"Final Lowest Energy Minima: {energies_trajectories.min().item()} Starting Point Energy: {energy_out[0, 0].item()} Average Distance: {torch.norm(latent_trajectories[:, 0, :], self.start_latent_point, dim=-1).mean().item()}, Lowest Harmonic Constraint: {H[:, 0, :].min().item()}, Start Mean Harmonic Constraint: {H[:, 0, :].mean().item()}")
            # print(f"Final Lowest Energy Minima: {energies_trajectories.min().item()} Starting Point Energy: {energy_out[0, 0].item()}, Lowest Harmonic Constraint: {H[:, 0, :].min().item()}, Start Mean Harmonic Constraint: {H[:, 0, :].mean().item()}")
            print(f"Final Lowest Energy Minima: {energies_trajectories.min().item()}, Lowest Harmonic Constraint: {H[:, 0, :].min().item()}, Start Mean Harmonic Constraint: {H[:, 0, :].mean().item()}")
        else:
            # print(f"Final Lowest Energy Minima: {energies_trajectories.min().item()} Starting Point Energy: {energy_out[:, 0].mean().item()} Average Distance: {torch.norm(latent_trajectories[:, 0, :] - start_points, dim=-1).mean().item()}, Lowest Harmonic Constraint: {H[:, 0, :].min().item()}, Start Mean Harmonic Constraint: {H[:, 0, :].mean().item()}")
            # print(energies_trajectories.device)
            # print(start_points.device)
            # print(latent_trajectories.device)
            # print(H.device)
            print(f"Final Lowest Energy Minima: {energies_trajectories.min().item()}, Average Distance: {torch.norm(latent_trajectories[:, 0, :] - start_points.to('cpu'), dim=-1).mean().item()}, Lowest Harmonic Constraint: {H[:, 0, :].min().item()}, Start Mean Harmonic Constraint: {H[:, 0, :].mean().item()}")

        # Get lowest energy structures
        out = self.decoder(latent_trajectories[:, 0, :].to(self.device))

        # generate self
        self.start_latent_point = latent_trajectories[:, 0, :].to(self.device)

        # Format them correctly
        out = out.detach()
        print("OUT SHAPE:", out.shape)
        torsion = self.calc_angles(out[:, :self.PARAMS["torsion_main_n"]*2])
        print("TORSION:", self.PARAMS["torsion_main_n"]*2)
        bond_angle = self.calc_angles(out[:, self.PARAMS["torsion_main_n"]*2:self.PARAMS["bond_angle_main_n"]*2*2])
        print("BOND ANGLES:", self.PARAMS["torsion_main_n"]*2, self.PARAMS["bond_angle_main_n"]*2*2)
        preds = torch.cat((torsion, bond_angle, out[:, self.PARAMS["torsion_main_n"]*2*2:-3]), dim=-1)
        print("bond angles:", self.PARAMS["torsion_main_n"]*2*2)
        preds = torch.cat((self.dof_traj[0, :].unsqueeze(0), preds), dim=0)
        # torsion = self.calc_angles(out[:, :116])
        # bond_angle = self.calc_angles(out[:, 116:116*2])
        # preds = torch.cat((torsion, bond_angle, out[:, 116*2:-3]), dim=-1)
        # preds = torch.cat((self.dof_traj[0, :].unsqueeze(0), preds), dim=0)
        torch.save(preds, "mcmcOutStructures.pt")
        if start_points == None:
            torch.save(
                {
                    "latent_traj": latent_trajectories, 
                    "energy_traj": energies_trajectories,
                    "start_energy": energy_out[0, -1],
                    "start_latent": self.start_latent_point,
                    "harmonic_traj": H,
                },
                "mcmctest.pt")
        return latent_trajectories, energies_trajectories

    def mcmc_step(
        self, 
        latent_trajectories: torch.tensor,
        sigma: float,
        H: torch.Tensor,
        ):
        """Perform a single MCMC step

        PARAMS
        ------
            latent_trajectories: torch.Tensor
                Latent Points to walk along
            sigma: float
                Noise to add to the walk
            H: torch.Tensor
                Harmonic Constraint Tensor
        """
        # Generate random walk noise
        epsilon = torch.normal(0, sigma, size=(B, self.latent_dim), dtype=torch.double)

        # Apply noise to previous t-1 point
        x_t = latent_trajectories + epsilon

        # decode points
        out = self.decoder(x_t)

        # Prediction Information
        torsion = out[:, :self.PARAMS["torsion_main_n"]*2]
        torsion_sin = torsion[:, :self.PARAMS["torsion_main_n"]]
        torsion_cos = torsion[:, self.PARAMS["torsion_main_n"]:]
        bond_angle = out[:, self.PARAMS["bond_angle_main_n"]*2:self.PARAMS["bond_angle_main_n"]*2*2]
        bond_angle_sin = bond_angle[:, :self.PARAMS["bond_angle_main_n"]]
        bond_angle_cos = bond_angle[:, self.PARAMS["bond_angle_main_n"]:]
        bond_length = out[:, self.PARAMS["bond_angle_main_n"]*2*2:-3]
        # torsion = out[:, :116]
        # torsion_sin = torsion[:, :58]
        # torsion_cos = torsion[:, 58:]
        # bond_angle = out[:, 116:116*2]
        # bond_angle_sin = bond_angle[:, :58]
        # bond_angle_cos = bond_angle[:, 58:]
        # bond_length = out[:, 116*2:-3]

        # Grab ref2015 energy from output
        energy_all = self.energy_scale.inverse_scale(out[:, -self.energy_n:])
        energy_t = energy_all[:, -2]

        # Harmonic Change for if none of these are true if they are then it will change
        harmonic_change = torch.zeros((B, 1), dtype=torch.double)
        harmonic_loss = torch.zeros((1, B), dtype=torch.double)

        if self.partial_optimization:
            # Harmonic Constraint
            print("SELF PARTIAL CHECK")
            harmonic_penalty_sin_torsion = torch.tensor(0.0)    if not self.freeze_torsion     else torch.pow(torch.cdist(torch.sin(real_t_sin[:, self.torsion_index]),   torsion_sin[:, self.torsion_index], p=1),2)
            harmonic_penalty_cos_torsion = torch.tensor(0.0)    if not self.freeze_torsion     else torch.pow(torch.cdist(torch.cos(real_t_cos[:, self.torsion_index]),   torsion_cos[:, self.torsion_index], p=1),2)
            harmonic_penalty_sin_bond_angle = torch.tensor(0.0) if not self.freeze_bond_angle  else torch.pow(torch.cdist(torch.sin(real_ba_sin[:, self.torsion_index]),  bond_angle_sin[:, self.torsion_index],p=1),2)
            harmonic_penalty_cos_bond_angle = torch.tensor(0.0) if not self.freeze_bond_angle  else torch.pow(torch.cdist(torch.cos(real_ba_cos[:, self.torsion_index]),  bond_angle_cos[:, self.torsion_index],p=1),2)
            harmonic_penalty_bond_length = torch.tensor(0.0)    if not self.freeze_bond_length else torch.pow(torch.cdist(real_bond_length[:, self.torsion_index],        bond_length[:, self.torsion_index],p=1),2)
            harmonic_loss = harmonic_penalty_sin_torsion + harmonic_penalty_cos_torsion + harmonic_penalty_cos_bond_angle + harmonic_penalty_sin_bond_angle + harmonic_penalty_bond_length
            harmonic_loss /= self.torsion_index.size(0)*torch.sum(torch.tensor([self.freeze_torsion, self.freeze_torsion, self.freeze_bond_angle, self.freeze_bond_angle, self.freeze_bond_length]))
            if t == 1:
                H[:, 0, :] = harmonic_loss.unsqueeze(-1)
                H[:, 1, :] = harmonic_loss.unsqueeze(-1)
                harmonic_change = harmonic_loss.squeeze(0).unsqueeze(-1)
            else:
                harmonic_change = H[:, 1, :] - harmonic_loss.squeeze(0).unsqueeze(-1)
        if self.retain_structure:
            harmonic_penalty_sin_torsion     = torch.pow(torch.cdist(torch.exp(torch.sin(real_t_sin)),   torch.exp(torsion_sin), p=2), 2)   
            harmonic_penalty_cos_torsion     = torch.pow(torch.cdist(torch.exp(torch.cos(real_t_cos)),   torch.exp(torsion_cos), p=2), 2)   
            harmonic_penalty_sin_bond_angle  = torch.pow(torch.cdist(torch.exp(torch.sin(real_ba_sin)),  torch.exp(bond_angle_sin),p=2), 2) 
            harmonic_penalty_cos_bond_angle  = torch.pow(torch.cdist(torch.exp(torch.cos(real_ba_cos)),  torch.exp(bond_angle_cos),p=2), 2) 
            harmonic_penalty_bond_length     = torch.pow(torch.cdist(torch.exp(real_bond_length),        torch.exp(bond_length),p=2), 2)    
            harmonic_loss = harmonic_penalty_sin_torsion + harmonic_penalty_cos_torsion + harmonic_penalty_cos_bond_angle + harmonic_penalty_sin_bond_angle + harmonic_penalty_bond_length
            harmonic_loss /= 5*self.PARAMS["torsion_main_n"]
            if t == 1:
                H[:, 0, :] = harmonic_loss.unsqueeze(-1)
                H[:, 1, :] = harmonic_loss.unsqueeze(-1)
                harmonic_change = harmonic_loss.squeeze(0).unsqueeze(-1)
            else:
                harmonic_change = H[:, 1, :] - harmonic_loss.squeeze(0).unsqueeze(-1)

        return x_t, H, harmonic_change, harmonic_loss, energy_t

    def generate_constraint_termbond(self, scaffolds: torch.tensor, terminal: bool):
        """Use dE/d(Latent Space) to generate constructs with and without termbond constraint

        PARAMS
        ------
        scaffolds: torch.tensor
            Inputs from inpainting with random sampling
        terminal: bool
            Whether to use the terminal or not

        RETURNS
        -------
        with_constraint: Tensor (B, D)
            DoF Matrix that was generated using the terminal bond constraint
        without_constraint: Tensor (B, D)
            DoF Matrix that was generated without the terminal bond constraint
        """
        scatter_latent = scaffolds
        start_range = 0
        # Init gradient requirement on the latent points
        scatter_latent.requires_grad_()

        # Define optimizer you want
        if self.adam_optim:
            optimizer = torch.optim.AdamW([scatter_latent], lr=self.lr)
        elif self.sgd_optim:
            optimizer = torch.optim.SGD([scatter_latent], lr=self.lr, momentum=0.9)

        for step in range(start_range, self.steps):
            # zero grad
            optimizer.zero_grad()

            # get the structure from the latent point
            out = self.decoder(scatter_latent)

            # grab the energies (ref2015 score)
            energies = self.energy_scale.inverse_scale(out[:, -self.energy_n:]).to(self.device)
            energy = energies[:, -1]

            if terminal:
                term_bond_loss, bah_loss, ahd_loss, ocnh_loss, ncaco_loss, clash_penalty, coordinates = self.term_bond.terminal_peptide_loss(
                    dof_inputs = out,
                    real_term = torch.ones(out.size(0), device=out.device, dtype=out.dtype) * 1.4,
                    params = self.PARAMS,
                    foldtree_root = self.term_bond.calculate_root(self.PARAMS["protein_len"]),
                    device = out.device,
                )

                coords_inter = torch.cat(
                    (
                        coordinates[:, :, 2, :].unsqueeze(2), # N
                        coordinates[:, :, 0, :].unsqueeze(2), # CA
                        coordinates[:, :, 1, :].unsqueeze(2), # C
                        coordinates[:, :, 3, :].unsqueeze(2), # O

                    ), dim = 2,
                )

                if not self.contiguous:
                    ca_motif = coordinates[:, self.freeze_residues, 0, :] # (B, motif, 3)
                    ref_ca_motif = ref_coordinates[:, self.freeze_residues, 0, :] # (B, motif, 3)
                    #### This is a PLACEHOLDER 
                    #### TODO: Make it so that self.ca_motif_initial is constructed from a pased PDB
                    #### This means reworking how initial motif DoFs are grabbed as well.
                    #### This is for the next version. However for testing this will be used
                    #### The distance needs to be precalculated
                    # ca_motif_initial = torch.norm(
                    #     torch.tensor([-26.782, -15.341, -19.5], dtype=torch.float32) - torch.tensor([-27.428, -16.801, -10.125], dtype=torch.float32),
                    # ).repeat(ca_motif.size(0)).to(self.device)
                    ref_ca_motif_initial = torch.cdist(ref_ca_motif, ref_ca_motif)
                    ca_distance_motif = torch.cdist(ca_motif, ca_motif)
                    # ca_distance_motif_initial = torch.pow(ca_motif[:, 0, -1] - ca_motif_initial, 2)
                    ca_distance_motif_initial = torch.pow(ref_ca_motif_initial[:, 0, -1] - ca_distance_motif[:, 0, -1], 2)
                    ca_distance_motif_loss = torch.pow(ca_distance_motif_initial, 2)*self.lambda_2
                else:
                    ca_distance_motif_loss = torch.zeros(1, device=self.device)

                # Align points
                coord_B, coord_L, coord_A, coord_D = coords_inter.shape
                coords = coords_inter.reshape(coord_B, coord_L * coord_A, coord_D).squeeze(0)
                if len(coords.shape) == 2:
                    coords = coords.unsqueeze(0)
                interface_total_loss = torch.zeros(1, device=energy.device)

                # Clone terms
                ocnh_loss_term = ocnh_loss.clone()
                ahd_loss_term = ahd_loss.clone()
                bah_loss_term = bah_loss.clone()
                ncaco_loss_term = ncaco_loss.clone()
                term_bond_loss_term = term_bond_loss.clone()

                combo_angles = ahd_loss_term + bah_loss_term + ncaco_loss_term + ocnh_loss_term
            else: 
                term_bond_loss_term = torch.tensor([0.0], dtype=torch.float64, device=self.device).requires_grad_(True)
                ca_distance_motif_loss = torch.tensor([0.0], dtype=torch.float64, device=self.device).requires_grad_(True)
                combo_angles = torch.tensor([0.0], dtype=torch.float64, device=self.device).requires_grad_(True)

            # term_bond_coeff = 10 if step >= int(self.steps*0.6) else 1
            term_bond_coeff = 100 if step >= int(self.steps*0.5) else 10

            # loss = (energy + term_bond_coeff*term_bond_loss).mean()
            loss = (energy + self.lambda_2*ca_distance_motif_loss + term_bond_coeff*term_bond_loss_term + term_bond_coeff*combo_angles).mean()

            if step % 10 == 0:
                print(f"Step: {step}, QM: {energies[:, 0].mean().item()}, Ref2015: {energies[:, -1].mean().item()}, Terminal Bond: {term_bond_loss_term.mean().item()}, Loss: {loss.item()}")
            # back prop
            loss.backward()
            optimizer.step()

        out = out.detach().clone()
        torsion = self.calc_angles(out[:, :self.PARAMS["torsion_main_n"]*2])
        bond_angle = self.calc_angles(out[:, self.PARAMS["torsion_main_n"]*2:self.PARAMS["bond_angle_main_n"]*2*2])
        preds = torch.cat((torsion, bond_angle, out[:, self.PARAMS["torsion_main_n"]*2*2:-3]), dim=-1)

        return preds


    # def generate_constraint_termbond(self, B: int):
    #     """Use dE/d(Latent Space) to generate constructs with and without termbond constraint
    #
    #     PARAMS
    #     ------
    #     B: int
    #         Batch Size (number of training examples)
    #
    #     RETURNS
    #     -------
    #     with_constraint: Tensor (B, D)
    #         DoF Matrix that was generated using the terminal bond constraint
    #     without_constraint: Tensor (B, D)
    #         DoF Matrix that was generated without the terminal bond constraint
    #     """
    #     # First inference generating is with term bond constraints
    #     term_bond_loop = True
    #     redo = True
    #     # set up tensor of random init values
    #     scatter_latent = torch.zeros((B, self.latent_dim), dtype=torch.float64, device=self.device)
    #     scatter_latent.uniform_(-0.3824, 0.3802) # This is set up for model 7731
    #     # store original points
    #     original_points = scatter_latent.clone().detach()
    #     start_range = 0
    #     # Init gradient requirement on the latent points
    #     scatter_latent.requires_grad_()
    #
    #     # Define optimizer you want
    #     if self.adam_optim:
    #         optimizer = torch.optim.AdamW([scatter_latent], lr=self.lr)
    #     elif self.sgd_optim:
    #         optimizer = torch.optim.SGD([scatter_latent], lr=self.lr, momentum=0.9)
    #
    #     # while redo:
    #     start_time = time.time()
    #     while redo:
    #         for step in range(start_range, self.steps):
    #             # zero grad
    #             optimizer.zero_grad()
    #
    #             # get the structure from the latent point
    #             out = self.decoder(scatter_latent)
    #
    #             # Prediction Information
    #             # torsion = out[:, :self.PARAMS["torsion_main_n"]*2]
    #             # torsion_sin = torsion[:, :self.PARAMS["torsion_main_n"]]
    #             # torsion_cos = torsion[:, self.PARAMS["torsion_main_n"]:]
    #             # bond_angle = out[:, self.PARAMS["bond_angle_main_n"]*2:self.PARAMS["bond_angle_main_n"]*2*2]
    #             # bond_angle_sin = bond_angle[:, :self.PARAMS["bond_angle_main_n"]]
    #             # bond_angle_cos = bond_angle[:, self.PARAMS["bond_angle_main_n"]:]
    #             # bond_length = out[:, self.PARAMS["bond_angle_main_n"]*2*2:-3]
    #             # preds = torch.cat((torsion, bond_angle, out[:, self.PARAMS["bond_angle_main_n"]*2*2:-3]), dim=-1)
    #
    #             # torsion = out[:, :116]
    #             # torsion_sin = torsion[:, :58]
    #             # torsion_cos = torsion[:, 58:]
    #             # bond_angle = out[:, 116:116*2]
    #             # bond_angle_sin = bond_angle[:, :58]
    #             # bond_angle_cos = bond_angle[:, 58:]
    #             # bond_length = out[:, 116*2:-3]
    #             # preds = torch.cat((torsion, bond_angle, out[:, 116*2:-3]), dim=-1)
    #
    #             if (step == start_range) & redo:
    #                 mid_out = out.detach().clone()
    #                 torsion = self.calc_angles(mid_out[:, :self.PARAMS["torsion_main_n"]*2])
    #                 bond_angle = self.calc_angles(mid_out[:, self.PARAMS["torsion_main_n"]*2:self.PARAMS["bond_angle_main_n"]*2*2])
    #                 preds = torch.cat((torsion, bond_angle, mid_out[:, self.PARAMS["torsion_main_n"]*2*2:-3]), dim=-1)
    #                 original_decode_dofs = preds
    #
    #             # grab the energies (ref2015 score)
    #             energies = self.energy_scale.inverse_scale(out[:, -self.energy_n:]).to(self.device)
    #             energy = energies[:, -1]
    #
    #             if term_bond_loop:
    #                 # # N-C Terminal Peptide Bond Loss
    #                 # term_bond_loss = self.term_bond.terminal_peptide_loss(
    #                 #     dof_inputs = out,
    #                 #     real_term = torch.ones(out.size(0), device=out.device, dtype=out.dtype) * 1.4,
    #                 #     params = self.PARAMS,
    #                 #     foldtree_root = self.term_bond.calculate_root(self.PARAMS["protein_len"]),
    #                 #     device = out.device,
    #                 # )
    #                 term_bond_loss, bah_loss, ahd_loss, ocnh_loss, ncaco_loss, clash_penalty, coordinates = self.term_bond.terminal_peptide_loss(
    #                     dof_inputs = out,
    #                     real_term = torch.ones(out.size(0), device=out.device, dtype=out.dtype) * 1.4,
    #                     params = self.PARAMS,
    #                     foldtree_root = self.term_bond.calculate_root(self.PARAMS["protein_len"]),
    #                     device = out.device,
    #                 )
    #
    #                 coords_inter = torch.cat(
    #                     (
    #                         coordinates[:, :, 2, :].unsqueeze(2), # N
    #                         coordinates[:, :, 0, :].unsqueeze(2), # CA
    #                         coordinates[:, :, 1, :].unsqueeze(2), # C
    #                         coordinates[:, :, 3, :].unsqueeze(2), # O
    #
    #                     ), dim = 2,
    #                 )
    #
    #                 if not self.contiguous:
    #                     ca_motif = coordinates[:, self.freeze_residues, 0, :] # (B, motif, 3)
    #                     ref_ca_motif = ref_coordinates[:, self.freeze_residues, 0, :] # (B, motif, 3)
    #                     #### This is a PLACEHOLDER 
    #                     #### TODO: Make it so that self.ca_motif_initial is constructed from a pased PDB
    #                     #### This means reworking how initial motif DoFs are grabbed as well.
    #                     #### This is for the next version. However for testing this will be used
    #                     #### The distance needs to be precalculated
    #                     # ca_motif_initial = torch.norm(
    #                     #     torch.tensor([-26.782, -15.341, -19.5], dtype=torch.float32) - torch.tensor([-27.428, -16.801, -10.125], dtype=torch.float32),
    #                     # ).repeat(ca_motif.size(0)).to(self.device)
    #                     ref_ca_motif_initial = torch.cdist(ref_ca_motif, ref_ca_motif)
    #                     ca_distance_motif = torch.cdist(ca_motif, ca_motif)
    #                     # ca_distance_motif_initial = torch.pow(ca_motif[:, 0, -1] - ca_motif_initial, 2)
    #                     ca_distance_motif_initial = torch.pow(ref_ca_motif_initial[:, 0, -1] - ca_distance_motif[:, 0, -1], 2)
    #                     ca_distance_motif_loss = torch.pow(ca_distance_motif_initial, 2)*self.lambda_2
    #                 else:
    #                     ca_distance_motif_loss = torch.zeros(1, device=self.device)
    #
    #                 # Align points
    #                 coord_B, coord_L, coord_A, coord_D = coords_inter.shape
    #                 coords = coords_inter.reshape(coord_B, coord_L * coord_A, coord_D).squeeze(0)
    #                 if len(coords.shape) == 2:
    #                     coords = coords.unsqueeze(0)
    #                 interface_total_loss = torch.zeros(1, device=energy.device)
    #
    #                 # Clone terms
    #                 ocnh_loss_term = ocnh_loss.clone()
    #                 ahd_loss_term = ahd_loss.clone()
    #                 bah_loss_term = bah_loss.clone()
    #                 ncaco_loss_term = ncaco_loss.clone()
    #                 term_bond_loss_term = term_bond_loss.clone()
    #
    #
    #                 combo_angles = ahd_loss_term + bah_loss_term + ncaco_loss_term + ocnh_loss_term
    #             else: 
    #                 term_bond_loss_term = torch.tensor([0.0], dtype=torch.float64, device=self.device).requires_grad_(True)
    #                 ca_distance_motif_loss = torch.tensor([0.0], dtype=torch.float64, device=self.device).requires_grad_(True)
    #                 combo_angles = torch.tensor([0.0], dtype=torch.float64, device=self.device).requires_grad_(True)
    #
    #             # term_bond_coeff = 10 if step >= int(self.steps*0.6) else 1
    #             term_bond_coeff = 100 if step >= int(self.steps*0.5) else 10
    #
    #             # loss = (energy + term_bond_coeff*term_bond_loss).mean()
    #             loss = (energy + self.lambda_2*ca_distance_motif_loss + term_bond_coeff*term_bond_loss_term + term_bond_coeff*combo_angles).mean()
    #
    #             if step % 10 == 0:
    #                 print(f"Step: {step}, QM: {energies[:, 0].mean().item()}, Ref2015: {energies[:, -1].mean().item()}, Terminal Bond: {term_bond_loss.mean().item()}, Loss: {loss.item()}")
    #             # back prop
    #             loss.backward()
    #             optimizer.step()
    #
    #         # Update Time
    #         batch_time = round(time.time() - start_time, 3)
    #         print()
    #         print('-------------------')
    #         print(f"Batch {B} Time: {batch_time}")
    #         print('-------------------')
    #         print()
    #
    #         if term_bond_loop:
    #             out = out.detach().clone()
    #             torsion = self.calc_angles(out[:, :self.PARAMS["torsion_main_n"]*2])
    #             bond_angle = self.calc_angles(out[:, self.PARAMS["torsion_main_n"]*2:self.PARAMS["bond_angle_main_n"]*2*2])
    #             preds = torch.cat((torsion, bond_angle, out[:, self.PARAMS["torsion_main_n"]*2*2:-3]), dim=-1)
    #             with_constraint = preds.cpu().detach()
    #             term_bond_loop = False
    #             scatter_latent = original_points.clone()
    #             scatter_latent.requires_grad_(True)
    #             # Define optimizer you want
    #             if self.adam_optim:
    #                 optimizer = torch.optim.AdamW([scatter_latent], lr=self.lr)
    #             elif self.sgd_optim:
    #                 optimizer = torch.optim.SGD([scatter_latent], lr=self.lr, momentum=0.9)
    #             term_time = batch_time
    #         else:
    #             out = out.detach().clone()
    #             torsion = self.calc_angles(out[:, :self.PARAMS["torsion_main_n"]*2])
    #             bond_angle = self.calc_angles(out[:, self.PARAMS["torsion_main_n"]*2:self.PARAMS["bond_angle_main_n"]*2*2])
    #             preds = torch.cat((torsion, bond_angle, out[:, self.PARAMS["torsion_main_n"]*2*2:-3]), dim=-1)
    #             without_constraint = preds.cpu().detach()
    #             redo = False
    #             non_term_time = batch_time
    #         
    #             # Save dataset
    #             torch.save(
    #                 {
    #                     "minimized_dofs": with_constraint,
    #                     "without_constraint": without_constraint,
    #                     "terminalconstraint_time": term_time,
    #                     "non_terminalconstraint_time": non_term_time,
    #                     "original_decode_dofs": original_decode_dofs,
    #                 }, f"termbond_original_vs_minimze_data_{B}_{datetime.date.today()}.pt"
    #             )

    def energy_steps(self, start_points: torch.Tensor = None, mcmc_minimize: bool = False, minimize_steps: int = 1000,
                     interface_full_dict = None, interface_cb_dict = None,
                     motif_dofs = None):
        """Model Eval Loop for GDS"""
        if start_points.size(0) == 1:
            self.start_latent_point = self.encoder(start_points).detach()
            start_range = 0
            self.path = torch.zeros((start_points.size(0),self.steps), device=self.device)
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
            # self.dof_traj = torch.zeros((start_points.size(0), self.steps, self.start_point_scaled.size(-1) - self.energy_n), dtype=torch.float64, device=self.device)
            self.dof_traj = torch.zeros((start_points.size(0), self.steps, self.dof_size - self.energy_n), dtype=torch.float64, device=self.device)
            start_range = 0
            # self.dof_traj[:, 0, :] = start_points[:, :-self.energy_n]
        original_point = self.start_latent_point.clone()
        original_shape = self.start_latent_point.shape[0]
        total_mask = torch.ones(self.start_latent_point.shape[0], dtype=torch.bool, device=self.start_latent_point.device)
        self.start_latent_point.requires_grad_()
        # Set our while minimize term
        minimize_still = True
        redo = 0

        # # Define optimizer you want
        # if self.adam_optim:
        #     optimizer = torch.optim.AdamW([self.start_latent_point], lr=self.lr)
        # elif self.sgd_optim:
        #     optimizer = torch.optim.SGD([self.start_latent_point], lr=self.lr, momentum=0.9)
        # if not self.contiguous:
        #     _, _, _, _, _, _, ref_coordinates = self.term_bond.terminal_peptide_loss(
        #         dof_inputs = self.start_point_dof.unsqueeze(0).to(self.device),
        #         real_term = torch.ones(1, device=out.device, dtype=out.dtype) * 1.4,
        #         params = self.ORIG_PARAMS,
        #         foldtree_root = self.term_bond.calculate_root(self.ORIG_PARAMS["protein_len"]),
        #         device = self.device,
        #         control = True,
        #     )

        # if self.sample_energy.size(0) == 3:
        #     print(f"Step: 0, QM: {self.sample_energy[0].item()}, Ref2015: {self.sample_energy[1].item()}, Ref2015+Solvent: {self.sample_energy[-1].item()}, Data Taken Straight From Input")
        # else:
        #     print(f"Step: 0, QM: {self.sample_energy[0].item()}, Ref2015: {self.sample_energy[-1].item()}, Data Taken Straight From Input")
        # # add energy path
        # if self.sample_energy.size(-1) == 1:
        #     self.energy_path[0] = self.sample_energy
        # else:
        #     self.energy_path[0] = self.sample_energy[1]

        while minimize_still: 
            original_point = self.start_latent_point.clone()
            # Define optimizer you want
            if self.adam_optim:
                optimizer = torch.optim.AdamW([self.start_latent_point], lr=self.lr)
            elif self.sgd_optim:
                optimizer = torch.optim.SGD([self.start_latent_point], lr=self.lr, momentum=0.9)

            for step in range(start_range, self.steps):
                # zero grad
                optimizer.zero_grad()

                # get the structure from the latent point
                out = self.decoder(self.start_latent_point)

                # Prediction Information
                torsion = out[:, :self.PARAMS["torsion_main_n"]*2]
                torsion_sin = torsion[:, :self.PARAMS["torsion_main_n"]]
                torsion_cos = torsion[:, self.PARAMS["torsion_main_n"]:]
                bond_angle = out[:, self.PARAMS["bond_angle_main_n"]*2:self.PARAMS["bond_angle_main_n"]*2*2]
                bond_angle_sin = bond_angle[:, :self.PARAMS["bond_angle_main_n"]]
                bond_angle_cos = bond_angle[:, self.PARAMS["bond_angle_main_n"]:]
                bond_length = out[:, self.PARAMS["bond_angle_main_n"]*2*2:-3]
                preds = torch.cat((torsion, bond_angle, out[:, self.PARAMS["bond_angle_main_n"]*2*2:-3]), dim=-1)
                # torsion = out[:, :116]
                # torsion_sin = torsion[:, :58]
                # torsion_cos = torsion[:, 58:]
                # bond_angle = out[:, 116:116*2]
                # bond_angle_sin = bond_angle[:, :58]
                # bond_angle_cos = bond_angle[:, 58:]
                # bond_length = out[:, 116*2:-3]
                # preds = torch.cat((torsion, bond_angle, out[:, 116*2:-3]), dim=-1)

                # Real Information
                real_t_sin = torch.sin(motif_dofs[0])
                real_t_cos = torch.cos(motif_dofs[0])
                real_ba_sin = torch.sin(motif_dofs[1])
                real_ba_cos = torch.cos(motif_dofs[1])
                real_bond_length = motif_dofs[2] 
                # real_t_sin = torch.sin(torch.deg2rad(self.start_point[:, :self.ORIG_PARAMS["torsion_main_n"]]))
                # real_t_cos = torch.cos(torch.deg2rad(self.start_point[:, :self.ORIG_PARAMS["torsion_main_n"]]))
                # real_ba_sin = torch.sin(torch.deg2rad(self.start_point[:, self.ORIG_PARAMS["bond_angle_main_n"]:2*self.ORIG_PARAMS["bond_angle_main_n"]]))
                # real_ba_cos = torch.cos(torch.deg2rad(self.start_point[:, self.ORIG_PARAMS["bond_angle_main_n"]:2*self.ORIG_PARAMS["bond_angle_main_n"]]))
                # real_bond_length = self.start_point[:, self.ORIG_PARAMS["bond_angle_main_n"]*2:]

                # grab the energies (ref2015 score)
                # energies = self.energy_scale.inverse_scale(out[:, -self.energy_n:]).to(self.device)
                energies = self.energy_scale.inverse_scale(out[:, -self.energy_n:]).to(self.device)
                energies = energies[:, :2]
                # print("ENRGIES:", energies)
                # print('OUT ENERGY:', out[:, -self.energy_n].mean())
                # print('energy number:', self.energy_n)
                # print('out shape', out.shape)
                if energies.size(-1) == 3:
                    print("--------------------")
                    print("--------------------")
                    print("--------------------")
                    print("HELLLOOOOO")
                    print("WRONG AREA")
                    print("--------------------")
                    print("--------------------")
                    print("--------------------")
                    qm = energies[:, 0]
                    energy = energies[:, 1]

                    dist = torch.norm(original_point - self.start_latent_point, dim=-1)
                    if self.partial_optimization:
                        harmonic_loss = self.harmonic_diff_inpaint(
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
                        )

                    elif self.retain_structure and (not self.partial_optimization):
                        harmonic_loss = self.harmonic_diff_inpaint(
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
                        )

                    else:
                        harmonic_loss = torch.tensor([0.0], device=self.device)

                    term_bond_loss, bah_loss, ahd_loss, ocnh_loss, ncaco_loss, clash_penalty = self.term_bond.terminal_peptide_loss(
                        dof_inputs = out,
                        real_term = torch.ones(out.size(0), device=out.device, dtype=out.dtype) * 1.4,
                        params = self.PARAMS,
                        foldtree_root = self.term_bond.calculate_root(self.PARAMS["protein_len"]),
                        device = out.device,
                    )
                    term_bond_coeff = 10 if step >= int(self.steps*0.5) else 1
                    # print("OCNH:", ocnh_loss)
                    # print("NCaCO:", ncaco_loss)

                    # print(clash_penalty.shape)
                    B_cp, N_cp, _ = clash_penalty.shape
                    clash_penalty = clash_penalty.sum(dim=(-1,-2)) * (1/(N_cp * N_cp))



                    loss = (energy + dist + self.lambda_2*harmonic_loss + term_bond_coeff*term_bond_loss + term_bond_coeff*bah_loss + term_bond_coeff*ahd_loss + 0*ocnh_loss + 0*ncaco_loss + torch.tensor(0.0)*clash_penalty).mean()
                    # # N-C Terminal Peptide Bond Loss
                    # term_bond_loss, bah_loss, ahd_loss = self.term_bond.terminal_peptide_loss(
                    #     dof_inputs = out,
                    #     real_term = torch.ones(out.size(0), device=out.device, dtype=out.dtype) * 1.4,
                    #     params = self.PARAMS,
                    #     foldtree_root = 4,
                    #     device = out.device,
                    # )
                    # term_bond_coeff = 10 if step >= int(self.steps*0.5) else 1
                    #
                    #
                    # loss = (energy + dist + self.lambda_2*harmonic_loss + term_bond_coeff*term_bond_loss + term_bond_coeff*bah_loss + term_bond_coeff*ahd_loss).mean()

                    self.path[step] = dist.detach()
                    self.energy_path[step] = energy.detach()
                    print(f"Step: {step}, QM: {energies[:, 0].item()}, Ref2015: {energies[:, 1].item()}, Dist: {dist.item()}, Harmonic Penalty: {harmonic_loss.item()}, Terminal Bond: {term_bond_loss.item()}, Loss: {loss.item()}")
                    # print(f"Step: {step}, QM: {energies[:, 0].item()} Ref2015: {energies[:, 1].item()}, Ref2015+Solvent: {energies[:, -1].item()}, Dist: {dist.item()}, Harmonic Penalty: {harmonic_loss.item()}, Loss: {loss.item()}")
                else:
                    energy = energies[:, -1]


                    dist = torch.norm(original_point - self.start_latent_point, dim=-1)
                        # harmonic_loss = self.harmonic_diff_inpaint(
                        # real_t_sin,
                        # real_t_cos,
                        # real_ba_sin,
                        # real_ba_cos,
                        # real_bond_length,
                        # torsion_sin,
                        # torsion_cos,
                        # bond_angle_sin,
                        # bond_angle_cos,
                        # bond_length,
                        # )

                    # elif self.retain_structure and (not self.partial_optimization):
                    #     harmonic_loss = self.harmonic_diff_inpaint(
                    #     real_t_sin,
                    #     real_t_cos,
                    #     real_ba_sin,
                    #     real_ba_cos,
                    #     real_bond_length,
                    #     torsion_sin,
                    #     torsion_cos,
                    #     bond_angle_sin,
                    #     bond_angle_cos,
                    #     bond_length,
                    #     )


                    # if step >= int(self.steps*0.8):
                    # N-C Terminal Peptide Bond Loss
                    # term_bond_loss = self.term_bond.terminal_peptide_loss(
                    #     dof_inputs = out,
                    #     real_term = torch.ones(out.size(0), device=out.device, dtype=out.dtype) * 1.4,
                    #     params = self.PARAMS,
                    #     foldtree_root = 4,
                    #     device = out.device,
                    # )
                    # # else: term_bond_loss = torch.tensor([0.0], dtype=torch.float64, device=self.device)
                    # term_bond_coeff = 10 if step >= int(self.steps*0.5) else 1
                    #
                    # loss = (energy + dist + self.lambda_2*harmonic_loss + term_bond_coeff*term_bond_loss).mean()
                    term_bond_loss, bah_loss, ahd_loss, ocnh_loss, ncaco_loss, HNCO, coordinates = self.term_bond.terminal_peptide_loss(
                        dof_inputs = out,
                        real_term = torch.ones(out.size(0), device=out.device, dtype=out.dtype) * 1.4,
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
                        harmonic_loss, _ = self.harmonic_diff_inpaint(
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
                            True,
                        )
                        # Calculate motif numbering
                        # HERE
                        self.motif, self.motif_input = calculate_motif_numbering_for_kabsch(
                            self.match_res_mask,
                            self.PARAMS["protein_len"],
                            self.freeze_residues.shape[0],
                            self.freeze_residues,
                            True,
                        )
                    else:
                        harmonic_loss = torch.tensor([0.0], device=self.device)

                    term_bond_coeff = 1
                    # term_bond_coeff = 100 if step >= int(self.steps*0.5) else 10


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



                        # interface_dict = interface_cb_dict if (step < self.steps//4) and (self.ramp_constraint) else interface_full_dict
                        # if step == start_range:
                            # aligned_coordinates = kabsch_algorithm(P = coords, Q = interface_dict["peptide"].double())
                            # print(interface_dict["peptide"].shape)
                            # print("coords:", coords.shape)
                        peptide_nonhydrogen_mask = interface_dict["epsilon_peptide"] != 0.901681
                        # print("MASK HYDROGENS:", peptide_nonhydrogen_mask)
                        # print("SUM MASK:", peptide_nonhydrogen_mask.sum())
                        if not self.contiguous:
                            aligned_coordinates = kabsch_rootatom(P = coords, Q = interface_dict["peptide"][peptide_nonhydrogen_mask].double(),
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
                            aligned_coordinates = kabsch_rootatom(P = coords, Q = interface_dict["peptide"][peptide_nonhydrogen_mask].double(),
                                                                  motif_output = self.motif,
                                                                  motif_input = self.motif_input,
                                                                  )


                        interface_nonhydrogen_mask = interface_dict["epsilon_interface"] != 0.901681
                        interface_nonorm_loss, energy_mask = interface_loss(interface_atoms = interface_dict["interface_atoms"][interface_nonhydrogen_mask].double(), recomputed_xyz = aligned_coordinates,
                                                              sigma_peptide = interface_dict["sigma_peptide"][peptide_nonhydrogen_mask],
                                                              epsilon_peptide = interface_dict["epsilon_peptide"][peptide_nonhydrogen_mask],
                                                              sigma_interface = interface_dict["sigma_interface"][interface_nonhydrogen_mask],
                                                              epsilon_interface = interface_dict["epsilon_interface"][interface_nonhydrogen_mask])

                        interface_total_loss = interface_nonorm_loss #/ aligned_coordinates.size(0)

                    else:
                        interface_total_loss = torch.zeros(1, device=energy.device)


                    # Clone terms
                    energy_pred = energy.clone()
                    # energy_pred = out[:, -self.energy_n].mean()
                    ocnh_loss_term = ocnh_loss.clone()
                    ahd_loss_term = ahd_loss.clone()
                    bah_loss_term = bah_loss.clone()
                    ncaco_loss_term = ncaco_loss.clone()
                    term_bond_loss_term = term_bond_loss.clone()

                    # Set terms to zero if necessary
                    # energy_pred[energy_mask] *= 0.0
                    # print(energy_pred)
                    # ahd_loss_term[energy_mask, :] *= 0.0
                    # bah_loss_term[energy_mask, :] *= 0.0
                    # term_bond_loss_term[energy_mask, :] *= 0.0
                    # clash_penalty = interface_loss(aligned_coordinates, aligned_coordinates, d_min = 1.0, d_max = 10.0) #/ aligned_coordinates.size(1)

                    # if step == start_range:
                    #     interface_total_loss, previous_distances = interface_loss(interface_atoms = interface_dict["interface_atoms"].double(), recomputed_xyz = aligned_coordinates, previous_distances = None, alpha = 1.0, beta=2.0, d_min = 2.5, d_max = 9.0, sigma=4) #/ interface_dict["interface_atoms"].size(0)
                    #     # interface_total_loss = interface_total_loss / interface_dict["interface_atoms"].size(0)
                    # else:
                    #     interface_total_loss, previous_distances = interface_loss(interface_atoms = interface_dict["interface_atoms"].double(), recomputed_xyz = aligned_coordinates, previous_distances = previous_distances, alpha = 1.0, beta=2.0, d_min = 2.5, d_max = 9.0, sigma=4) #/ interface_dict["interface_atoms"].size(0)
                    #     # interface_total_loss = interface_total_loss / interface_dict["interface_atoms"].size(0)
                    # print('INTERFACE LOSS SHAPE:', interface_total_loss.shape)
                    # clash_penalty = interface_loss(aligned_coordinates, aligned_coordinates, d_min = 1.0, d_max = 10.0) #/ aligned_coordinates.size(1)

                    combo_angles = ahd_loss_term + bah_loss_term + ncaco_loss_term + ocnh_loss_term

                    # loss = (energy_pred + 0*dist + self.lambda_2*harmonic_loss + term_bond_coeff*term_bond_loss_term + term_bond_coeff*10*bah_loss_term + term_bond_coeff*10*ahd_loss_term + 10*interface_total_loss + 0*clash_penalty).mean()
                    if redo == 0:
                        # loss = (energy_pred + 0*dist + self.lambda_2*harmonic_loss + term_bond_coeff*term_bond_loss_term + term_bond_coeff*10*bah_loss_term + term_bond_coeff*10*ahd_loss_term + interface_total_loss + 0*clash_penalty).mean()
                        # loss = (energy_pred + 0*dist + self.lambda_2*harmonic_loss + term_bond_coeff*term_bond_loss_term + term_bond_coeff*100*ocnh_loss_term + term_bond_coeff*100*combo_angles + interface_total_loss + 0*clash_penalty).mean()
                        loss = (energy_pred + self.lambda_2*harmonic_loss + term_bond_coeff*discontig_loss + term_bond_coeff*term_bond_loss_term + term_bond_coeff*combo_angles + interface_total_loss).mean()
                        # loss = (energy_pred + self.lambda_2*harmonic_loss + term_bond_coeff*term_bond_loss_term + term_bond_coeff*ocnh_loss_term + term_bond_coeff*interface_total_loss + 0*clash_penalty).mean()
                    else:
                        loss = (term_bond_coeff*term_bond_loss_term + interface_total_loss*(redo+1)).mean()
                    # loss = (energy + dist + (term_bond_coeff*100)*harmonic_loss + term_bond_coeff*term_bond_loss + term_bond_coeff*bah_loss + term_bond_coeff*ahd_loss + 0*ocnh_loss + 0*ncaco_loss + 0*clash_penalty).mean()
                    # loss = (energy + dist + self.lambda_2*harmonic_loss + term_bond_coeff*term_bond_loss + term_bond_coeff*bah_loss + term_bond_coeff*ahd_loss + term_bond_coeff*ocnh_loss + torch.tensor(0.0)*ncaco_loss).mean()
                    # loss = (energy + dist + self.lambda_2*harmonic_loss + term_bond_coeff*term_bond_loss + 2*bah_loss + 2*ahd_loss + 2*ocnh_loss + torch.tensor(0.0)*ncaco_loss).mean()


                    if mcmc_minimize:
                        self.path[:, step] = dist.detach()
                        self.energy_path[:, step] = energy.detach()
                        print(f"Step: {step}, QM: {energies[:, 0].mean().item()}, Ref2015: {energies[:, -1].mean().item()}, Dist: {dist.mean().item()}, Harmonic Penalty: {harmonic_loss.mean().item()}, Terminal Bond: {term_bond_loss.mean().item()}, AHD: {ahd_loss.mean().item()}, BAH: {bah_loss.mean().item()}, Interface Loss: {interface_total_loss.mean().item()}, Lowest Harmonic Penalty: {harmonic_loss.min().item()}, Loss: {loss.item()}")
                    elif start_points == None:
                        self.path[step] = dist.detach()
                        self.energy_path[step] = energy.detach()
                        # print(f"Step: {step}, QM: {energies[:, 0].item()}, Ref2015: {energies[:, -1].item()}, Dist: {dist.item()}, Harmonic Penalty: {harmonic_loss.item()}, Terminal Bond: {term_bond_loss.item()}, AHD: {ahd_loss.item()}, BAH: {bah_loss.item()}, CaCNCa Planar: {ncaco_loss.item()}, LJ Loss: {clash_penalty.item()}, Lowest Harmonic Penalty: {harmonic_loss.item()}, Loss: {loss.item()}")
                        # print(f"Step: {step}, QM: {energies[:, 0].item()}, Ref2015: {energies[:, -1].item()}, Dist: {dist.item()}, Harmonic Penalty: {harmonic_loss.item()}, Terminal Bond: {term_bond_loss.item()}, AHD: {ahd_loss.item()}, BAH: {bah_loss.item()}, LJ penalty: {clash_penalty.item()}, Interface Loss: {interface_total_loss.mean().item()}, Lowest Harmonic Penalty: {harmonic_loss.item()}, Loss: {loss.item()}")
                        print(f"Step: {step}, QM: {energies[:, 0].item()}, Ref2015: {energy.item()}, Dist: {dist.item()}, Harmonic Penalty: {harmonic_loss.item()}, Terminal Bond: {term_bond_loss.item()}, AHD: {ahd_loss.item()}, BAH: {bah_loss.item()}, LJ penalty: {clash_penalty.item()}, Interface Loss: {interface_total_loss.mean().item()}, Lowest Harmonic Penalty: {harmonic_loss.item()}, Loss: {loss.item()}")
                        # print(f"Step: {step}, QM: {energies[:, 0].item()}, Ref2015: {energies[:, -1].item()}, Dist: {dist.item()}, Harmonic Penalty: {harmonic_loss.item()}, Terminal Bond: {term_bond_loss.item()}, Loss: {loss.item()}")
                    else:
                        self.path[total_mask, step] = dist.detach().to(torch.float32)
                        self.energy_path[total_mask, step] = energy.detach().to(torch.float32)
                        # self.path[:, step] = dist.detach()
                        # self.energy_path[:, step] = energy.detach()
                        print(f"Step: {step}, QM: {energies[:, 0].mean().item()}, Ref2015: {energies[:, -1].mean().item()}, Dist: {dist.mean().item()}, Harmonic Penalty: {harmonic_loss.mean().item()}, Discontig Loss: {discontig_loss.mean().item()} Terminal Bond: {term_bond_loss.mean().item()}, OCNH Plane: {ocnh_loss_term.mean().item()} AHD: {ahd_loss.mean().item()}, BAH: {bah_loss.mean().item()}, Interface Loss: {interface_total_loss.mean().item()}, Lowest Harmonic Penalty: {harmonic_loss.min().item()}, Loss: {loss.item()}")
                        # print(f"Step: {step}, QM: {energies[:, 0].mean().item()}, Ref2015: {energies[:, -1].mean().item()}, Dist: {dist.mean().item()}, Harmonic Penalty: {harmonic_loss.mean().item()}, Terminal Bond: {term_bond_loss.mean().item()}, AHD: {ahd_loss.mean().item()}, BAH: {bah_loss.mean().item()}, HNCO Planar: {ocnh_loss.mean().item()}, LJ Loss: {clash_penalty.mean().item()}, Lowest Harmonic Penalty: {harmonic_loss.min().item()}, Loss: {loss.item()}")
                        # print(f"Step: {step}, QM: {energies[:, 0].mean().item()}, Ref2015: {energies[:, -1].mean().item()}, Dist: {dist.mean().item()}, Harmonic Penalty: {harmonic_loss.mean().item()}, Terminal Bond: {term_bond_loss.mean().item()}, Lowest Harmonic Penalty: {harmonic_loss.min().item()} Loss: {loss.item()}")

                # back prop
                loss.backward()
                optimizer.step()

                # if (step == self.steps-1) and (sum(interface_total_loss >=10) == 0):
                #     minimize_still = False
                # elif (step == self.steps-1):
                #     clash_mask = interface_total_loss > 10
                #     if redo == 0:
                #         indices = torch.argwhere(clash_mask).flatten()
                #     inverse_mask = clash_mask == False
                #     print(f"Number of successful outputs (based on Interface non-clashing < 10): {sum(inverse_mask)} proportion: {sum(inverse_mask) / inverse_mask.flatten().shape[0]}")
                #     out = out.detach().clone()
                #     torsion = self.calc_angles(out[inverse_mask, :self.PARAMS["torsion_main_n"]*2])
                #     bond_angle = self.calc_angles(out[inverse_mask, self.PARAMS["torsion_main_n"]*2:self.PARAMS["bond_angle_main_n"]*2*2])
                #     preds = torch.cat((torsion, bond_angle, out[inverse_mask, self.PARAMS["torsion_main_n"]*2*2:-self.energy_n]), dim=-1)
                #     self.start_latent_point = self.start_latent_point[clash_mask].detach().requires_grad_()
                #     ### WHAT WE NEED HERE IS AN OVERALL MASK THAT KEEPS TRACK OF THE CURRENT BAD THINGS
                #     if redo > 0:
                #         print(clash_mask)
                #         print(indices)
                #         print(indices[clash_mask])
                #         indices = indices[clash_mask == True]
                #         total_mask = torch.zeros(original_shape, dtype=torch.bool, device=self.start_latent_point.device)
                #         total_mask[indices] = True
                #     else:
                #         total_mask = torch.zeros(original_shape, dtype=torch.bool, device=self.start_latent_point.device)
                #         total_mask[indices] = True
                #     print("NUMBER OF CLASHING POINTS:", self.start_latent_point.shape)
                #     redo += 1
                if (step == self.steps-1):
                    minimize_still = False
                    # clash_mask = interface_total_loss > 20
                    if interface_total_loss.size(0) != 1:
                        clash_mask = interface_total_loss > self.interface_cutoff
                        if redo == 0:
                            indices = torch.argwhere(clash_mask).flatten()
                        inverse_mask = clash_mask == False
                        print(f"Number of successful outputs (based on Interface non-clashing < {self.interface_cutoff}): {sum(inverse_mask)} proportion: {sum(inverse_mask) / inverse_mask.flatten().shape[0]}")
                        out = out.detach().clone()
                        torsion = self.calc_angles(out[inverse_mask, :self.PARAMS["torsion_main_n"]*2])
                        bond_angle = self.calc_angles(out[inverse_mask, self.PARAMS["torsion_main_n"]*2:self.PARAMS["bond_angle_main_n"]*2*2])
                        preds = torch.cat((torsion, bond_angle, out[inverse_mask, self.PARAMS["torsion_main_n"]*2*2:-self.energy_n]), dim=-1)
                    else:
                        out = out.detach().clone()
                        torsion = self.calc_angles(out[:, :self.PARAMS["torsion_main_n"]*2])
                        bond_angle = self.calc_angles(out[:, self.PARAMS["torsion_main_n"]*2:self.PARAMS["bond_angle_main_n"]*2*2])
                        preds = torch.cat((torsion, bond_angle, out[:, self.PARAMS["torsion_main_n"]*2*2:-self.energy_n]), dim=-1)
                else:
                    out = out.detach().clone()
                    torsion = self.calc_angles(out[:, :self.PARAMS["torsion_main_n"]*2])
                    bond_angle = self.calc_angles(out[:, self.PARAMS["torsion_main_n"]*2:self.PARAMS["bond_angle_main_n"]*2*2])
                    preds = torch.cat((torsion, bond_angle, out[:, self.PARAMS["torsion_main_n"]*2*2:-self.energy_n]), dim=-1)



        # out = out.detach().clone()
        # torsion = self.calc_angles(out[:, :self.PARAMS["torsion_main_n"]*2])
        # bond_angle = self.calc_angles(out[:, self.PARAMS["torsion_main_n"]*2:self.PARAMS["bond_angle_main_n"]*2*2])
        # preds = torch.cat((torsion, bond_angle, out[:, self.PARAMS["torsion_main_n"]*2*2:-self.energy_n]), dim=-1)
        # torsion = self.calc_angles(out[:, :116])
        # bond_angle = self.calc_angles(out[:, 116:116*2])
        # preds = torch.cat((torsion, bond_angle, out[:, 116*2:-3]), dim=-1)

                if mcmc_minimize:
                    self.dof_traj[:, step] = preds
                elif start_points == None:
                    self.dof_traj[step] = preds
                else:
                    if (interface_total_loss.size(0) != 1) and (step == self.steps-1):
                        self.dof_traj[inverse_mask, step] = preds
                        self.dof_traj = self.dof_traj[inverse_mask]
                    else:
                        self.dof_traj[:, step] = preds


    def output_GDS(
        self, 
        dofsummary: DoFSummary,
        skip_individual = False,
        silentfile: bool = False,
        score: bool = False,
        output_path: str = "./", 
        random_outputs: bool = False,
        minimize_pose: bool = False,
        ) -> int:
        """Output the GDS trajectory"""
        # file name
        if (len(self.dof_traj.shape) == 3) & (not random_outputs) & (not skip_individual):
            for sample in range(self.dof_traj.size(0)):
                file = f"finalstep{self.outstep}-{datetime.date.today()}-sample{sample}-trajectories.pt"
                out = self.dof_traj[sample]
                torch.save(out, 
                           os.path.join(output_path, file)
                           )
            file = f"finalstep{self.outstep}-{datetime.date.today()}-sampleall-trajectories.pt"
            torch.save(self.dof_traj[:, -1, :], 
                       os.path.join(output_path, file)
                       )
        elif random_outputs or skip_individual:
            file = f"finalstep{self.outstep}-{datetime.date.today()}-trajectories.pt"
            torch.save(self.dof_traj[:, -1, :], 
                       os.path.join(output_path, file)
                       )
        else:
            file = f"finalstep{self.outstep}-{datetime.date.today()}-trajectories.pt"
            torch.save(self.dof_traj, 
                       os.path.join(output_path, file)
                       )

        # Now thread and place our DoFs back in the context we took it from
        if self.dof_traj.size(0) == 1:
            thread_data = {'x': self.dof_traj[0, :, :],
                           'params': self.PARAMS,}
        else:
            thread_data = {'x': self.dof_traj[:, -1, :],
                           'params': self.PARAMS,}
        dofsummary.thread_dofs_to_poses(
            thread_data,
            silentfile = silentfile,
            score_outputs = score,
            output_path = output_path,
            random_outputs = random_outputs,
            minimize_pose = minimize_pose,
        )

        return 0

        

if __name__ == "__main__":
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--steps", type=int, default=1000, help="Search Steps Max")
    p.add_argument("--minima", type=float, default=-4.775, help="Minima energy to search for")
    p.add_argument("--prefix-outfile", type=str, default="test", help="PDB/SilentFile prefix")
    p.add_argument("--silentfile", action="store_true", help="Save Model outputs in a SilentFile format \
                        if not specified then individual PDBs are produced.")
    p.add_argument("--score-outputs", action="store_true", help="Score each Model output and save in \
                        SilentFile (not implemented yet to save score to separate file when Silent not set).")
    p.add_argument("--output-path", type=str, default="./", help="Determines location PDBs/SilentFile are saved.")
    p.add_argument("--extract-minimize-input", action="store_true", help="Minimize a given input only")
    # p.add_argument("--start-point", type=int, default=14, help="This will be updated to take a pdb in the future")
    p.add_argument("--weights-load", type=str, help="Specify Model Weights To Use")
    p.add_argument("--out-size", type=int, help="Specify the output size")
    p.add_argument("--lr", type=float, default=0.001, help="Optimizer Learning Rate")
    # p.add_argument("--dataset", type=str, default="./binary_import/dof_vectors2024-04-10torch_tensor.pt", help="Dataset to use")
    p.add_argument("--cutoff-true", action="store_true", help="Apply a distance cutoff")
    p.add_argument("--cutoff", type=float, default=0.005, help="Specify Euclidean Distance Cutoff in Latent Space to Search")
    p.add_argument("--latent-dim", type=int, default=3, help="Latent Size of the model")
    p.add_argument("--partial-optimization", action="store_true", help="Apply partial optimization to a structure")
    p.add_argument("--freeze-residues", nargs="+", type=int, help="Used wth partial-optimization. If no sub freeze flags specified all are frozen \
                        Specify residues in a whitespace manner to freeze (e.g. 1 2 3)")
    p.add_argument("--freeze-torsion", action="store_true", help="Use with partial-optimization")
    p.add_argument("--freeze-bond-angle", action="store_true", help="Use with partial-optimization")
    p.add_argument("--freeze-bond-length", action="store_true", help="Use with partial-optimization")
    p.add_argument("--lambda-partial", type=float, default=1.0, help="Reweight Partial Freezing")
    p.add_argument("--energy-search", action="store_true", help="Gradient Search on dE/d(latent space)")
    p.add_argument("--mcmc", action="store_true", help="Markov Chain Monte Carlo Metropolis Hastings Search")
    p.add_argument("--batch-size", type=int, default=10, help="Used with mcmc. Set number of trajectories")
    p.add_argument("--sigma", type=float, default=0.01, help="Used with mcmc. Set sigma of normal movement to add to latent point")
    p.add_argument("--temperature", type=float, default=1, help="Used with mcmc. Set Temperature")
    p.add_argument("--initial-steps", type=int, default=10, help="Used with mcmc. Set initial steps to accept no matter what")
    p.add_argument("--ramp-down", type=int, default=100, help="Used with mcmc. Steps to go from temperature to 1")
    p.add_argument("--retain-structure", action="store_true", help="Used with mcmc. Walks path closer to input structure")
    p.add_argument("--inpaint", action="store_true", help="Provide Frozen Residues and generate scaffolds")
    p.add_argument("--inpaint-dist", type=float, default=300.0, help="Cutoff distance (p1norm) to consider a scaffold a success")
    p.add_argument("--scaffold-num", type=int, default=30, help="Number of Inpainted Scaffolds to generate")
    p.add_argument("--sgd-optim", action="store_true", help="Use SGD optimizer with momentum 0.9. AdamW is used by default")
    p.add_argument("--gpu", action="store_true", help="Use a GPU for inference")
    p.add_argument("--termbond-test", action="store_true", help="Run Terminal Bond Constraint Generation")
    p.add_argument("--minimize-mcmc", action="store_true", help="Minimize after MCMC")
    p.add_argument("--minimize-steps", type=int, default=1000, help="Number of energy minimization steps to take when using --minimize-mcmc")
    p.add_argument("--mcmc-steps", type=int, default=0, help="MCMC Steps")
    p.add_argument("--mcmc-paths", type=int, default=10, help="MCMC Number of Paths when doing inpainting")
    p.add_argument("--pdb-interface", type=str, default=None, help="Pass a pdb with the Target (Chain A) and Peptide (Chain B)")
    p.add_argument("--random-sample", action="store_true", help="Sample latent space randomly, without a refernce.")
    p.add_argument("--skip-individual", action="store_true", help="For skipping individual construct trajectory outputting during inpainting")
    p.add_argument("--interface-cutoff", type=float, default=20.0, help="Metric for accepting a not clashing peptide [10.0 for compact pockets, 20.0 for others] ")
    p.add_argument("--peptide-chain", type=int, default=2, help="Chain Index (1-index) that contains our motif of interest")
    p.add_argument("--interface-distance-cutoff", type=float, default=15.0, help="Atoms within this distance cutoff of the target will be considered \
                        when gradient-marching peptide")
    p.add_argument("--fullpocket-minimize", action="store_true", help="Specify if pocket context takes into account all non-hydrgoen bb and side-chain atoms \
                        or only bb-heavy + Cb atoms (which is default if you don't specify this flag)")
    p.add_argument("--ramp-interface-constraint", action="store_true", help="This sets the first 0.50 of the energy minimize steps to bb-heavy + Cb pocket context. \
                        Then for the last 0.50 switches to full pocket context (all heavy atoms). If not set then uses passed pocket specification throughout")
    args = p.parse_args()

    use_device = "cuda:0" if args.gpu else "cpu"

    # init the start path with X tensor 14 
    energySearch = EnergyPath(
        steps = args.steps, 
        minima = args.minima, 
        # start_point = torch.tensor(args.start_point), 
        weight_passed = args.weights_load, 
        lr=args.lr,
        # dataset = args.dataset, 
        cutoff = args.cutoff, 
        cutoff_true = args.cutoff_true,
        partial_optimization = args.partial_optimization,
        freeze_residues = torch.tensor(args.freeze_residues) - 1 if args.inpaint else None,
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
    )

    # Init dofsummary which is used in both the beginning and end of the script
    dofsummary = DoFSummary(prefix=args.prefix_outfile)

    # generate our output path if not yet created
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

    # Search And Output
    # if args.energy_search:
    #     if args.pdb_interface != None:
    #         atomdict = extract_pdb_xyz(args.pdb_interface)
    #         full, target, peptide, sigma_target, epsilon_target, sigma_peptide, epsilon_peptide, color_target = gen_xyz_tensor(atomdict)
    #         interface, mask = grab_interface(target, peptide, dist_cutoff = 10.0)
    #         interface_dict = {"target": target, "peptide": peptide, "interface_atoms": interface,
    #                           "sigma_peptide": sigma_peptide, "epsilon_peptide": epsilon_peptide,
    #                           "sigma_interface": sigma_target[mask], "epsilon_interface": epsilon_target[mask]}
    #         # print("INTERFACE SHAPE TENSOR:", interface.shape)
    #         energySearch.energy_steps(interface_dict=interface_dict)
    #     else:
    #         energySearch.energy_steps()
    #     energySearch.dist_energy_plot()
    #     energySearch.output_GDS()
    # elif args.termbond_test:
    #     scaffolds, latent_points = energySearch.inpainting(args.batch_size, args.random_sample)
    #     terminal_out = energySearch.generate_constraint_termbond(latent_points, True)
    #     # Non-Terminal Bond
    #     scaffolds, latent_pointss = energySearch.inpainting(args.batch_size, args.random_sample, terminal=False)
    #     noterminal_out = energySearch.generate_constraint_termbond(latent_points, False)
    #     torch.save({"with_constraint": terminal_out, "without_constraint": noterminal_out}, f"termbond_without_vs_with_{datetime.date.today()}.pt" )
    # elif args.mcmc:
    #     energySearch.metropolis_hastings(args.batch_size, args.sigma)
    #     if args.minimize_mcmc:
    #         energySearch.energy_steps(mcmc_minimize=True, minimize_steps=args.minimize_steps)
    #         energySearch.output_GDS()

    ## Inpainting Task 
    if args.inpaint:
        # Checks before proceeding
        assert args.pdb_interface != None, "You did not pass a PDB File for inpainting. \
               Please Provide a path to one with `--pdb-interface`"

        #### Here we are using our custom functions to extract Target and PDB
        #### information for inpainting and gradient-marching
        # All atoms are Extracted within cutoff distance
        atomdict = extract_pdb_xyz(args.pdb_interface, full=True)
        full, target, peptide, sigma_target, epsilon_target, sigma_peptide, epsilon_peptide, color_target = gen_xyz_tensor(atomdict)
        interface, mask = grab_interface(target, peptide, dist_cutoff = args.interface_distance_cutoff)
        # torch.save(color_target[mask], "interface_color.pt")
        # torch.save(interface, "interfacetensor.pt")
        interface_full_dict = {"target": target.to(use_device), "peptide": peptide.to(use_device), "interface_atoms": interface.to(use_device),
                          "sigma_peptide": sigma_peptide.to(use_device), "epsilon_peptide": epsilon_peptide.to(use_device),
                          "sigma_interface": sigma_target[mask].to(use_device), "epsilon_interface": epsilon_target[mask].to(use_device)}
        # Cb Pocket Context Extract
        atomdict = extract_pdb_xyz(args.pdb_interface, full=False)
        full, target, peptide, sigma_target, epsilon_target, sigma_peptide, epsilon_peptide, color_target = gen_xyz_tensor(atomdict)
        interface, mask = grab_interface(target, peptide, dist_cutoff = args.interface_distance_cutoff)
        interface_cb_dict = {"target": target.to(use_device), "peptide": peptide.to(use_device), "interface_atoms": interface.to(use_device),
                          "sigma_peptide": sigma_peptide.to(use_device), "epsilon_peptide": epsilon_peptide.to(use_device),
                          "sigma_interface": sigma_target[mask].to(use_device), "epsilon_interface": epsilon_target[mask].to(use_device)}
        
        #### Here we are extracting the DoFs needed for Harmonic DoF Inpainting Regression
        #### these are taken from our passed PDB
        motif_dofs = dofsummary.extract_pdb_pose_and_dof(
            input_pdb = args.pdb_interface,
            peptide_chain = args.peptide_chain,
            residues_of_interest = torch.tensor(args.freeze_residues),
            include_torsion = args.freeze_torsion if args.partial_optimization else True,
            include_bondangle = args.freeze_bond_angle if args.partial_optimization else True,
            include_bondlength = args.freeze_bond_length if args.partial_optimization else True,
        )

        #### Based on Target Information perform inpainting
        if args.fullpocket_minimize:
            scaffolds, latent_points = energySearch.inpainting(args.batch_size, args.random_sample, interface_dict = interface_full_dict,
                                                               motif_dofs = motif_dofs)
        else:
            scaffolds, latent_points = energySearch.inpainting(args.batch_size, args.random_sample, interface_dict = interface_cb_dict,
                                                               motif_dofs = motif_dofs)

        #### Take either inpaint output and use for MCMC/Gradient-march or just graident-march
        if args.mcmc_steps >= 1:
            latent_traj, _ = energySearch.metropolis_hastings(scaffolds.size(0), args.sigma, latent_points, mcmc_steps=args.mcmc_steps)
            energySearch.energy_steps(latent_traj[:, 0, :], interface_full_dict = interface_full_dict, interface_cb_dict = interface_cb_dict,
                                      motif_dofs = motif_dofs)
        elif args.mcmc_steps < 1:
            energySearch.energy_steps(latent_points, interface_full_dict = interface_full_dict, interface_cb_dict = interface_cb_dict,
                                      motif_dofs = motif_dofs)

    #### Here we are just randomly sampling the latent space to get constructs.
    #### There are no conditions being applied
    elif args.random_sample:
        if args.mcmc_steps < 1:
            scaffolds, latent_points = energySearch.inpainting(args.batch_size, args.random_sample)
            energySearch.energy_steps(latent_points)
        else: 
            scaffolds, latent_points = energySearch.inpainting(args.batch_size, args.random_sample)
            latent_traj, _ = energySearch.metropolis_hastings(scaffolds.size(0), args.sigma, latent_points, mcmc_steps=args.mcmc_steps)
            energySearch.energy_steps(latent_points)


    # Write our outputs out to this location
    energySearch.output_GDS(skip_individual = args.skip_individual,
                            dofsummary = dofsummary,
                            silentfile = args.silentfile,
                            score = args.score_outputs,
                            output_path = args.output_path,
                            random_outputs = args.random_sample,
                            minimize_pose = args.extract_minimize_input,
                            )
