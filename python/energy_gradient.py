#!/usr/bin/env python
#
# @author: Andrew Powers (apowers@flatironinstitute.org)
# @brief: Function to carry out Gradient Descent Marching along energy
#       term, during evaluation mode.
#
#


# native Packages
import os, sys
import datetime
import argparse
import ast

# import torch
import torch
import torch.nn as nn

# graphing
import matplotlib.pyplot as plt

# import model and functions
from test_case import ConvCase4
from model_functions import scale, ScaleOutputs
from terminal_error import TerminalPeptide

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
        start_point: torch.Tensor,
        weight_passed: str,
        lr: float,
        dataset: str,
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
        adam_optim: bool = True,
        sgd_optim: bool = False,
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
            adam_optim: bool
                Use Adam optimizer for energy search
            sgd_optim: bool
                Use SGD optimizer for energy search
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
        self.freeze_bond_angle = freeze_bond_angle
        self.freeze_bond_length = freeze_bond_length
        if self.partial_optimization and (torch.sum(torch.tensor([self.freeze_torsion, self.freeze_bond_angle, self.freeze_bond_length])).item() == 0):
            self.freeze_torsion = True
            self.freeze_bond_angle = True
            self.freeze_bond_length = True
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

        # Grab values required for model and other parts
        dataset = torch.load(dataset)
        self.PARAMS = dataset["params"]
        X = dataset["x"].double()
        self.energy_n = self.PARAMS["energy_n"]

        self.minima_energy = X[:, -self.energy_n:].min(dim=0).values

        # start point select without pdb input
        start_point_dof = X[start_point, :]

        # generate holding tensor and input the beginning traj
        self.dof_traj = torch.zeros((steps, start_point_dof.size(-1) - self.energy_n), dtype=torch.float64, device=device)
        self.dof_traj[0, :] = start_point_dof[:-self.energy_n]
        # Store points 
        self.start_point = start_point_dof[:-self.energy_n]
        self.start_point = self.start_point.unsqueeze(0)
        self.start_point = self.start_point.to(device)
        # Energy store
        self.sample_energy = start_point_dof[-self.energy_n:]


        # init the model and eval mode
        self.model = ConvCase4(
            encoder_layers    = 3,
            decoder_layers    = 3,
            latent_space_dim  = latent_dim,
            decomposition_n   = 1,
            dropout           = 0.1,
            noise_std         = 3,
            cyclic            = True,
            pe_linear         = False,
            convolution_layer = True,
            device            = device, #"cpu",
            **self.PARAMS,
        # ).to("cuda" if torch.cuda.is_available() else "cpu").double()
        ).to(device).double()
        # weights = torch.load(weight_passed, map_location="cuda" if torch.cuda.is_available() else "cpu")
        weights = torch.load(weight_passed, map_location=device)
        self.model.load_state_dict(weights["model_state_dict"])
        self.model.eval()

        # scale start point data
        X_start_point = torch.cat((X, start_point_dof.unsqueeze(0)), dim=0)
        scaled_X_start_point = scale(X_start_point, self.PARAMS) 

        #### Since right now we are not inputting a pdb to search we will select a 
        #### start point for the beginning of the search
        self.path = torch.zeros(self.steps)
        self.energy_path = torch.zeros(self.steps)

        # Pull out scaled start point
        self.start_point_scaled = scaled_X_start_point[-1, :]
        self.start_point_scaled = self.start_point_scaled.unsqueeze(0)
        # self.start_point = self.start_point.to("cuda" if torch.cuda.is_available() else "cpu")
        self.start_point_scaled = self.start_point_scaled.to(device)

        # scale energy for conversion importance
        self.energy_scale = ScaleOutputs(X[:, -self.energy_n:])

        if self.partial_optimization:
            # Set up atoms to start with per residue
            freeze_one = freeze_residues == 0
            freeze_eight = freeze_residues == 7
            atoms_per_residue = torch.ones(freeze_residues.shape) * 7
            atoms_per_residue[freeze_one] += 1
            atoms_per_residue[freeze_eight] += 2
            freeze_mas = freeze_residues > 0

            # generate start locations
            start_torsion_index = freeze_residues * 7
            # start_torsion_index[freeze_mas] += 1
            start_bond_angle_index = start_torsion_index + 58
            start_bond_length_index = start_bond_angle_index + 58

            # generate end locations
            end_torsion_index = start_torsion_index + atoms_per_residue
            end_bond_angle_index = start_bond_angle_index + atoms_per_residue
            end_bond_length_index = start_bond_length_index + atoms_per_residue

            # index tensors
            residue_len = freeze_residues.numel()
            # iter through the residues to build lists
            for res in range(residue_len):
                if res == 0:
                    torsion_index = torch.arange(start_torsion_index[res], end_torsion_index[res], device=self.device)
                    bond_angle_index = torch.arange(start_bond_angle_index[res], end_bond_angle_index[res], device=self.device)
                    bond_length_index = torch.arange(start_bond_length_index[res], end_bond_length_index[res], device=self.device)
                else:
                    torsion_index = torch.cat((torsion_index, torch.arange(start_torsion_index[res], end_torsion_index[res])))
                    bond_angle_index = torch.cat((bond_angle_index, torch.arange(start_bond_angle_index[res], end_bond_angle_index[res])))
                    bond_length_index = torch.cat((bond_length_index, torch.arange(start_bond_length_index[res], end_bond_length_index[res])))

            self.torsion_index = torsion_index.to(torch.long)
            # combine them
            self.idx = torch.cat(
                (
                    torsion_index,
                    bond_angle_index,
                    bond_length_index,
                )
            ).to(torch.long)

            if freeze_torsion and freeze_bond_angle:
                self.idx = self.idx[:torsion_index.numel()+bond_angle_index.numel()]
            elif freeze_torsion:
                self.idx = self.idx[:torsion_index.numel()]
            elif freeze_bond_length:
                self.idx = self.idx[torsion_index.numel():bond_angle_index.numel()]

        else:
            self.torsion_index = slice(None)

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
        ):
        """Harmonic Difference"""
        harmonic_penalty_sin_torsion = torch.tensor(0.0, device=self.device)    if not self.freeze_torsion     else torch.pow(real_t_sin[:, self.torsion_index] - torsion_sin[:, self.torsion_index],2, device=self.device).mean(dim=-1, device=self.device)
        harmonic_penalty_cos_torsion = torch.tensor(0.0, device=self.device)    if not self.freeze_torsion     else torch.pow(real_t_cos[:, self.torsion_index] - torsion_cos[:, self.torsion_index],2, device=self.device).mean(dim=-1, device=self.device)
        harmonic_penalty_sin_bond_angle = torch.tensor(0.0, device=self.device) if not self.freeze_bond_angle  else torch.pow(real_ba_sin[:, self.torsion_index] - bond_angle_sin[:, self.torsion_index],2, device=self.device).mean(dim=-1, device=self.device)
        harmonic_penalty_cos_bond_angle = torch.tensor(0.0, device=self.device) if not self.freeze_bond_angle  else torch.pow(real_ba_cos[:, self.torsion_index] - bond_angle_cos[:, self.torsion_index],2, device=self.device).mean(dim=-1, device=self.device)
        harmonic_penalty_bond_length = torch.tensor(0.0, device=self.device)    if not self.freeze_bond_length else torch.pow(real_bond_length[:, self.torsion_index] - bond_length[:, self.torsion_index],2, device=self.device).mean(dim=-1, device=self.device)
        harmonic_loss = harmonic_penalty_sin_torsion + harmonic_penalty_cos_torsion + harmonic_penalty_cos_bond_angle + harmonic_penalty_sin_bond_angle + harmonic_penalty_bond_length
        harmonic_loss /= torch.sum(torch.tensor([self.freeze_torsion, self.freeze_torsion, self.freeze_bond_angle, self.freeze_bond_angle, self.freeze_bond_length]))
        return harmonic_loss

    def inpainting(self, B: int):
        """given frozen residues search for scaffolds to hold motif residues

        PARAMS
        ------
            B: int
                Batch size. Number of places to search within the latent space

        RETURNS
        -------
            scaffolds: torch.Tensor
                DoF Structures that are close in proximity to the motif residues
        """
        no_scaffolds = True
        run=0
        scaffolds = 0
        points = 0

        while no_scaffolds:
            # set up tensor of random init values
            scatter_latent = torch.zeros((B, self.latent_dim), dtype=torch.float64, device=self.device)
            scatter_latent.uniform_(-0.3784, 0.3822)
            
            # run the points through the decoder
            out = self.decoder(scatter_latent)

            # input information
            real_t_sin = torch.sin(torch.deg2rad(self.start_point[:, :58]))
            real_t_cos = torch.cos(torch.deg2rad(self.start_point[:, :58]))
            real_ba_sin = torch.sin(torch.deg2rad(self.start_point[:, 58:116]))
            real_ba_cos = torch.cos(torch.deg2rad(self.start_point[:, 58:116]))
            real_bond_length = self.start_point[:, 116:]

            # Prediction Information
            torsion = out[:, :116]
            torsion_sin = torsion[:, :58]
            torsion_cos = torsion[:, 58:]
            bond_angle = out[:, 116:116*2]
            bond_angle_sin = bond_angle[:, :58]
            bond_angle_cos = bond_angle[:, 58:]
            bond_length = out[:, 116*2:-3]

            # generate tensor of preds
            preds = torch.cat(
                (
                    self.calc_angles(torsion),
                    self.calc_angles(bond_angle),
                    bond_length,
                ), dim=-1
            )

            # Grab ref2015 energy from output
            energy_all = self.energy_scale.inverse_scale(out[:, -self.energy_n:])
            energy_ref = energy_all[:, -2]

            # Distance Check
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

            # Determine which points are within the latent space cutoff
            mask = harmonic_loss <= self.inpaint_dist

            # add to set of scaffolds
            if not torch.is_tensor(scaffolds):
                scaffolds = preds[mask, :-3]
                points = scatter_latent[mask]
            else:
                scaffolds = torch.cat(
                    (
                        scaffolds,
                        preds[mask, :-3],
                    ), dim=0,
                )
                points = torch.cat(
                    (
                        points,
                        scatter_latent[mask],
                    ), dim=0,
                )

            # Report run statistics
            print(f'BATCH: {run}, FOUND SCAFFOLD START POINTS: {mask.sum().item()}, BATCH LOW: {harmonic_loss.min().item():.2f}, BATCH MEAN: {harmonic_loss.mean().item():.2f}, CUTOFF: {self.inpaint_dist}, SCAFFOLDS FOUND: {points.size(0)}')
                

            # If scaffolds found Stop While Loop
            run+=1
            if run > 100 or scaffolds.size(0) > 30: # 100
                no_scaffolds = False

        # Save outputs
        if torch.is_tensor(scaffolds):
            print(f"Found {scaffolds.size(0)} To Proceed With")
            torch.save({"scaffold": scaffolds, "latent_point": points}, "inpaintStartOutStructures.pt")
        else:
            print(f"No Scaffolds Found with residue similarity as {self.inpaint_dist} in {run-1} iterations")
        return scaffolds, points

    def metropolis_hastings(self, B: int, sigma: torch.float, start_points: torch.Tensor = None):
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
        else:
            F = 30
            latent_trajectories = torch.zeros((B, F, 2, self.latent_dim), dtype=torch.double)
            energies_trajectories = torch.zeros((B, F, 2, 1), dtype=torch.double)
            H = torch.zeros((B*F, 2, 1), dtype=torch.double)
            latent_trajectories[:, :, 0, :] = start_points.unsqueeze(1).repeat(1, F, 1)
            latent_trajectories[:, :, -1, :] = start_points.unsqueeze(1).repeat(1, F, 1)
            # reshpae for model input
            latent_trajectories = latent_trajectories.reshape(B*F, 2, self.latent_dim)


        # set initial energy to everything (specific Ref2015 energies)
        encode_decode_input = self.model(self.start_point_scaled, False)[0]
        energy_out = self.energy_scale.inverse_scale(encode_decode_input[:, -self.energy_n:])
        # energies_trajectories[:, 0, :] = energy_out[:, -1]

        energies_trajectories[:, :, 0, :] = energy_out[:, -1].unsqueeze(1).repeat(B, F, 1)
        #reshape energy
        energies_trajectories = energies_trajectories.reshape(B*F, 2, 1)

        # updated temperature
        temp_linspace = torch.linspace(1, self.temp, steps=self.rampdown) if self.temp != 1 else torch.tensor([1])

        # Harmonic Dampening on Structural Similarity Variables
        # Real Information
        real_t_sin = torch.sin(torch.deg2rad(self.start_point[:, :58]))
        real_t_cos = torch.cos(torch.deg2rad(self.start_point[:, :58]))
        real_ba_sin = torch.sin(torch.deg2rad(self.start_point[:, 58:116]))
        real_ba_cos = torch.cos(torch.deg2rad(self.start_point[:, 58:116]))
        real_bond_length = self.start_point[:, 116:] #-self.energy_n]

        # Perform MCMC for n steps
        for t in range(1, self.steps):
            
            # Generate random walk noise
            epsilon = torch.normal(0, sigma, size=(B*F, self.latent_dim), dtype=torch.double)
            # Apply noise to previous t-1 point
            # x_t = latent_trajectories[:, t-1, :] + epsilon
            x_t = latent_trajectories[:, -1, :] + epsilon

            # decode points
            out = self.decoder(x_t)

            # Prediction Information
            torsion = out[:, :116]
            torsion_sin = torsion[:, :58]
            torsion_cos = torsion[:, 58:]
            bond_angle = out[:, 116:116*2]
            bond_angle_sin = bond_angle[:, :58]
            bond_angle_cos = bond_angle[:, 58:]
            bond_length = out[:, 116*2:-3]

            # Grab ref2015 energy from output
            energy_all = self.energy_scale.inverse_scale(out[:, -self.energy_n:])
            energy_t = energy_all[:, -2]

            # Harmonic Change for if none of these are true if they are then it will change
            harmonic_change = torch.zeros((B, 1), dtype=torch.double)
            harmonic_loss = torch.zeros((1, B), dtype=torch.double)
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
                    harmonic_change = H[:, 1, :] - harmonic_loss.squeeze(0).unsqueeze(-1)
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
                # print('energy', energies_trajectories[:, -1, :].shape)
                # print('harmonic change', harmonic_change.shape)
                alpha = torch.minimum(torch.ones(energy_t.shape), torch.exp( -( ( (energies_trajectories[:, -1, :].squeeze(-1) - energy_t ) + self.lambda_2*harmonic_change.squeeze(-1) ) / temp_coeff ) ) )
                u = torch.ones((B*F))
                u.uniform_(0, 1)
                # generate bool of T/F
                condition_u = u > alpha
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
                H[mask, 0, :] = harmonic_loss[mask].unsqueeze(-1)
            # Accept all movements that are within the initial set range
            else:
                energies_trajectories[:, -1, :] = energy_t.view(-1, 1)
                latent_trajectories[:, -1, :] = x_t
                H[:, 1, :] = harmonic_loss.unsqueeze(-1)

            if t % 100 == 0:
                # print(f"Current Time Step: {t} Current Lowest Energy Minima: {energies_trajectories[:, :t, :].min().item()} Starting Point Energy: {energies_trajectories[0, 0, :].item()}")
                print(f"Current Time Step: {t} Current Lowest Energy Minima: {energies_trajectories.min().item()}, Starting Point Energy: {energy_out[0, -1].item()}, Current Lowest Harmonic Constraint: {H[:, -1, :].min().item()}, Start Harmonic Constraint: {H[0, 0, 0].item()}")


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
            energy_min = energies_trajectories.min(dim=1).indices
            latent_trajectories = latent_trajectories[:, energy_min, :, :]

        if start_points == None:
            print(f"Final Lowest Energy Minima: {energies_trajectories.min().item()} Starting Point Energy: {energy_out[0, 0].item()} Average Distance: {torch.norm(latent_trajectories[:, 0, :], self.start_latent_point, dim=-1).mean().item()}, Lowest Harmonic Constraint: {H[:, 0, :].min().item()}, Start Mean Harmonic Constraint: {H[:, 0, :].mean().item()}")
        else:
            print(f"Final Lowest Energy Minima: {energies_trajectories.min().item()} Starting Point Energy: {energy_out[:, 0].mean().item()} Average Distance: {torch.norm(latent_trajectories[:, 0, :] - start_points, dim=-1).mean().item()}, Lowest Harmonic Constraint: {H[:, 0, :].min().item()}, Start Mean Harmonic Constraint: {H[:, 0, :].mean().item()}")

        # Get lowest energy structures
        out = self.decoder(latent_trajectories[:, 0, :])

        # Format them correctly
        out = out.detach()
        torsion = self.calc_angles(out[:, :116])
        bond_angle = self.calc_angles(out[:, 116:116*2])
        preds = torch.cat((torsion, bond_angle, out[:, 116*2:-3]), dim=-1)
        preds = torch.cat((self.dof_traj[0, :].unsqueeze(0), preds), dim=0)
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
        torsion = out[:, :116]
        torsion_sin = torsion[:, :58]
        torsion_cos = torsion[:, 58:]
        bond_angle = out[:, 116:116*2]
        bond_angle_sin = bond_angle[:, :58]
        bond_angle_cos = bond_angle[:, 58:]
        bond_length = out[:, 116*2:-3]

        # Grab ref2015 energy from output
        energy_all = self.energy_scale.inverse_scale(out[:, -self.energy_n:])
        energy_t = energy_all[:, -2]

        # Harmonic Change for if none of these are true if they are then it will change
        harmonic_change = torch.zeros((B, 1), dtype=torch.double)
        harmonic_loss = torch.zeros((1, B), dtype=torch.double)

        if self.partial_optimization:
            # Harmonic Constraint
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
            harmonic_loss /= 5*58
            if t == 1:
                H[:, 0, :] = harmonic_loss.unsqueeze(-1)
                H[:, 1, :] = harmonic_loss.unsqueeze(-1)
                harmonic_change = harmonic_loss.squeeze(0).unsqueeze(-1)
            else:
                harmonic_change = H[:, 1, :] - harmonic_loss.squeeze(0).unsqueeze(-1)

        return x_t, H, harmonic_change, harmonic_loss, energy_t


    def energy_steps(self, start_points: torch.Tensor = None):
        """Model Eval Loop for GDS"""
        if start_points == None:
            self.start_latent_point = self.encoder(self.start_point_scaled).detach()
            start_range = 1
        else:
            self.start_latent_point = start_points
            self.path = torch.zeros((start_points.size(0), self.steps), device=self.device)
            self.energy_path = torch.zeros((start_points.size(0), self.steps), device=self.device)
            # generate holding tensor and input the beginning traj
            self.dof_traj = torch.zeros((start_points.size(0), self.steps, self.start_point_scaled.size(-1) - self.energy_n), dtype=torch.float64, device=self.device)
            start_range = 0
            # self.dof_traj[0, :] = start_points[:, :-self.energy_n]
        original_point = self.start_latent_point.clone()
        self.start_latent_point.requires_grad_()

        # Define optimizer you want
        if self.adam_optim:
            optimizer = torch.optim.AdamW([self.start_latent_point], lr=self.lr)
        elif self.sgd_optim:
            optimizer = torch.optim.SGD([self.start_latent_point], lr=self.lr, momentum=0.9)


        if self.sample_energy.size(0) == 3:
            print(f"Step: 0, QM: {self.sample_energy[0].item()}, Ref2015: {self.sample_energy[1].item()}, Ref2015+Solvent: {self.sample_energy[-1].item()}, Data Taken Straight From Input")
        else:
            print(f"Step: 0, QM: {self.sample_energy[0].item()}, Ref2015: {self.sample_energy[-1].item()}, Data Taken Straight From Input")
        # add energy path
        self.energy_path[0] = self.sample_energy[1]

        for step in range(start_range, self.steps):
            # zero grad
            optimizer.zero_grad()

            # get the structure from the latent point
            out = self.decoder(self.start_latent_point)

            # Prediction Information
            torsion = out[:, :116]
            torsion_sin = torsion[:, :58]
            torsion_cos = torsion[:, 58:]
            bond_angle = out[:, 116:116*2]
            bond_angle_sin = bond_angle[:, :58]
            bond_angle_cos = bond_angle[:, 58:]
            bond_length = out[:, 116*2:-3]
            preds = torch.cat((torsion, bond_angle, out[:, 116*2:-3]), dim=-1)

            # Real Information
            real_t_sin = torch.sin(torch.deg2rad(self.start_point[:, :58]))
            real_t_cos = torch.cos(torch.deg2rad(self.start_point[:, :58]))
            real_ba_sin = torch.sin(torch.deg2rad(self.start_point[:, 58:116]))
            real_ba_cos = torch.cos(torch.deg2rad(self.start_point[:, 58:116]))
            real_bond_length = self.start_point[:, 116:] #-self.energy_n]

            # grab the energies (ref2015 score)
            energies = self.energy_scale.inverse_scale(out[:, -self.energy_n:]).to(self.device)
            if energies.size(-1) == 3:
                qm = energies[:, 0]
                energy = energies[:, 1]

                dist = torch.norm(original_point - self.start_latent_point, dim=-1)
                if self.partial_optimization:

                    # Harmonic Constraint
                    harmonic_penalty_sin_torsion = torch.tensor(0.0)    if not self.freeze_torsion     else torch.pow(torch.norm(real_t_sin[:, self.torsion_index] - torsion_sin[:, self.torsion_index], dim=-1)*10,2)
                    harmonic_penalty_cos_torsion = torch.tensor(0.0)    if not self.freeze_torsion     else torch.pow(torch.norm(real_t_cos[:, self.torsion_index] - torsion_cos[:, self.torsion_index], dim=-1)*10,2)
                    harmonic_penalty_sin_bond_angle = torch.tensor(0.0) if not self.freeze_bond_angle  else torch.pow(torch.norm(real_ba_sin[:, self.torsion_index] - bond_angle_sin[:, self.torsion_index], dim=-1)*10,2)
                    harmonic_penalty_cos_bond_angle = torch.tensor(0.0) if not self.freeze_bond_angle  else torch.pow(torch.norm(real_ba_cos[:, self.torsion_index] - bond_angle_cos[:, self.torsion_index], dim=-1)*10,2)
                    harmonic_penalty_bond_length = torch.tensor(0.0)    if not self.freeze_bond_length else torch.pow(torch.norm(real_bond_length[:, self.torsion_index] - bond_length[:, self.torsion_index], dim=-1),2)
                    harmonic_loss = harmonic_penalty_sin_torsion + harmonic_penalty_cos_torsion + harmonic_penalty_cos_bond_angle + harmonic_penalty_sin_bond_angle + harmonic_penalty_bond_length
                    harmonic_loss /= self.torsion_index.size(0)*torch.sum(torch.tensor([self.freeze_torsion, self.freeze_torsion, self.freeze_bond_angle, self.freeze_bond_angle, self.freeze_bond_length]))

                elif self.retain_structure and (not self.partial_optimization):
                    harmonic_penalty_sin_torsion     = torch.pow(torch.norm(real_t_sin - torsion_sin, dim=-1)*10, 2)   
                    harmonic_penalty_cos_torsion     = torch.pow(torch.norm(real_t_cos - torsion_cos, dim=-1)*10, 2)   
                    harmonic_penalty_sin_bond_angle  = torch.pow(torch.norm(real_ba_sin - bond_angle_sin, dim=-1)*10, 2) 
                    harmonic_penalty_cos_bond_angle  = torch.pow(torch.norm(real_ba_cos - bond_angle_cos, dim=-1)*10, 2) 
                    harmonic_penalty_bond_length     = torch.pow(torch.norm(real_bond_length - bond_length, dim=-1), 2)    
                    harmonic_loss = harmonic_penalty_sin_torsion + harmonic_penalty_cos_torsion + harmonic_penalty_cos_bond_angle + harmonic_penalty_sin_bond_angle + harmonic_penalty_bond_length
                    harmonic_loss /= 58*5

                else:
                    harmonic_loss = torch.tensor([0.0], device=energy.device)

                # N-C Terminal Peptide Bond Loss
                term_bond_loss = self.term_bond.terminal_peptide_loss(
                    dof_inputs = out,
                    real_term = torch.ones(out.size(0), device=out.device, dtype=out.dtype) * 3.,
                    params = self.PARAMS,
                    foldtree_root = 4,
                    device = out.device,
                )

                loss = (energy + dist + self.lambda_2*harmonic_loss + self.lambda_2*term_bond_loss).mean()

                print(f"Step: {step}, QM: {energies[:, 0].item()} Ref2015: {energies[:, 1].item()}, Ref2015+Solvent: {energies[:, -1].item()}, Dist: {dist.item()}, Harmonic Penalty: {harmonic_loss.item()}, Loss: {loss.item()}")
            else:
                energy = energies[:, -1]


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
                    # harmonic_penalty_sin_torsion     = torch.pow(torch.norm(real_t_sin - torsion_sin, dim=-1)*10, 2)   
                    # harmonic_penalty_cos_torsion     = torch.pow(torch.norm(real_t_cos - torsion_cos, dim=-1)*10, 2)   
                    # harmonic_penalty_sin_bond_angle  = torch.pow(torch.norm(real_ba_sin - bond_angle_sin, dim=-1)*10, 2) 
                    # harmonic_penalty_cos_bond_angle  = torch.pow(torch.norm(real_ba_cos - bond_angle_cos, dim=-1)*10, 2) 
                    # harmonic_penalty_bond_length     = torch.pow(torch.norm(real_bond_length - bond_length, dim=-1), 2)    
                    # harmonic_loss = harmonic_penalty_sin_torsion + harmonic_penalty_cos_torsion + harmonic_penalty_cos_bond_angle + harmonic_penalty_sin_bond_angle + harmonic_penalty_bond_length
                    # harmonic_loss /= 5

                else:
                    harmonic_loss = torch.tensor([0.0], device=self.device)

                if step >= int(self.steps*0.8):
                    # N-C Terminal Peptide Bond Loss
                    term_bond_loss = self.term_bond.terminal_peptide_loss(
                        dof_inputs = out,
                        real_term = torch.ones(out.size(0), device=out.device, dtype=out.dtype) * 1.4,
                        params = self.PARAMS,
                        foldtree_root = 4,
                        device = out.device,
                    )
                else: term_bond_loss = torch.tensor([0.0], dtype=torch.float64, device=self.device)

                loss = (energy + dist + self.lambda_2*harmonic_loss + self.lambda_2*term_bond_loss).mean()

                if start_points == None:
                    self.path[step] = dist.detach()
                    self.energy_path[step] = energy.detach()
                    print(f"Step: {step}, QM: {energies[:, 0].item()}, Ref2015: {energies[:, -1].item()}, Dist: {dist.item()}, Harmonic Penalty: {harmonic_loss.item()}, Terminal Bond: {term_bond_loss.item()}, Loss: {loss.item()}")
                else:
                    self.path[:, step] = dist.detach()
                    self.energy_path[:, step] = energy.detach()
                    print(f"Step: {step}, QM: {energies[:, 0].mean().item()}, Ref2015: {energies[:, -1].mean().item()}, Dist: {dist.mean().item()}, Harmonic Penalty: {harmonic_loss.mean().item()}, Terminal Bond: {term_bond_loss.item()}, Lowest Harmonic Penalty: {harmonic_loss.min().item()} Loss: {loss.item()}")

            # back prop
            loss.backward()
            optimizer.step()



            out = out.detach().clone()
            torsion = self.calc_angles(out[:, :116])
            bond_angle = self.calc_angles(out[:, 116:116*2])
            preds = torch.cat((torsion, bond_angle, out[:, 116*2:-3]), dim=-1)

            if start_points == None:
                self.dof_traj[step] = preds
            else:
                self.dof_traj[:, step] = preds

            # stop check
            # if (dist >= self.cutoff) and (self.cutoff_true):
            #     self.outstep = step
            #     print(f"Distance cutoff: {self.cutoff} found at Step: {step}")
            #     self.dof_traj = self.dof_traj[:step, :]
            #     break
            

    def output_GDS(self):
        """Output the GDS trajectory"""
        # file name
        if len(self.dof_traj.shape) == 3:
            for sample in range(self.dof_traj.size(0)):
                file = f"finalstep{self.outstep}-{datetime.date.today()}-sample{sample}-trajectories.pt"
                out = self.dof_traj[sample]
                torch.save(out, file)
            file = f"finalstep{self.outstep}-{datetime.date.today()}-sampleall-trajectories.pt"
            torch.save(self.dof_traj[:, -1, :], file)
        else:
            file = f"finalstep{self.outstep}-{datetime.date.today()}-trajectories.pt"
            torch.save(self.dof_traj, file)
            torch.save({"distance": self.path, "energy": self.energy_path}, 'trajpath.pt')
        

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=1000, help="Search Steps Max (default: 1000)")
    p.add_argument("--minima", type=float, default=-4.775, help="Minima energy to search for (default: -5.0)")
    p.add_argument("--start-point", type=int, default=14, help="This will be updated to take a pdb in the future")
    p.add_argument("--weights-load", type=str, help="Specify Model Weights To Use")
    p.add_argument("--lr", type=float, default=0.001, help="Optimizer Learning Rate (default: 1e-3)")
    p.add_argument("--dataset", type=str, default="./binary_import/dof_vectors2024-04-10torch_tensor.pt", help="Dataset to use (default: dof_vectors2024-04-10torch_tensor)")
    p.add_argument("--cutoff-true", action="store_true", help="Apply a distance cutoff")
    p.add_argument("--cutoff", type=float, default=0.005, help="Specify Euclidean Distance Cutoff in Latent Space to Search (default: 0.005)")
    p.add_argument("--latent-dim", type=int, default=3, help="Latent Size of the model (default: 3)")
    p.add_argument("--partial-optimization", action="store_true", help="Apply partial optimization to a structure")
    p.add_argument("--freeze-residues", type=str, help="Used wth partial-optimization. If no sub freeze flags specified all are frozen \
                        Specify residue in a list (zero index) to freeze (e.g. '[1,2,3]')")
    p.add_argument("--freeze-torsion", action="store_true", help="Use with partial-optimization")
    p.add_argument("--freeze-bond-angle", action="store_true", help="Use with partial-optimization")
    p.add_argument("--freeze-bond-length", action="store_true", help="Use with partial-optimization")
    p.add_argument("--lambda-partial", type=float, default=1.0, help="Reweight Partial Freezing (Default: 1.0)")
    p.add_argument("--energy-search", action="store_true", help="Gradient Search on dE/d(latent space)")
    p.add_argument("--mcmc", action="store_true", help="Markov Chain Monte Carlo Metropolis Hastings Search")
    p.add_argument("--batch-size", type=int, default=10, help="Used with mcmc. Set number of trajectories (default: 10)")
    p.add_argument("--sigma", type=float, default=0.01, help="Used with mcmc. Set sigma of normal movement to add to latent point (default: 0.01)")
    p.add_argument("--temperature", type=float, default=1, help="Used with mcmc. Set Temperature (default: 1)")
    p.add_argument("--initial-steps", type=int, default=10, help="Used with mcmc. Set initial steps to accept no matter what (default: 10)")
    p.add_argument("--ramp-down", type=int, default=100, help="Used with mcmc. Steps to go from temperature to 1 (default: 100)")
    p.add_argument("--retain-structure", action="store_true", help="Used with mcmc. Walks path closer to input structure")
    p.add_argument("--inpaint", action="store_true", help="Provide Frozen Residues and generate scaffolds")
    p.add_argument("--inpaint-dist", type=float, default=300.0, help="Cutoff distance (p1norm) to consider a scaffold a success (default: 300.0)")
    p.add_argument("--sgd-optim", action="store_true", help="Use SGD optimizer with momentum 0.9. AdamW is used by default")
    p.add_argument("--gpu", action="store_true", help="Use a GPU for inference")
    args = p.parse_args()

    use_device = "cuda:0" if args.gpu else "cpu"

    # init the start path with X tensor 14 
    energySearch = EnergyPath(
        steps = args.steps, 
        minima = args.minima, 
        start_point = torch.tensor(args.start_point), 
        weight_passed = args.weights_load, 
        lr=args.lr,
        dataset = args.dataset, 
        cutoff = args.cutoff, 
        cutoff_true = args.cutoff_true,
        partial_optimization = args.partial_optimization,
        freeze_residues = torch.tensor(ast.literal_eval(args.freeze_residues)) if args.partial_optimization else None,
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
        adam_optim = False if args.sgd_optim else True,
        sgd_optim = args.sgd_optim,
    )

    # Search And Output
    if args.energy_search:
        energySearch.energy_steps()
        energySearch.dist_energy_plot()
        energySearch.output_GDS()
    if args.mcmc:
        energySearch.metropolis_hastings(args.batch_size, args.sigma)
    if args.inpaint:
        scaffolds, latent_points = energySearch.inpainting(args.batch_size)
        latent_traj, _ = energySearch.metropolis_hastings(scaffolds.size(0), args.sigma, latent_points)
        energySearch.energy_steps(latent_traj[:, 0, :])
        energySearch.output_GDS()




