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

        # THIS IS CHANGED FOR THE UNIT CIRCLE NORM TRY
        # sin_targets = targets[:, :targets.size(-1)//2]
        # cos_targets = targets[:, targets.size(-1)//2:]
        # sin_preds, cos_preds = preds[:, ::2], preds[:, 1::2]

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
        warmup: int,
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
            warmup: int,
                Number of steps to go from a low temp to high temp
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
        self.warmup = warmup

        # Grab values required for model and other parts
        dataset = torch.load(dataset)
        PARAMS = dataset["params"]
        X = dataset["x"].double()
        self.energy_n = PARAMS["energy_n"]

        self.minima_energy = X[:, -self.energy_n:].min(dim=0).values

        # start point select without pdb input
        start_point_dof = X[start_point, :]

        # generate holding tensor and input the beginning traj
        self.dof_traj = torch.zeros((steps, start_point_dof.size(-1) - self.energy_n), dtype=torch.float64)
        self.dof_traj[0, :] = start_point_dof[:-self.energy_n]
        self.sample_energy = start_point_dof[-self.energy_n:]


        # init the model and eval mode
        self.model = ConvCase4(
            encoder_layers    = 3,
            decoder_layers    = 3,
            latent_space_dim  = 3,
            decomposition_n   = 1,
            dropout           = 0.1,
            noise_std         = 3,
            cyclic            = True,
            pe_linear         = False,
            convolution_layer = True,
            device            = "cpu",
            **PARAMS,
        # ).to("cuda" if torch.cuda.is_available() else "cpu").double()
        ).double()
        # weights = torch.load(weight_passed, map_location="cuda" if torch.cuda.is_available() else "cpu")
        weights = torch.load(weight_passed, map_location="cpu")
        self.model.load_state_dict(weights["model_state_dict"])
        self.model.eval()

        # scale start point data
        X_start_point = torch.cat((X, start_point_dof.unsqueeze(0)), dim=0)
        scaled_X_start_point = scale(X_start_point, PARAMS) 

        #### Since right now we are not inputting a pdb to search we will select a 
        #### start point for the beginning of the search
        self.path = torch.zeros(self.steps)
        self.energy_path = torch.zeros(self.steps)

        # Pull out scaled start point
        self.start_point = scaled_X_start_point[-1, :]
        self.start_point = self.start_point.unsqueeze(0)
        # self.start_point = self.start_point.to("cuda" if torch.cuda.is_available() else "cpu")
        self.start_point = self.start_point.to("cpu")

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
                    torsion_index = torch.arange(start_torsion_index[res], end_torsion_index[res])
                    bond_angle_index = torch.arange(start_bond_angle_index[res], end_bond_angle_index[res])
                    bond_length_index = torch.arange(start_bond_length_index[res], end_bond_length_index[res])
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

    def metropolis_hastings(self, B: int, sigma: torch.float):
        """MCMC Search for energy minima structures
        PARAMS
        ------
            B: int,
                Batch Size. Amount of attempts to try in parallel
            sigma: torch.float
                Amount of noise for normal distribution

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
        latent_trajectories = torch.zeros((B, 2, 3), dtype=torch.double)

        # Lowest energy holder for the trajectories. This determines which point to return and decode
        # energies_trajectories = torch.zeros((B, self.steps, 1), dtype=torch.double)
        energies_trajectories = torch.zeros((B, 2, 1), dtype=torch.double)

        # minima structures
        # self.minima_structures = torch.zeros((B, self.start_point.size(-1)-self.energy_n), dtype=torch.float64)

        # get starting point and enable grad
        self.start_latent_point = self.encoder(self.start_point).detach()

        # Set this as t_0 for all trajectories
        # latent_trajectories[:, 0, :] = self.start_latent_point
        latent_trajectories[:, 0, :] = self.start_latent_point.repeat(B, 1)

        # set initial energy to everything (specific Ref2015 energies)
        encode_decode_input = self.model(self.start_point, False)[0]
        energy_out = self.energy_scale.inverse_scale(encode_decode_input[:, -self.energy_n:])
        # energies_trajectories[:, 0, :] = energy_out[:, -1]
        energies_trajectories[:, 0, :] = energy_out[:, -1].repeat(B, 1)

        # updated temperature
        temp_linspace = torch.linspace(1, self.temp, steps=self.warmup) if self.temp != 1 else torch.tensor([1])

        # Perform MCMC for n steps
        for t in range(1, self.steps):
            
            # Generate random walk noise
            epsilon = torch.normal(0, sigma, size=(B, 3), dtype=torch.double)
            # Apply noise to previous t-1 point
            # x_t = latent_trajectories[:, t-1, :] + epsilon
            x_t = latent_trajectories[:, -1, :] + epsilon

            # decode points
            out = self.decoder(x_t)

            # Grab ref2015 energy from output
            energy_all = self.energy_scale.inverse_scale(out[:, -self.energy_n:])
            energy_t = energy_all[:, -2]

            # Get alpha 
            # alpha = torch.minimum(torch.ones(energy_t.shape), torch.exp(energies_trajectories[:, t-1, :].squeeze(-1) - energy_t))
            temp_coeff = temp_linspace[t-1] if t <= self.warmup else 0.01
            alpha = torch.minimum(torch.ones(energy_t.shape), torch.exp((energies_trajectories[:, -1, :].squeeze(-1) - energy_t) / temp_coeff))

            # accept or not
            if t > self.initial_set:
                u = torch.ones((B))
                u.uniform_(0, 1)
                # generate bool of T/F
                condition_u = u <= alpha
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
            # Accept all movements that are within the initial set range
            else:
                energies_trajectories[:, -1, :] = energy_t.view(-1, 1)
                latent_trajectories[:, -1, :] = x_t

            if t % 500 == 0:
                # print(f"Current Time Step: {t} Current Lowest Energy Minima: {energies_trajectories[:, :t, :].min().item()} Starting Point Energy: {energies_trajectories[0, 0, :].item()}")
                print(f"Current Time Step: {t} Current Lowest Energy Minima: {energies_trajectories.min().item()} Starting Point Energy: {energy_out[0, -1].item()}")

        print(f"Final Lowest Energy Minima: {energies_trajectories.min().item()} Starting Point Energy: {energy_out[0, -1].item()} Average Distance: {torch.cdist(latent_trajectories[:, 0, :], self.start_latent_point, p=2).mean().item()}")
        # Get lowest energy structures
        out = self.decoder(latent_trajectories[:, 0, :])
        # Format them correctly
        out = out.detach()
        torsion = self.calc_angles(out[:, :116])
        bond_angle = self.calc_angles(out[:, 116:116*2])
        preds = torch.cat((torsion, bond_angle, out[:, 116*2:-3]), dim=-1)
        preds = torch.cat((self.dof_traj[0, :].unsqueeze(0), preds), dim=0)
        torch.save(preds, "mcmcOutStructures.pt")
        torch.save(
            {
                "latent_traj": latent_trajectories, 
                "energy_traj": energies_trajectories,
                "start_energy": energy_out[0, -1],
                "start_latent": self.start_latent_point,
            },
            "mcmctest.pt")
        return latent_trajectories, energies_trajectories



    def energy_steps(self):
        """Model Eval Loop for GDS"""
        self.start_latent_point = self.encoder(self.start_point).detach()
        original_point = self.start_latent_point.clone()
        self.start_latent_point.requires_grad_()
        # optimizer = torch.optim.SGD([self.start_latent_point], lr=self.lr, momentum=0.9)
        optimizer = torch.optim.AdamW([self.start_latent_point], lr=self.lr)

        # Loss function weights
        lambda_1 = (1 - self.lambda_2)#20 if self.lambda_2 <= 2 else 1

        if self.sample_energy.size(0) == 3:
            print(f"Step: 0, QM: {self.sample_energy[0].item()}, Ref2015: {self.sample_energy[1].item()}, Ref2015+Solvent: {self.sample_energy[-1].item()}, Data Taken Straight From Input")
        else:
            print(f"Step: 0, QM: {self.sample_energy[0].item()}, Ref2015: {self.sample_energy[-1].item()}, Data Taken Straight From Input")
        # add energy path
        self.energy_path[0] = self.sample_energy[1]

        for step in range(1, self.steps):
            # zero grad
            optimizer.zero_grad()

            # out from model
            # out, _, latent_space, _ = self.model(self.start_point, False)
            # get the structure from the latent point
            out = self.decoder(self.start_latent_point)
            # torsion = self.calc_angles(out[:, :116])
            # bond_angle = self.calc_angles(out[:, 116:116*2])

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
            real_t_sin = torch.sin(self.start_point[:, :58])
            real_t_cos = torch.cos(self.start_point[:, :58])
            real_ba_sin = torch.sin(self.start_point[:, 58:116])
            real_ba_cos = torch.cos(self.start_point[:, 58:116])
            real_bond_length = self.start_point[:, 116:-self.energy_n]
            # latent_space = latent_space.detach()
            # if step == 1 or redo:
            #     original_point = latent_space
            #     redo = False

            # grab the energies (ref2015 score)
            energies = self.energy_scale.inverse_scale(out[:, -self.energy_n:])
            if energies.size(-1) == 3:
                qm = energies[:, 0]
                energy = energies[:, 1]

                dist = torch.cdist(original_point, self.start_latent_point, p=2)
                if self.partial_optimization:

                    #### THIS NEEDS TO BE FIXED COMPLETELY
                    harmonic_penalty_sin_torsion = torch.cdist(torch.sin(real_t_sin[:, self.torsion_index]), torsion_sin[:, self.torsion_index], p=2)
                    harmonic_penalty_cos_torsion = torch.cdist(torch.cos(real_t_cos[:, self.torsion_index]), torsion_cos[:, self.torsion_index], p=2)
                    harmonic_penalty_sin_bond_angle = torch.cdist(torch.sin(real_ba_sin[:, self.torsion_index]),  bond_angle_sin[:, self.torsion_index], p=2)
                    harmonic_penalty_cos_bond_angle = torch.cdist(torch.cos(real_ba_cos[:, self.torsion_index]),  bond_angle_cos[:, self.torsion_index], p=2)
                    harmonic_penalty_bond_length = torch.cdist(real_bond_length[:, self.torsion_index], bond_length[:, self.torsion_index], p=2)
                    harmonic_loss = harmonic_penalty_sin_torsion + harmonic_penalty_cos_torsion + harmonic_penalty_cos_bond_angle + harmonic_penalty_sin_bond_angle + harmonic_penalty_bond_length
                    harmonic_loss /= 5

                else:
                    harmonic_loss = torch.tensor([0.0])

                loss = lambda_1*energy + dist + self.lambda_2*harmonic_loss

                print(f"Step: {step}, QM: {energies[:, 0].item()} Ref2015: {energies[:, 1].item()}, Ref2015+Solvent: {energies[:, -1].item()}, Dist: {dist.item()}, Harmonic Penalty: {harmonic_loss.item()}, Loss: {loss.item()}")
            else:
                energy = energies[:, -1]


                dist = torch.cdist(original_point, self.start_latent_point, p=2)
                if self.partial_optimization:
                    harmonic_penalty_sin_torsion = torch.tensor(0.0)    if not self.freeze_torsion     else torch.pow(torch.cdist(torch.sin(real_t_sin[:, self.torsion_index]),   torsion_sin[:, self.torsion_index], p=1),2)
                    harmonic_penalty_cos_torsion = torch.tensor(0.0)    if not self.freeze_torsion     else torch.pow(torch.cdist(torch.cos(real_t_cos[:, self.torsion_index]),   torsion_cos[:, self.torsion_index], p=1),2)
                    harmonic_penalty_sin_bond_angle = torch.tensor(0.0) if not self.freeze_bond_angle  else torch.pow(torch.cdist(torch.sin(real_ba_sin[:, self.torsion_index]),  bond_angle_sin[:, self.torsion_index],p=1),2)
                    harmonic_penalty_cos_bond_angle = torch.tensor(0.0) if not self.freeze_bond_angle  else torch.pow(torch.cdist(torch.cos(real_ba_cos[:, self.torsion_index]),  bond_angle_cos[:, self.torsion_index],p=1),2)
                    harmonic_penalty_bond_length = torch.tensor(0.0)    if not self.freeze_bond_length else torch.pow(torch.cdist(real_bond_length[:, self.torsion_index],        bond_length[:, self.torsion_index],p=1),2)
                    harmonic_loss = harmonic_penalty_sin_torsion + harmonic_penalty_cos_torsion + harmonic_penalty_cos_bond_angle + harmonic_penalty_sin_bond_angle + harmonic_penalty_bond_length
                    harmonic_loss /= torch.sum(torch.tensor([self.freeze_torsion, self.freeze_torsion, self.freeze_bond_angle, self.freeze_bond_angle, self.freeze_bond_length]))
                else:
                    harmonic_penalty_sin_torsion     = torch.pow(torch.cdist(torch.exp(torch.sin(real_t_sin[:, self.torsion_index])),   torch.exp(torsion_sin[:, self.torsion_index]), p=2), 2)   
                    harmonic_penalty_cos_torsion     = torch.pow(torch.cdist(torch.exp(torch.cos(real_t_cos[:, self.torsion_index])),   torch.exp(torsion_cos[:, self.torsion_index]), p=2), 2)   
                    harmonic_penalty_sin_bond_angle  = torch.pow(torch.cdist(torch.exp(torch.sin(real_ba_sin[:, self.torsion_index])),  torch.exp(bond_angle_sin[:, self.torsion_index]),p=2), 2) 
                    harmonic_penalty_cos_bond_angle  = torch.pow(torch.cdist(torch.exp(torch.cos(real_ba_cos[:, self.torsion_index])),  torch.exp(bond_angle_cos[:, self.torsion_index]),p=2), 2) 
                    harmonic_penalty_bond_length     = torch.pow(torch.cdist(torch.exp(real_bond_length[:, self.torsion_index]),        torch.exp(bond_length[:, self.torsion_index]),p=2), 2)    
                    harmonic_loss = harmonic_penalty_sin_torsion + harmonic_penalty_cos_torsion + harmonic_penalty_cos_bond_angle + harmonic_penalty_sin_bond_angle + harmonic_penalty_bond_length
                    harmonic_loss /= 5
                    # harmonic_loss = torch.tensor([0.0])

                loss = lambda_1*energy + dist + self.lambda_2*harmonic_loss

                self.path[step] = dist.detach()
                self.energy_path[step] = energy.detach()
                print(f"Step: {step}, QM: {energies[:, 0].item()}, Ref2015: {energies[:, -1].item()}, Dist: {dist.item()}, Harmonic Penalty: {harmonic_loss.item()}, Loss: {loss.item()}")

            # back prop
            loss.backward()
            # if self.partial_optimization:
            #     with torch.no_grad():
            #         self.start_point.grad[:, self.idx] = 0
            #         # print("GRAD", self.start_point.grad)
            optimizer.step()
            
            # test replace values
            # if self.partial_optimization:
                # with torch.no_grad():
                #     self.start_point[:, self.idx] = self.dof_traj[1, self.idx]
            #         print(self.idx)
            #         original_vals = self.dof_traj[0, self.idx]
            #         self.start_point[:, self.idex] = original_vals



            out = out.detach().clone()
            torsion = self.calc_angles(out[:, :116])
            bond_angle = self.calc_angles(out[:, 116:116*2])
            preds = torch.cat((torsion, bond_angle, out[:, 116*2:-3]), dim=-1)
            # diff = self.start_point[:, :-2].detach() - self.dof_traj[0, :]
            # print("T",  diff[:, :58])
            # print("BA", diff[:, 58:116])
            # print("BL", diff[:, 116:])

            self.dof_traj[step] = preds

            # stop check
            if (dist >= self.cutoff) and (self.cutoff_true):
                self.outstep = step
                print(f"Distance cutoff: {self.cutoff} found at Step: {step}")
                self.dof_traj = self.dof_traj[:step, :]
                break
            

    def output_GDS(self):
        """Output the GDS trajectory"""
        # file name
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
    p.add_argument("--partial-optimization", action="store_true", help="Apply partial optimization to a structure")
    p.add_argument("--freeze-residues", type=str, help="Used wth partial-optimization. If no sub freeze flags specified all are frozen \
                        Specify residue in a list (zero index) to freeze (e.g. '[1,2,3]')")
    p.add_argument("--freeze-torsion", action="store_true", help="Use with partial-optimization")
    p.add_argument("--freeze-bond-angle", action="store_true", help="Use with partial-optimization")
    p.add_argument("--freeze-bond-length", action="store_true", help="Use with partial-optimization")
    p.add_argument("--lambda-partial", type=float, default=0.5, help="Reweight Partial Freezing (Default: 0.5)")
    p.add_argument("--energy-search", action="store_true", help="Gradient Search on dE/d(latent space)")
    p.add_argument("--mcmc", action="store_true", help="Markov Chain Monte Carlo Metropolis Hastings Search")
    p.add_argument("--batch-size", type=int, default=10, help="Used with mcmc. Set number of trajectories (default: 10)")
    p.add_argument("--sigma", type=float, default=0.01, help="Used with mcmc. Set sigma of normal movement to add to latent point (default: 0.01)")
    p.add_argument("--temperature", type=float, default=1, help="Used with mcmc. Set Temperature (default: 1)")
    p.add_argument("--initial-steps", type=int, default=10, help="Used with mcmc. Set initial steps to accept no matter what (default: 10)")
    p.add_argument("--warmup", type=int, default=100, help="Used with mcmc. Steps to go from temperature to 1 (default: 100)")
    args = p.parse_args()


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
        warmup = args.warmup,
    )

    # Search And Output
    if args.energy_search:
        energySearch.energy_steps()
        energySearch.dist_energy_plot()
        energySearch.output_GDS()
    if args.mcmc:
        energySearch.metropolis_hastings(args.batch_size, args.sigma)



