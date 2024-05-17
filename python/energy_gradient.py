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
            free_torsion, bond_angle, bond_length : bool, bool, bool
                Specify the DoF to freeze if not specified then all are frozen
        """
        # Pull up points
        self.steps = steps
        self.minima = minima
        self.outstep = steps
        self.lr = lr
        self.cutoff = cutoff
        self.cutoff_true = cutoff_true
        self.partial_optimization = partial_optimization
        self.freeze_residues = freeze_residues

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
        # self.start_point.requires_grad_()

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
            print("ATOMS PER RES:", atoms_per_residue)
            print('Start T', start_torsion_index)
            print('End T',  end_torsion_index)
            print("Start BL", start_bond_angle_index)
            print("End BL", end_bond_angle_index)
            print("Start BD", start_bond_length_index)
            print("End BD", end_bond_length_index)

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

    def energy_steps(self):
        """Model Eval Loop for GDS"""
        # init optimizer
        # if self.partial_optimization:
        #     self.start_point = self.generate_mask(self.start_point, self.idx).unsqueeze(0)
        #     self.start_point.requires_grad_()
        #     print(self.start_point.shape)
        #     optimizer = torch.optim.SGD([self.start_point], lr=self.lr, momentum=0.9)
        # else:
        self.start_point.requires_grad_()
        print("START POINT SHAPE:", self.start_point.shape)
        # optimizer = torch.optim.SGD([self.start_point], lr=self.lr, momentum=0.9)
        optimizer = torch.optim.Adam([self.start_point], lr=self.lr)
        redo = False

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
            out, _, latent_space, _ = self.model(self.start_point, False)
            latent_space = latent_space.detach()
            if step == 1 or redo:
                original_point = latent_space
                redo = False

            # grab the energies (ref2015 score)
            energies = self.energy_scale.inverse_scale(out[:, -self.energy_n:])
            if energies.size(-1) == 3:
                qm = energies[:, 0]
                energy = energies[:, 1]
                dist = torch.cdist(latent_space, original_point, p=2)
                loss = torch.add(*energy, 200*dist)

                print(f"Step: {step}, QM: {energies[:, 0].item()} Ref2015: {energies[:, 1].item()}, Ref2015+Solvent: {energies[:, -1].item()}, Dist: {dist.item()}, To Min Loss: {energy.item()}, Loss: {loss.item()}")
            else:
                energy = energies[:, -1]
                dist = torch.cdist(latent_space, original_point, p=2)
                loss = torch.add(energy, 200*dist)
                self.path[step] = dist
                self.energy_path[step] = energy.detach()
                print(f"Step: {step}, QM: {energies[:, 0].item()}, Ref2015: {energies[:, -1].item()}, Dist: {dist.item()}, To Min Loss: {energy.item()}, Loss: {loss.item()}")

            # back prop
            loss.backward()
            if self.partial_optimization:
                with torch.no_grad():
                    self.start_point.grad[:, self.idx] = 0
                    # self.start_point -= self.lr * self.start_point.grad
                    # print("GRAD", self.start_point.grad)
            optimizer.step()
            
            # test replace values
            if self.partial_optimization:
                with torch.no_grad():
                    self.start_point[:, self.idx] = self.dof_traj[1, self.idx]
            #         print(self.idx)
            #         original_vals = self.dof_traj[0, self.idx]
            #         self.start_point[:, self.idex] = original_vals



            out = out.detach().clone()
            torsion = self.calc_angles(out[:, :116])
            bond_angle = self.calc_angles(out[:, 116:116*2])
            preds = torch.cat((torsion, bond_angle, out[:, 116*2:-3]), dim=-1)
            diff = self.start_point[:, :-2].detach() - self.dof_traj[0, :]
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
    )

    # Search And Output
    energySearch.energy_steps()
    energySearch.dist_energy_plot()
    energySearch.output_GDS()



