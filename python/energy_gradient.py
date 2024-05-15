#!/usr/bin/env python
#
# @author: Andrew Powers (apowers@flatironinstitute.org)
# @brief: Function to carry out Gradient Descent Marching along energy
#       term, during evaluation mode.
#
#


# Import Packages
import os, sys
import datetime
import argparse

# import torch
import torch
import torch.nn as nn

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
        """
        # Pull up points
        self.steps = steps
        self.minima = minima
        self.outstep = steps
        self.lr = lr
        self.cutoff = cutoff

        # Grab values required for model and other parts
        dataset = torch.load(dataset)
        PARAMS = dataset["params"]
        X = dataset["x"].double()
        self.energy_n = PARAMS["energy_n"]
        # if X.size(-1) != 177:
        #     X = torch.cat((X, ))

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

        # Pull out scaled start point
        self.start_point = scaled_X_start_point[-1, :]
        self.start_point = self.start_point.unsqueeze(0)
        # self.start_point = self.start_point.to("cuda" if torch.cuda.is_available() else "cpu")
        self.start_point = self.start_point.to("cpu")
        self.start_point.requires_grad_()
        print("Check to see if Tensor requires Grad now: ", self.start_point.requires_grad)

        # scale energy for conversion importance
        self.energy_scale = ScaleOutputs(X[:, -self.energy_n:])

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

    def energy_steps(self):
        """Model Eval Loop for GDS"""
        # init optimizer
        optimizer = torch.optim.SGD([self.start_point], lr=self.lr, momentum=0.9)

        print('Eval Mode:', not self.model.training)
        if self.sample_energy.size(0) == 3:
            print(f"Step: 0, QM: {self.sample_energy[0].item()}, Ref2015: {self.sample_energy[1].item()}, Ref2015+Solvent: {self.sample_energy[-1].item()}, Data Taken Straight From Input")
        else:
            print(f"Step: 0, QM: {self.sample_energy[0].item()}, Ref2015: {self.sample_energy[-1].item()}, Data Taken Straight From Input")

        for step in range(1, self.steps):
            # zero grad
            optimizer.zero_grad()

            # out from model
            out, _, latent_space, _ = self.model(self.start_point, False)
            latent_space = latent_space.detach()
            if step == 1:
                original_point = latent_space
            # grab the energies (ref2015 score)
            energies = self.energy_scale.inverse_scale(out[:, -self.energy_n:])
            if energies.size(-1) == 3:
                qm = energies[:, 0]
                energy = energies[:, 1]
                loss_from_min = torch.mean(torch.pow(0.0 - (energies[:, 1] - self.minima_energy[1]), 2))
                dist_loss, dist = self.euclidean_measure(
                    cutoff = self.cutoff, 
                    latent_space = latent_space,
                    original_point = original_point,
                )
                loss = torch.add(loss_from_min, dist_loss)

                print(f"Step: {step}, QM: {energies[:, 0].item()} Ref2015: {energies[:, 1].item()}, Ref2015+Solvent: {energies[:, -1].item()}, Dist: {dist_loss.item()}, To Min Loss: {loss_from_min.item()}, Loss: {loss.item()}")
            else:
                energy = energies[:, -1]
                loss_from_min = torch.mean(torch.pow(0.0 - (energies[:, 1] - self.minima_energy[1]), 2))
                dist_loss, dist = self.euclidean_measure(
                    cutoff = self.cutoff, 
                    latent_space = latent_space,
                    original_point = original_point,
                )
                self.path[step] = dist
                loss = torch.add(loss_from_min, dist_loss)
                print(f"Step: {step}, QM: {energies[:, 0].item()}, Ref2015: {energies[:, -1].item()}, Dist: {dist_loss.item()}, To Min Loss: {loss_from_min.item()}, Loss: {loss.item()}")

            # Turn the search around to keep it local
            if dist_loss.item() > 0:
                print(f"Step {step}: Condition met, adjusting path")

                # Compute the gradient manually
                grad = torch.autograd.grad(outputs=loss, inputs=self.start_point, create_graph=True)[0]

                noise = torch.randn_like(self.start_point) * 10

                # Adjust the gradient direction to steer the point back within the cutoff distance
                adjustment = self.start_point / dist * ((dist - self.cutoff) * 2) + noise

                # Update the starting point manually
                with torch.no_grad():
                    self.start_point += 1e-2 * (grad + adjustment)
            else:
                # back prop
                loss.backward()
                optimizer.step()


            out = out.detach().clone()
            torsion = self.calc_angles(out[:, :116])
            bond_angle = self.calc_angles(out[:, 116:116*2])
            preds = torch.cat((torsion, bond_angle, out[:, 116*2:-3]), dim=-1)

            self.dof_traj[step] = preds

            # stop check
            if energy <= self.minima:
                self.outstep = step
                print("Minima {self.minima} found at Step: {step}")
                break
            
            # reinstantiate the start point
            # self.start_point = self.start_point.detach().clone().requires_grad_()

    def output_GDS(self):
        """Output the GDS trajectory"""
        # file name
        file = f"finalstep{self.outstep}-{datetime.date.today()}-trajectories.pt"
        torch.save(self.dof_traj, file)
        torch.save(self.path, 'trajpath.pt')
        

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=1000, help="Search Steps Max (default: 1000)")
    p.add_argument("--minima", type=float, default=-4.775, help="Minima energy to search for (default: -5.0)")
    p.add_argument("--start-point", type=int, default=14, help="This will be updated to take a pdb in the future")
    p.add_argument("--weights-load", type=str, help="Specify Model Weights To Use")
    p.add_argument("--lr", type=float, default=0.001, help="Optimizer Learning Rate (default: 1e-3)")
    p.add_argument("--dataset", type=str, default="./binary_import/dof_vectors2024-04-10torch_tensor.pt", help="Dataset to use (default: dof_vectors2024-04-10torch_tensor)")
    p.add_argument("--cutoff", type=float, default=0.005, help="Specify Euclidean Distance Cutoff in Latent Space to Search (default: 0.005)")
    args = p.parse_args()


    # init the start path with X tensor 14 
    energySearch = EnergyPath(steps = args.steps, minima = args.minima, start_point = torch.tensor(args.start_point), weight_passed = args.weights_load, lr=args.lr,
                              dataset = args.dataset, cutoff = args.cutoff)

    # Search And Output
    energySearch.energy_steps()
    energySearch.output_GDS()



