#!/usr/bin/env python
#
# @author: Andrew Powers (apowers@flatironinstitute.org)
# @brief: Create a trajectory through the latent space and decode poses
#
#

### Packages

# torch packages
import torch
import torch.nn as nn

# numpy for choice
import numpy as np

# native packages
from collections import defaultdict
import os, sys
import argparse
import math
import ast

# Import loss function
# from model_functions import sin_cos_loss # this is wrong and needs to be fixed
from model_utils import ConformationalMLPAutoEncoder

#### Class

class DecodeTrajectories:
    def __init__(
        self,
        model_weights: torch.Tensor,
        trajectories: torch.Tensor,
        model_params: dict,
        real_data: torch.Tensor,
        y_indicies: torch.Tensor = None,
        num_samples: int = 0,
        error_calc: bool = True,
        encode_decode: bool = False,
        generate_traj: bool = False,
        start_point: list = [0., 0., 0.],
        end_point: list = [0., 0., 0.],
        ):
        # Assign params
        self.encode_decode_bool = encode_decode
        if len(trajectories.shape) == 3:
            self.trajectories = trajectories[:, -1, :].double()
        else:
            self.trajectories = trajectories.double()
        self.y_indicies = y_indicies

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # If we are calculating error of a model and not 
        # actually generating dofs
        if error_calc:
            # Extract out information 
            x = real_data["x"]
            params = real_data["params"]
            torsion_main = params["torsion_main_n"]

            # Assign y test case and trajectories
            self.y = x[y_indicies, :torsion_main].double()
            if not self.encode_decode_bool:
                self.trajectories = trajectories[y_indicies, -1, :].double()


        # Load in model weigths
        weights = torch.load(model_weights, map_location=self.device)

        # load in model with params
        self.model = ConformationalMLPAutoEncoder(
            **model_params,
        ).to(self.device)

        # assign weights to model
        self.model.load_state_dict(weights["model_state_dict"])

        # Convert model to eval mode, so there is no confusion
        self.model.eval()

        # calculate trajectories if encode_decode
        if self.encode_decode_bool:
            self.predictions = self.encode_decode

        if generate_traj:
            self.trajectories = self.interpolate_traj(
            start_point = torch.tensor(start_point),
            end_point = torch.tensor(end_point),
            num_points = num_samples,
            )
            print(self.trajectories)


    def interpolate_traj(
        self, 
        start_point: torch.Tensor,
        end_point: torch.Tensor,
        num_points: int
        ):
        """Interpolate a linear path of points from one point to another along a straight line"""
        # Init points list
        points = torch.zeros((num_points, 3), device="cuda" if torch.cuda.is_available() else "cpu")
        for i in range(num_points):
            # get time step value
            t = i / num_points
            # get interpolated point
            C = start_point + (end_point - start_point) * t
            # Add to tensor
            points[i, :] = C
        return points.double()



    @property
    def decode(self):
        """Take a set of (N,3) points in R^3 within the latent space and
        decode them into DoF vectors.
        
        PARAMS
        ------
            trajectories: torch.Tensor
                (N,3) Tensor that are points within the latent space
        """
        if self.encode_decode_bool:
            return self.predictions[0]
        else:
            post_latent_skip = self.model.decoder_4_64_skip(self.trajectories)
            post_latent = self.model.decoder_4_64(post_latent_skip)
            return self.model.output(self.model.decoder(
                torch.add(post_latent, post_latent_skip)
            ))

    @property
    def encode_decode(self):
        """Take a DoF Vector of (N, D) samples and encode/decode them"""
        return self.model(self.y.to(self.device))

    def calculate_error(self, model_out: torch.Tensor):
        """Calculate the Error between the two vectors for evaluation

        PARAMS
        ------
            model_out: torch.Tensor
                (N, D) Tensor that has final model predictions for the latent
                space trajectory points

            self.y: torch.Tensor
                (N, D//2) Tensor that has the real angles from the passed in DoF
                Vector
        """
        # Grab sin and cos values separately
        sin_out, cos_out = model_out[:, :model_out.size(-1)//2], model_out[:, model_out.size(-1)//2:]

        # Extract sin and cos from angle
        sin_y, cos_y = torch.sin(torch.deg2rad(self.y)), torch.cos(torch.deg2rad(self.y))

        # Calculate diff
        sin_diff = torch.pow( sin_y - sin_out, 2)
        cos_diff = torch.pow( cos_y - cos_out, 2)

        # calculate total mean diff
        mean_diff = torch.mean(sin_diff + cos_diff)

        return mean_diff.detach().cpu().item(), torch.mean(sin_diff).detach().cpu().item(), torch.mean(cos_diff).detach().cpu().item()

    def calc_angles(self, preds: torch.Tensor):
        """Display the Computed Angles"""
        sin_pred = preds[:, :preds.size(-1)//2]
        cos_pred = preds[:, preds.size(-1)//2:]

        return torch.rad2deg(torch.atan2(sin_pred, cos_pred))

    def calculate_accuracy(self, model_out: torch.Tensor, tolerance: int = 5):
        """Calculate the Accuracy between the two vectors for evaluation

        PARAMS
        ------
            model_out: torch.Tensor
                (N, D) Tensor that has final model predictions for the latent
                space trajectory points

            self.y: torch.Tensor
                (N, D//2) Tensor that has the real angles from the passed in DoF
                Vector

            tolerance: int
                How much away from each other the values are still considered okay
        """
        # Convert to int for comparison
        model_out_angles = self.calc_angles(model_out)

        # check exact equal measurement
        abs_diff = torch.abs(model_out_angles - self.y)

        # check how many fall in the range
        correct_predictions = abs_diff <= tolerance

        # Return accuracy mean
        return correct_predictions.float().mean().item()


def display_angles(preds: torch.Tensor):
    """Display the Computed Angles"""
    sin_pred = preds[:, :preds.size(-1)//2]
    cos_pred = preds[:, preds.size(-1)//2:]

    return torch.rad2deg(torch.atan2(sin_pred, cos_pred))
        
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--trajectories", type=str, default="./outputs/data_tensor/model_run_2763_latent_tensor.pt",
                   help="Specify Latent Points, must be Tensor (N,3) \
                   (default: ./outputs/data_tensor/model_run_2763_latent_tensor.pt)")
    p.add_argument("--y-tensor", type=str, default="./binary_import/dof_vectors_clusters_firstten2024-03-01torch_tensor.pt",
                   help="Dataset to compare against (default: ./binary_import/dof_vectors_clusters_firstten2024-03-01torch_tensor.pt)")
    p.add_argument("--model-weights", type=str, default="./outputs/model_weights/model_2763_current_weights.pt",
                   help="Model weights to use (default: ./outputs/model_weights/model_2763_current_weights.pt)")
    p.add_argument("--calculate-error", action="store_true", help="This is for evaluation purposes and generates error terms.")
    p.add_argument("--calculate-accuracy", action="store_true", help="This is for evaluation purposes and generates accuracy term.")
    p.add_argument("--random-indicies-num", type=int, default=10, help="Number of indicies to randomly select for error calculation \
                    (default: 10)")
    p.add_argument("--encode-decode", action="store_true", help="Run an unseen dataset for evaluation metric")
    p.add_argument("--tolerance", type=int, default=5, help="Amount +- around angle that is still considered correct (default: 5)")
    p.add_argument("--generate-traj", action="store_true", help="Interpolate random-indicies-num amount of points. \
                    Arguments start-point and end-point need to be set as well.")
    p.add_argument("--start-point", type=str, default="[0., 0., 0.,]", help="Pass float list of start-point")
    p.add_argument("--end-point", type=str, default="[0., 0., 0.,]", help="Pass float list of end-point")
    args = p.parse_args()

    # load in real data
    real_data = torch.load(args.y_tensor)
    trajectories = torch.load(args.trajectories)

    print('Y tensor size:', real_data["x"].shape)

    # Pull out random values for testing 
    if (args.calculate_error) or (args.calculate_accuracy):
        random_tensor = torch.tensor(
            np.random.choice(real_data["x"].size(0), size=args.random_indicies_num, replace=False)
        )
    else:
        random_tensor = None

    # Explicit model params, but will change the trajectories tensor to include it in the future
    # model_params = {"decoder_layers": 1, "encoder_layers": 6, "latent_space_dim": 3, "decomposition_n": 2}
    model_params = {"decoder_layers": 6, "encoder_layers": 6, "latent_space_dim": 3, "decomposition_n": 4}
    model_params.update(real_data["params"])

    decode_class = DecodeTrajectories(
        model_weights = args.model_weights,
        trajectories = trajectories,
        model_params = model_params,
        real_data = real_data,
        y_indicies = random_tensor,
        num_samples = args.random_indicies_num,
        error_calc = args.calculate_error,
        encode_decode = args.encode_decode,
        generate_traj = args.generate_traj,
        start_point = ast.literal_eval(args.start_point),
        end_point = ast.literal_eval(args.end_point),
    )

    predictions = decode_class.decode

    torch.save(
        display_angles(predictions),
        "prediction_traj_{}.pt".format(args.random_indicies_num)
    )

    if args.calculate_accuracy:
        print() # add spacing for easier reading
        print("***Accuracy for {} Samples***".format(args.random_indicies_num))
        accuracy = decode_class.calculate_accuracy(predictions.cpu(), tolerance=args.tolerance)
        print("Mean Accuracy w/ {} tolerance:".format(args.tolerance), accuracy)
        print()# add spacing for easier reading

    if args.calculate_error:
        print("***Error for {} Samples***".format(args.random_indicies_num))
        error, sin_error, cos_error = decode_class.calculate_error(predictions.cpu())
        print("Total Error:", error)
        print("Sin Error:", sin_error)
        print("Cos Error:", cos_error)
    print()# add spacing for easier reading
