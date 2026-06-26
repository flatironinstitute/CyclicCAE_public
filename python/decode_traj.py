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
import datetime

# Import loss function
# from model_functions import sin_cos_loss # this is wrong and needs to be fixed
from model_utils import ConformationalMLPAutoEncoder
from test_case import TestCase1, ConvCase1, ConvCase3, ConvCase3_prefixed, ConvCase4
from model_functions import ScaleOutputs

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
        error_calc: bool = False,
        error_accuracy: bool = False,
        encode_decode: bool = False,
        generate_traj: bool = False,
        start_point: list = [0., 0., 0.],
        end_point: list = [0., 0., 0.],
        full_data: bool = False,
        torsion_only: bool = False,
        ):
        # Assign params
        self.encode_decode_bool = encode_decode
        if not encode_decode:
            if len(trajectories.shape) == 3:
                self.trajectories = trajectories[:, -1, :].double()
            else:
                self.trajectories = trajectories.double()

        self.y_indicies = y_indicies

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # If we are calculating error of a model and not 
        # actually generating dofs
        if error_calc or error_accuracy:
            # Extract out information 
            x = real_data["x"]
            params = real_data["params"]
            torsion_main = params["torsion_main_n"]

            # Assign y test case and trajectories
            if torsion_only:
                if full_data:
                    self.y = x[:, :torsion_main].double()
                else:
                    self.y = x[y_indicies, :torsion_main].double()
                if not self.encode_decode_bool:
                    self.trajectories = trajectories[y_indicies, -1, :].double()
            else:
                if full_data:
                    self.y = x[:, :].double()
                else:
                    self.y = x[y_indicies, :].double()
                if not self.encode_decode_bool:
                    self.trajectories = trajectories[y_indicies, -1, :].double()

        # Load in model weigths
        weights = torch.load(model_weights, map_location=self.device)

        # load in model with params
        # self.model = ConformationalMLPAutoEncoder(
        #     **model_params,
        # ).to(self.device)
        # self.model = ConvCase1(
        # self.model = ConvCase3(
        # self.model = ConvCase3_prefixed(
        # self.model = TestCase1(
        self.model = ConvCase4(
            **model_params,
        ).double().to(self.device)

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
        points = torch.zeros((num_points, 3), device="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.float64)
        for i in range(num_points):
            # get time step value
            t = i / num_points
            # get interpolated point
            C = start_point + (end_point - start_point) * t
            # Add to tensor
            points[i, :] = C
        return points



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
            return self.predictions #[0]
        else:
            # post_latent_skip = self.model.decoder_4_64_skip(self.trajectories)
            # post_latent = self.model.decoder_n_64(post_latent_skip)
            # return self.model.output(self.model.decoder(
            #     torch.add(post_latent, post_latent_skip)
            # ))
            ##### DECODER
            energy_x = self.model.energy_decoder1(self.trajectories)
            energy_skip1 = energy_x.clone()
            # Decoder latent to decoder blocks size 3 -> 64
            x = self.model.decoder_pyramid(self.trajectories)
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

    @property
    def encode_decode(self):
        """Take a DoF Vector of (N, D) samples and encode/decode them"""
        return self.model(self.y.to(self.device), False)

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
        sin_out_torsion, cos_out_torsion = model_out[:, :model_out.size(-1)//2], model_out[:, model_out.size(-1)//2:]

        # Extract sin and cos from angle
        sin_y, cos_y = torch.sin(torch.deg2rad(self.y[:, :58*2])), torch.cos(torch.deg2rad(self.y[:, :58*2]))

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

    def calculate_accuracy(self, model_out: torch.Tensor, angle_out: bool = False, energy: bool = False, tolerance: float = 5., index_in: int = 0, index_out = 58):
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
        if angle_out:
            model_out_angles = self.calc_angles(model_out)
            print("*********************************")
            print('Y-hat', model_out_angles[0, :5])
            print('Y', self.y[0, index_in:index_in+5])
            print("*********************************")
        else:
            model_out_angles = model_out

        # check exact equal measurement
        if energy:
            low_indicies = self.y[:, index_in:] <= 0
            low_abs_diff = torch.abs(model_out_angles[low_indicies] - self.y[:, index_in:][low_indicies])
            mask_high = self.y[:, index_in:] > 0
            high_abs_diff = torch.abs(model_out_angles[mask_high] - self.y[:, index_in:][mask_high])
            abs_diff = torch.abs(model_out_angles - self.y[:, index_in:])
        else:
            abs_diff = torch.abs(model_out_angles - self.y[:, index_in:index_out])

        # check how many fall in the range
        correct_predictions = abs_diff <= tolerance

        if energy:
            low_correct = low_abs_diff <= tolerance
            high_correct = high_abs_diff <= tolerance
            print("low correct:", low_correct)
            print("PRED ACCURACY ENERGY BELOW <= 0:", low_correct.float().mean().item())
            print("PRED ACCURACY ENERGY ABOVE > 0:", high_correct.float().mean().item())

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
    p.add_argument("--cyclic", action="store_true", help="Cyclic PE Encodings")
    p.add_argument("--convolution", action="store_true", help="Apply convolution")
    p.add_argument("--full-data", action="store_true", help="Encode/Decode The Full Dataset")
    p.add_argument("--latent-dim", type=int, default=3, help="Specify Model Latent Dim if Needed (default: 3)")
    args = p.parse_args()

    # load in real data
    real_data = torch.load(args.y_tensor)
    if not args.encode_decode:
        trajectories = torch.load(args.trajectories)
    else:
        trajectories = None

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
    model_params = {"encoder_layers": 3, "decoder_layers": 3, "latent_space_dim": args.latent_dim, "decomposition_n": 1,
                    "cyclic": args.cyclic, 'convolution_layer': args.convolution}
    model_params.update(real_data["params"])

    decode_class = DecodeTrajectories(
        model_weights = args.model_weights,
        trajectories = trajectories,
        model_params = model_params,
        real_data = real_data,
        y_indicies = random_tensor,
        num_samples = args.random_indicies_num,
        error_calc = args.calculate_error,
        error_accuracy = args.calculate_accuracy,
        encode_decode = args.encode_decode,
        generate_traj = args.generate_traj,
        start_point = ast.literal_eval(args.start_point),
        end_point = ast.literal_eval(args.end_point),
        full_data = args.full_data,
    )

    # Scaling for Outputs
    scaleout = ScaleOutputs(real_data["x"][:, -3:])

    if args.encode_decode:
        predictions, _, _, true_y = decode_class.decode
    else:
        predictions = decode_class.decode
    
    # print(predictions[2].max(dim=0))
    # print(predictions[2].min(dim=0))
    angles = predictions[:, :116*2]
    torsion_angles = display_angles(angles[:, :angles.size(-1)//2].cpu())
    bond_angles = display_angles(angles[:, angles.size(-1)//2:].cpu())
    bond_length = predictions[:, 116*2: (116*2 + 58)].cpu()
    # energies = predictions[:, 116*2 + 58:].cpu() #scaleout.inverse_scale(predictions[:, (116*2 + 58):].cpu())
    energies = scaleout.inverse_scale(predictions[:, (116*2 + 58):].cpu())
    print("Energies Preview:")
    print(energies[:3, :])
    angles_out = torch.cat((torsion_angles, bond_angles), dim=-1)
    # angles_out = display_angles(angles)

    # out = torch.cat((angles_out, bond_length, energies), dim=-1)
    out = torch.cat((angles_out, bond_length), dim=-1)

    # print(angles_out[:10, :5])
    # print(angles_out[1, :])
    # print(angles_out[-2, :])
    # print(angles_out[-10:, :10])

    # print(angles_out[-1, :] - angles_out[0, :])

    torch.save(
        # display_angles(predictions),
        out,
        "prediction_traj_{}_{}.pt".format(args.random_indicies_num, datetime.date.today())
    )

    if args.calculate_accuracy:
        print() # add spacing for easier reading
        print("***Torsion Accuracy for {} Samples***".format(args.random_indicies_num))
        accuracy = decode_class.calculate_accuracy(predictions[:, :116].cpu(), tolerance=args.tolerance, index_in=0, index_out=58, angle_out=True)
        print("Mean Accuracy w/ {} tolerance:".format(args.tolerance), accuracy)
        print()# add spacing for easier reading
        print("***Bond Angle Accuracy for {} Samples***".format(args.random_indicies_num))
        accuracy = decode_class.calculate_accuracy(predictions[:, 116:116*2].cpu(), tolerance=args.tolerance, index_in=58, index_out=116, angle_out=True)
        print("Mean Accuracy w/ {} tolerance:".format(args.tolerance), accuracy)
        print()# add spacing for easier reading
        print("***Bond Length Accuracy for {} Samples***".format(args.random_indicies_num))
        accuracy = decode_class.calculate_accuracy(predictions[:, 116*2: (116*2 + 58)].cpu(), tolerance=0.1, index_in=116, index_out=116+58, angle_out=False)
        print("Mean Accuracy w/ {} tolerance:".format(0.01), accuracy)
        print()# add spacing for easier reading
        print("***Energy Ref2015 & Ref2015+fa_sal Accuracy for {} Samples***".format(args.random_indicies_num))
        accuracy = decode_class.calculate_accuracy(energies[:, -2:], tolerance=2, index_in=116+58+1, energy=True, index_out=116+58+2, angle_out=False)
        print("Mean Accuracy w/ {} tolerance:".format(2), accuracy)
        print()# add spacing for easier reading

    if args.calculate_error:
        print("***Torsion Error for {} Samples***".format(args.random_indicies_num))
        error, sin_error, cos_error = decode_class.calculate_error(predictions[:, :116].cpu())
        print("Total Error:", error)
        print("Sin Error:", sin_error)
        print("Cos Error:", cos_error)
        print("***Bond Angle Error for {} Samples***".format(args.random_indicies_num))
        error, sin_error, cos_error = decode_class.calculate_error(predictions[:, 116:].cpu())
        print("Total Error:", error)
        print("Sin Error:", sin_error)
        print("Cos Error:", cos_error)
    print()# add spacing for easier reading
