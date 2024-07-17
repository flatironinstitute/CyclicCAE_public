#!/usr/bin/env python
#
# @brief: File for separate housing of train/test functions of the model.
# @author: Andrew Powers (apowers@flatironinstitute.org) 
#

##################################################################################################################
#                                               Packages 
##################################################################################################################

# base python packages
from collections import defaultdict
import os
import random

# ml imports
import torch
import torch.nn.init as init
import torch.nn as nn

# function import
# from terminal_error import terminal_peptide_loss

##################################################################################################################
#                                               Classes
##################################################################################################################

class ScaleOutputs:
    def __init__(
        self,
        input_tensor: torch.Tensor,
        ):
        """Take the predicted value and scale them, but store min/max for inverse scaling

        PARAMS
        ------
        input_tensor : Tensor
            Input tensors that will be predicted, but not encoded.
        """
        # store input tensor
        self.input_tensor = input_tensor
        # create holding min/max values
        self.max = torch.max(input_tensor, dim=0).values
        self.min = torch.min(input_tensor, dim=0).values
        # scaled values
        self.scaled_vals = self.scale_values.to("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def scale_values(self):
        """Perform Scaling to [0,1]"""
        return (self.input_tensor - self.min) / (self.max - self.min)

    def inverse_scale(self, scaled_input: torch.Tensor):
        """Perform Inverse Scaling which goes from [0,1] back to the original magnitude."""
        return scaled_input.cpu() * (self.max - self.min) + self.min

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

##################################################################################################################
#                                                Functions
##################################################################################################################

def scale(x: torch.Tensor, params: dict):
    """Scale all input values"""
    # clone
    out = x.clone()

    # starting spot in tensor
    linear_terms_start = params["torsion_main_n"] + params["torsion_side_n"] + params["bond_angle_main_n"] + params["bond_angle_side_n"]

    # calculate bond length size
    bond_length_size = params["bond_length_main_n"] + params["bond_length_side_n"]

    # grab out each section
    # Scale Bond Length
    out[:, linear_terms_start: linear_terms_start + bond_length_size ] = normalize_vals(
        out[:, linear_terms_start: linear_terms_start + bond_length_size ] 
    )

    # Scale Jumps if any
    if params["jump_n"] != 0:
        jump_end = linear_terms_start + bond_length_size + params["jump_n"]
        jump_start = linear_terms_start + bond_length_size
        out[:, jump_start : jump_end ] = normalize_vals(
            out[:, jump_start : jump_end ]
        )

    # Scale Energy if any
    if params["energy_n"] != 0:
        energy_end = linear_terms_start + bond_length_size + params["jump_n"] + params["energy_n"]
        energy_start = linear_terms_start + bond_length_size + params["jump_n"]
        out[:, energy_start : energy_end ] = normalize_vals(
            out[:, energy_start : energy_end ]
        )

    return out


def normalize_vals(x: torch.Tensor, norm_min:float = -1., norm_max:float = 1.):
    """Set terms between -1 and 1"""
    min_val = x.min()
    max_val = x.max()
    norm_feature = (x - min_val) / (max_val - min_val) * (norm_max - norm_min) + norm_min
    return norm_feature
    

def sklearn_scale(x: torch.Tensor):
    """pytorch implementation of sklearn scaler

    Found here: https://discuss.pytorch.org/t/pytorch-tensor-scaling/38576"""
    m = x.mean(0, keepdim=True)
    s = x.std(0, unbiased=False, keepdim=True)
    x -= m
    x /= s
    return x


def angle_loss_term(y_hat: torch.Tensor, y: torch.Tensor):
    """Angle Loss function taken from here:
    https://discuss.pytorch.org/t/custom-loss-function-for-discontinuous-angle-calculation/58579/13
    """
    sin_hat, sin_y = y_hat[:, :y_hat.size(-1)//2], y[:, :y.size(-1)//2]
    cos_hat, cos_y = y_hat[:, y_hat.size(-1)//2:], y[:, y.size(-1)//2:]
    # calc angles
    angle_hat = torch.rad2deg(torch.atan2(sin_hat, cos_hat))
    angle_y = torch.rad2deg(torch.atan2(sin_y, cos_y))
    # clamp with where
    angle_y = torch.where(angle_y.abs()<0.9, torch.tensor(0.0), angle_y)
    angle_hat = torch.where(angle_hat.abs()<0.9, torch.tensor(0.0), angle_hat)
    # return torch.mean(torch.sqrt(1. - torch.cos(y - y_hat)))
    arccos_diff = torch.arccos(torch.cos(angle_y - angle_hat))
    return torch.mean(torch.pow(arccos_diff, 2))

def linear_loss_term(y_hat: torch.Tensor, y: torch.Tensor, scale:int = 1):
    """General L2 Linear Loss"""
    return torch.mean(torch.pow((y - y_hat)*scale, 2))

def linear_energy_loss_term(y_hat: torch.Tensor, y: torch.Tensor, scale:int = 1):
    """General Energy Exponential L2 Linear Loss"""
    return torch.mean(torch.pow(torch.exp(y - y_hat)*scale, 2))

def angle_sin2_cos2_enforce(x: torch.Tensor):
    """Enforce that the angle decomposition values sum to 1"""
    # grab values
    sinx = x[:, :x.size(-1)//2]
    cosx = x[:, x.size(-1)//2:]
    # Extract Values
    sin2 = torch.pow(sinx, 2)
    cos2 = torch.pow(cosx, 2)
    # compute sum
    theta = sin2 + cos2
    theta = theta.view(-1)
    return torch.mean( torch.pow( theta - 1., 2) )

def SSE_latent_RMSD_loss_term(rmsd_real: torch.Tensor, rmsd_latent: torch.Tensor, device: str):
    """Custom (B) loss function to enforce latent space RMSD distances to be comparable 
    with real RMSD distances.

    PARAMS
    ------
        rmsd_real: torch.Tensor
            The batch susbet RMSDs all vs all from the actual structures

        rmsd_latent: torch.Tensor
            The batch subset RMSDs all vs all from the latent space structures

    RETURNS
    -------
        loss: torch.float
            Summed Square Error between RMSD_real and RMSD_latent
    """
    # extract sizes
    B, _ = rmsd_latent.shape
    
    # make bool mask
    mask = torch.eye(B, B, dtype=torch.bool).to(device)

    # Make sure that i /= j within the tensors. Therefore we can block
    # out the diagonal with zeros on both tensors
    # rmsd_real.masked_fill_(mask, 0)
    # rmsd_latent.masked_fill_(mask, 0)
    rmsd_real[mask] = 0.
    rmsd_latent[mask] = 0.

    # Need to set lambda some how right now I will just randomly fill in with values
    lambda_real   = 2
    lambda_latent = 2

    # Compute each component of the function sum((real_component - latent_component)**2)
    # Real: e^{-(RMSDreal_{ij} / lambda_real)**2}
    real_component = torch.exp( -( torch.pow( torch.div( rmsd_real, lambda_real ), 2 ) ) )

    # Latent: e^{-(RMSDlatent_{ij,latent} / lambda_latent)**2}
    latent_component = torch.exp( -( torch.pow( torch.div( rmsd_latent, lambda_latent ), 2 ) ) )

    # fill nans
    # real_component = torch.nan_to_num(real_component, nan=torch.nan_to_num(real_component).max())
    # latent_component = torch.nan_to_num(latent_component, nan=torch.nan_to_num(latent_component).max())

    # now sum the information
    loss = torch.sum( torch.pow( real_component - latent_component, 2 ) )
    loss = torch.div(loss, rmsd_latent.size(0)**2)
    # loss = torch.mean( torch.pow( real_component - latent_component, 2 ) )
    return loss

def lambda_weight_decay(epoch: int, start_weight: int, end_weight: int, start_decay_weight: int, end_decay_weight: int):
    if epoch < start_decay_weight:
        return end_weight, start_weight
    else:
        # decay rate
        decay_rate = (start_weight - end_weight) / (end_decay_weight - start_decay_weight)
        # apply 
        lambda_latent = max(1000 - decay_rate * (epoch - start_decay_weight), end_weight)
        lambda_rec = min(1000 - lambda_latent, 100)
    return lambda_rec, lambda_latent

def loss_terms_reg(term1: torch.float32, term2: torch.float32, threshold:int=10):
    """Compute terms to keep terms on the same scale for equivalent backprop

    PARAMS
    ------
        term1: torch.float32
            reconstruction loss term

        term2: torch.float32
            Latent space enforcing term

        threshold: int
            The magnitude that we do not want to go over
    """
    # get ratio of terms
    ratio = max(term1, term2) / min(term1, term2)

    # if less than threshold return 1 since nothign needs to be changed
    if ratio < threshold:
        return 1, 1
    
    # if one is less than other that needs reweighting
    if term1 < term2:
        return ratio, 1
    else:
        return 1, ratio

def triangle_inequality(latent_space: torch.Tensor):
    """Perform Triangle Inequality Enforcement In Loss

    PARAMS
    ------
        latent_batch: torch.Tensor
            Latent points of shape (B, 3)
    """
    # Samples ins Batch
    B = latent_space.size(0)

    # Calculate distances
    dists = torch.cdist(latent_space, latent_space, p=2)

    # generate sample numbers
    samples = torch.arange(B)

    samples = samples.view(-1, 1).repeat(1, B)

    # get loss
    loss = torch.max(dists - dists[samples, (samples.T + 1) % B] - dists[(samples + 1) % B, samples.T], torch.zeros_like(dists))
    return loss.mean()

def energy_loss_term(min_vals: torch.Tensor, pred_energy: torch.Tensor, real_energy: torch.Tensor, z: torch.Tensor):
    """Exponential Function to enforce low energies are low and high energies blow up

    PARAMS
    ------
        min_vals: torch.Tensor
            min_vals to use for scaling the values to go to zero.

        pred_energy: torch.Tensor
            Model energy Predictions

        real_energy: torch.Tensor
            Datset Rosetta predicted energy terms
    """
    # Boltzman Temp Constant
    kbt = 0.63 #2

    # scale inputs
    pred_energy = torch.subtract(pred_energy, min_vals)
    real_energy = torch.subtract(real_energy, min_vals)

    # Sum of Entire Dataset
    # z = torch.pow(torch.sum(real_energy / kbt, dim=0), 2)
    # print('z nan:', z.isnan().sum())

    # Sum differences
    # kbt_real = -real_energy / kbt
    # kbt_pred = -pred_energy / kbt
    # print('real kbt', kbt_real)
    # print('pred kbt', kbt_pred)

    real_e = torch.exp(-(real_energy / kbt))
    pred_e = torch.exp(-(pred_energy / kbt))
    # print('real e', real_e)
    # print('pred e', pred_e)

    # take the difference of each other
    diff_e = torch.pow(real_e - pred_e, 2)
    # print('diff e', diff_e)

    # normalize by z
    return torch.sum(torch.sum(diff_e, dim=0) / z)

def modulo_loss(pred: torch.Tensor, real: torch.Tensor):
    """Range loss between angles to be [-pi, pi] using modulo operator on euclidean space"""
    sin_preds, sin_real = pred[:, :pred.size(-1)//2], real[:, :real.size(-1)//2]   # Extract sin from vector
    cos_preds, cos_real = pred[:, pred.size(-1)//2:], real[:, real.size(-1)//2:]   # Extract cos from vector
    diff = torch.atan2(sin_preds, cos_preds) - torch.atan2(sin_real, cos_real)        # Calculate radian difference (might be fine if in degrees as well)
    modulo = (diff + (2*torch.pi)) % (2*torch.pi)                                    # Scale the diff by 2*pi and take the modulos [0, 2pi)
    mask = modulo > torch.pi                                                         # Grab indices of values larger then pi
    modulo[mask] -= 2*torch.pi                                                       # Scale these values by -2*pi making the range [-pi, pi]
    return torch.pow( torch.norm(modulo, dim=-1), 2).sum()


def loss_function(
    y_hat_angle: torch.Tensor,
    y_angle: torch.Tensor,
    # decomposed_angle: torch.Tensor,
    bond_lengths_hat: torch.Tensor,
    bond_lengths: torch.Tensor,
    energy_hat: torch.Tensor,
    energy: torch.Tensor,
    # y_hat_linear: torch.Tensor,
    # y_linear: torch.Tensor,
    rmsd_real: torch.Tensor,
    rmsd_latent: torch.Tensor,
    device: str,
    epoch: int,
    angle_size: int,
    min_vals: torch.Tensor,
    z : torch.Tensor,
    term_class,
    real_terminal_lengths: torch.Tensor,
    dof_out: torch.Tensor,
    params: dict,
    ):
    """Combination of Loss terms take the mean"""
    # Set up loss function
    sin_cos_linear_loss = CustomSinCosLoss(
        sin_weight=1.0,
        cos_weight=1.0,
    )

    # terminal peptide regression loss
    terminal_loss = term_class.terminal_peptide_loss(dof_out, real_terminal_lengths, params, foldtree_root=4, device=device).mean()

    # Compute individual angle loss terms
    # torsion_loss = angle_loss_term(y_hat_angle[:, :y_hat_angle.size(-1)//2], y_angle[:, :y_hat_angle.size(-1)//2])
    # bond_angle_loss = angle_loss_term(y_hat_angle[:, y_hat_angle.size(-1)//2:], y_angle[:, y_hat_angle.size(-1)//2:])
    # torsion_loss = linear_loss_term(y_hat_angle[:, :y_hat_angle.size(-1)//2], y_angle[:, :y_hat_angle.size(-1)//2])
    # bond_angle_loss = linear_loss_term(y_hat_angle[:, y_hat_angle.size(-1)//2:], y_angle[:, y_hat_angle.size(-1)//2:])
    # torsion_loss = sin_cos_linear_loss(y_hat_angle[:, :y_hat_angle.size(-1)//2], y_angle[:, :y_hat_angle.size(-1)//2])
    # print("Full y_hat_angle", y_hat_angle.shape)
    # print("torsion y_hat_angle", y_hat_angle[:, y_hat_angle.size(-1)//2:].shape)
    torsion_loss = sin_cos_linear_loss(y_hat_angle[:, :y_hat_angle.size(-1)//2], y_angle[:, :y_angle.size(-1)//2]) * 100_000
    # torsion_loss = modulo_loss(y_hat_angle[:, :y_hat_angle.size(-1)//2], y_angle[:, :y_angle.size(-1)//2]) #* 100_000
    # torsion_loss += angle_loss_term(y_hat_angle, y_angle) * 10000

    # Uncomment This
    # bond_angle_loss = torch.tensor(0.0) 
    # print("bond angle y_hat_angle", y_hat_angle[:, :y_hat_angle.size(-1)//2].shape)
    bond_angle_loss = sin_cos_linear_loss(y_hat_angle[:, y_hat_angle.size(-1)//2:], y_angle[:, y_angle.size(-1)//2:]) * 100_000
    # bond_angle_loss = modulo_loss(y_hat_angle[:, y_hat_angle.size(-1)//2:], y_angle[:, y_angle.size(-1)//2:]) #* 100_000

    # This is constrain loss for sin/cos pairs
    # constrain_loss_torsion = angle_sin2_cos2_enforce(y_hat_angle[:, :y_hat_angle.size(-1)//2])
    # constrain_loss_bond_angle = angle_sin2_cos2_enforce(y_hat_angle[:, y_hat_angle.size(-1)//2:])
    # constrain_loss = constrain_loss_torsion + constrain_loss_bond_angle
    # constrain_loss = torch.tensor(0.0)
    # constrain_loss = linear_loss_term(enforce_theta, ones.to(device))

    # Compute individual linear loss terms
    bond_length_loss = linear_loss_term(bond_lengths_hat, bond_lengths) * 1000
    # bond_length_loss = torch.tensor(0.0)
    # energy_loss = torch.tensor(0.0)
    energy_loss = linear_loss_term(energy_hat, energy) * 10_000
    # energy_loss = energy_loss_term(min_vals, energy_hat, energy, z) * 100_000_000
    # print("ENERGY LOSS:", energy_loss)

    # extra latent space loss term
    enforce_latent_rmsd = SSE_latent_RMSD_loss_term(rmsd_real, rmsd_latent, device) * 100 #* 100_000
    # enforce_latent_rmsd = torch.tensor(0.0)
    # triangle_inequality_term = triangle_inequality(rmsd_latent) * 100
    triangle_inequality_term = torch.tensor(0.0)

    # reconstruction individual weights
    lambda_lin, lambda_ang = 1, 1

    # linear loss
    linear_loss = (bond_length_loss) * lambda_lin

    # torsion & bond angle weights
    lambda_tor, lambda_bond = 1, 1

    # angle loss
    angle_loss = (torsion_loss*lambda_tor + bond_angle_loss*lambda_bond) * lambda_ang

    # Combine Loss Terms for reconstruction
    reconstruction_loss = angle_loss + linear_loss

    # Combine Loss Terms for LatentEnforce
    latent_enforce = enforce_latent_rmsd + energy_loss

    # compute lambda weighting terms for full loss
    lambda_rec, lambda_latent = 1, 1

    # print("LOSS NO TERMINAL:", reconstruction_loss*lambda_rec + latent_enforce*lambda_latent)
    # print("LOSS W/ TERMINAL:", reconstruction_loss*lambda_rec + latent_enforce*lambda_latent + terminal_loss)
    # print("TERMINAL LOSS:", terminal_loss)

    return reconstruction_loss*lambda_rec + latent_enforce*lambda_latent + terminal_loss, angle_loss, linear_loss, enforce_latent_rmsd, energy_loss, terminal_loss
    # return enforce_latent_rmsd, torch.tensor([0]), torch.tensor([0]), enforce_latent_rmsd

    

def train(
    model: torch.nn.Module,
    train_dataset: torch.utils.data.DataLoader,
    val_dataset: torch.utils.data.DataLoader,
    epochs: int,
    begin_epoch: int,
    device: str,
    optimizer: torch.optim,
    scheduler: torch.optim.lr_scheduler,
    angle_n: int,
    linear_n: int,
    file_out_track: str,
    output_dir: str,
    weight_save_n: int,
    file_num: int,
    real_rmsd_mat: torch.Tensor,
    latent_plotly: int,
    sample_n: int,
    real_X: torch.Tensor,
    early_stop: object,
    latent_dim: int,
    energy_n: int,
    overfit: bool,
    noise_iterations: int,
    scale_outputs,
    min_vals: torch.Tensor,
    z: torch.Tensor,
    params: dict,
    real_terminal_lengths: torch.Tensor,
    term_class,
    ):
    """General Train function for the AutoEncoder Model

    PARAMS
    ------
        model: torch.nn.Module
            An Auto-Encoder based pytorch model

        dataset: torch.utils.data.DataLoader
            dataset loaded into batches. The dataset
            has only an X portion, since we are 
            trying to recreate X

        epochs: int
            Number of epochs to train for

        begin_epoch: int
            Starting Place of training epoch
            
        device: str
            This will be either cpu or cuda

        optimizer: torch.optim
            Particular optimizer that you want to use

        scheduler: torch.optim.lr_scheduler
            A Learning rate scheduler object

        angle_n: int
            Number of angles within the dataset

        linear_n: int
            Number of linear terms within the dataset

        file_out_track: str
            Specific file with tracked info path

        output_dir: str
            Path to output dir where weights are saved

        weight_save_n: int
            Every N epoch save the model's weights

        file_num: int
            Random generated number for specification between files

        real_rmsd_mat: torch.Tensor
            The RMSD between all i,j structures within the real space.

        latent_plotly: int
            What N epochs to extract latent space coordinates for each sample

        sample_n: int
            Number of samples within X

    RETURNS
    -------
        mean_loss: float
            The mean loss of the total run after some N
            number of epochs.

        lowest_loss: float
            The lowest loss encountered when training

        epoch_lowest_loss: int
            The epoch of when the lowest loss occurred

        latent_plotly_tensor: torch.Tensor
            Extracted latent space tensor for each sample at every N epoch
    """
    # init total loss track
    loss_all_track = list()

    # init lowest loss tracker
    lowest_loss = float('inf')
    epoch_lowest_loss = 0

    #init step tracker
    step_size = 0

    # set device and model to train
    model = model.to(device)
    model.train(True)

    # init model weights
    # model.apply(xavier_init)

    # latent holding tensor
    latent_plotly_tensor = torch.zeros((sample_n, (epochs//latent_plotly)+1, latent_dim), dtype=torch.double) if latent_plotly else torch.zeros((sample_n, latent_dim), dtype=torch.double)
    extract_latent_epoch = latent_plotly if latent_plotly else epochs
    plotly_index = 0
    old_epoch = 0

    # init mean loss
    loss_all = list()
    val_loss_all = list()
    a_loss = list()
    l_loss = list()
    lat_loss = list()
    triangle_list = list()
    terminal_list = list()
    best_mean_val_loss = float('inf')
    best_weights_epoch = 0

    # init loss
    loss_noise = 0
    angle_noise_loss  = 0
    linear_noise_loss = 0
    latent_noise_loss = 0
    triangle_noise_loss = 0
    terminal_loss_noise = 0

    # print(model)
    energy_out_resize = energy_n - 3 if energy_n !=3 else -3


    # loop through epochs
    for e in range(begin_epoch, epochs+begin_epoch):

        # loop through dataset 
        for n, (data, indices) in enumerate(train_dataset):

            # optimizer zero out grad
            optimizer.zero_grad()

            # send to device
            x = data.to(device)

            # run model with noise
            # out, angles_decomposed, latent_rmsd_mat, latent_space, decomposed_x = model(x)
            if noise_iterations > 0:
                for noise_n in range(noise_iterations):
                    out, latent_rmsd_mat, latent_space, decomposed_x = model(x, True)

                    # Create output tensor to check
                    y_linear = real_X[indices, angle_n:].to(device)
                    y_angle = real_X[indices, :angle_n].to(device)

                    # linear terms
                    bond_lengths = y_linear[:, :-energy_n]
                    # energy = y_linear[:, -energy_n:]
                    energy = scale_outputs.scaled_vals[indices, :]

                    args = {"y_hat_angle": out[:, :angle_n*2], # Torsion only we get rid of the *2
                            "y_angle": decomposed_x[:, :angle_n*2], # Torsion only we get rid of the *2
                            # "bond_lengths_hat": out[:, angle_n*2:-energy_n], # This currently is because we don't have energy #angle_n*2 + (linear_n-energy_n)],
                            "bond_lengths_hat": out[:, angle_n*2:-3], # This currently is because we don't have energy #angle_n*2 + (linear_n-energy_n)],
                            "bond_lengths": bond_lengths, 
                            # "energy_hat": out[:, -energy_n:], # None,
                            "energy_hat": out[:, energy_out_resize:], # None,
                            "energy": energy, # None,
                            "rmsd_real": real_rmsd_mat[indices.view(-1, 1), indices].to(device),
                            "rmsd_latent": latent_rmsd_mat,
                            "device": device,
                            "epoch": e+1,
                            "angle_size": angle_n,
                            "min_vals": min_vals,
                            "z": z,
                            "real_terminal_lengths": real_terminal_lengths[indices,:], 
                            "dof_out": out,
                            "params": params,
                            "term_class": term_class,
                            }

                    # generate loss and append
                    loss, angle_loss, linear_loss, latent_loss, triangle_loss, terminal_loss = loss_function(**args)

                    # loss track
                    loss_noise += loss
                    angle_noise_loss += angle_loss
                    linear_noise_loss += linear_loss
                    latent_noise_loss += latent_loss
                    triangle_noise_loss += triangle_loss
                    terminal_loss_noise += terminal_loss

            # # Complete loss
            # loss_all.append(loss.item())
            #
            # # Extra loss terms
            # a_loss.append(angle_loss.item())
            # l_loss.append(linear_loss.item())
            # lat_loss.append(latent_loss.item())
            # triangle_list.append(triangle_loss.item())

            ### Run same set without noise
            out, latent_rmsd_mat, latent_space, decomposed_x = model(x, False)

            # Extract out the latent tensor if multiple middle dimensions
            if ((e+1) % extract_latent_epoch == 0) and (extract_latent_epoch != epochs):

                # Keep track of the epoch and where we are in the middle index
                if old_epoch != (e+1):
                    plotly_index += 1
                    old_epoch = (e+1)
                latent_plotly_tensor[indices, plotly_index, :] = latent_space.detach().cpu()

            # pull out initial  0 epoch latent
            elif (e == 0) and (extract_latent_epoch != epochs):
                # pass
                latent_plotly_tensor[indices, plotly_index, :] = latent_space.detach().cpu()

            # if not passed then only extract the last epochs tensor
            elif ((e+1) % extract_latent_epoch == 0) and (extract_latent_epoch == epochs):
                # pass
                latent_plotly_tensor[indices, :] = latent_space.detach().cpu()

            # Create output tensor to check
            y_linear = real_X[indices, angle_n:].to(device)
            y_angle = real_X[indices, :angle_n].to(device)

            # linear terms
            bond_lengths = y_linear[:, :-energy_n]
            # energy = y_linear[:, -energy_n:]
            energy = scale_outputs.scaled_vals[indices, :]

            # args input for loss
            args = {"y_hat_angle": out[:, :angle_n*2], 
                    "y_angle": decomposed_x[:, :angle_n*2], 
                    # "bond_lengths_hat": out[:, angle_n*2:-energy_n], # This currently is because we don't have energy #angle_n*2 + (linear_n-energy_n)],
                    "bond_lengths_hat": out[:, angle_n*2:-3], # This currently is because we don't have energy #angle_n*2 + (linear_n-energy_n)],
                    "bond_lengths": bond_lengths, 
                    # "energy_hat": out[:, -energy_n:], # None,
                    "energy_hat": out[:, energy_out_resize:], # None,
                    "energy": energy, # None
                    "rmsd_real": real_rmsd_mat[indices.view(-1, 1), indices].to(device),
                    "rmsd_latent": latent_rmsd_mat,
                    "device": device,
                    "epoch": e+1,
                    "angle_size": angle_n,
                    "min_vals": min_vals,
                    "z": z,
                    "real_terminal_lengths": real_terminal_lengths[indices,:], 
                    "dof_out": out,
                    "params": params,
                    "term_class": term_class,
                    }

            # generate loss and append
            loss_nonoise, angle_loss_nonoise, linear_loss_nonoise, latent_loss_nonoise, triangle_loss_nonoise, terminal_loss_nonoise = loss_function(**args)
            if noise_iterations > 0:
                # Complete loss
                loss_all.append(((loss_noise + loss_nonoise)/(noise_iterations+1)).item())

                # Extra loss terms
                a_loss.append(((angle_loss_nonoise + angle_noise_loss)/(noise_iterations+1)).item())
                l_loss.append(((linear_loss_nonoise + linear_noise_loss)/(noise_iterations+1)).item())
                lat_loss.append(((latent_loss_nonoise + latent_noise_loss)/(noise_iterations+1)).item())
                triangle_list.append(((triangle_loss_nonoise + triangle_noise_loss)/(noise_iterations+1)).item())
                terminal_list.append(((terminal_loss_nonoise + terminal_loss_noise)/(noise_iterations+1)).item())
                # backprop weights and clip
                loss = (loss_nonoise + loss_noise)/noise_iterations
            else:
                # Complete loss
                loss_all.append(loss_nonoise.item())

                # Extra loss terms
                a_loss.append(angle_loss_nonoise.item())
                l_loss.append(linear_loss_nonoise.item())
                lat_loss.append(latent_loss_nonoise.item())
                triangle_list.append(triangle_loss_nonoise.item())
                terminal_list.append(terminal_loss_nonoise.item())

                loss = loss_nonoise
            loss.backward() 
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # optimizer step and scheduler step
            optimizer.step()
            # print('grad 4_4', model.energy_decoder4[4].weight.grad)
            # print('weight 4_4', model.energy_decoder4[4].weight)

            # scheduler.step(e+n / len(dataset))
            scheduler.step_cycleLR(step_size) # This is for the custom scheduler not for overfitting
            # scheduler.step()

            # increment step size for tracking
            step_size += 1

            # clear loss
            loss_noise = 0
            angle_noise_loss  = 0
            linear_noise_loss = 0
            latent_noise_loss = 0
            triangle_noise_loss = 0
            terminal_loss_noise = 0

        # calc mean loss
        mean_loss = sum(loss_all) / len(loss_all)
        a_mean = sum(a_loss) / len(a_loss)
        l_mean = sum(l_loss) / len(l_loss)
        lat_mean = sum(lat_loss) / len(lat_loss)
        cons_mean = sum(triangle_list) / len(triangle_list)
        terminal_mean = sum(terminal_list) / len(terminal_list)

        # This is only for overfitting values
        # scheduler.step_reduceLR(mean_loss)

        ##########
        # VALIDATION
        ##########

        # if it is actual training and not sanity check then run eval
        if not overfit:

            model.eval()

            # loop through dataset 
            for n, (data, indices) in enumerate(val_dataset):

                # send to device
                x = data.to(device)

                # run model
                # out, angles_decomposed, latent_rmsd_mat, latent_space, decomposed_x = model(x)
                out, latent_rmsd_mat, latent_space, decomposed_x = model(x, False)

                # Extract out the latent tensor if multiple middle dimensions
                if ((e+1) % extract_latent_epoch == 0) and (extract_latent_epoch != epochs):

                    # Keep track of the epoch and where we are in the middle index
                    if old_epoch != (e+1):
                        plotly_index += 1
                        old_epoch = (e+1)
                    latent_plotly_tensor[indices, plotly_index, :] = latent_space.detach().cpu()

                # pull out initial  0 epoch latent
                elif (e == 0) and (extract_latent_epoch != epochs):
                    # pass
                    latent_plotly_tensor[indices, plotly_index, :] = latent_space.detach().cpu()

                # if not passed then only extract the last epochs tensor
                elif ((e+1) % extract_latent_epoch == 0) and (extract_latent_epoch == epochs):
                    # pass
                    latent_plotly_tensor[indices, :] = latent_space.detach().cpu()

                # Create output tensor to check
                y_linear = real_X[indices, angle_n:].to(device)
                y_angle = real_X[indices, :angle_n].to(device)

                # linear terms
                bond_lengths = y_linear[:, :-energy_n]
                # energy = y_linear[:, -energy_n:]
                energy = scale_outputs.scaled_vals[indices, :]

                # args input for loss
                args = {"y_hat_angle": out[:, :angle_n*2], # Torsion only we get rid of the *2
                        "y_angle": decomposed_x[:, :angle_n*2], # Torsion only we get rid of the *2
                        # "bond_lengths_hat": out[:, angle_n*2:-energy_n], # This currently is because we don't have energy #angle_n*2 + (linear_n-energy_n)],
                        "bond_lengths_hat": out[:, angle_n*2:-3], # This currently is because we don't have energy #angle_n*2 + (linear_n-energy_n)],
                        "bond_lengths": bond_lengths, 
                        # "energy_hat": out[:, -energy_n:], # None,
                        "energy_hat": out[:, energy_out_resize:], # None,
                        "energy": energy, #None,
                        "rmsd_real": real_rmsd_mat[indices.view(-1, 1), indices].to(device),
                        "rmsd_latent": latent_rmsd_mat,
                        "device": device,
                        "epoch": e+1,
                        "angle_size": angle_n,
                        "min_vals": min_vals,
                        "z": z,
                        "real_terminal_lengths": real_terminal_lengths[indices,:], 
                        "dof_out": out,
                        "params": params,
                        "term_class": term_class,
                        }

                # generate loss and append
                val_loss, _, _, _, _, _ = loss_function(**args)

                # Complete loss
                val_loss_all.append(val_loss.item())

            # calculate val loss
            mean_val_loss = sum(val_loss_all) / len(val_loss_all)

            # clear val loss list
            val_loss_all.clear()

            model.train(True)

            # UpdateLR on Plateau Scheduler step This is the custom scheduler
            scheduler.step_reduceLR(mean_val_loss)
        else:
            scheduler.step_reduceLR(mean_loss)
            mean_val_loss = 0

        # check if loss is lower
        if mean_loss < lowest_loss:
            lowest_loss = mean_loss
            epoch_lowest_loss = e+1


        if output_dir != None:

            # Write progress to tracking file
            with open(file_out_track, 'a') as specific_out:
                print(f"{e+1},{mean_loss},{mean_val_loss},{a_mean},{l_mean},{lat_mean},{cons_mean}", file=specific_out)

            # First check if dirs exist if not then create
            first_layer_dir = os.path.join(output_dir, "model_weights/")
            if not os.path.exists(first_layer_dir):
                os.mkdir(first_layer_dir)

            # save current epochs weights
            current_weights = os.path.join(first_layer_dir, f"model_{file_num}_current_weights.pt")
            torch.save(
                {
                    "epoch": e+1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    # "scheduler_state_dict": scheduler.state_dict(),
                    "loss": loss,
                },
                current_weights,
            )

            if mean_val_loss < best_mean_val_loss:
                # update mean val loss
                best_mean_val_loss = mean_val_loss
                best_weights_epoch = e+1

                # save best epoch weights
                current_weights = os.path.join(first_layer_dir, f"model_{file_num}_best_weights.pt")
                torch.save(
                    {
                        "epoch": e+1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        # "scheduler_state_dict": scheduler.state_dict(),
                        "loss": loss,
                    },
                    current_weights,
                )

            # Save weights for passed epoch numbers
            if (e+1) % weight_save_n == 0:
                specific_weights = os.path.join(first_layer_dir, f"epoch_{e+1}_weights_{file_num}.pt")
                torch.save(
                    {
                        "epoch": e+1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        # "scheduler_state_dict": scheduler.state_dict(),
                        "loss": loss,
                    },
                    specific_weights,
                )

        # Print out progress for each epoch
        print(f"Epoch: {e+1}, Step: {step_size}, LR: {optimizer.param_groups[0]['lr']}, Train Loss: {mean_loss}, Val Loss: {mean_val_loss}, Angel Loss: {a_mean}, Linear Loss: {l_mean}, Latent RMSD Loss: {lat_mean}, Energy Loss: {cons_mean}, Terminal Loss (Cyclic): {terminal_mean}")
        # Visualize Output vs Real values for qualitative check
        if (e+1) % 50 == 0:
            angles = out[:, :angle_n*2]
            torsion = angles[:, :angle_n]
            sin_torsion = torsion[:, :angle_n//2]
            cos_torsion = torsion[:, angle_n//2:]
            computed_angle = torch.rad2deg(torch.atan2(sin_torsion, cos_torsion))
            print("Out vs Real Angles")
            # Angles
            print(computed_angle[:3, :10])
            print(y_angle[:3, :10])
            print("Pred Sin/Cos")
            # predicted sin /cos
            print(sin_torsion[:3, :5])
            print(cos_torsion[:3, :5])
            print("Real Sin/Cos")
            # sin /cos real
            print(decomposed_x[:3, :5])
            print(decomposed_x[:3, angle_n//2:(angle_n//2)+5])
            print("Linear Terms Out vs Real")
            # Linear Terms
            print(scale_outputs.inverse_scale(out[:3, -energy_n:]))
            print(scale_outputs.inverse_scale(energy[:3, :]))

        # clear loss lists
        loss_all.clear()
        a_loss.clear()
        l_loss.clear()
        lat_loss.clear()
        triangle_list.clear()
        terminal_list.clear()

        # # add loss_all_track
        # loss_all_track.append(mean_loss)

        # Early Stopping
        # if early_stop.early_stop(mean_loss):
        #     return mean_loss, lowest_loss, epoch_lowest_loss, latent_plotly_tensor, e+1
    print(f"Best Model Weights For {file_num} Are From Epoch: {best_weights_epoch}")
    return mean_loss, lowest_loss, epoch_lowest_loss, latent_plotly_tensor, e+1


def test(
    model: torch.nn.Module,
    dataset: torch.utils.data.DataLoader,
    ):
    """General Test function for the AutoEncoder Model

    This is a placeholder for a test function if we need one."""
    pass


##### Model Testing

# checking problems
def print_weights(module, input, output):
    for name, param in module.named_parameters():
        if 'weight' in name:
            print(f"Weights of {name}: {param.data}")
        elif 'bias' in name:
            print(f"Biases of {name}: {param.data}")

# init weight xavier
def xavier_init(layer):
    if isinstance(layer, torch.nn.Linear):
        init.xavier_uniform_(layer.weight)
        if layer.bias is not None:
            init.constant_(layer.bias, 0.1)

# print out grad infromation
def print_grad(module, input, output):
    for name, param in module.named_parameters():
        if param.grad is not None:
            print(f"Gradient of {name}: {param.grad}")


