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

##################################################################################################################
#                                               Classes
##################################################################################################################

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
    rmsd_real.masked_fill_(mask, 0)
    rmsd_latent.masked_fill_(mask, 0)

    # Need to set lambda some how right now I will just randomly fill in with values
    lambda_real = 2
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
    ):
    """Combination of Loss terms take the mean"""
    # Set up loss function
    sin_cos_linear_loss = CustomSinCosLoss(
        sin_weight=1.0,
        cos_weight=1.0,
    )

    # Compute individual angle loss terms
    # torsion_loss = angle_loss_term(y_hat_angle[:, :y_hat_angle.size(-1)//2], y_angle[:, :y_hat_angle.size(-1)//2])
    # bond_angle_loss = angle_loss_term(y_hat_angle[:, y_hat_angle.size(-1)//2:], y_angle[:, y_hat_angle.size(-1)//2:])
    # torsion_loss = linear_loss_term(y_hat_angle[:, :y_hat_angle.size(-1)//2], y_angle[:, :y_hat_angle.size(-1)//2])
    # bond_angle_loss = linear_loss_term(y_hat_angle[:, y_hat_angle.size(-1)//2:], y_angle[:, y_hat_angle.size(-1)//2:])
    # torsion_loss = sin_cos_linear_loss(y_hat_angle[:, :y_hat_angle.size(-1)//2], y_angle[:, :y_hat_angle.size(-1)//2])
    # print("Full y_hat_angle", y_hat_angle.shape)
    # print("torsion y_hat_angle", y_hat_angle[:, y_hat_angle.size(-1)//2:].shape)
    torsion_loss = sin_cos_linear_loss(y_hat_angle[:, :y_hat_angle.size(-1)//2], y_angle[:, :y_angle.size(-1)//2]) * 100_000
    # torsion_loss += angle_loss_term(y_hat_angle, y_angle) * 10000

    # Uncomment This
    # bond_angle_loss = torch.tensor(0.0) 
    # print("bond angle y_hat_angle", y_hat_angle[:, :y_hat_angle.size(-1)//2].shape)
    bond_angle_loss = sin_cos_linear_loss(y_hat_angle[:, y_hat_angle.size(-1)//2:], y_angle[:, y_angle.size(-1)//2:]) * 100_000

    # This is constrain loss for sin/cos pairs
    # constrain_loss_torsion = angle_sin2_cos2_enforce(y_hat_angle[:, :y_hat_angle.size(-1)//2])
    # constrain_loss_bond_angle = angle_sin2_cos2_enforce(y_hat_angle[:, y_hat_angle.size(-1)//2:])
    # constrain_loss = constrain_loss_torsion + constrain_loss_bond_angle
    constrain_loss = torch.tensor(0.0)
    # constrain_loss = linear_loss_term(enforce_theta, ones.to(device))

    # Compute individual linear loss terms
    bond_length_loss = linear_loss_term(bond_lengths_hat, bond_lengths) * 1000
    # bond_length_loss = torch.tensor(0.0)
    energy_loss = torch.tensor(0.0)
    # energy_loss = linear_loss_term(energy_hat, energy)

    # extra latent space loss term
    enforce_latent_rmsd = SSE_latent_RMSD_loss_term(rmsd_real, rmsd_latent, device) * 100 #* 100_000
    # enforce_latent_rmsd = torch.tensor(0.0)

    # reconstruction individual weights
    lambda_lin, lambda_ang = 1, 1

    # linear loss
    linear_loss = (bond_length_loss + energy_loss) * lambda_lin

    # torsion & bond angle weights
    lambda_tor, lambda_bond = 1, 1

    # angle loss
    angle_loss = (torsion_loss*lambda_tor + bond_angle_loss*lambda_bond + constrain_loss) * lambda_ang

    # Combine Loss Terms for reconstruction
    reconstruction_loss = angle_loss + linear_loss

    # compute lambda weighting terms for full loss
    lambda_rec, lambda_latent = 1, 1

    return reconstruction_loss*lambda_rec + enforce_latent_rmsd*lambda_latent, angle_loss, linear_loss, enforce_latent_rmsd, constrain_loss
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
    constrain_list = list()
    best_mean_val_loss = float('inf')

    # init loss
    loss_noise = 0
    angle_noise_loss  = 0
    linear_noise_loss = 0
    latent_noise_loss = 0
    constrain_noise_loss = 0


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
                    energy = y_linear[:, -energy_n]

                    args = {"y_hat_angle": out[:, :angle_n*2], # Torsion only we get rid of the *2
                            "y_angle": decomposed_x[:, :angle_n*2], # Torsion only we get rid of the *2
                            "bond_lengths_hat": out[:, angle_n*2: ], # This currently is because we don't have energy #angle_n*2 + (linear_n-energy_n)],
                            "bond_lengths": bond_lengths, 
                            "energy_hat": None, #out[:, -energy_n],
                            "energy": None, #energy,
                            "rmsd_real": real_rmsd_mat[indices.view(-1, 1), indices].to(device),
                            "rmsd_latent": latent_rmsd_mat,
                            "device": device,
                            "epoch": e+1,
                            "angle_size": angle_n,
                            }

                    # generate loss and append
                    loss, angle_loss, linear_loss, latent_loss, constrain_loss = loss_function(**args)

                    # loss track
                    loss_noise += loss
                    angle_noise_loss += angle_loss
                    linear_noise_loss += linear_loss
                    latent_noise_loss += latent_loss
                    constrain_noise_loss += constrain_loss

            # # Complete loss
            # loss_all.append(loss.item())
            #
            # # Extra loss terms
            # a_loss.append(angle_loss.item())
            # l_loss.append(linear_loss.item())
            # lat_loss.append(latent_loss.item())
            # constrain_list.append(constrain_loss.item())

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
            energy = y_linear[:, -energy_n]

            # args input for loss
            args = {"y_hat_angle": out[:, :angle_n*2], 
                    "y_angle": decomposed_x[:, :angle_n*2], 
                    "bond_lengths_hat": out[:, angle_n*2: ], # This currently is because we don't have energy #angle_n*2 + (linear_n-energy_n)],
                    "bond_lengths": bond_lengths, 
                    "energy_hat": None, #out[:, -energy_n],
                    "energy": None, #energy,
                    "rmsd_real": real_rmsd_mat[indices.view(-1, 1), indices].to(device),
                    "rmsd_latent": latent_rmsd_mat,
                    "device": device,
                    "epoch": e+1,
                    "angle_size": angle_n,
                    }

            # generate loss and append
            loss_nonoise, angle_loss_nonoise, linear_loss_nonoise, latent_loss_nonoise, constrain_loss_nonoise = loss_function(**args)
            if noise_iterations > 0:
                # Complete loss
                loss_all.append(((loss_noise + loss_nonoise)/noise_iterations).item())

                # Extra loss terms
                a_loss.append(((angle_loss_nonoise + angle_noise_loss)/noise_iterations).item())
                l_loss.append(((linear_loss_nonoise + linear_noise_loss)/noise_iterations).item())
                lat_loss.append(((latent_loss_nonoise + latent_noise_loss)/noise_iterations).item())
                constrain_list.append(((constrain_loss_nonoise + constrain_noise_loss)/noise_iterations).item())
                # backprop weights and clip
            else:
                # Complete loss
                loss_all.append(loss_nonoise.item())

                # Extra loss terms
                a_loss.append(angle_loss_nonoise.item())
                l_loss.append(linear_loss_nonoise.item())
                lat_loss.append(latent_loss_nonoise.item())
                constrain_list.append(constrain_loss_nonoise.item())

                loss = loss_nonoise
            loss.backward() 
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # optimizer step and scheduler step
            optimizer.step()

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
            constrain_noise_loss = 0

        # calc mean loss
        mean_loss = sum(loss_all) / len(loss_all)
        a_mean = sum(a_loss) / len(a_loss)
        l_mean = sum(l_loss) / len(l_loss)
        lat_mean = sum(lat_loss) / len(lat_loss)
        cons_mean = sum(constrain_list) / len(constrain_list)

        # This is only for overfitting values
        # scheduler.step(mean_loss)

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
                energy = y_linear[:, -energy_n]


                # args input for loss
                args = {"y_hat_angle": out[:, :angle_n*2], # Torsion only we get rid of the *2
                        "y_angle": decomposed_x[:, :angle_n*2], # Torsion only we get rid of the *2
                        "bond_lengths_hat": out[:, angle_n*2: ], # This currently is because we don't have energy #angle_n*2 + (linear_n-energy_n)],
                        "bond_lengths": bond_lengths, 
                        "energy_hat": None, #out[:, -energy_n],
                        "energy": None, #energy,
                        "rmsd_real": real_rmsd_mat[indices.view(-1, 1), indices].to(device),
                        "rmsd_latent": latent_rmsd_mat,
                        "device": device,
                        "epoch": e+1,
                        "angle_size": angle_n,
                        }

                # generate loss and append
                val_loss, _, _, _, _ = loss_function(**args)

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
                # save best epoch weights
                current_weights = os.path.join(first_layer_dir, f"model_{file_num}_epoch_{e+1}_best_weights.pt")
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
        print(f"Epoch: {e+1}, Step: {step_size}, LR: {optimizer.param_groups[0]['lr']}, Train Loss: {mean_loss}, Val Loss: {mean_val_loss}, Angel Loss: {a_mean}, Linear Loss: {l_mean}, Latent RMSD Loss: {lat_mean}, Constrain Sin/Cos Loss: {cons_mean}")
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
            print("Linear Terms")
            # Linear Terms
            print(out[:3, -1])
            print(bond_lengths[:3, -1])

        # clear loss lists
        loss_all.clear()
        a_loss.clear()
        l_loss.clear()
        lat_loss.clear()
        constrain_list.clear()

        # # add loss_all_track
        # loss_all_track.append(mean_loss)

        # Early Stopping
        # if early_stop.early_stop(mean_loss):
        #     return mean_loss, lowest_loss, epoch_lowest_loss, latent_plotly_tensor, e+1
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


