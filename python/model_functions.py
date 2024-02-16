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

# ml imports
import torch
import torch.nn.init as init

##################################################################################################################
#                                                Functions
##################################################################################################################

def scale(x: torch.Tensor, params: dict):
    """Scale all input values"""
    # starting spot in tensor
    linear_terms_start = params["torsion_main_n"] + params["torsion_side_n"] + params["bond_angle_main_n"] + params["bond_angle_side_n"]

    # calculate bond length size
    bond_length_size = params["bond_length_main_n"] + params["bond_length_side_n"]

    # grab out each section
    # Scale Bond Length
    print("BOND LENGTH")
    x[:, linear_terms_start: linear_terms_start + bond_length_size ] = normalize_vals(
        x[:, linear_terms_start: linear_terms_start + bond_length_size ] 
    )

    # Scale Jumps if any
    if params["jump_n"] != 0:
        jump_end = linear_terms_start + bond_length_size + params["jump_n"]
        jump_start = linear_terms_start + bond_length_size
        x[:, jump_start : jump_end ] = normalize_vals(
            x[:, jump_start : jump_end ]
        )

    # Scale Energy if any
    if params["energy_n"] != 0:
        energy_end = linear_terms_start + bond_length_size + params["jump_n"] + params["energy_n"]
        energy_start = linear_terms_start + bond_length_size + params["jump_n"]
        print("ENERGY")
        x[:, energy_start : energy_end ] = normalize_vals(
            x[:, energy_start : energy_end ]
        )

    return x


def normalize_vals(x: torch.Tensor, norm_min:float = -1., norm_max:float = 1.):
    """Set terms between -1 and 1"""
    min_val = x.min()
    max_val = x.max()
    norm_feature = (x - min_val) / (max_val - min_val) * (norm_max - norm_min) + norm_min
    print(norm_feature)
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
    return torch.mean(1. - torch.cos(y - y_hat))

def linear_loss_term(y_hat: torch.Tensor, y: torch.Tensor):
    """General L2 Linear Loss"""
    return torch.mean(torch.pow(y - y_hat, 2))

def SSE_latent_RMSD_loss_term(rmsd_real: torch.Tensor, rmsd_latent: torch.Tensor):
    """Custom (B) loss function to enforce latent space RMSD distances to be comparable 
    with real RMSD distances.

    PARAMS
    ------
        rmsd_real: torch.Tensor
            The RMSDs all vs all from the actual structures

        rmsd_latent: torch.Tensor
            The RMSDs all vs all from the latent space structures

    RETURNS
    -------
        loss: torch.float
            Summed Square Error between RMSD_real and RMSD_latent
    """
    # extract sizes
    B, L, _ = rmsd_real.shape
    
    # make bool mask
    mask = torch.eye(L, L, dtype=torch.bool)

    # Make sure that i /= j within the tensors. Therefore we can block
    # out the diagonal with zeros on both tensors
    rmsd_real.masked_fill_(mask, 0)
    rmsd_latent.masked_fill_(mask, 0)

    # Need to set lambda some how right now I will just randomly fill in with values
    lambda_real = 1.
    lambda_latent = 0.1

    # Compute each component of the function sum((real_component - latent_component)**2)
    # Real: e^{-(RMSDreal_{ij} / lambda_real)**2}
    real_component = torch.exp( -( torch.pow( rmsd_real/lambda_real, 2 ) ) )

    # Latent: e^{-(RMSDlatent_{ij,latent} / lambda_latent)**2}
    latent_component = torch.exp( -( torch.pow( rmsd_latent/lambda_latent, 2 ) ) )

    # now sum the information
    loss = torch.sum( torch.pow( real_component - latent_component, 2 ) )
    return loss

def loss_function(
    y_hat_angle: torch.Tensor,
    y_angle: torch.Tensor,
    y_hat_linear: torch.Tensor,
    y_linear: torch.Tensor,
    ):
    """Combination of Loss terms take the mean"""
    angle_loss = angle_loss_term(y_hat_angle, y_angle)
    linear_loss = linear_loss_term(y_hat_linear, y_linear)
    return angle_loss + linear_loss

    

def train(
    model: torch.nn.Module,
    dataset: torch.utils.data.DataLoader,
    epochs: int,
    device: str,
    optimizer: torch.optim,
    scheduler: torch.optim.lr_scheduler,
    angle_n: int,
    linear_n: int,
    file_out_track: str,
    output_dir: str,
    weight_save_n: int,
    file_num: int,
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

    RETURNS
    -------
        mean_loss: float
            The mean loss of the total run after some N
            number of epochs.
    """
    # init loss list
    loss_all = list()

    #init step tracker
    step_size = 0

    # set device and model to train
    model = model.to(device)
    model.train()

    # register the hook
    # model.encoder.register_forward_hook(print_grad)
    # model.register_full_backward_hook(print_grad)

    # init model weights
    # model.apply(xavier_init)

    # loop through epochs
    for e in range(epochs):

        # loop through dataset 
        for data in dataset:

            print("***********************")
            print("***********************")
            print("***********************")
            print("NEW DATASET")
            print("***********************")
            print("***********************")
            print("***********************")
            # optimizer zero out grad
            optimizer.zero_grad()

            # send to device
            x = data.to(device)

            # run model
            out = model(x)

            # args input for loss
            args = {"y_hat_angle": out[:, :angle_n],
                    "y_angle": x[:, :angle_n],
                    "y_hat_linear": out[:, angle_n:],
                    "y_linear": x[:, angle_n:],
                    }

            # generate loss and append
            loss = loss_function(**args)
            loss_all.append(loss.item())

            print("LOSS: ", loss)


            # backprop weights and step
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step_size += 1

        # calc mean loss
        mean_loss = sum(loss_all) / len(loss_all)

        # Write progress to tracking file
        with open(file_out_track, 'a') as specific_out:
            print(f"{e+1},{mean_loss}", file=specific_out)

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
                },
                specific_weights,
            )

        # Print out progress for each epoch
        print(f"Epoch: {e+1}, Step: {step_size}, LR: {optimizer.param_groups[0]['lr']}, Train Loss: {mean_loss}")
    return mean_loss


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

def print_grad(module, input, output):
    for name, param in module.named_parameters():
        if param.grad is not None:
            print(f"Gradient of {name}: {param.grad}")


def test(
    model: torch.nn.Module,
    dataset: torch.utils.data.DataLoader,
    ):
    """General Test function for the AutoEncoder Model

    This is a placeholder for a test function if we need one."""
    pass
