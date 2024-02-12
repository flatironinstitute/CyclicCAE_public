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

# ml imports
import torch

##################################################################################################################
#                                                Functions
##################################################################################################################

def MSE_angle_loss_term(y_hat: torch.Tensor, y: torch.Tensor):
    """Angle Loss function taken from here:
    https://discuss.pytorch.org/t/custom-loss-function-for-discontinuous-angle-calculation/58579/13
    """
    return torch.mean(1. - torch.cos(y - y_hat))

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

def train(
    model: torch.nn.Module,
    dataset: torch.utils.data.DataLoader,
    epochs: int,
    device: str,
    optimizer: torch.optim,
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
    RETURNS
    -------
        mean_loss: float
            The mean loss of the total run after some N
            number of epochs.
    """
    # init loss list
    loss_all = list()
    # set device and model to train
    model = model.to(device)
    model.train()

    # loop through epochs
    for e in range(epochs):

        # loop through dataset 
        for data in dataset:

            # send to device
            x = data.to(device)

            # run model
            out = model(x)

            # generate loss and append
            loss = MSE_angle_loss_term(out, x)
            loss_all.append(loss.item())

            # optimizer zero out grad
            optimizer.zero_grad()

            # backprop weights and step
            loss.backward()
            optimizer.step()

        # calc mean loss
        mean_loss = sum(loss_all) / len(loss_all)

        # Print out progress for each epoch
        print(f"Epoch: {e+1}, Train Loss: {mean_loss}")
    return mean_loss



def test(
    model: torch.nn.Module,
    dataset: torch.utils.data.DataLoader,
    ):
    """General Test function for the AutoEncoder Model

    This is a placeholder for a test function if we need one."""
    pass
