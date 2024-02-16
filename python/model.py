


# base python packages
import sys, os
from collections import defaultdict
import time
import argparse

# torch
import torch

# train 
from model_functions import train as mse_train

# Model 
from model_utils import ConformationalMLPAutoEncoder



#### Train
def main():
    """Run training"""
    # Grab passed in values from CLI
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_tensor", required=True, type=str, default="./binary_import/dof_vectorstorch_tensor.pt",
                   help="Path to .pt dataset file. (defualt: ./binary_import/dof_vectorstorch_tensor.pt)")
    p.add_argument("--epochs", type=int, default=1000, help="Number of epochs to train (default: 1000)")
    p.add_argument("--num_encoder_layers", type=int, default=3,
                   help="Number of encoder layers in the model (default: 3)")
    p.add_argument("--num_decoder_layers", type=int, default=3,
                   help="Number of decoder layers in the model (default: 3)")
    p.add_argument("--latent_space", type=int, default=3,
                   help="Latent Space R dimension (default: 3)")
    args = p.parse_args()

    # load in dataset
    dataset = torch.load(args.dataset_tensor)

    # set global vars
    PARAMS           = dataset["params"]
    X                = dataset["x"]
    N_ENCODER_LAYERS = args.num_encoder_layers
    N_DECODER_LAYERS = args.num_decoder_layers
    EPOCHS           = args.epochs
    LATENT_SPACE     = args.latent_space




