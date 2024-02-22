


# base python packages
import sys, os
from collections import defaultdict
import time
import argparse
import random

# torch
import torch

# train 
from model_functions import train as mse_train
from model_functions import scale

# Model 
from model_utils import ConformationalMLPAutoEncoder, MyDataset



#### Train
def main():
    """Run training"""
    # Grab passed in values from CLI
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_tensor", type=str, default="./binary_import/dof_vectors2024-02-20torch_tensor.pt",
                   help="Path to .pt dataset file. (defualt: ./binary_import/dof_vectors2024-02-20torch_tensor.pt)")
    p.add_argument("--epochs", type=int, default=1000, help="Number of epochs to train (default: 1000)")
    p.add_argument("--num_encoder_layers", type=int, default=3,
                   help="Number of encoder layers in the model (default: 3)")
    p.add_argument("--num_decoder_layers", type=int, default=3,
                   help="Number of decoder layers in the model (default: 3)")
    p.add_argument("--latent_space", type=int, default=3,
                   help="Latent Space R dimension (default: 3)")
    p.add_argument("--batch_size", type=int, default=32,
                   help="Specify model batch size (default: 32)")
    p.add_argument("--learning_rate", type=float, default=1e-3, help="Specify the LR (default: 1e-3)")
    p.add_argument("--output_weights_every_n", type=int, default=1000,
                   help="Output saved model weights every N number of epochs (default: 1000)")
    p.add_argument("--path_to_output_dir", type=str, default="./outputs/",
                   help="Specify model output specific directory (default: ./outputs/)")
    args = p.parse_args()

    # load in dataset
    dataset = torch.load(args.dataset_tensor)

    # set global vars
    PARAMS           = dataset["params"]
    X                = dataset["x"]
    RMSD_REAL        = dataset["rmsd"]
    N_ENCODER_LAYERS = args.num_encoder_layers
    N_DECODER_LAYERS = args.num_decoder_layers
    EPOCHS           = args.epochs
    LATENT_SPACE     = args.latent_space

    # scale data
    X_scaled = scale(X, PARAMS)


    # pass to custom dataset class
    dataset = MyDataset(X_scaled)

    # load data into a specific dataloader class
    dataload = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size)

    model = ConformationalMLPAutoEncoder(
        encoder_layers   = N_ENCODER_LAYERS,
        decoder_layers   = N_DECODER_LAYERS,
        latent_space_dim = LATENT_SPACE,
        **PARAMS,
    )

    # Specify angle and linear number for L2 loss specifics
    angle_number = PARAMS["torsion_main_n"] + PARAMS["torsion_side_n"] + PARAMS["bond_angle_main_n"] + PARAMS["bond_angle_side_n"]
    linear_number = PARAMS["input_size"] - angle_number

    # init AdamW optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)

    # init scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           patience = 8,
                                                           mode = "min",
                                                           factor = 0.8,
                                                           threshold = 1e-4,
                                                           min_lr = 1e-8,
                                                           )

    # Model Run specific number classifier for output files
    random_num = random.randint(0, 10000)

    # Generate output for run
    file_out_name = os.path.join(args.path_to_output_dir, f"model_run_{random_num}.csv")
    with open(file_out_name, "w") as file_out:
        print("epoch,loss,angle_loss,linear_loss,latent_rmsd_loss", file=file_out)

    # Run Model
    print(f"***MODEL {random_num} TRAIN***")
    final_loss, lowest_loss, epoch_lowest_loss = mse_train(
        model = model,
        dataset = dataload,
        epochs = EPOCHS,
        device = "cuda",
        optimizer = optimizer,
        scheduler = scheduler,
        angle_n = angle_number,
        linear_n = linear_number,
        file_out_track = file_out_name,
        output_dir = args.path_to_output_dir,
        weight_save_n = args.output_weights_every_n,
        file_num = random_num,
        real_rmsd_mat = RMSD_REAL,
    )

    # Save to combo file
    md_combo_out_file = os.path.join(args.path_to_output_dir, "results.md")

    # if it exists save to it
    if os.path.exists(md_combo_out_file):
        with open(md_combo_out_file, 'a') as md_out:
            print(f"|{random_num}|{EPOCHS}|{args.learning_rate}|{N_ENCODER_LAYERS}|{N_DECODER_LAYERS}|{lowest_loss}|{epoch_lowest_loss}|{final_loss}|", file=md_out)
    else:
        with open(md_combo_out_file, 'w') as md_out:
            print(f"|File Number|Epochs|LR|Num Encoding Layers|Num Decoding Layers|Lowest Loss|Epoch Lowest Loss|Final Loss|", file=md_out)
            print(f"|---|---|---|---|---|---|---|---|", file=md_out)
            print(f"|{random_num}|{EPOCHS}|{args.learning_rate}|{N_ENCODER_LAYERS}|{N_DECODER_LAYERS}|{lowest_loss}|{epoch_lowest_loss}|{final_loss}|", file=md_out)

    return 0


if __name__ == "__main__":
    main()

