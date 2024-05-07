


# base python packages
import sys, os
from collections import defaultdict
import time
import argparse
import random
import math

# torch
import torch

#numpy
import numpy as np

# pytorch
import torch.nn as nn

# train 
from model_functions import train as mse_train
from model_functions import scale, xavier_init, ScaleOutputs

# Model 
from model_utils import ConformationalMLPAutoEncoder, MyDataset, EarlyStopper, CustomScheduler, subset_dataset, rotate_samples
from test_case import TestCase1, HingeModel, ConvCase1, ConvCase2, ConvCase3, ConvCase4

def compare_model_state_dicts(state_dict1, state_dict2):
    for key in state_dict1:
        if key in state_dict2:
            if not torch.equal(state_dict1[key], state_dict2[key]):
                return False
        else:
            return False
    return True

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
    p.add_argument("--load_weights", type=str, default=None, 
                   help="Load in specific weights from known stopping position. (default: None)")
    p.add_argument("--path_to_output_dir", type=str, default="/mnt/ceph/users/apowers/outputs_old/",
                   help="Specify model output specific directory (default: /mnt/ceph/users/apowers/outputs_old/)")
    p.add_argument("--plotly_latent_animate", type=int, default=0,
                   help="Generate plotly latent 3D matrix. Specify the N epochs to take snapshot \
                   (default: 0)")
    p.add_argument("--angle-decomposition", type=int, default=1, help="Decompose the angle into \
                    sin/cos N times (default: 1)")
    p.add_argument("--override-lr", action="store_true", help="Pass flag if you don't want to load \
                    weights LR in.")
    p.add_argument("--dropout", type=float, default=0.0, help="Apply Dropout (default: 0.0)")
    p.add_argument("--noise-std", type=float, default=5.0, help="Training noise added to degrees (default: 5.0)")
    p.add_argument("--cyclic", action="store_true", help="Use Cyclic Positional Encodings")
    p.add_argument("--linear-pe", action="store_true", help="Use Linear Positional Encodings")
    p.add_argument("--overfit", action="store_true", help="Select only a couple samples to overfit")
    p.add_argument("--overfit-n", type=int, default=0, help="Select sample amount (default: 0)")
    p.add_argument("--convolution", action="store_true", help="Include Convolution after PE embedding")
    p.add_argument("--rotation-permutation", action="store_true", help="Include Rotation Permutations of Data only Torsion currently")
    p.add_argument("--noise-iterations", type=int, default=0, help="Number of rounds of noise during a batch (default: 0)")
    args = p.parse_args()

    # load in dataset
    dataset = torch.load(args.dataset_tensor)

    # set global vars
    PARAMS           = dataset["params"]
    X                = dataset["x"].double()
    RMSD_REAL        = dataset["rmsd"].double()
    N_ENCODER_LAYERS = args.num_encoder_layers
    N_DECODER_LAYERS = args.num_decoder_layers
    EPOCHS           = args.epochs
    LATENT_SPACE     = args.latent_space
    PLOTLY_LATENT    = args.plotly_latent_animate
    WEIGHTS          = args.load_weights
    BEGIN_EPOCH      = 0
    DECOMPOSE_ANGLE  = args.angle_decomposition
    LR               = args.learning_rate
    DROPOUT          = args.dropout
    NOISE_STD        = args.noise_std
    CYCLIC           = args.cyclic
    LINEAR           = args.linear_pe
    CONVOLUTION      = args.convolution

    assert EPOCHS % PLOTLY_LATENT == 0, "Epochs for Plotly Number does not have remainder 0 for passed EPOCH number"

    # scale data
    X_scaled = scale(X, PARAMS)

    if args.rotation_permutation:
        # rotate samples
        X_scaled = rotate_samples(X_scaled)


    # pass to custom dataset class
    dataset = MyDataset(X_scaled)

    # grab scaling values
    factor_vals_energy = ScaleOutputs(X[:, -PARAMS["energy_n"]:])


    if args.overfit:
        # Generate train/val split
        train_idx = np.random.choice(X_scaled.size(0), size=args.overfit_n, replace=False)

        # load data into a specific dataloader class
        train_loader = torch.utils.data.DataLoader(subset_dataset(dataset, train_idx), batch_size=args.batch_size, shuffle=False)
        val_loader = torch.utils.data.DataLoader(subset_dataset(dataset, train_idx), batch_size=args.batch_size, shuffle=False)

    else:
        # Generate train/val split
        train_idx = np.random.choice(X_scaled.size(0), size=int(X_scaled.size(0)*.9), replace=False)
        val_idx = list( set(range(X_scaled.size(0))) - set(train_idx))

        # load data into a specific dataloader class
        train_loader = torch.utils.data.DataLoader(subset_dataset(dataset, train_idx), batch_size=args.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(subset_dataset(dataset, val_idx), batch_size=args.batch_size, shuffle=True)

    # model = ConformationalMLPAutoEncoder(
    # model = HingeModel(
    # model = ConvCase1(
    # model = ConvCase3(
    model = ConvCase4(
        encoder_layers    = N_ENCODER_LAYERS,
        decoder_layers    = N_DECODER_LAYERS,
        latent_space_dim  = LATENT_SPACE,
        decomposition_n   = DECOMPOSE_ANGLE,
        dropout           = DROPOUT,
        noise_std         = NOISE_STD,
        cyclic            = CYCLIC,
        pe_linear         = LINEAR,
        convolution_layer = CONVOLUTION,
        **PARAMS,
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu")).double()

    # init AdamW optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                  betas=(0.9, 0.999), weight_decay=0.1,
                                  eps=1e-8,)

    # load weights if specified
    if WEIGHTS != None:
        # grab specific state before loading
        orig_state_dict = model.state_dict()

        # load in weights
        weights = torch.load(WEIGHTS, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        # set weights
        model.load_state_dict(weights["model_state_dict"])
        # scheduler.load_state_dict(weights["scheduler_state_dict"])
        # optimizer.load_state_dict(weights["optimizer_state_dict"])

        # Check model has been updated
        loaded_state_dict = model.state_dict()

        if compare_model_state_dicts(orig_state_dict, loaded_state_dict):
            print("Model State Has NOT CHANGED")
        else:
            print("Model State Has Updated")
        # set epoch
        BEGIN_EPOCH = weights["epoch"]
        if not args.override_lr:
            # grab LR
            LR = weights["optimizer_state_dict"]["param_groups"][0]["lr"]
    else:
        model.apply(xavier_init)

    # print(model.named_children())

    ##### Model 0.6
    # Apply spectral normalization to all linear and convolutional layers
    def apply_spectral_norm(model, lipschitz_constant=1):
        for name, module in model.named_children():
            if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):  # Check if the layer type is Conv2d or Linear
                # Apply spectral norm
                # print('module name', module)
                # print('module list', module.children())
                nn.utils.spectral_norm(module, n_power_iterations=1)
                with torch.no_grad():
                    module.weight_u *= lipschitz_constant
            elif len(list(module.children())) > 0:  # If the module has children, recursively apply to its children
                apply_spectral_norm(module, lipschitz_constant)

    # Apply spectral normalization to the entire network
    # apply_spectral_norm(model, 100)
    ##### Model 0.6

    # Specify angle and linear number for L2 loss specifics
    angle_number = PARAMS["torsion_main_n"] + PARAMS["torsion_side_n"] + PARAMS["bond_angle_main_n"] + PARAMS["bond_angle_side_n"]
    linear_number = PARAMS["input_size"] - angle_number


    # early stopping class
    early_stopping = EarlyStopper(patience=50, min_delta=0.1)

    warmup_steps = 1000
    base_lr = 1e-8
    initial_lr = LR

    def warmup(current_step: int):
        if current_step < 1000:  # current_step / warmup_steps * base_lr
            return float((current_step+1) / 1000)
        else:                                 # (num_training_steps - current_step) / (num_training_steps - warmup_steps) * base_lr
            return max(0.0, float(len(dataset)*EPOCHS - current_step) / float(max(1, len(dataset)*EPOCHS - 1000)))


    # Create the scheduler
    scheduler = CustomScheduler(
        warmup_steps = 5000, #10000, #20000, #5000, # 10000,
        base_lr = 1e-10,
        max_lr = LR,
        patience = 10,
        cooldown = 5,
        optimizer = optimizer,
    )

    # init scheduler
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
    #                                                        patience = 10,
    #                                                        mode = "min",
    #                                                        factor = 0.8,
    #                                                        threshold = 1e-3,
    #                                                        min_lr = 1e-20,
    #                                                        )

    # scheduler = None

    # Model Run specific number classifier for output files
    random_num = random.randint(0, 10000)

    # Generate output for run
    file_out_name = os.path.join(args.path_to_output_dir, f"model_run_{random_num}.csv")
    with open(file_out_name, "w") as file_out:
        print("epoch,loss,val_loss,angle_loss,linear_loss,latent_rmsd_loss,constrain_angle_loss", file=file_out)

    # Run Model
    print(f"***MODEL {random_num} TRAIN***")
    final_loss, lowest_loss, epoch_lowest_loss, latent_plotly, epoch_final = mse_train(
        model = model,
        train_dataset = train_loader,
        val_dataset = val_loader,
        epochs = EPOCHS,
        begin_epoch = BEGIN_EPOCH,
        device = torch.device("cuda"),
        optimizer = optimizer,
        scheduler = scheduler,
        angle_n = angle_number,
        linear_n = linear_number,
        file_out_track = file_out_name,
        output_dir = args.path_to_output_dir,
        weight_save_n = args.output_weights_every_n,
        file_num = random_num,
        real_rmsd_mat = RMSD_REAL,
        latent_plotly = PLOTLY_LATENT,
        sample_n = X_scaled.size(0) if args.rotation_permutation else X.size(0),
        real_X = X_scaled if args.rotation_permutation else X,
        early_stop = early_stopping,
        latent_dim = LATENT_SPACE,
        energy_n = PARAMS["energy_n"],
        overfit = args.overfit,
        noise_iterations = args.noise_iterations,
        scale_outputs = factor_vals_energy,
    )

    # saved extract latent tensor
    torch.save(latent_plotly, os.path.join("/mnt/ceph/users/apowers/outputs_old/data_tensor", f"model_run_{random_num}_latent_tensor.pt"))

    # Save to combo file
    md_combo_out_file = os.path.join(args.path_to_output_dir, "results.md")

    # if it exists save to it
    if os.path.exists(md_combo_out_file):
        with open(md_combo_out_file, 'a') as md_out:
            print(f"|{random_num}|{EPOCHS}|{epoch_final}|{args.learning_rate}|{N_ENCODER_LAYERS}|{N_DECODER_LAYERS}|{lowest_loss}|{epoch_lowest_loss}|{final_loss}|", file=md_out)
    else:
        with open(md_combo_out_file, 'w') as md_out:
            print(f"|File Number|Epochs|Final Epoch|LR|Num Encoding Layers|Num Decoding Layers|Lowest Loss|Epoch Lowest Loss|Final Loss|", file=md_out)
            print(f"|---|---|---|---|---|---|---|---|---|", file=md_out)
            print(f"|{random_num}|{EPOCHS}|{epoch_final}|{args.learning_rate}|{N_ENCODER_LAYERS}|{N_DECODER_LAYERS}|{lowest_loss}|{epoch_lowest_loss}|{final_loss}|", file=md_out)

    return 0


if __name__ == "__main__":
    main()

