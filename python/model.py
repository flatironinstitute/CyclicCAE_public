


# base python packages
import sys, os
from collections import defaultdict
import time
import argparse
import random
import math
import glob
from multiprocessing import shared_memory

# torch
import torch
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

#numpy
import numpy as np

# pytorch
import torch.nn as nn

# train 
from model_functions import train as mse_train
from model_functions import scale, xavier_init, ScaleOutputs, StochasticSampler, SharedMemoryDataset, MyDatasetShared

# Model 
from model_utils import ConformationalMLPAutoEncoder, MyDataset, EarlyStopper, CustomScheduler, subset_dataset, rotate_samples
# from test_case import TestCase1, HingeModel, ConvCase1, ConvCase2, ConvCase3, ConvCase4
from test_case import ConvCase4

# model terminal peptide
from terminal_error import TerminalPeptide

def compare_model_state_dicts(state_dict1, state_dict2):
    for key in state_dict1:
        if key in state_dict2:
            if not torch.equal(state_dict1[key], state_dict2[key]):
                return False
        else:
            return False
    return True

def ddp_setup(rank: int, world_size: int):
    """Setup our system for using DDP

    PARAMS
    ------
    :rank: Unique identifier of each process
    :world_size: Total number of epochs
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    torch.cuda.empty_cache() # IDK if this is necessary
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

#### Train
# def main(rank, world_size, dataset, args, rmsd, params):
def main(rank, world_size, shm_x, shm_rmsd, xshape, xdtype, rmsdshape, rmsddtype, params, factor_vals_energy, z, args):
    """Run training"""

    # Set up ddp
    ddp_setup(rank, world_size)

    # load in dataset
    # dataset = torch.load(args.dataset_tensor)

    # set global vars
    PARAMS           = params  #dataset["params"]
    X                = MyDatasetShared(shm_x, xshape, xdtype) #dataset["x"].double()
    RMSD_REAL        = MyDatasetShared(shm_rmsd, rmsdshape, rmsddtype)    #dataset["rmsd"].double()
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

    # assert EPOCHS % PLOTLY_LATENT == 0, "Epochs for Plotly Number does not have remainder 0 for passed EPOCH number"

    # # scale data
    # X_scaled = scale(X, PARAMS)
    #
    # if args.rotation_permutation:
    #     # rotate samples
    #     X_scaled = rotate_samples(X_scaled)

    # pass to custom dataset class
    # dataset = MyDataset(X_scaled)
    dataset = SharedMemoryDataset(shm_x, xshape, xdtype)

    # grab scaling values
    # factor_vals_energy = ScaleOutputs(X[:, -PARAMS["energy_n"]:])
    # factor_vals_energy.scaled_vals.to(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    # energy_scaled = torch.subtract(X[:, -PARAMS["energy_n"]:], X[:, -PARAMS["energy_n"]:].min(dim=0).values)
    # z = torch.pow(torch.sum(energy_scaled / 0.63, dim=0), 2) # 2


    if args.overfit:
        # Generate train/val split
        train_idx = np.random.choice(X_scaled.size(0), size=args.overfit_n, replace=False)

        # load data into a specific dataloader class
        train_loader = torch.utils.data.DataLoader(subset_dataset(dataset, train_idx), batch_size=args.batch_size, shuffle=False)
        val_loader = torch.utils.data.DataLoader(subset_dataset(dataset, train_idx), batch_size=args.batch_size, shuffle=False)

    else:
        # Generate train/val split
        # train_idx = np.random.choice(X_scaled.size(0), size=int(X_scaled.size(0)*.9), replace=False)
        # val_idx = list( set(range(X_scaled.size(0))) - set(train_idx))
        train_idx = np.random.choice(len(X), size=int(len(X)*.9), replace=False)
        val_idx = list( set(range(len(X))) - set(train_idx))

        # load data into a specific dataloader class
        if args.shuffle_samples:
            sampler = StochasticSampler(train_idx.shape[0], args.subset_size)
            train_loader = torch.utils.data.DataLoader(subset_dataset(dataset, train_idx), batch_size=args.batch_size, sampler=sampler)
        else:
            train_pair = subset_dataset(dataset, train_idx)
            # train_pair = subset_dataset(X, train_idx)
            train_sampler = DistributedSampler(train_pair, num_replicas=world_size, rank=rank, shuffle=True)
            train_loader = torch.utils.data.DataLoader(train_pair, batch_size=args.batch_size, shuffle=False, sampler=train_sampler)
        val_pair = subset_dataset(dataset, val_idx)
        # val_pair = subset_dataset(X, val_idx)
        val_loader = torch.utils.data.DataLoader(val_pair, batch_size=args.batch_size, shuffle=False, sampler=DistributedSampler(val_pair, num_replicas=world_size, rank=rank, shuffle=True))

    # model = ConformationalMLPAutoEncoder(
    # model = HingeModel(
    # model = ConvCase1(
    # model = ConvCase3(
    model = DDP(ConvCase4(
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
    ).to(torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")).double(), broadcast_buffers=False, find_unused_parameters=True, device_ids=[rank]) #.to(torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")).double()
    # model = model.to(f"cuda:{rank}") # This is for DDP

    # init AdamW optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                  betas=(0.9, 0.999), weight_decay=0.1,
                                  eps=1e-8,)

    # load weights if specified
    if WEIGHTS != "None":
        # grab specific state before loading
        orig_state_dict = model.module.state_dict()

        # load in weights
        # weights = torch.load(WEIGHTS, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        weights = torch.load(WEIGHTS, map_location=torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu"))
        # set weights
        model.module.load_state_dict(weights["model_state_dict"])
        # scheduler.load_state_dict(weights["scheduler_state_dict"])
        optimizer.load_state_dict(weights["optimizer_state_dict"])

        # Check model has been updated
        loaded_state_dict = model.module.state_dict()

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

    # def warmup(current_step: int):
    #     if current_step < 1000:  # current_step / warmup_steps * base_lr
    #         return float((current_step+1) / 1000)
    #     else:                                 # (num_training_steps - current_step) / (num_training_steps - warmup_steps) * base_lr
    #         return max(0.0, float(len(dataset)*EPOCHS - current_step) / float(max(1, len(dataset)*EPOCHS - 1000)))


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
    #                                                        patience = 40,
    #                                                        mode = "min",
    #                                                        factor = 0.8,
    #                                                        threshold = 1e-3,
    #                                                        min_lr = 1e-20,
    #                                                        )

    # scheduler = None

    # pull out values
    torsion_full = X[:, :PARAMS["torsion_main_n"]]
    bond_angles = X[:, PARAMS["torsion_main_n"]:PARAMS["torsion_main_n"]+PARAMS["bond_angle_main_n"]]
    bond_lengths = X[:, PARAMS["torsion_main_n"]+PARAMS["bond_angle_main_n"]:-PARAMS["energy_n"]]

    # init class
    term_pep_class = TerminalPeptide()

    # root
    root = term_pep_class.calculate_root(PARAMS["protein_len"])
    print("ROOT INDEX: ", root)

    # index tensors
    phi_psi, omega_idx = term_pep_class.torsion_extract(torsion_full.size(-1), PARAMS["protein_len"], root)
    omega = torsion_full[:, omega_idx]
    torsion = torsion_full[:, phi_psi]

    # compute coordinates
    coordinates = term_pep_class.terminal_peptide(torsion, omega, bond_angles, bond_lengths, root_idx=root)
    n_coords = coordinates[:, :, 2, :]
    c_coords = coordinates[:, :, 1, :]
    n_roll = torch.roll(n_coords, shifts=-1, dims=1)

    # real terminal length calc
    bond_length = torch.norm(n_roll - c_coords, dim=-1)
    real_terminal_length = bond_length[:, -1].view(-1, 1)

    # print('MODEL X REAL PEP LENGTH')
    # print(real_terminal_length)

    # Model Run specific number classifier for output files
    random_num = random.randint(0, 10000)

    # Generate output for run
    if not os.path.exists(args.path_to_output_dir):
        os.makedirs(args.path_to_output_dir)

    file_out_name = os.path.join(args.path_to_output_dir, f"model_run_{random_num}.csv")
    with open(file_out_name, "w") as file_out:
        print("epoch,loss,val_loss,angle_loss,linear_loss,latent_rmsd_loss,constrain_angle_loss,terminal_loss", file=file_out)

    # Run Model
    print(f"***MODEL {random_num} TRAIN***")
    final_loss, lowest_loss, epoch_lowest_loss, latent_plotly, epoch_final = mse_train(
        model = model,
        train_dataset = train_loader,
        val_dataset = val_loader,
        epochs = EPOCHS,
        begin_epoch = BEGIN_EPOCH,
        device = torch.device(f"cuda:{rank}") if torch.cuda.is_available() else torch.device("cpu"),
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
        # sample_n = X_scaled.size(0) if args.rotation_permutation else X.size(0),
        sample_n = X_scaled.size(0) if args.rotation_permutation else len(X),
        real_X = X_scaled if args.rotation_permutation else X,
        early_stop = early_stopping,
        latent_dim = LATENT_SPACE,
        energy_n = PARAMS["energy_n"],
        overfit = args.overfit,
        noise_iterations = args.noise_iterations,
        scale_outputs = factor_vals_energy,
        min_vals = factor_vals_energy.min.to(f"cuda:{rank}" if torch.cuda.is_available() else "cpu"),
        z = z.to(f"cuda:{rank}" if torch.cuda.is_available() else "cpu"),
        params = PARAMS,
        real_terminal_lengths = real_terminal_length,
        term_class = term_pep_class,
        train_sampler = train_sampler,
    )
    destroy_process_group()

    # saved extract latent tensor
    # torch.save(latent_plotly, os.path.join("/mnt/ceph/users/apowers/outputs_old/data_tensor", f"model_run_{random_num}_latent_tensor.pt"))

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
    # main()
    # Grab passed in values from CLI
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-tensor", type=str, default="./binary_import/dof_vectors2024-02-20torch_tensor.pt",
                   help="Path to .pt dataset file. (defualt: ./binary_import/dof_vectors2024-02-20torch_tensor.pt)")
    p.add_argument("--epochs", type=int, default=1000, help="Number of epochs to train (default: 1000)")
    p.add_argument("--num-encoder-layers", type=int, default=3,
                   help="Number of encoder layers in the model (default: 3)")
    p.add_argument("--num-decoder-layers", type=int, default=3,
                   help="Number of decoder layers in the model (default: 3)")
    p.add_argument("--latent-space", type=int, default=3,
                   help="Latent Space R dimension (default: 3)")
    p.add_argument("--batch-size", type=int, default=32,
                   help="Specify model batch size (default: 32)")
    p.add_argument("--learning-rate", type=float, default=1e-3, help="Specify the LR (default: 1e-3)")
    p.add_argument("--output_weights_every_n", type=int, default=1000,
                   help="Output saved model weights every N number of epochs (default: 1000)")
    p.add_argument("--load-weights", type=str, default=None, 
                   help="Load in specific weights from known stopping position. (default: None)")
    p.add_argument("--path-to-output-dir", type=str, default="/mnt/ceph/users/apowers/outputs_old/",
                   help="Specify model output specific directory (default: /mnt/ceph/users/apowers/outputs_old/)")
    p.add_argument("--plotly-latent-animate", type=int, default=0,
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
    p.add_argument("--shuffle-samples", action="store_true", help="We don't use the full dataset each epoch, but a subset. Shuffles each epoch")
    p.add_argument("--subset-size", type=int, default=50_000, help="Works in conjunction with shuffle-samples")
    p.add_argument("--shards", action="store_true", help="If the dataset is large indicate this, and we will search for the 'shards' files in the same location.")
    args = p.parse_args()

    # load in dataset
    dataset = torch.load(args.dataset_tensor)
    print("dataset fully loaded...")
    params = dataset["params"]

    # grab scaling values
    factor_vals_energy = ScaleOutputs(dataset["x"][:, -params["energy_n"]:])
    # factor_vals_energy.scaled_vals.to(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    energy_scaled = torch.subtract(dataset["x"][:, -params["energy_n"]:], dataset["x"][:, -params["energy_n"]:].min(dim=0).values)
    z = torch.pow(torch.sum(energy_scaled / 0.63, dim=0), 2) # 2

    dataset["x"] = dataset["x"].double().numpy()
    dataset["rmsd"] = dataset["rmsd"].double().numpy()

    # scale data
    # X_scaled = scale(dataset["x"], params)

    # if args.rotation_permutation:
    #     # rotate samples
    #     X_scaled = rotate_samples(X_scaled)
    #
    # dataset = SharedMemoryDataset(X_scaled)


    x_shape, x_dtype = dataset["x"].shape, np.float64
    rmsd_shape, rmsd_dtype, numbytes = dataset["rmsd"].shape, np.float64, dataset["rmsd"].nbytes


    shm_x = shared_memory.SharedMemory(create=True, size=dataset["x"].nbytes)
    shm_x_array = np.ndarray(x_shape, dtype=x_dtype, buffer=shm_x.buf)

    np.copyto(shm_x_array, dataset["x"])
    print("Transfered X...")

    if args.shards:
        # Delete the dataset
        del dataset

        # Generate our holding rmsd dataset
        shm_rmsd = shared_memory.SharedMemory(create=True, size=numbytes)
        shm_rmsd_array = np.ndarray(rmsd_shape, dtype=rmsd_dtype, buffer=shm_rmsd.buf)

        # Increment through
        begin = 0
        end = 0
        file_in = args.dataset_tensor.split('.pt')[0]
        files_glob = glob.glob(f"{file_in}*shard*.pt")
        # Sort them, so that they are loaded in order
        files_glob.sort()
        print("Iterate through glob files...")
        print(files_glob)
        for i, f in enumerate(files_glob):
            rmsd = torch.load(f)
            end += rmsd.shape[0]
            if i < len(files_glob)-1:
                shm_rmsd_array[begin:end] = rmsd.numpy()
            else:
                shm_rmsd_array[begin:] = rmsd.numpy()
            print(f"Transfered {f} to shared memory allocation...")
            print("Section:", begin, ":", end)
            begin += rmsd.shape[0]
            del rmsd
    else:
        # Generate our holding rmsd dataset
        shm_rmsd = shared_memory.SharedMemory(create=True, size=numbytes)
        shm_rmsd_array = np.ndarray(rmsd_shape, dtype=rmsd_dtype, buffer=shm_rmsd.buf)
        np.copyto(shm_rmsd_array, dataset["rmsd"])
        del dataset

    world_size = torch.cuda.device_count()
    try:
        mp.spawn(main, args=(world_size, shm_x.name, shm_rmsd.name, x_shape, x_dtype, rmsd_shape, rmsd_dtype, params, factor_vals_energy, z, args), nprocs=world_size, join=True)
    finally:
        shm_x.close()
        shm_x.unlink()
        shm_rmsd.close()
        shm_rmsd.unlink()

