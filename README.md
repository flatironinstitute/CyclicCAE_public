# Conformational autoencoder

## Authors

Vikram K. Mulligan (vmulligan@flatironinstitute.org)

Andrew Powers (apowers@flatironinstitute.org)

## Description
![](./pictures/backbones.gif)

Given an anchor, failed design, or motif allow the heterochiral design/redesign of backbone/scaffold for macrocycles.
Structure minimzation and Markov Chain Monte Carlo (MCMC) structure search.

## Inference

You can perform inference for any of the methods below:

- Inpaint (Motif Scaffolding)
    `--inpaint`
- MCMC Metropolis-Hastings
    `--mcmc`
- Energy Minimization
    `--energy-search`

The general approach for running inference is as such:
```
# base run
python ./python/energy_gradient.py --weights-load <path/to/weights> --dataset <binary/file/of/input/dofs> --<method> --<extra method specific params>
```

All of the method specifc params are shown here:

```
p = argparse.ArgumentParser()
p.add_argument("--steps", type=int, default=1000, help="Search Steps Max (default: 1000)")
p.add_argument("--minima", type=float, default=-4.775, help="Minima energy to search for (default: -5.0)")
p.add_argument("--start-point", type=int, default=14, help="This will be updated to take a pdb in the future")
p.add_argument("--weights-load", type=str, help="Specify Model Weights To Use")
p.add_argument("--lr", type=float, default=0.001, help="Optimizer Learning Rate (default: 1e-3)")
p.add_argument("--dataset", type=str, default="./binary_import/dof_vectors2024-04-10torch_tensor.pt", help="Dataset to use (default: dof_vectors2024-04-10torch_tensor)")
p.add_argument("--cutoff-true", action="store_true", help="Apply a distance cutoff")
p.add_argument("--cutoff", type=float, default=0.005, help="Specify Euclidean Distance Cutoff in Latent Space to Search (default: 0.005)")
p.add_argument("--latent-dim", type=int, default=3, help="Latent Size of the model (default: 3)")
p.add_argument("--partial-optimization", action="store_true", help="Apply partial optimization to a structure")
p.add_argument("--freeze-residues", type=str, help="Used wth partial-optimization. If no sub freeze flags specified all are frozen \
                    Specify residue in a list (zero index) to freeze (e.g. '[1,2,3]')")
p.add_argument("--freeze-torsion", action="store_true", help="Use with partial-optimization")
p.add_argument("--freeze-bond-angle", action="store_true", help="Use with partial-optimization")
p.add_argument("--freeze-bond-length", action="store_true", help="Use with partial-optimization")
p.add_argument("--lambda-partial", type=float, default=1.0, help="Reweight Partial Freezing (Default: 1.0)")
p.add_argument("--energy-search", action="store_true", help="Gradient Search on dE/d(latent space)")
p.add_argument("--mcmc", action="store_true", help="Markov Chain Monte Carlo Metropolis Hastings Search")
p.add_argument("--batch-size", type=int, default=10, help="Used with mcmc. Set number of trajectories (default: 10)")
p.add_argument("--sigma", type=float, default=0.01, help="Used with mcmc. Set sigma of normal movement to add to latent point (default: 0.01)")
p.add_argument("--temperature", type=float, default=1, help="Used with mcmc. Set Temperature (default: 1)")
p.add_argument("--initial-steps", type=int, default=10, help="Used with mcmc. Set initial steps to accept no matter what (default: 10)")
p.add_argument("--ramp-down", type=int, default=100, help="Used with mcmc. Steps to go from temperature to 1 (default: 100)")
p.add_argument("--retain-structure", action="store_true", help="Used with mcmc. Walks path closer to input structure")
p.add_argument("--inpaint", action="store_true", help="Provide Frozen Residues and generate scaffolds")
p.add_argument("--inpaint-dist", type=float, default=300.0, help="Cutoff distance (p1norm) to consider a scaffold a success (default: 300.0)")
p.add_argument("--sgd-optim", action="store_true", help="Use SGD optimizer with momentum 0.9. AdamW is used by default")
```


## Training Model

If you would like to retrain the model currently you can run the script `model.py`:

```
# Train model with default parameters
python ./python/model.py --dataset-tensor <path/to/dataset>
```

The curent model was trained with first training on `./python/binary_import/dof_vectors_clusters_firstten2024-04-11torch_tensor.pt`
then `./python/binary_import/dof_vectors2024-04-10torch_tensor.pt`

You can change the model's default parameters with the flags below:

```
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
p.add_argument("--batch-size", type=int, default=2064,
               help="Specify model batch size (default: 2064)")
p.add_argument("--learning-rate", type=float, default=1e-3, help="Specify the LR (default: 1e-3)")
p.add_argument("--output_weights_every_n", type=int, default=1000,
               help="Output saved model weights every N number of epochs (default: 1000)")
p.add_argument("--load-weights", type=str, default=None, 
               help="Load in specific weights from known stopping position. (default: None)")
p.add_argument("--path-to-output-dir", type=str, default="./",
               help="Specify model output specific directory (default: ./)")
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
```

