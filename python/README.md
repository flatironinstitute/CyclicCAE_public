# Python Folder Information

A simple informative overlay of what each file is and what they do/house.

## Files

- `model_utils.py`:
    Houses all functions and classes involved in the model architecture. In 
    particular the Auto-Encoder model classes.

- `model_functions.py`:
    Functions related to the Loss and Train/Test are within this file.

- `model.py`:
    Training specific file. This file takes in the already constructed dataset
    tensor and passes it with params to the model. Scheduler and Optimizer are
    specified here. As well as loss per epoch training run file, and every $n$ epoch
    weights saved tensor.

- `graph_loss.py`:
    Plot the train and validatin loss of a particular model run. There are additional
    options that allow you to zoom in on a specific $x$-axis region.

- `graph_latentspace.py`:
    Plot the latent space $f(X+E)$, where $X$ are our inputs, $E$ is some noise added
    to it, and $f(\cdot)$ is the encoding function. You can also, animate the particular
    latent space trajectory through its various epochs. This animation can be manipulated.

- `decode_traj.py`:
    Take the weights of a particular model run and use them to generate trajectories from
    a particular latent space from $(A, B)$ points. You can linearlly interpolate between these
    two points. Allowing you to animate how the protein/peptide moves, as you time step through
    the latent space. Also, you can calculate an error and accuracy metric if you want to see how
    closely we are recapitulating the angles and sin/cos values.

- `binary_import/BinaryDoFCreater.py`:
    Takes a trajectory generated from `decode_traj.py` and converts it into a binary files that can
    be read by Rosetta to reconstruct the decoded protien/peptide.

- `binary_import/BinaryDoFReader.py`:
    Takes binary generated datasets from Rosetta and converts them into DoF `.pt` torch files that
    can be used for training with the ML model.

- `energy_search.py`:
    Given a tensor dataset generated from `binary_import/BinaryDoFReader.py`, perform either of these
    operations on a selected peptide:
    - Inpaint: Keep a given anchor/motif and generate scaffolds to best accomidate that motif.
    - dE/d(latent_space): Minimize the energy of the peptide. This could be large or small structural changes
        depending on the learning rate used. However, this is deterministic.
    - MCMC Metropolis-Hastings: From a given structure explore paths around that point, to generate
        multiple outputs per given input.

- `insolution_genkic.py`
    Sample macrocycle conformations in isolation using the Rosetta Generalized Kinematic Closure (GeneralizedKIC)
    algorithm.  This is useful as a point of comparison for CyclicCAE.


## Folders

- `outputs/`:
    General folder that holds the model run loss csv files.

    - `model_weights/`:
        Specific folder for model specific weights. Both current epoch and every $n$ epoch
        saved weights.

- `binary_import/`:
    Houses the scripts for converting Binary to `.pt` and `.pt` to binary. Also, houses dataset
    that were computed to `.pt` from binary.

