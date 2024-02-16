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

## Folders

- `outputs/`:
    General folder that holds the model run loss csv files.

    - `model_weights/`:
        Specific folder for model specific weights. Both current epoch and every $n$ epoch
        saved weights.
