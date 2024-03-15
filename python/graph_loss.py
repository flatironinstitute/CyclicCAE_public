#!/usr/bin/env python
#
#
#

# packages
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse


# Define function for plotting

def plot_loss(data: pd.DataFrame, columns: list, title: str):
    """Plot a specific columns loss"""
    fig, ax = plt.subplots(1, 1, figsize=(12,8))

    # extract epochs
    epochs = data["epoch"]

    for loss in columns:
        loss_col = data[loss]
        ax.plot(epochs, loss_col, label=loss)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("MSE Loss Run {}".format(title))
    plt.legend()
    plt.savefig("loss_run_{}.png".format(title))
    plt.show() 
    

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--loss-csv", type=str, help="CSV Loss file to graph")
    p.add_argument("--include-linear-loss", action="store_true", help="Graph Linear Term Loss")
    args = p.parse_args()

    data = pd.read_csv(args.loss_csv)

    title = args.loss_csv.split("_")[-1].split('.')[0]

    if args.include_linear_loss:
        cols = ['loss','angle_loss','linear_loss']
    else:
        cols = ['loss','angle_loss']

    plot_loss(data, cols, title)
