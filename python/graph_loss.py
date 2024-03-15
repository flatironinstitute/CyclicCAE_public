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

def plot_loss(
    data: pd.DataFrame,
    columns: list,
    title: str,
    zoom: bool = False,
    lower_x: int = 0,
    upper_x: int = 0,
    ):
    """Plot a specific columns loss"""
    fig, ax = plt.subplots(1, 1, figsize=(12,8))

    # extract epochs
    epochs = data["epoch"]

    for loss in columns:
        loss_col = data[loss]
        ax.plot(epochs, loss_col, label=loss)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")

    if zoom:
        # extract zoomed in info
        zoom_df = data[(data["epoch"] >= lower_x) & (data["epoch"] <= upper_x)]
        ymin, ymax = zoom_df["loss"].min(), zoom_df["loss"].max()

        ax.set_ylim(ymin, ymax)
        ax.set_xlim(lower_x, upper_x)
        ax.set_title("MSE Zoom ({}, {}) Loss Run {}".format(lower_x, upper_x, title))

    else:
        ax.set_title("MSE Loss Run {}".format(title))

    plt.legend()
    plt.savefig("loss_run_{}_zoom{}.png".format(title, zoom))
    plt.show() 
    

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--loss-csv", type=str, help="CSV Loss file to graph")
    p.add_argument("--include-linear-loss", action="store_true", help="Graph Linear Term Loss")
    p.add_argument("--zoom-lower-x", type=int, default=0, help="Set zoom for graph lower x-lim")
    p.add_argument("--zoom-upper-x", type=int, default=0, help="Set zoom for graph upper x-lim")
    p.add_argument("--zoom-graph", action="store_true", help="Zoom in on graph")

    args = p.parse_args()

    data = pd.read_csv(args.loss_csv)

    title = args.loss_csv.split("_")[-1].split('.')[0]

    if args.include_linear_loss:
        cols = ['loss','angle_loss','linear_loss']
    else:
        cols = ['loss','angle_loss']

    plot_loss(data, cols, title, args.zoom_graph, args.zoom_lower_x, args.zoom_upper_x)
