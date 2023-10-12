from prodict import Prodict
import matplotlib.pyplot as plt
from sklearn import metrics
import utils
import pandas as pd
import numpy as np


def plot_loss_curves(cfg: Prodict) -> None:
    """Plots Loss and R2 curves during model training.

    Parameters
    ----------
    cfg : Prodict
        A configuration file describing the parameters for training TDNODE.
    """
    print("Plotting Loss and R2 curves.")
    run_dir = cfg.TEST.RUN_ID
    plot_dir = f"{run_dir}/overall_analysis"
    utils.make_dir(plot_dir)
    metrics_df = pd.read_csv(f"{run_dir}/metrics.csv", sep="\t")

    # plot the loss curve.
    plt.plot(metrics_df["Epoch"], metrics_df["Training Loss"])
    plt.plot(metrics_df["Epoch"], metrics_df["Validation Loss"])

    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.legend(["Training Loss", "Validation Loss"])
    plt.title("TDNODE Loss")
    plt.savefig(f"{plot_dir}/loss_curve.png")
    plt.close()

    # plot the R2 score.
    plt.plot(metrics_df["Epoch"], metrics_df["Training R2"])
    plt.plot(metrics_df["Epoch"], metrics_df["Validation R2"])
    plt.xlabel("Epoch")
    plt.ylabel("R2")
    plt.legend(["Training R2", "Validation R2"])
    plt.title("TDNODE R2")
    plt.savefig(f"{plot_dir}/R2_curve.png")
    plt.close()
    return
