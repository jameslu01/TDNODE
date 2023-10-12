import sys

sys.path.extend(["src"])
from src.predict import evaluate_model
import matplotlib.pyplot as plt
import utils

import numpy as np
from prodict import Prodict
import matplotlib.lines as mlines


def plot_select_cases(cfg: Prodict) -> None:
    """Generates patient-level plots comparing TDNODE-generated predictions with SLD observations.

    Parameters
    ----------
    cfg : Prodict
        A configuration file describing the parameters of the evaluation.
    """
    print("Plotting Individual Cases")
    (_, _, _, _, performance_by_pt, _) = evaluate_model(cfg)

    save_dir = f"{cfg.DATA.OUTPUT_DIR}/{cfg.TEST.RUN_ID}/e{cfg.TEST.EPOCH_TO_TEST}/plots/individual_cohort-{cfg.DATA.COHORT}_plotpred-{cfg.TEST.INDIVIDUAL.PLOT_PRED}"
    utils.make_dir(save_dir)
    plt.rcParams['axes.facecolor'] = '#D3D3D3'

    save_name = f"{cfg.NAME}_pt-individual_prediction_cutoff-{cfg.DATA.CUTOFF}-grouped.png"
    pt_ids = cfg.TEST.INDIVIDUAL.PATIENT_IDS if not cfg.TEST.INDIVIDUAL.GET_ALL_PTS else set(performance_by_pt.keys())
    for (idx, nmid) in enumerate(pt_ids):
        fig = plt.figure(figsize=(12, 10))
        fig.patch.set_alpha(0.0)
        plt.rc("font", size=40)
        save_name = f"{cfg.NAME}_pt-individual_prediction_cutoff-{cfg.DATA.CUTOFF}_nmid-{nmid}_psuedoid-{idx}.png"
        save_path = f"{save_dir}/{save_name}"
        print(f"Generating graph for pt {nmid}")
        try:
            pt_data = performance_by_pt[nmid]
        except KeyError:
            print(f"Patient {nmid} is not found.")
            continue
        labels = np.array(pt_data.labels)
        times = np.array(pt_data.label_times)
        cutoff_idx = np.argmax(times > cfg.DATA.CUTOFF)
        if cutoff_idx == 0:
            cutoff_idx = len(labels)
        labels_before = labels[:cutoff_idx]
        labels_after = labels[cutoff_idx:]
        times_before = times[:cutoff_idx]
        times_after = times[cutoff_idx:]

        plt.scatter(
            times_before,
            labels_before,
            facecolors='r', #cfg.TEST.INDIVIDUAL.PATIENT_COLORS[idx],
            edgecolors='r', #cfg.TEST.INDIVIDUAL.PATIENT_COLORS[idx],
            s=150,
            linewidths=2
        )
        plt.scatter(
            times_after,
            labels_after,
            facecolors="none",
            edgecolors='r', #cfg.TEST.INDIVIDUAL.PATIENT_COLORS[idx],
            s=150,
            linewidths=2
        )
        if cfg.TEST.INDIVIDUAL.PLOT_PRED:
            plt.plot(
                pt_data.pred_times, pt_data.preds, color='r') #cfg.TEST.INDIVIDUAL.PATIENT_COLORS[idx]
        
        if not cfg.TEST.INDIVIDUAL.GET_ALL_PTS:
            close_plot(cfg, save_path, idx)
    #if not cfg.TEST.INDIVIDUAL.GET_ALL_PTS:
    #close_plot(cfg, save_path, idx)


def close_plot(cfg: Prodict, save_path: str, nmid: int) -> None:
    """Closes the current patient-level plot with the selected labels & axes options.

    Parameters
    ----------
    cfg : Prodict
        A configuration file describing the parameters of the evaluation.
    save_path : str
        A path referencing where the plot will be saved.
    """
    line_marker = mlines.Line2D([], [], color="red", linestyle="-", markersize=10)
    point_marker = mlines.Line2D([], [], color="red", marker="o", linestyle="None", markersize=10)
    hollow_marker = mlines.Line2D(
        [], [], color="red", marker="o", linestyle="None", markersize=10, fillstyle="none"
    )
    obs_marker = mlines.Line2D([], [], color="black", linestyle="--", markersize=10)
    plt.xlabel("Time", labelpad=25)
    plt.ylabel("SLD", labelpad=25)
    plt.yticks([])
    plt.xticks([])
    # plt.tick_params(
    #     axis='x',          # changes apply to the x-axis
    #     which='both',      # both major and minor ticks are affected
    #     bottom=False,      # ticks along the bottom edge are off
    #     top=False,         # ticks along the top edge are off
    #     labelbottom=False) # labels along the bottom edge are off
    # plt.tick_params(
    #     axis='y',          # changes apply to the x-axis
    #     which='both',      # both major and minor ticks are affected
    #     left=False,      # ticks along the bottom edge are off
    #     top=False,         # ticks along the top edge are off
    #     labelbottom=False) # labels along the bottom edge are off
    if cfg.DATA.CUTOFF != 1000:

        plt.axvline(x=cfg.DATA.CUTOFF, linestyle="--", color="black")
    #plt.title(f"Subject {nmid}")
    # if cfg.TEST.INDIVIDUAL.PLOT_PRED:
    #     plt.legend(
    #         [line_marker, point_marker, hollow_marker, obs_marker],
    #         ["Predictions", "Observations", "Unseen", "Obs Window"]
    #     # loc="upper center",
    #     )
    # else:
    #     plt.legend(
    #         [point_marker, hollow_marker, obs_marker],
    #         ["Observations", "Unseen", "Obs Window"]
    #     )


    plt.savefig(save_path)
    plt.close()
