import matplotlib.pyplot as plt
import sys

sys.path.append("src")

import pandas as pd
from prodict import Prodict
import utils
import numpy as np


def produce_survival_curves(cfg: Prodict):

    print("ready to make survival curves")
    run_path = f"{cfg.DATA.OUTPUT_DIR}/{cfg.TEST.RUN_ID}/e{cfg.TEST.EPOCH_TO_TEST}"
    # only works on test set (Objective: utilize for all cohorts)
    pred_path = f"{run_path}/tables/{cfg.TEST.SURVIVAL_CURVE.PREDICTION_PATH}"
    plot_path = f"{run_path}/plots/survival_curve"
    print(pred_path)
    utils.make_dir(plot_path)
    pred_df = pd.read_csv(pred_path)
    print(pred_df.head())

    #print(np.sort(pred_df.y_test.values))
    # need the gt plots - maybe hold off til I get response (not correct)
    survival_gt = pred_df.y_test.values
    event = pred_df.y_test.values
    predictions = pred_df["-predictions"].values


    # 1. get info for a single pt
    pt_data = pred_df.loc[pred_df.NMID == cfg.TEST.SURVIVAL_CURVE.PATIENT_IDS[0]]
    print(pt_data)

    # do some exploration: event happens means that the hazard rate is higher
    print(np.mean(pred_df.loc[pred_df.event == 0]))
    print(np.mean(pred_df.loc[pred_df.event == 1]))

    return
    predictions = predictions / np.max(predictions)
    # bootstrap survival probs using the HRs for each arm

    pred_survival_curves = list()

    for iteration in range(cfg.TEST.SURVIVAL_CURVE.N_BS):
        print("performing bootstrapping here")
        time = 0
        num_patients = survival_gt.shape[0]
        max_time = np.max(survival_gt)
        pt_alive = num_patients
        times = np.linspace(0,max_time,num=int(max_time))
        pt_alive_arr = []
        alive_mask = np.ones(survival_gt.shape)
        for t in times:
            # for each patient: non negative lower number decrements
            for idx, p in enumerate(survival_gt):
                if alive_mask[idx] == 1 and p > 0 and p < t:
                    # survival probability
                    event_status = np.random.uniform() < predictions[idx]
                    if event_status:
                        pt_alive -= 1
                        alive_mask[idx] = 0
            pt_alive_arr.append(pt_alive)
        
        pred_survival_curves.append(pt_alive_arr)

        

    # sort and get bounds
    sorted_vals = np.sort(pred_survival_curves, axis=0)
    bound = int(cfg.TEST.SURVIVAL_CURVE.N_BS * (1 - 0.99) / 2)
    y_lower = sorted_vals[bound - 1]
    y_upper = sorted_vals[-bound]
    median = sorted_vals[int(cfg.TEST.SURVIVAL_CURVE.N_BS / 2)]
    # get CI
    time = 0
    num_patients = survival_gt.shape[0]
    max_time = np.max(survival_gt)
    pt_alive = num_patients
    times = np.linspace(0,max_time,num=int(max_time))
    pt_alive_arr = []
    alive_mask = np.ones(survival_gt.shape)
    for t in times:
        # for each patient: non negative lower number decrements
        for idx, p in enumerate(survival_gt):
            if alive_mask[idx] == 1 and p > 0 and p < t:
                pt_alive -= 1
                alive_mask[idx] = 0
        pt_alive_arr.append(pt_alive)
    

    fig = plt.figure(figsize=(10, 10))
    fig.patch.set_alpha(0.0)
    plt.rc("font", size=40)

    plt.plot(times, pt_alive_arr, linewidth=3)
    plt.plot(times, median, '--', linewidth=3)
    plt.fill_between(times, y_lower, y_upper, alpha=0.4)
    plt.xlabel("Time", labelpad=25)
    plt.ylabel("Survival", labelpad=25)
    plt.ylim(0, 230)
    plt.xticks([])
    plt.yticks([])
    plt.savefig("survival_curve_test.png")


    return
