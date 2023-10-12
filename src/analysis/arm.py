from prodict import Prodict

import json
import pandas as pd
import os
import sys

sys.path.append("src")
import utils
from predict import plot_predictions
from log_res import generate_log_pred_plots

def plot_performance_by_arm(cfg: Prodict) -> None:
    """Same functionality as `evaluate_model`, but targeted towards obtaining performance for
    subjects with a predefined treatment arm.

    Parameters
    ----------
    cfg : Prodict
        A configuration file specifying the parameters of the evaluation.
    """
    print(f"Plotting performance by treatment arm for arm {cfg.DATA.ARM}")
    df = pd.read_csv(cfg.DATA.TGI_DATA)
    ptids = Prodict.from_dict(json.load(open(cfg.TEST.ARM.PTID_PATH, "r")))
    (train_df, test_df) = (df[df.NMID.isin(ptids.train)], df[df.NMID.isin(ptids.test)])
    col = ["NMID", "TIME", "SLD"]
    arm_dir = f"{cfg.DATA.ROOT_DIR}/arm{cfg.DATA.ARM}"
    utils.make_dir(arm_dir)
    train_df[col].copy()[train_df.ARM == cfg.DATA.ARM].to_csv(f"{arm_dir}/train.csv", index=False)
    test_df[col].copy()[test_df.ARM == cfg.DATA.ARM].to_csv(f"{arm_dir}/test.csv", index=False)
    original_root_dir = cfg.DATA.ROOT_DIR
    cfg.DATA.ROOT_DIR = arm_dir
    #plot_predictions(cfg)

    # log residual
    print(f"Plotting residuals by treatment arm for arm {cfg.DATA.ARM}")
    generate_log_pred_plots(cfg)
    cfg.DATA.ROOT_DIR = original_root_dir
