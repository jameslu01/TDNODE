from prodict import Prodict
import sys


sys.path.append("src")

from args import args

from predict import plot_predictions
from src.analysis.explore_encodings import explore_encondings
from src.analysis.arm import plot_performance_by_arm
from src.analysis.bootstrap import perform_bootstrap
from src.analysis.encodings import get_pt_encodings
from src.analysis.predict_single import plot_select_cases
from src.analysis.vary_encodings import make_feature_dependence_plots
from src.analysis.loss_curve import plot_loss_curves
from src.analysis.log_res import generate_log_pred_plots
from src.analysis.survival_curves import produce_survival_curves
import numpy as np
import torch
import utils
from typing import List
import logging

from src.analysis.OS import perform_survival_analysis


def perform_model_assessment(cfg: Prodict):
    """Dynamic model analysis utility function. 

    Parameters
    ----------
    cfg : Prodict
        A configuration file describing the parameters of the evaluation.
    """
    analysis_dict = {
        "cumulative": plot_predictions,
        "bootstrap": perform_bootstrap,
        "individual": plot_select_cases,
        "encodings": get_pt_encodings,
        "OS": perform_survival_analysis,
        "feature_dependence": make_feature_dependence_plots,
        "loss_curve": plot_loss_curves,
        "log_residual": generate_log_pred_plots,
        "arm": plot_performance_by_arm,
        "explore_encodings": explore_encondings,
        "survival_curves": produce_survival_curves,
    }

    utils.log_message(f"Assessing model run ID {cfg.TEST.RUN_ID}.")
    utils.log_message("making directories")
    plot_path = f"{cfg.DATA.OUTPUT_DIR}/{cfg.TEST.RUN_ID}/e{cfg.TEST.EPOCH_TO_TEST}/plots"
    table_path = f"{cfg.DATA.OUTPUT_DIR}/{cfg.TEST.RUN_ID}/e{cfg.TEST.EPOCH_TO_TEST}/tables"
    utils.make_dir(plot_path)
    utils.make_dir(table_path)

    for cutoff in cfg.ANALYSIS.WINDOWS:
        utils.log_message(f"\nGenerating results with cutoff {cutoff}")
        cfg = adjust_cfg(cfg, "DATA", "CUTOFF", cutoff)
        for fn in cfg.ANALYSIS.FUNCTIONS:
            if fn == "encodings":
                cfg = adjust_cfg(cfg, "DATA", "CUTOFF", cfg.TEST.ENCODINGS.CUTOFF_TO_SET)
            elif fn == "OS":
                np.random.seed(42)  
            utils.log_message(f"Performing {fn} analysis")
            analysis_dict[fn](cfg)
            cfg = adjust_cfg(cfg, "DATA", "CUTOFF", cutoff)
            np.random.seed(cfg.DATA.SEED)
            utils.log_message(f"Function {fn} successfully executed.")
    print("done!")


def adjust_cfg(cfg: Prodict, key1: str, key2: str, value: any):
    """Adjust a specific parameter in the configuration file.
    Parameters
    ----------
    cfg : _type_
        A configuration file describing the perameters of evaluation.
    keylist : List
        A list of keys to access the value to be adjusted.
    value : any
        The new value to set.

    Returns
    -------
    Prodict
    
        The updated configuration dictionary.
    """
    cfg[key1][key2] = value
    return cfg


if __name__ == "__main__":
    cfg = utils.load_cfg(args.config)
    save_path = f"{cfg.DATA.OUTPUT_DIR}/{cfg.TEST.RUN_ID}"
    logging.basicConfig(filename=f'{save_path}/TDNODE_analysis.log', level=logging.INFO)
    torch.manual_seed(cfg.DATA.SEED)
    np.random.seed(cfg.DATA.SEED)
    utils.log_message("Performing model assessment")
    perform_model_assessment(cfg)
