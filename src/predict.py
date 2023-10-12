import torch
import matplotlib.pyplot as plt
import utils
import sys

sys.path.extend(["../TDNODE", "src"])
from src.data.data_prepare import prepare_TGI
from src.constants import title_dict
from model import model_dict
import numpy as np
from prodict import Prodict
import pandas as pd
import utils
from typing import Tuple


def evaluate_model(
    cfg: Prodict, feature_dependence: bool = False
) -> Tuple[str, float, float, Prodict,]:
    """Generates model predictions and calculates numerical performance.

    Parameters
    ----------
    cfg : Prodict
        A configuration file describing the parameters of the evaluation.
    feature_dependence: bool
        Indicator for whether to obtain predictions with systematically perturbed encodings.
    Returns
    -------
    save_path: str
        A relative path that points to a dataframe containing a series of labels, observation times,
        subject IDS, and TDNODE-generated predictions.
    r2: float
        The R2 score measuring the degree of correlation between the labels and predictions.
    rmse: float
        The root-mean-square error (RMSE) score quantifying the average squared residual between
        predictions and labels.
    res.encodings: Prodict
        A dictionary containing a series of patient encodings.
    performance_by_pth: Prodict
        A dictionary containing a series of continuous patient predictions. To be used when
        generating patient-level plots.
    varied_encoding_data: Prodict
        A dictionary containing a series of continuous predictions with systematically perturbed
        parameter encodings.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    results_path = f"{cfg.DATA.OUTPUT_DIR}/{cfg.TEST.RUN_ID}"
    ckpt_path = f"{results_path}/e{cfg.TEST.EPOCH_TO_TEST}/e{cfg.TEST.EPOCH_TO_TEST}.ckpt"
    save_name = f"epoch-{cfg.TEST.EPOCH_TO_TEST}_cutoff-{cfg.DATA.CUTOFF}_cohort-{cfg.DATA.COHORT}_arm-{cfg.DATA.ARM}_evalall-{cfg.TEST.EVAL_ALL}_results.csv"
    save_path = f"{results_path}/e{cfg.TEST.EPOCH_TO_TEST}/tables/{save_name}"
    TGI_dataset = prepare_TGI(
        data_root_path=cfg.DATA.ROOT_DIR,
        augment=cfg.DATA.AUGMENTATION,
        cutoff=cfg.DATA.CUTOFF,
        get_all=cfg.TEST.EVAL_ALL,
        batch_size=1,
    )
    model_dims = utils.load_cfg(f"{results_path}/{cfg.MODEL.PARAMETERS.PARAMETER_PATH}")
    model = model_dict[cfg.MODEL.NAME](model_dims, cfg.MODEL.TOL, device=device)
    utils.load_model(ckpt_path, model, device)
    with torch.no_grad():
        (res, performance_by_pt, varied_encoding_data) = utils.compute_val_loss(
            model,
            cfg,
            TGI_dataset[cfg.DATA.COHORT],
            device=device,
            feature_dependence=feature_dependence,
        )

        print(len(res.times))
        r2 = res.r2
        rmse = res.loss
        df = {"NMID": res.NMID, "times": res.times, "labels": res.labels, "preds": res.preds}
        pd.DataFrame(df).to_csv(save_path, index=False)
    return (
        save_path,
        "%.3f" % r2,
        "%.4f" % rmse,
        res.encodings,
        performance_by_pt,
        varied_encoding_data,
    )


def plot_predictions(cfg: Prodict) -> None:
    """Plots the TDNODE-generated predictions against the observed data.

    Parameters
    ----------
    cfg : Prodict
        A configuration file specifying the parameters of the evaluation.
    """
    (df, _, _, _, _, _) = evaluate_model(cfg)
    results = pd.read_csv(df)
    save_name = f"epoch-{cfg.TEST.EPOCH_TO_TEST}_cumulative_plot_cohort-{cfg.DATA.COHORT}_cutoff-{cfg.DATA.CUTOFF}_arm-{cfg.DATA.ARM}_evalall-{cfg.TEST.EVAL_ALL}.png"
    save_dir = f"{cfg.DATA.OUTPUT_DIR}/{cfg.TEST.RUN_ID}/e{cfg.TEST.EPOCH_TO_TEST}/plots/cumulative"
    utils.make_dir(save_dir)
    save_path = f"{save_dir}/{save_name}"
    x = np.linspace(0, 275, 20)
    fig = plt.figure(figsize=(10, 10))
    fig.patch.set_alpha(0.0)
    plt.rc("font", size=25)
    plt.scatter(results["labels"], results["preds"], s=11, color="#003087")
    plt.plot(x, x, "--", color="black")
    plt.xlabel("SLD Data (mm)")
    plt.ylabel("TDNODE Predictions (mm)")
    plt.xlim(0, 250)
    plt.ylim(0, 250)
    plt.title(f"{title_dict[cfg.DATA.COHORT]} {cfg.TEST.CUMULATIVE.PLOT_TITLE}")
    plt.savefig(save_path)
    plt.close()
