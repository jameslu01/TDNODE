import pandas as pd
from prodict import Prodict
import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.append("src")
import utils
from args import args
from src.constants import title_dict
from predict import evaluate_model
import statsmodels.api as sm
from typing import Tuple


def generate_log_pred_plots(cfg: Prodict) -> None:
    """Generates a log-residual plot comparing the residuals between TDNODE-generated predictions
    and observed data with respect to the observed time of measurement.

    Parameters
    ----------
    cfg : _type_
        A configuration file specifying the parameters of the evaluation.
    """
    print("Generating log-residual plot.")
    table_dir = f"{cfg.DATA.OUTPUT_DIR}/{cfg.TEST.RUN_ID}/e{cfg.TEST.EPOCH_TO_TEST}/tables/residual"
    utils.make_dir(table_dir)
    res_table_path = f"{table_dir}/epoch-{cfg.TEST.EPOCH_TO_TEST}_cutoff-{cfg.DATA.CUTOFF}_cohort-{cfg.DATA.COHORT}_residuals.csv"
    plot_dir = (
        f"{cfg.DATA.OUTPUT_DIR}/{cfg.TEST.RUN_ID}/e{cfg.TEST.EPOCH_TO_TEST}/plots/log_residual"
    )
    utils.make_dir(plot_dir)
    res_plot_path = f"{plot_dir}/epoch-{cfg.TEST.EPOCH_TO_TEST}_cutoff-{cfg.DATA.CUTOFF}_cohort-{cfg.DATA.COHORT}_arm-{cfg.DATA.ARM}_residuals.png"

    cfg.TEST.EVAL_ALL = True
    (df, _, _, _, _, _) = evaluate_model(cfg)

    results = pd.read_csv(df)
    residual = np.log(results.labels) - np.log(results.preds)
    results["residual"] = residual

    residual[~np.isfinite(residual)] = 0
    xvals = np.linspace(0, 150, 150)
    y_lowess = sm.nonparametric.lowess(
        exog=results.times.values, endog=residual.values, xvals=xvals, frac=0.2
    )

    (y_lower, y_upper) = get_lowess_ci(
        results.times.values,
        residual.values,
        xnew=xvals,
        ci=cfg.TEST.LOG_RESIDUAL.CONFIDENCE_INTERVAL,
    )
    plot_log_residual(cfg, results, y_lowess, y_lower, y_upper, xvals, res_plot_path)

    results.sort_values(by=["residual"]).to_csv(res_table_path, index=False)


def get_lowess_ci(
    times: np.ndarray,
    residual: np.ndarray,
    xnew: np.ndarray = np.linspace(0, 150, 150),
    num_samples: int = 200,
    ci: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray,]:
    """Generates a confidence interval of LOWESS curves via bootstrapping.

    Parameters
    ----------
    times : np.ndarray
        The array of observation times.
    residual : np.ndarray
        The array of residuals between TDNODE-generated predictions and observed data.
    xnew : np.ndarray, optional
        The x-axis against which the bootstrapped LOWESS curves will be plotted, by default
        np.linspace(0, 150, 150)
    num_samples : int, optional
        The number of bootstrapping operations performed, by default 200
    ci : float, optional
        The desired confidence interval, by default 0.95

    Returns
    -------
    y_lower: np.ndarray
        The `100 * (1 - ci / 2)`'th percentile of LOWESS curves generated during the bootstrapping
        operation.
    y_upper: np.ndarray
        The `100 * (1 - (1 - ci) / 2)`'th percentile of LOWESS curves generated during the
        bootstrapping operation.
    """
    lowess_bootstrap_y = np.zeros((num_samples, len(xnew)))
    for sample in range(num_samples):
        idx = np.random.choice(len(times), len(times), replace=True)
        sample_x = times[idx]
        sample_y = residual[idx]
        lowess_bootstrap_y[sample] = sm.nonparametric.lowess(
            exog=sample_x, endog=sample_y, xvals=xnew, frac=0.2
        )
    sorted_vals = np.sort(lowess_bootstrap_y, axis=0)
    bound = int(num_samples * (1 - ci) / 2)
    y_lower = sorted_vals[bound - 1]
    y_upper = sorted_vals[-bound]
    return (y_lower, y_upper)


def plot_log_residual(
    cfg: Prodict,
    results: pd.DataFrame,
    y_lowess: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
    xvals: np.ndarray,
    res_plot_path: str,
) -> None:
    """Plots the log-residual curve with LOWESS smoothing and confidence interval estimation.

    Parameters
    ----------
    cfg : Prodict
        A configuration file specifying the parameters of the evaluation.
    results : pd.DataFrame
        A Dataframe containing the residual and its corresponding observation time.
    y_lowess : np.ndarray
        An array containing the LOWESS smoothed curve that captures trends in the residual.
    y_lower : np.ndarray
        The `100 * (1 - ci / 2)`'th percentile of LOWESS curves generated during the
        bootstrapping operation.
    y_upper : np.ndarray
        The `100 * (1 - (1 - ci) / 2)`'th percentile of LOWESS curves generated during the
        bootstrapping operation.
    res_plot_path : str
        A relative path specifying the location to save the plot.
    """
    fig = plt.figure(figsize=(10, 10))
    fig.patch.set_alpha(0.0)
    results_observed = results[results.times <= cfg.DATA.CUTOFF]
    results_unseen = results[results.times > cfg.DATA.CUTOFF]
    plt.rc("font", size=25)
    plt.plot(np.linspace(0, 150, 2), np.zeros((2)), "--", color="black")
    l = plt.plot(xvals, y_lowess, c="#8B4000", linewidth=5, label="LOWESS")
    f = plt.fill_between(
        xvals,
        y_lower,
        y_upper,
        alpha=0.4,
        color="#8B4000",
        label=f"{int(cfg.TEST.LOG_RESIDUAL.CONFIDENCE_INTERVAL*100)}% CI",
    )

    plt.xlabel("Time (weeks)")
    plt.title(f"{title_dict[cfg.DATA.COHORT]} {cfg.TEST.CUMULATIVE.PLOT_TITLE}")
    plt.ylabel("log(SLD) - log(PRED)")
    plt.ylim(-2, 2)
    plt.xlim(-5, 140)

    o = plt.scatter(
        results_observed.times if cfg.DATA.COHORT == "test" else results.times,
        results_observed.residual if cfg.DATA.COHORT == "test" else results.residual,
        s=11,
        c=cfg.TEST.LOG_RESIDUAL.COLOR,
        label="Observed",
    )
    if cfg.DATA.COHORT == "test":
        u = plt.scatter(results_unseen.times, results_unseen.residual, c="r", s=11, label="Unseen")
        (handles, labels) = plt.gca().get_legend_handles_labels()
        order = [2, 3, 0, 1]
        plt.legend(
            [handles[idx] for idx in order], [labels[idx] for idx in order], loc="upper right"
        )
    else:
        (handles, labels) = plt.gca().get_legend_handles_labels()
        order = [2, 0, 1]
        plt.legend(
            [handles[idx] for idx in order], [labels[idx] for idx in order], loc="upper right"
        )

    plt.tight_layout()
    plt.savefig(res_plot_path)
    plt.close()
