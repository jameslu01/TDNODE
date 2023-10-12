from prodict import Prodict

import json
import matplotlib.pyplot as plt
import numpy as np
import sys
import torch
import utils
from typing import Dict, List, Optional
from matplotlib import colors
import matplotlib.lines as mlines
from matplotlib.cm import ScalarMappable

sys.path.extend(["src", "../TDNODE"])

import torch.nn as nn
from src.predict import evaluate_model

nmid_to_subjct = {
    184: 85,
    120: 62,
    679: 77,
    230: 106,
    768: 119,
    191: 90,
    718: 98,
    851: 158,
    575: 32,
    672: 76,
    574: 31,
    437: 190
}

nmid_to_subjct_train = {
    176: 694,
    126: 370,
    207: 308,
    888: 274,
    330: 410,
    18: 829,
    765: 439,
    608: 465,
    812: 834,
    16: 558,
    481: 162,
    162: 193
}

def make_feature_dependence_plots(cfg: Prodict):
    """Plots Encoding feature dependence plots.

    Parameters
    ----------
    cfg : cfg
        A configuration file describing the parameters of the evaluation.
    """
    (_, _, _, _, _, varied_encoding_data) = evaluate_model(cfg, feature_dependence=True)
    cmap_grad = colors.LinearSegmentedColormap.from_list("blue_to_red", ["blue", "red"])

    var_value_sampler = dict()
    for idx, (nmid, values) in enumerate(varied_encoding_data.items()):
        plt.rcParams['axes.facecolor'] = '#D3D3D3'

        var_value_sampler[nmid] = dict()
        print(f"plotting nmid {nmid}")
        fig = plt.figure(figsize=(15, 10))
        fig.patch.set_alpha(0.0)

        plt.rc("font", size=36)

        i = 0
        utils.make_dir(values["plot_path"])
        learned_value = values["learned_value"]
        if cfg.TEST.FEATURE_DEPENDENCE.MULTIVAR:
            learned_value = "".join(
                [f"minus{str(l)[1:]}-" if l < 0 else f"{str(l)}-" for l in learned_value]
            )
        else:
            if learned_value < 0:
                learned_value = f"minus{str(learned_value)[1:]}"
        save_name = f'{values["plot_path"]}/latentvalue-{learned_value}_NMID-{nmid}_var{values["var"]}_cutoff-{cfg.DATA.CUTOFF}_psuedoid-{idx}.png'
        if cfg.TEST.FEATURE_DEPENDENCE.MULTIVAR:
            # TODO: integrate PCA
            save_name = f'{values["plot_path"]}/PCA-{"NA"}_latentvalue-{learned_value}_NMID-{nmid}_var{values["var"]}_dir-{cfg.TEST.FEATURE_DEPENDENCE.MULTIVAR_DIRECTION}_cutoff-{cfg.DATA.CUTOFF}_mag-{cfg.TEST.FEATURE_DEPENDENCE.MULTIVAR_MAGNITUDE}.png'

        for (var, pred) in values["preds"].items():
            var_value_sampler[nmid][var] = {
                "preds": pred["preds"][::4],
                "times": values["pred_times"].tolist()[::4],
                "encoding_used": pred["encoding_used"],
            }
            #print(var, pred["encoding_used"], i)
            if var == "control":
                plt.plot(
                    values["pred_times"].tolist(), pred["preds"], c="black", zorder=3, linewidth=3
                )
            else:
                plt.plot(values["pred_times"].tolist(), pred["preds"], c=cmap_grad(i), zorder=1)

            i += 1 / cfg.TEST.FEATURE_DEPENDENCE.NUMBER_PERTURBED

        json.dump(
            var_value_sampler,
            open(
                f"{values['plot_path']}/var-{cfg.TEST.FEATURE_DEPENDENCE.VARIABLE_IDX}_prediction_mapping.json",
                "w",
            ),
            indent=4,
        )

        point_marker = mlines.Line2D(
            [], [], color="black", marker="o", linestyle="None", markersize=10
        )
        hollow_marker = mlines.Line2D(
            [], [], color="black", marker="o", linestyle="None", markersize=10, fillstyle="none"
        )
        legend = list()
        legend.append("labels")
        times = values["label_times"]
        labels = values["labels"]
        cutoff_idx = np.argmax(times > cfg.DATA.CUTOFF)
        if cutoff_idx == 0:
            cutoff_idx = len(times)

        plt.scatter(
            times[:cutoff_idx],
            labels[:cutoff_idx],
            facecolors="black",
            edgecolors="#003087",
            s=150,
            color="black",
            zorder=2,
            linewidths=2
        )
        plt.scatter(
            times[cutoff_idx:],
            labels[cutoff_idx:],
            facecolors="none",
            edgecolors="#003087",
            s=150,
            color="black",
            zorder=2,
            linewidths=2
        )

        plt.legend([point_marker], ["Observations"])
        plt.xlabel("Time (weeks)")
        plt.ylabel("SLD (mm)")
        # plt.xticks([])
        # plt.yticks([])
        #plt.ylim(19, 40)
        #plt.axvline(x=32, linestyle="--", color = 'black')
        plt.title(f"Subject {nmid_to_subjct_train[nmid]}")
        fig.colorbar(
            ScalarMappable(
                cmap=cmap_grad,
                norm=plt.Normalize(
                    cfg.TEST.FEATURE_DEPENDENCE.RANGE[0], cfg.TEST.FEATURE_DEPENDENCE.RANGE[-1]
                ),
            )
        )
        #fig.axes[-1].tick_params(labelsize=0) # color bar access
        fig.axes[-1].set_ylabel("Feature Change", size=32)

        plt.savefig(save_name)
        plt.close()
    return


def multivar_perturb_encodings(
    encoding: torch.Tensor,
    var_indices: List[int],
    var_range: List[float],
    directions: List[str],
    num: int,
    magnitude: List[int],
) -> Prodict:
    """Generates a set of predictions with systematically perturbed encoding inputs, perturbing one
    or more variables at a time.

    Parameters
    ----------
    encoding : torch.Tensor
        The learned parameter encoding generated by TDNODE.
    var_indices : List[int]
        A list of parameter encoding variables IDs to perturb.
    var_range : List[float]
        A list containing the degrees to which to encodings will be perturbed.
    directions : List[str]
        A list of directions in which to perturb the encodings. directions[i] corresponds to the
        directionality in which encoding[i] is perturbed.
    num : int
        The number of perturbed predictions to generate.
    magnitude : List[int]
        The magnitude of perturbations to amke. magnitude[i] corresponds to the magnitude of
        perturbation for encoding[i].

    Returns
    -------
    Prodict
        A dictionary containing the TDNODE-generated predictions with perturbed encodings.
    """
    perturbed_encodings = Prodict(control=encoding)
    current_vals = [encoding.clone()[var - 1] for var in var_indices]
    for v in np.linspace(var_range[0], var_range[1], num):
        _encoding = encoding.clone()
        for (current_val_idx, var) in enumerate(var_indices):
            _encoding[int(var) - 1] = (
                current_vals[current_val_idx] + v * magnitude[current_val_idx]
                if directions[current_val_idx] == "up"
                else current_vals[current_val_idx] - v * magnitude[current_val_idx]
            )
        perturbed_encodings[v] = _encoding
    return perturbed_encodings


def perturb_encodings(
    encoding: torch.Tensor, var_idx: int, var_range: List[float], num: int
) -> Prodict:
    """Generates a set of predictions with systematically perturbed encoding inputs, but ONLY for
    a single variable.

    Parameters
    ----------
    encoding : torch.Tensor
        The learned parameter encoding generated by TDNODE.
    var_idx : int
        The variable ID (0-indexed) to be perturbed.
    var_range : List[int]
        A length-2 array containing the degree of perturbation to be performed.
    num : int
        The number of perturbed TDNODE-generated predictions to produce.

    Returns
    -------
    perturbed_encodings: Prodict
        A dictionary containing the TDNODE-generated predictions originating from perturbed
        encodings.
    """
    perturbed_encodings = Prodict(control=encoding)
    current_val = encoding.clone()[var_idx]
    for v in np.linspace(var_range[0], var_range[1], num):
        _encoding = encoding.clone()
        _encoding[int(var_idx)] = current_val + v
        perturbed_encodings[v] = _encoding
    return perturbed_encodings


def obtain_varied_encoding_data(
    model: nn.Module,
    cfg: Prodict,
    sld: torch.Tensor,
    baseline: torch.Tensor,
    pt_level_times: torch.Tensor,
    learned_encoding: torch.Tensor,
    plot_path: str,
    pt_time_scale: float,
    mean: float,
    std: float,
    labels: torch.Tensor,
    nmid: int,
) -> Optional[Dict[str, any,]]:
    """Given a trained model, configuration, and learned_encoding, generates patient SLD predictions
    with systematically perturbed encodings.

    Parameters
    ----------
    model : nn.Module
        A trained instantiation of TDNODE.
    cfg : Prodict
        A configuration file describing the parameters of the evaluation.
    sld : torch.Tensor
        A partitioned tensor containing truncated time series post-treatment observations.
    baseline : torch.Tensor
        A tensor containing time one or more series pre-treatment measurements
    pt_level_times : torch.Tensor
        The times that correspond to observed SLD data.
    learned_encoding : torch.Tensor
        The learned parameter encoding that TDNODE produces by default for each patient.
    plot_path : str
        A string that references the location in which the feature dependence plot will be saved.
    pt_time_scale : float
        A single value equivalent to the last observed time of measurement.
    mean : float
        The mean SLD value of the training cohort.
    std : float
        The standard deviation of SLD values in the training cohort.
    labels : torch.Tensor
        A tensor containing the patient's unnormlaized labels.
    nmid : int
        An integer referencing the patient's assigned NMID

    Returns
    -------
    Optional[Dict[str, any]]
        A dictionary containing the necessary components for plotting feature dependence plots.
    """
    print(plot_path)
    if nmid not in cfg.TEST.FEATURE_DEPENDENCE.PATIENT_IDS and not cfg.TEST.FEATURE_DEPENDENCE.GET_ALL_PTS:
        return "Patient not in cfg.TEST.FEATURE_DEPENDENCE.PATIENT_IDS"
    print(f"evaluating patient {nmid}")
    preds_dict = Prodict()
    varied_nmid_path = f"{plot_path}"
    if nmid in cfg.TEST.FEATURE_DEPENDENCE.PATIENT_IDS:
        utils.make_dir(varied_nmid_path)

    # TODO: integrate PCA
    # pca_component = round(float(encodings_df[encodings_df.nmid == nmid]["pca-1"].values[0]), 4)
    if cfg.TEST.FEATURE_DEPENDENCE.MULTIVAR:
        learned_value = [
            round(float(learned_encoding[c - 1]), 3)
            for c in cfg.TEST.FEATURE_DEPENDENCE.MULTIVAR_IDX
        ]
    else:
        learned_value = round(float(learned_encoding[cfg.TEST.FEATURE_DEPENDENCE.VARIABLE_IDX]), 3)
    perturbed_encodings = perturb_encodings(
        learned_encoding,
        cfg.TEST.FEATURE_DEPENDENCE.VARIABLE_IDX,
        cfg.TEST.FEATURE_DEPENDENCE.RANGE,
        cfg.TEST.FEATURE_DEPENDENCE.NUMBER_PERTURBED,
    )
    if cfg.TEST.FEATURE_DEPENDENCE.MULTIVAR:
        perturbed_encodings = multivar_perturb_encodings(
            learned_encoding,
            cfg.TEST.FEATURE_DEPENDENCE.MULTIVAR_IDX,
            cfg.TEST.FEATURE_DEPENDENCE.RANGE,
            cfg.TEST.FEATURE_DEPENDENCE.MULTIVAR_DIRECTION,
            cfg.TEST.FEATURE_DEPENDENCE.NUMBER_PERTURBED,
            cfg.TEST.FEATURE_DEPENDENCE.MULTIVAR_MAGNITUDE,
        )

    for (var, encoding) in perturbed_encodings.items():
        model.set_cached_encoding(encoding.reshape(1, -1))
        pred_times = torch.linspace(pt_level_times[0], pt_level_times[-1], 25)
        #print(sld.shape, baseline.shape, pred_times.shape)
        preds = utils.rescale_metric(
            model(sld, baseline, pred_times, pt_time_scale=pt_time_scale)[0], mean, std
        )
        scaled_pred_times = pred_times * pt_time_scale
        preds_dict[var] = {
            "encoding_used": encoding.reshape(-1, 1).tolist(),
            "preds": preds.tolist(),
        }
    return {
        "pred_times": scaled_pred_times,
        "label_times": pt_level_times * pt_time_scale,
        "preds": preds_dict,
        "labels": labels.tolist(),
        "plot_path": varied_nmid_path,
        "var": cfg.TEST.FEATURE_DEPENDENCE.VARIABLE_IDX
        if not cfg.TEST.FEATURE_DEPENDENCE.MULTIVAR
        else cfg.TEST.FEATURE_DEPENDENCE.MULTIVAR_IDX,
        "learned_value": learned_value,
    }
