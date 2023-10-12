import sys
sys.path.extend(["../TDNODE", "src"])
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score
import os
from datetime import datetime
import pytz
import pandas as pd
from prodict import Prodict
from yaml import safe_load, YAMLError
import json
from typing import Tuple
import shutil

import logging


def compute_loss_on_train(
    criterion: nn.Module,
    labels: torch.Tensor,
    preds: torch.Tensor,
    mean: float,
    std: float,
    mask: torch.Tensor,
):
    """Computes the loss with a provided batch of predictions and labels.

    Parameters
    ----------
    criterion : nn.Module
        The loss criterion (ex: MSELoss) used to optimize TDNODE.
    labels : torch.Tensor
        A tensor of labels against which to compare predictions. Shape: B x T, where B is the batch
        size and T is the number of time points with corresponding predictions.
    preds : torch.Tensor
        A tensor of TDNODE-generated predictions.
    mean : float
        The arithmetic mean of SLD measurements in the training cohort.
    std : float
        The standard deviation of SLD measurements in the training cohort.
    mask : torch.Tensor
        A binary tensor indicating which positions in the predictions/labels tensors to evaluate.
        Shape: B x T.

    Returns
    -------
    torch.Tensor
        The computed loss value. Contains gradients used for backpropagation.
    """
    preds = torch.add(torch.mul(preds, std.SLD), mean.SLD) * mask
    return torch.sqrt(criterion(preds, labels))


def inf_generator(iterable: any):
    """Creates an iterator object that yields a single element of the input object until all
    elements have been fetched.

    Parameters
    ----------
    iterable : any
        An iterable object, such as a list, dict, or set, that is composed of an arbitrary set of
        elements.

    Yields
    ------
    Any
        A single element originating from iterable.
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


def init_network_weights(net: nn.Module, std: float = 0.1):
    """Randomizes weights of an instantiation of TDNODE with 0-mean and adjustable variance.

    Parameters
    ----------
    net : nn.Module
        An instantiation of TDNODE.
    std : float, optional
        The standard deviation of the distribution to sample for each weight in the input net, by
        default 0.1.
    """
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=std)
            nn.init.constant_(m.bias, val=0)


def compute_val_loss(
    model: nn.Module,
    cfg: Prodict,
    dataset: Prodict,
    device=torch.device("cpu"),
    pt_time_norm: str = "patient",
    feature_dependence: bool = False,
) -> Tuple[Prodict, Prodict, Prodict]:
    """Main function for evaluation. Computes validation loss for TDNODE, obtains learned parameter
    encodings, and optionally fetches TDNODE-generated predictions with systematically perturbed
    encodings.

    Parameters
    ----------
    model : nn.Module
        An instantiation of TDNODE.
    cfg : Prodict
        A configruation file describing the parameters of the evaluation.
    dataset : Prodict
        A dictionary containing the preprocessed training and test sets with accompanying metadata.
    device : str, optional
        A string stating the device in which to process this function, by default
        torch.device("cpu")
    pt_time_norm : str, optional
        The time-normalization method to utilize, by default "patient"
    feature_dependence : bool, optional
        Indicator for whether to generate predictions with systematically perturbed encodings, by
        default False

    Returns
    -------
    Prodict
        A dictionary containing the cumulative numerical performance of TDNODE.
    performance_by_pt: Prodict
        A dictionary containing near-continuous time series predictions. Used for comparing
        predicted dynamics with observed SLD data.
    varied_encoding_data: Prodict
        A dictionary containing time series predictions with systematically perturbed encodings.
        Used for observing how changes in learned parameter encodings impact tumor dynamic
        predictions.
    """
    import analysis.vary_encodings as fd

    ptnums = list()
    _times = torch.Tensor([]).to(device=device)
    predictions = torch.Tensor([]).to(device=device)
    ground_truth = torch.Tensor([]).to(device=device)
    encoding_data = Prodict(nmid=list(), latent_encoding=list(), IC_encoding=list())
    performance_by_pt = Prodict()
    varied_encoding_data = dict()
    for _ in range(dataset.n_batches):

        (
            id,
            sld,
            times,
            labels,
            cutoff_idx,
            baseline,
            mask,
            pt_time_scale,
            indices,
        ) = remove_augmentation(dataset.dataloader.__next__())
        if id.shape[0] == 0:  # all pts augmented
            continue
        if torch.cuda.is_available():
            preds = model(sld.cuda(), baseline.cuda(), times.cuda(), pt_time_scale=pt_time_scale)
        else:
            preds = model(sld, baseline, times, pt_time_scale=pt_time_scale)

        pt_level_times = torch.linspace(times[0], times[-1], 250).reshape(1, -1).squeeze()
        pt_level_preds = model(sld, baseline, pt_level_times, pt_time_scale=pt_time_scale)

        for idx in range(len(id)):

            sample_times = times[indices[idx]][cutoff_idx[idx] :]
            if pt_time_norm == "patient":
                sample_times = sample_times * pt_time_scale[idx]
            _times = torch.cat((_times, sample_times.to(device)))
            sample_pred = torch.mul(preds[idx], mask[idx].to(device))
            predictions = torch.cat(
                (predictions, sample_pred[indices[idx]][cutoff_idx[idx] :].to(device)), dim=0
            )

            ground_truth = torch.cat(
                (ground_truth, labels[idx][indices[idx]][cutoff_idx[idx] :].to(device)), dim=0
            )

            ptnums.extend([int(id[idx])] * sample_times.shape[0])
            encoding_data.nmid.append(int(id[idx]))
            encoding_data.latent_encoding.append(model.encoding[idx].squeeze().cpu().numpy())
            encoding_data.IC_encoding.append(model.IC_encoding[idx].squeeze().cpu().numpy())

            performance_by_pt[int(id[idx])] = Prodict(
                preds=rescale_metric(
                    pt_level_preds[idx], dataset.mean.SLD, dataset.std.SLD
                ).tolist(),
                pred_times=pt_level_times * pt_time_scale[idx],
                labels=labels[idx].tolist(),
                label_times=times * pt_time_scale[idx],
            )

            if feature_dependence:
                model.use_cached_encoding = True
                param_encoding = model.encoding[idx]
                plot_path = f"{cfg.DATA.OUTPUT_DIR}/{cfg.TEST.RUN_ID}/e{cfg.TEST.EPOCH_TO_TEST}/plots/varied_encodings_cohort-{cfg.DATA.COHORT}_var-{cfg.TEST.FEATURE_DEPENDENCE.VARIABLE_IDX}-range-{cfg.TEST.FEATURE_DEPENDENCE.RANGE}_F4_20230315"
                if cfg.TEST.FEATURE_DEPENDENCE.MULTIVAR:
                    plot_path = f"{cfg.DATA.OUTPUT_DIR}/{cfg.TEST.RUN_ID}/e{cfg.TEST.EPOCH_TO_TEST}/plots/varied_encodings_cohort-{cfg.DATA.COHORT}_var-{cfg.TEST.FEATURE_DEPENDENCE.MULTIVAR_IDX}_dir-{cfg.TEST.FEATURE_DEPENDENCE.MULTIVAR_DIRECTION}-range-{cfg.TEST.FEATURE_DEPENDENCE.RANGE}_mag-{cfg.TEST.FEATURE_DEPENDENCE.MULTIVAR_MAGNITUDE}20230404_train"
                pt_encoding_data = fd.obtain_varied_encoding_data(
                    model,
                    cfg,
                    torch.unsqueeze(sld[idx], 0),
                    torch.unsqueeze(baseline[idx], 0),
                    times[indices[idx]],
                    param_encoding,
                    plot_path,
                    pt_time_scale[idx],
                    dataset.mean.SLD,
                    dataset.std.SLD,
                    labels[idx],
                    int(id[idx]),
                )
                if pt_encoding_data != "Patient not in cfg.TEST.FEATURE_DEPENDENCE.PATIENT_IDS":
                    varied_encoding_data[int(id[idx])] = pt_encoding_data
                model.use_cached_encoding = False

    if pt_time_norm == "cohort":
        _times = _times * dataset.std.TIME + dataset.mean.TIME
    predictions = predictions.squeeze() * dataset.std.SLD + dataset.mean.SLD
    ground_truth = ground_truth.squeeze()

    try:
        rmse_loss = mean_squared_error(
            ground_truth.cpu().numpy(), predictions.cpu().numpy(), squared=False
        )
        r2 = r2_score(ground_truth.cpu().numpy(), predictions.cpu().numpy())
    except ValueError:
        print(
            "No labels to predict after observation window. Consider run failed unless collecting encodings."
        )
        rmse_loss = 0
        r2 = 0
    return (
        Prodict(
            NMID=[int(id) for id in ptnums],
            times=_times.cpu().tolist(),
            labels=ground_truth.cpu().tolist(),
            preds=predictions.cpu().tolist(),
            loss=rmse_loss,
            r2=r2,
            encodings=encoding_data,
        ),
        performance_by_pt,
        varied_encoding_data,
    )


def rescale_metric(arr: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    """Restores SLD measurements to unnormalized values.

    Parameters
    ----------
    arr : torch.Tensor
        A tensor of Z-score normalized SLD values.
    mean : float
        The cohort mean of the training set of SLD data used to train TDNODE.
    std : float
        The cohort standard deviation of the training set of SLD data used to train TDNODE.

    Returns
    -------
    torch.Tensor
        A tensor of unnormalized SLD values.
    """
    return arr * std + mean


def remove_augmentation(batch: Tuple) -> Tuple:
    """Remove augmented subjects from a batch of subjects.

    Parameters
    ----------
    batch : Tuple
        A batch of subjects generated by the TGI Dataloader.

    Returns
    -------
    Tuple
        A batch of subjects with augmented subjects removed.
    """
    (indices, _) = torch.where(torch.remainder(batch[0], 1) == 0)
    batch = tuple([batch[b][indices] if b != 2 else batch[b] for b in range(len(list(batch)))])
    return batch


def load_model(ckpt_path: str, model=None, device="cpu") -> None:
    """Loads an instantiation of TDNODE to memory. Handles excpetion in which file is not found.

    Parameters
    ----------
    ckpt_path : str
        A relative reference to the weights to be leaded into a TDNODE instantiation.
    model : nn.Module, optional
        An instantiation of TDNODE, by default None.
    device : str, optional
        The device to which to load the model, by default "cpu".

    Raises
    ------
    Exception
        Raised when ckpt path is not found or does not exist.
    """
    if not os.path.exists(ckpt_path):
        raise Exception("Checkpoint " + ckpt_path + " does not exist.")

    checkpt = torch.load(ckpt_path)

    if model is not None:
        model_state = checkpt["model"]
        model.load_state_dict(model_state)
        model.to(device)
    return


def save_model(ckpt_path: str, model: nn.Module, epoch: int) -> None:
    """Saves TDNODE to a specified directory.

    Parameters
    ----------
    ckpt_path : str
        The path to the ckpt directory to be created, created relative to root dir of repository.
    model : nn.Module
        An instantiation of TDNODE to be saved.
    epoch : int
        The epoch number of the training run.
    """
    make_dir(ckpt_path)
    save_path = f"{ckpt_path}/e{epoch}.ckpt"
    torch.save({"model": model.state_dict()}, f"{ckpt_path}/e{epoch}.ckpt")
    log_message(f"Model saved: {save_path}")


def prepare_new_model_directory(name: str, results_path: str, cutoff: int) -> str:
    """Prepares a new model directory in which model checkpoints and analyses will be stored.

    Parameters
    ----------
    name : str
        The name of the directory that will store model checkpoints/analyses.
    results_path : str
        The parent directory of the new model run path.
    cutoff : int
        The observation window of the current model run.

    Returns
    -------
    save_path: str
        The path to the new model directory, relative to the root directory of the repository.
    """
    make_dir(results_path)
    time = datetime.now(pytz.UTC).strftime("%Y-%m-%d %H:%M:%S")
    save_path = f"{results_path}/{time}_{name}_cutoff-{cutoff}"
    make_dir(save_path)
    return save_path


def print_model_performance(epoch_stats: Prodict) -> None:
    """Prints model performance stdout.

    Parameters
    ----------
    epoch_stats: Prodict
        A dictionary containing the numerical performance metrics for the given epoch.
    """
    message = """
            Epoch {:04d} | Training loss {:.6f} | Training R2 {:.6f} | Validation loss {:.6f} | Validation R2 {:.6f}
            Best loss {:.6f} | Best epoch {:04d}
            """.format(
        epoch_stats.epoch,
        epoch_stats.train_loss,
        epoch_stats.train_r2,
        epoch_stats.validation_loss,
        epoch_stats.validation_r2,
        epoch_stats.best_rmse,
        epoch_stats.best_epochs,
    )
    log_message(message)


def save_model_performance(epoch_stats: Prodict, save_path: str) -> None:
    """Saves model performance metrics in a .csv file, which contain the performance data for each
    epoch.

    Parameters
    ----------
    epoch_stats : Prodict
        A dictionary containing the numerical performance metrics for the given epoch.
    save_path : str
        A path referencing the location in which to save the .csv file.
    """
    res_df = pd.DataFrame(
        {
            "Epoch": int(epoch_stats.epoch),
            "Training Loss": [epoch_stats.train_loss],
            "Training R2": [epoch_stats.train_r2],
            "Validation Loss": [epoch_stats.validation_loss],
            "Validation R2": [epoch_stats.validation_r2],
            "Best Loss": [epoch_stats.best_rmse],
            "Best Epoch": [epoch_stats.best_epochs],
            "Epoch Duration": [epoch_stats.epoch_duration],
            "Average Epoch Duration": [epoch_stats.average_duration],
        }
    )
    if epoch_stats.epoch != 0:
        metrics = pd.read_csv(f"{save_path}/metrics.csv", sep="\t")
        metrics = metrics.append(res_df, ignore_index=True)
    else:
        metrics = res_df
    metrics.to_csv(f"{save_path}/metrics.csv", index=False, sep="\t")


def make_dir(path: str) -> None:
    """Creates a new directory. Exhibits silent behavior when the specified path already exists.

    Parameters
    ----------
    path : str
        The path referencing the new directory to create, if it does not already exist.
    """
    if not os.path.exists(path):
        print(path)
        os.mkdir(path)


def load_cfg(cfg: str) -> Prodict:
    """Loads a .yaml configuration file as a dictionary.

    Parameters
    ----------
    cfg : str
        A path referencing the .yaml file to load into memory.

    Returns
    -------
    Prodict
        A dictionary containing the contents of the .yaml configuration file.
    """
    with open(cfg, "r") as stream:
        try:
            _C = safe_load(stream)
        except YAMLError as e:
            print(e)
    return Prodict.from_dict(_C)


def cache_configs(cfg: Prodict, save_path: str) -> None:
    """Caches hyperparameter and model configurations at the beginning of a training run.

    Parameters
    ----------
    cfg : Prodict
        A configuration file describing the parameters of training/evaluation.
    save_path : str
        A path referencing the directory in which the configurations will be saved.
    """
    json.dump(cfg.MODEL.PARAMETERS, open(f"{save_path}/model_config.json", "w"), indent=2)
    json.dump(cfg, open(f"{save_path}/run_config.json", "w"), indent=2)
    return


def from_pretrain(cfg: Prodict, device: str, save_path: str, model: nn.Module) -> nn.Module:
    """Loads an instantiation of TDNODE from a previous training run. Migrates existing
    performance metrics, plots, and configurations

    Parameters
    ----------
    cfg : Prodict
        A configuration file describing the parameters for training TDNODE.
    device : str
        A string that describes the device to which to load the model.
    save_path: str
        A directory in which all results from further optimization of the pretrained model will be
        saved.
    model: str
        An instantiated pytorch model

    Returns
    -------
    nn.Module
        An instantiation of TDNODE with the pre-trained weights.
    """
    print("Loading pre-trained model.")
    model_dims = load_cfg(f"{cfg.DATA.OUTPUT_DIR}/{cfg.MODEL.FROM_PRETRAINED_ID}/model_config.json")
    model = model(model_dims, cfg.MODEL.TOL, device=device)
    load_model(
        f"{cfg.DATA.OUTPUT_DIR}/{cfg.MODEL.FROM_PRETRAINED_ID}/e{cfg.MODEL.FROM_PRETRAINED_EPOCH_START}/e{cfg.MODEL.FROM_PRETRAINED_EPOCH_START}.ckpt",
        model,
        device,
    )
    shutil.copy(
        f"{cfg.DATA.OUTPUT_DIR}/{cfg.MODEL.FROM_PRETRAINED_ID}/metrics.csv",
        f"{save_path}/metrics.csv",
    )
    shutil.copy(
        f"{cfg.DATA.OUTPUT_DIR}/{cfg.MODEL.FROM_PRETRAINED_ID}/run_config.json",
        f"{save_path}/run_config.json",
    )
    shutil.copy(
        f"{cfg.DATA.OUTPUT_DIR}/{cfg.MODEL.FROM_PRETRAINED_ID}/model_config.json",
        f"{save_path}/model_config.json",
    )
    return model

def log_message(msg: str, print_msg: bool = True) -> None:
    """_summary_

    Parameters
    ----------
    msg : str
        Message to log
    print_msg : bool, optional
        indicator for whether to print message to console, by default True
    """
    logging.info(msg)
    if print_msg:
        print(f'{msg}\n')
