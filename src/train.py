from prodict import Prodict
from args import args
import sys

sys.path.extend(["src", "../TDNODE"])

from src.model import model_dict
from data.data_prepare import prepare_TGI
from data.data_split import split_datasets

from analysis.loss_curve import plot_loss_curves
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
import utils
import logging


def train(cfg: Prodict, save_path: str) -> None:
    """Central training function for TDNODE.

    Parameters
    ----------
    cfg : Prodict
        A configuration file describing the parameters for training TDNODE.
    save_path: str
        The directory name relative to the root directory in which to save model results
    """
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    utils.log_message(f"Using device: {device}")
    cfg.TEST.RUN_ID = save_path

    utils.log_message(f"Loading Model: {cfg.MODEL.NAME}")
    model = model_dict[cfg.MODEL.NAME](
        cfg.MODEL.PARAMETERS,
        cfg.MODEL.TOL,
        device=device
                                        )
    utils.log_message(f"Model loaded.\nParameters: {json.dumps(cfg.MODEL.PARAMETERS, indent=2)}")
    utils.log_message(f"Model Summary: {model}")
    utils.log_message(f"Tolerance: {cfg.MODEL.TOL}")
    utils.log_message(f"Using cached encodings: {model.use_cached_encoding}")
    utils.log_message(f"Using device: {model.device}")

    utils.cache_configs(cfg, save_path)
    utils.log_message(f"CUDA Availabile: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        model.cuda()

    weights = list(list(model.parameters()))

    utils.log_message(f"Using Pretrained Model: {cfg.TRAIN.USE_PRETRAIN}")
    if cfg.TRAIN.USE_PRETRAIN:
        model = utils.from_pretrain(cfg, device, model)

    optimizer = optim.Adam(
        weights, lr=cfg.TRAIN.LR * np.sqrt(cfg.TRAIN.BATCH_SIZE), weight_decay=cfg.TRAIN.L2
    )
    utils.log_message(f'Loaded optimizer. optimizer: Adam, lr: {cfg.TRAIN.LR}, L2: {cfg.TRAIN.L2}')
    
    utils.log_message(f"Loading dataset from {cfg.DATA.ROOT_DIR}. Augmentation: {cfg.DATA.AUGMENTATION}")
    utils.log_message(f"Observation Window: {cfg.DATA.CUTOFF}, Evaluate only unseen data: {not cfg.TEST.EVAL_ALL}")
    utils.log_message(f"Batch size: {cfg.TRAIN.BATCH_SIZE}")
    TGI_dataset = prepare_TGI(
        cfg.DATA.ROOT_DIR,
        cfg.DATA.AUGMENTATION,
        cfg.DATA.CUTOFF,
        cfg.TEST.EVAL_ALL,
        cfg.TRAIN.BATCH_SIZE,
        train_all=cfg.DATA.TRAIN_ALL,
        norm_method=cfg.DATA.NORM_METHOD,
        device=device,
    )
    utils.log_message(f"Dataset loaded: {TGI_dataset}")

    criterion = nn.MSELoss().to(device=device)
    best_rmse = float("inf")
    TGI_train = TGI_dataset.train
    model.train()
    utils.save_model(f"{save_path}/e{-1}", model, -1)
    average_duration = 0
    utils.log_message(f"Using loss criterion: {criterion}")

    epoch_start = 0 if not cfg.TRAIN.USE_PRETRAIN else cfg.MODEL.FROM_PRETRAINED_EPOCH_START + 1
    epoch_end = (
        cfg.TRAIN.NUM_EPOCH
        if not cfg.TRAIN.USE_PRETRAIN
        else cfg.MODEL.FROM_PRETRAINED_EPOCH_START + cfg.TRAIN.NUM_EPOCH
    )
    utils.log_message(f"Using start epoch {epoch_start}, end epoch {epoch_end}")

    for epoch in range(epoch_start, epoch_end):
        start = time.time()
        
        for _iter in tqdm(range(TGI_train.n_batches)[:5]):
            utils.log_message(f"Epoch: {epoch}, Iteration: {_iter}", print_msg=False)
            optimizer.zero_grad()
            (
                id,
                sld,
                times,
                labels,
                _,
                baseline,
                masks,
                pt_time_scale,
                _,
            ) = TGI_train.dataloader.__next__()
            if torch.cuda.is_available():
                preds = model(sld, baseline, times, pt_time_scale=pt_time_scale)
            else:
                preds = model(sld, baseline, times, pt_time_scale=pt_time_scale)
            loss = utils.compute_loss_on_train(
                criterion, labels, preds, TGI_train.mean, TGI_train.std, masks
            )
            utils.log_message(f"Loss: {loss}", print_msg=False)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            train_res, _, _ = utils.compute_val_loss(
                model, cfg, TGI_train, device=device, pt_time_norm=cfg.DATA.NORM_METHOD
            )
            validation_res, _, _ = utils.compute_val_loss(
                model, cfg, TGI_dataset.test, device, pt_time_norm=cfg.DATA.NORM_METHOD
            )

            utils.save_model(f"{save_path}/e{epoch}", model, epoch)
            if validation_res.loss < best_rmse:
                best_rmse = validation_res.loss
                best_epochs = epoch
            end = time.time()
            duration = end - start
            average_duration = (average_duration * (epoch) + (duration)) / (epoch + 1)
            epoch_stats = Prodict(
                epoch=epoch,
                train_loss=train_res.loss,
                train_r2=train_res.r2,
                validation_loss=validation_res.loss,
                validation_r2=validation_res.r2,
                best_rmse=best_rmse,
                best_epochs=best_epochs,
                epoch_duration=duration,
                average_duration=average_duration,
            )

            utils.print_model_performance(epoch_stats)
            utils.save_model_performance(epoch_stats, save_path)

            plot_loss_curves(cfg)


if __name__ == "__main__":

    cfg = utils.load_cfg(args.config)
    torch.manual_seed(cfg.DATA.SEED)
    np.random.seed(cfg.DATA.SEED)
    save_path = utils.prepare_new_model_directory(
        cfg.NAME, cfg.DATA.OUTPUT_DIR, cfg.DATA.CUTOFF
    )

    logging.basicConfig(filename=f'{save_path}/TDNODE_train.log', level=logging.INFO)
    utils.log_message(f'Loaded configuration: {json.dumps(cfg, indent=2)}')
    utils.log_message(f'TDNODE model run directory created: {save_path}')

    split_datasets(cfg.DATA.ROOT_DIR, cfg.DATA.SEED, cfg.DATA.SPLIT)
    train(cfg, save_path)
