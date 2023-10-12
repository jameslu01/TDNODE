import sys
import numpy as np

sys.path.extend(["../TDNODE", "src"])

import torch
from typing import Tuple

import warnings
import constants

warnings.filterwarnings("ignore")


def make_batch(batch: Tuple[Tuple[any]], device: str):
    """Collate function to process multiple patients at a time during training and evaluation.
    Performs the following steps:
    1. Creates a left-padded batched tensor of partitioned SLD data, where the pad value is the first
    observation of the respective subject's post-treatment time series.
    2. Creates a left-padded batched tensor of partitioned baseline data, where the pad value is the
    first observation of the respective subject's pre-treatment time series.
    3. Merges observation times to a single 1D tensor. Creates mask tensor indicating which time
    observations correspond to which patient.
    4. Creates intermittent zero-padded label tensor for correct loss calculation.
    5. Stacks non-tensors of patient IDS, scale factors, and cutoff indices for reference during
    training/evaluation.

    Parameters
    ----------
    batch : Tuple[Tuple[any]]
        A batch of tuples containing preprocessed data for each patient.
    device : str
        A string indicating which device to which to load batched data.

    Returns
    -------
    ids: torch.Tensor
        A stacked tensor of patient IDS. Shape: B x 1, where B is the selected batch size.
    slds: torch.Tensor
        A stacked left-padded tensor of post-treatment truncated paritioned time series data.
        Shape: B x (L_pmax - 1) x 4., where L_pmax is the maximum number of post-treatment
        observations in the batch of patients.
    times: torch.Tensor
        A flattened, sorted tensor of time values with which to perform the evaluation, calculated
        as the union of the time values in each patient time series in the current batch. Shape:
        L_T, where L_T is the number of unique time measurements for all patients in the current
        batch.
    labels: torch.Tensor
        A dynamically zero-padded tensor of label values, where zeros represent positions to not
        evaluate during optimization/evaluation. Shape: B x L_T.
    cuts: torch.Tensor
        A stacked tensor of patient cutoff indices to be used during evaluation. Shape: B x 1.
    baselines: torch Tensor
        A stacked, left-padded tensor of pre-treatment time series data. Shape: L_b x 2, where L_b
        is the number of pre-treatment observations for a given patient.
    masks: torch.Tensor
        A dynamically allocated binary tensor, where zeros represent positions to not evaluate
        during evaluation/optimization. Shape: B x L_T.
    scale_factors: torch.Tensor
        A stacked tensor of patient scale factors, used to scale observation times. Shape: B x 1.
    indices: np.ndarray
        A an array of index tensors, used to parse for labels, predictions, and times of a given
        patient. Shape: B x <any>.

    """
    sld_max_dim = max([batch[i][1].shape[0] for i in range(len(batch))])
    baseline_max_dim = max([batch[i][5].shape[0] for i in range(len(batch))])

    ids = list()
    slds = list()
    times = list()
    labels = list()
    cuts = list()
    baselines = list()
    scale_factors = list()
    indices = list()

    masks = list()
    times = torch.cat([batch[i][2] for i in range(len(batch))]).flatten()
    (times, _) = torch.sort(torch.unique(times[times != constants.TIME_PAD_VALUE]))

    for sample in batch:

        # sld
        sld_pad = sld_max_dim - sample[1].shape[0]
        padded_sld = torch.stack([sample[1][0]] * sld_max_dim)
        padded_sld[sld_pad:] = sample[1]
        slds.append(padded_sld)

        # baseline
        baseline_pad = baseline_max_dim - sample[5].shape[0]
        padded_baseline = torch.stack([sample[5][0]] * baseline_max_dim)
        padded_baseline[baseline_pad:] = sample[5]
        baselines.append(padded_baseline)

        # labels and indices
        sample_indices = np.where(np.in1d(times, sample[2]))[0]
        (label, mask) = torch.zeros(times.shape), torch.zeros(times.shape)
        label[sample_indices] = sample[3]
        mask[sample_indices] = torch.ones(sample[3].shape[0])
        labels.append(label)
        masks.append(mask)

        # non tensors
        scale_factors.append(torch.Tensor([sample[-1]]))
        ids.append(torch.Tensor([sample[0]]))
        cuts.append(torch.LongTensor([sample[4]]))
        indices.append([torch.LongTensor(sample_indices).numpy()])

    ids = torch.stack(ids)
    slds = torch.stack(slds).to(device)  # B, L, 4
    labels = torch.stack(labels).to(device)  # B T, times is shape T
    cuts = torch.stack(cuts)
    baselines = torch.stack(baselines).to(device)  # B, L, 2
    masks = torch.stack(masks).to(device)
    scale_factors = torch.stack(scale_factors).to(device)
    indices = np.array(indices)
    return (ids, slds, times, labels, cuts, baselines, masks, scale_factors, indices)
