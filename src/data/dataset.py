import sys
import numpy as np

sys.path.extend(["../TDNODE", "src"])
from pandas.core.frame import DataFrame

import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import List, Tuple
import json

from prodict import Prodict
from utils import log_message


class TGIData(Dataset):
    def __init__(
        self,
        data_path: str,
        cutoff: int = 32,
        cohort: str = "train",
        augment: bool = False,
        train_mean: DataFrame = pd.DataFrame({}),
        train_std: DataFrame = pd.DataFrame({}),
        norm_method: str = "patient",
        get_all: bool = False,
        train_all: bool = False,
        enrollment_cuts: bool = True
    ):
        """A torch Dataset class that prepares the training and test sets for training/evaluation.

        Parameters
        ----------
        data_path : str
            A relative path to the dataset to be preprocessed.
        cutoff : int, optional
            An integer specifying the observation window to use during preprocessing, by default 32.
        cohort : str, optional
            The cohort that is being preprocessed, by default "train"
        augment : bool, optional
            Indicator for whether to perform augmentation on the dataset, by default False.
        train_mean : DataFrame, optional
            The mean SLD and TIME values of the training dataset, by default pd.DataFrame({})
        train_std : DataFrame, optional
            The SLD and TIME standard deviations of the training set, by default pd.DataFrame({}).
        norm_method : str, optional
            The normalization method to apply to TIME values, by default "patient"
        get_all : bool, optional
            Indicator for whether to evaluate all patients during evaluation, by default False
        train_all : bool, optional
            Augmentation indicator for whether to perform subsampling on observations after
            `cutoff`, by default False
        """
        self.raw_data = pd.read_csv(data_path)
        self.cohort = cohort
        self.augment = augment
        self.norm_dict = {"cohort": self.normalize, "patient": self.normalize_by_patient}
        self.norm_method = norm_method
        self.get_all = get_all
        self.train_all = train_all
        self.cohort = cohort
        (self.SLD_mean, self.SLD_std) = (train_mean, train_std)
        if train_mean.empty:
            (self.SLD_mean, self.SLD_std) = self.collect_stats(self.raw_data)
        self.data = self.prepare_data(self.raw_data.copy(), cutoff)

    def __len__(self) -> int:
        """Returns the size of the dataset (i.e. the number of subjects)

        Returns
        -------
        int
            The number of subjects in the dataset.
        """
        return len(self.data)

    def __getitem__(self, index: int):
        """Obtains preprocessed data from a single subject.

        Parameters
        ----------
        index : int
            An integer specifying the ID of the subject to fetch.

        Returns
        -------
        torch.Tensor
            The NMID of the subject.
        torch.Tensor
            A partitioned tensor of time series data describing post-treatment measurements up to
            `cutoff`. Shape: (L_p - 1) x 4, where L_p is the number of post-treatment measurements.
        torch.Tensor
            A tensor containing the normalized observation times for the patient. Length: L, where L
            is the number of measurements for the patient.
        torch.Tensor
            A tensor containing the unnormalized labels for the patient. Length: L.
        torch.Tensor
            A tensor containing the cutoff index for the patient. Indices less than the cutoff index
            reference observed post-treatment times, whereas indices greater than or equal to the
            cutoff index reference unseen post-treatment times. `get_all` arbitrarily sets this
            value to 0 when True.
        torch.Tensor
            A tensor of pre-treatment time series data. Shape, L_b x 2, where L_b is the number of
            pre-treatment measurements for the subject.
        torch.Tensor
            A tensor containing the scale factor to use for the patient. Used to scale between
            normalized and unnormalized times. Only active when `norm_method` is set to 'patient'.
        """
        return (
            self.data[index].id,
            self.data[index].sld,
            self.data[index].times,
            self.data[index].labels,
            self.data[index].cut_idx if not self.get_all else 0,
            self.data[index].baseline,
            self.data[index].pt_time_scale,
        )

    def normalize_by_patient(
        self, data: DataFrame, SLD_mean: Prodict, SLD_std: Prodict, pt_scale_factor: int = 0
    ) -> DataFrame:
        """Normalizes patient SLD data using the mean and standard deviation of the cohort, and
        normalizes patient TIME data using the last observed measurement.

        Parameters
        ----------
        data : DataFrame
            A DataFrame containing time series data for a single patient.
        SLD_mean : Prodict
            A dictionary containing the mean SLD and TIME (not used) values of the training cohort.
        SLD_std : Prodict
            A dictionary containing the SLD and TIME (not use) standard deviations of the training
            cohort.
        pt_scale_factor : int, optional
            The last observed measurement for a single patient, by default 0

        Returns
        -------
        DataFrame
            A DataFrame containing the normalized SLD and TIME data for a single patient.
        """
        data.TIME = data.TIME / pt_scale_factor
        data.SLD = (data.SLD - SLD_mean["SLD"]) / SLD_std["SLD"]
        return data

    def prepare_data(
        self, data: DataFrame, cutoff: int = 32, normalize: bool = True
    ) -> List[Prodict]:
        """Prepares TGIData for training/evaluation. Optionally handles augmentation, varying
        `cutoff` selection, and normalization method

        Parameters
        ----------
        data : DataFrame
            A DataFrame containing the time series measurements for all patients in the cohort.
        cutoff : int, optional
            An integer specifying the desired observation window of the study, by default 32.
        normalize : bool, optional
            Indicator for whether to normalize data, by default True.

        Returns
        -------
        List[Prodict]
            A list of dictionaries such that each dictionary contains the preprocessed data for a
            single patient.
        """
        TGI_dataset = list()
        log_message(f"Num Pts for {self.cohort} cohort: {len(data.NMID.unique())}")
        exlcuded = list()
        for ptid in data.NMID.unique():

            # get the pt time series data
            pt_data = data[data.NMID == ptid].reset_index(drop=True)

            # exclude any patients with less than two measurements
            if len(pt_data) <= 1:
                exlcuded.append(ptid)
                continue

            # get the index such that times referenced at lower indices are pre-treatment
            baseline_idx = pt_data.TIME.gt(0).idxmax()
            pre_tx_exists = True
            if baseline_idx == 0:
                baseline_idx = 1
                pre_tx_exists = False  # indicator that patient had no pre-treatment measurements

            # get the index such that times referenced at lower indices are within this fixed value
            cutoff_idx = pt_data.TIME.gt(cutoff).idxmax()
                       
            # if cutoff_idx = 0, use all measurements 
            if cutoff_idx == 0:
                cutoff_idx = len(pt_data)

            # instantiate the list that will contain the dicts of pt data
            samples = list()

            # Augmentation requires that the below operation happens with the above cutoffs. However, we adjust cutoff_idx first then perform
            for cut in range(
                2 if self.augment else cutoff_idx,
                cutoff_idx + 1 if not self.train_all else len(pt_data) + 1,
            ):

                # set the first time equal to 0, and the first SLD measurement to be the last pre-treatment measurement
                sld_start = pd.DataFrame(
                    {"NMID": [ptid], "TIME": [0], "SLD": pt_data.SLD.values[baseline_idx - 1]}
                )

                # concatenate all observed measurements to the starting measurement
                sld_obs = pd.concat(
                    [sld_start, pt_data[baseline_idx:cut].copy(deep=True)], ignore_index=True
                )

                # get the last observation time. This is the patient-dependent scaling
                x_obs = sld_obs.TIME.values[-1]
                assert x_obs == np.max(sld_obs.TIME.values)

                # skip this patient (or augmented variant) if this time is pre-treatment
                if x_obs <= 0:
                    continue

                # get the unseen time series
                sld_extrapolate = pt_data[cut:].copy(deep=True)

                # collect the unnormalized SLD labels to be used for evaluation
                labels = torch.Tensor(
                    np.concatenate([sld_obs.SLD.values, sld_extrapolate.SLD.values])
                )

                # perform normalization using the scale factor for time and the cohort mean and std for SLD measurements
                if normalize:
                    norm_pt_data = self.norm_dict[self.norm_method](
                        pt_data.copy(deep=True), self.SLD_mean, self.SLD_std, pt_scale_factor=x_obs
                    )
                    norm_sld_obs = self.norm_dict[self.norm_method](
                        sld_obs.copy(deep=True), self.SLD_mean, self.SLD_std, pt_scale_factor=x_obs
                    )
                    norm_sld_extrapolate = self.norm_dict[self.norm_method](
                        sld_extrapolate.copy(deep=True),
                        self.SLD_mean,
                        self.SLD_std,
                        pt_scale_factor=x_obs,
                    )

                # collect the normalized times of evaluation
                times = torch.Tensor(
                    np.concatenate([norm_sld_obs.TIME.values, norm_sld_extrapolate.TIME.values])
                )
                assert times.shape == labels.shape
                assert (
                    times[1:] >= times[:-1]
                ).all(), f"t must increase --> {times} {ptid} {x_obs} {data[data.NMID == ptid].reset_index(drop=True)}"

                baseline_pt_data = torch.from_numpy(
                    norm_pt_data[:baseline_idx][["TIME", "SLD"]].to_numpy(
                        dtype=np.float32, copy=True
                    )
                )

                # if the first measurement is post-treatment, set the time to 0 (holds regadless of scale factor)
                if not pre_tx_exists:
                    baseline_pt_data[0][0] = 0

                partitioned_sld = self.partition(norm_sld_obs)

                # generate ID. ID % 0 --> real patient, else augmented patient
                id = ptid if cut == cutoff_idx else ptid + cut * 0.01

                pt_item = Prodict(
                    id=id,
                    sld=partitioned_sld,
                    labels=labels,
                    times=times,
                    cut_idx=cut,
                    baseline=baseline_pt_data,
                    pt_time_scale=x_obs,
                )
                log_message(f'Patient {id} data: {pt_item}', print_msg=False)

                # store the calculated values
                samples.append(pt_item)

            TGI_dataset.extend(samples)

        return TGI_dataset
    
    def compute_cutoff(self, pt_data: pd.DataFrame, perc: int) -> List[float]:
        col_name = f'MDV_{perc}'
        cutoff = pt_data[col_name].ge(1).idxmax()
        if perc == 100:
            cutoff = len(pt_data)
        return cutoff


    def partition(self, pt_data: DataFrame) -> torch.Tensor:
        """Concatenates time-series data 2 to a row. Shape transformation: L_p x 2 --> (L_p - 1) x 4

        Parameters
        ----------
        pt_data : DataFrame
            A DataFrame containing time series data for a single patient.

        Returns
        -------
        torch.Tensor
            A tensor containing partioned SLD data with shape L_p x 4.
        """
        pt_partitioned_data = list()
        for (ind1, row1), (ind2, row2) in zip(pt_data[:-1].iterrows(), pt_data[1:].iterrows()):
            pt_partitioned_data.append([row1[1], row1[2], row2[1], row2[2]])
        return torch.from_numpy(np.array(pt_partitioned_data, dtype=np.float32))

    def normalize(self, data: DataFrame, mean: Prodict, STD: Prodict) -> DataFrame:
        """Performs Z-score normalization on SLD and TIME data using the mean and standard
        deviation of the training set.

        Parameters
        ----------
        data : DataFrame
            A DataFrame containing the unnormalized time series data for a single patient.
        mean : Prodict
            A dictionary containing the mean TIME and SLD values of the training cohort.
        STD : Prodict
            A dictionary containing the TIME and SLD standard deviations of the training cohort.

        Returns
        -------
        DataFrame
            A DataFrame containing the normalized time series data for a single patient.
        """
        for col in data.drop(["NMID"], axis=1).columns:
            data[col] = (data[col] - mean[col]) / STD[col]
        return data

    def collect_stats(self, data: DataFrame) -> Tuple[Prodict, Prodict,]:
        """Collects the mean and standard deviations of TIME and SLD values in the entire cohort.
        Should only be called when utilizing training set.

        Parameters
        ----------
        data : DataFrame
            A DataFrame containing time series data for all patients in the cohort.

        Returns
        -------
        Tuple[Prodict, Prodict,]
            A tuple of dictionaries containing the means and standard deviations of the TIME and SLD
            values in the cohort.
        """
        data = data.drop(["NMID"], axis=1)
        data_mean = Prodict.from_dict(data.mean())
        data_std = Prodict.from_dict(data.std())
        return (data_mean, data_std)
