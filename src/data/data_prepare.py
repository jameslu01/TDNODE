import sys

sys.path.extend(["../TDNODE", "src"])

from torch.utils.data import DataLoader
from data.dataset import TGIData
from data.collate import make_batch
import src.utils
import warnings
from prodict import Prodict

warnings.filterwarnings("ignore")


def prepare_TGI(
    data_root_path: str,
    augment: bool = False,
    cutoff: int = 32,
    get_all: bool = False,
    batch_size: int = 1,
    device: str = "cpu",
    train_all: bool = False,
    norm_method: str = "patient",
) -> Prodict:
    """Preprocesses TGI data for training/evaluation.

    Parameters
    ----------
    data_root_path : str
        A string that references the the directory in which the training and test sets reside.
    augment : bool, optional
        Indicator for whether to augment the training set, by default False
    cutoff : int, optional
        An integer describing the desired observation window to use during preprocessing,
        by default 32.
    get_all : bool, optional
        Indicator for whether to obtain all observations regardless of the value of `cutoff`,
        by default False.
    batch_size : int, optional
        An integer the specifies the desired batch size (number of samples to process per iteration)
        during training and evaluation, by default 1.
    device : str, optional
        A string specifying the device on which to load the data, by default "cpu"
    train_all : bool, optional
        Augmentation indicator for whether to perform subsampling on observations after `cutoff`,
        by default False
    norm_method : str, optional
        Time normalization method to use, by default "patient"

    Returns
    -------
    Prodict
        A dictionary containing the preprocessed training and test sets, with accompanying metadata.
    """

    train_data_path = f"{data_root_path}/train.csv"
    test_data_path = f"{data_root_path}/test.csv"

    train = TGIData(
        train_data_path,
        cohort="train",
        augment=augment,
        cutoff=cutoff,
        get_all=True,
        train_all=train_all,
        norm_method=norm_method,
    )

    test = TGIData(
        test_data_path,
        cohort="test",
        train_mean=train.SLD_mean,
        train_std=train.SLD_std,
        cutoff=cutoff,
        get_all=get_all,
        norm_method=norm_method,
    )

    train_dataloader = DataLoader(
        train,
        batch_size=batch_size,
        collate_fn=lambda batch: make_batch(batch, device),
        shuffle=True,
    )
    test_dataloader = DataLoader(
        test,
        batch_size=batch_size,
        collate_fn=lambda batch: make_batch(batch, device),
        shuffle=False,
    )

    TGI_train = Prodict(
        dataloader=src.utils.inf_generator(train_dataloader),
        n_batches=len(train_dataloader),
        mean=train.SLD_mean,
        std=train.SLD_std,
    )

    TGI_test = Prodict(
        dataloader=src.utils.inf_generator(test_dataloader),
        n_batches=len(test_dataloader),
        mean=train.SLD_mean,
        std=train.SLD_std,
    )

    return Prodict(train=TGI_train, test=TGI_test)
