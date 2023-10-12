from prodict import Prodict
import numpy as np
import sys
import pandas as pd

sys.path.append("src")
import utils
from predict import evaluate_model
from model import *


def get_pt_encodings(cfg: Prodict) -> None:
    """Generates the initial condition and parameter encodings for each patient.

    Parameters
    ----------
    cfg : Prodict
        A configuration file specifying the parameters of the evaluation.
    """
    print("Generating patient encodings")
    save_folder = (
        f"{cfg.DATA.OUTPUT_DIR}/{cfg.TEST.RUN_ID}/e{cfg.TEST.EPOCH_TO_TEST}/tables/encodings"
    )
    utils.make_dir(save_folder)
    cfg.DATA.CUTOFF = cfg.TEST.ENCODINGS.CUTOFF_TO_SET
    print(f"Generating encodings with observation window set to {cfg.DATA.CUTOFF} weeks.")
    (_, _, _, encoding_data, _, _) = evaluate_model(cfg)
    save_path = f"{save_folder}/epoch-{cfg.TEST.EPOCH_TO_TEST}_cutoff-{cfg.TEST.ENCODINGS.CUTOFF_TO_SET}_cohort-{cfg.DATA.COHORT}_encodings_var.csv"
    encoding_df = pd.DataFrame(dict(encoding_data))
    latent_encodings = encoding_df["latent_encoding"].apply(pd.Series)
    latent_encodings = latent_encodings.rename(columns=lambda x: f"latent-{x+1}")

    IC_encodings = encoding_df["IC_encoding"].apply(pd.Series)
    IC_encodings = IC_encodings.rename(columns=lambda x: f"IC-{x+1}")

    final_df = pd.concat([encoding_df["nmid"], latent_encodings[:], IC_encodings[:]], axis=1)
    final_df.sort_values(by=["nmid"]).to_csv(save_path, index=False)
