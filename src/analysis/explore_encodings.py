from unittest import result
import pandas as pd
import matplotlib.pyplot as plt
import utils
from prodict import Prodict
import numpy as np
from scipy import stats


def explore_encondings(cfg: Prodict):

    encodings_path = f"{cfg.DATA.OUTPUT_DIR}/{cfg.TEST.RUN_ID}/e{cfg.TEST.EPOCH_TO_TEST}/tables/encodings/epoch-{cfg.TEST.EPOCH_TO_TEST}_cutoff-{cfg.DATA.CUTOFF}_cohort-{cfg.DATA.COHORT}_encodings_var.csv"
    results_path = (
        f"{cfg.DATA.OUTPUT_DIR}/{cfg.TEST.RUN_ID}/e{cfg.TEST.EPOCH_TO_TEST}/encoding_exploration"
    )
    utils.make_dir(results_path)
    encodings_df = pd.read_csv(encodings_path).drop(
        columns=["IC-0", "IC-1", "IC-2", "IC-3", "nmid"]
    )
    for col_x in range(len(encodings_df.columns)):
        for col_y in range(col_x + 1, len(encodings_df.columns)):
            col_x_name = encodings_df.columns[col_x]
            col_y_name = encodings_df.columns[col_y]
            corr = stats.pearsonr(encodings_df[col_x_name].values, encodings_df[col_y_name].values)
            plt.scatter(encodings_df[col_x_name].values, encodings_df[col_y_name].values, s=3)
            plt.title(f"Correlation  {col_x_name} vs. {col_y_name}: r= {round(corr[0], 3)}")
            plt.xlabel(f"{col_x_name}")
            plt.ylabel(f"{col_y_name}")
            plt.savefig(f"{results_path}/colx-{col_x_name}_coly-{col_y_name}_scatter.png")
            plt.close()
