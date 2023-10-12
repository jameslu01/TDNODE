from prodict import Prodict
from sklearn.metrics import mean_squared_error

import json
import numpy as np
import pandas as pd
import random
import sys

sys.path.append("src")


def get_pt_stats(params: Prodict) -> None:

    table_path = f"{params.data.src_path}/{params.data.identifier}/{params.data.epoch}/tables/{params.data.epoch}_{params.data.cutoff}_{params.data.cohort}_arm{params.data.arm}_results.csv"
    results_path = f"{params.data.src_path}/{params.data.identifier}/{params.data.epoch}/tables/{params.data.epoch}_{params.data.cutoff}_ptstats.csv"

    df = pd.read_csv(table_path)

    nmids = np.array([])
    rmse = np.array([])
    num_obs = np.array([])
    for ptid in df.NMID.unique():
        ptid_data = df[df.NMID == ptid].reset_index(drop=True)
        pt_rmse = mean_squared_error(ptid_data.labels, ptid_data.preds, squared=False)
        nmids = np.append(nmids, ptid)
        rmse = np.append(rmse, pt_rmse)
        num_obs = np.append(num_obs, len(ptid_data.preds))

    pd.DataFrame({"NMID": nmids, "RMSE": rmse, "N_preds": num_obs}).sort_values(by=["RMSE"]).to_csv(
        results_path, sep="\t"
    )


if __name__ == "__main__":
    from args import args

    params = Prodict.from_dict(json.load(open(args.params, "r")))
    random.seed(params.data.seed)
    np.random.seed(params.data.seed)
    get_pt_stats(params)
