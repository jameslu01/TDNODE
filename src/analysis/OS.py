import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
import pickle
import warnings
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import shap
from typing import Tuple

import xgboost as xgb
from lifelines.utils import concordance_index

plt.style.use("bmh")

np.random.seed(42)
import sys

sys.path.append("src")


warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", 500)
from prodict import Prodict
import json
import utils
from args import args

import shap


def perform_survival_analysis(cfg: Prodict):
    """Uses TDNODE-generated encodings to predict OS via XGBoost. Evaluates OS predictions and
    performs SHAP analyses to identify salient features.

    Parameters
    ----------
    cfg : Prodict
        A configuration file specifying the parameters of the evaluation.
    """
    print("Performing Survival Analysis.")
    (TGIOS_Data_NMID, latent_dim, run_path, OS_path) = load_data(cfg)
    (
        X_XGB_Train,
        Y_XGB_Train,
        X_XGB_Test,
        Y_XGB_Test,
        X_XGB_Train_NMID,
        X_XGB_Test_NMID,
        eventQ_Train,
        eventQ_test,
    ) = preprocess_data(cfg, run_path, OS_path, TGIOS_Data_NMID)

    n_folds = 5
    kf = KFold(n_splits=n_folds, shuffle=True)
    num_boost_round = 1000
    XGB_best = perform_XGBoost(
        cfg,
        X_XGB_Train,
        Y_XGB_Train,
        X_XGB_Test,
        Y_XGB_Test,
        X_XGB_Train_NMID,
        X_XGB_Test_NMID,
        eventQ_Train,
        eventQ_test,
        OS_path,
        kf,
        num_boost_round,
    )

    X_XGB_All = pd.concat([X_XGB_Train, X_XGB_Test]).drop("ARM", axis=1)
    Y_XGB_All = pd.concat([Y_XGB_Train, Y_XGB_Test])
    X_XGB_All_NMID = pd.concat([X_XGB_Train_NMID, X_XGB_Test_NMID])
    perform_SHAP_analysis(
        cfg,
        X_XGB_All,
        Y_XGB_All,
        X_XGB_All_NMID,
        kf,
        num_boost_round,
        OS_path,
        latent_dim,
        XGB_best,
    )
    return


def perform_SHAP_analysis(
    cfg: Prodict,
    X_XGB_All: pd.DataFrame,
    Y_XGB_All: pd.DataFrame,
    X_XGB_All_NMID: pd.DataFrame,
    kf: KFold,
    num_boost_round: int,
    OS_path: str,
    latent_dim: int,
    XGB_best: xgb.Booster,
) -> None:
    """Performs SHAP using the best XGBoost model obtained during model training.

    Parameters
    ----------
    cfg : Prodict
        A configuration file specifying the parameters of the evaluation.
    X_XGB_All : pd.DataFrame
        A DataFrame containing the input features for XGBoost from all cohorts.
    Y_XGB_All : pd.DataFrame
        A DataFrame containing the output feature for XGBoost from all cohorts.
    X_XGB_All_NMID : pd.DataFrame
        A DataFrame containing the NMIDs from all cohorts
    kf : KFold
        A helper class that assists in data splitting for 5-fold cross-validation.
    num_boost_round : int
        An XGBoost parameter (TODO: explain).
    OS_path : str
        A path that references the location of the survival analysis directory, where results and
        cached data will be stored.
    latent_dim : int
        The dimensionality of the parameter encoding, obtained automatically from model_cfg.
    XGB_best : xgb.Booster
        The XGB model trained during 5-fold cross-validation that exhibited the predictivity of
        patient OS.
    """
    print("Performing SHAP Analysis.")
    X_XGB_All_NMID.to_csv(f"{OS_path}/X_XGB_All_NMID.csv")
    X_XGB_All.to_csv(f"{OS_path}/X_XGB_All.csv")

    list_shap_values = list()
    list_val_sets = list()
    list_pred = list()
    list_X_val_concat = list()
    list_Y_val_concat = list()
    list_parity_NMID = list()

    for (f, (idx_train, idx_val)) in enumerate(kf.split(X_XGB_All)):

        print(f"Performing SHAP Analysis: Current Fold = {f}.")
        X_tr = X_XGB_All.iloc[idx_train]
        X_val = X_XGB_All.iloc[idx_val]

        X_val_NMID = X_XGB_All_NMID.iloc[idx_val]

        y_tr = Y_XGB_All.iloc[idx_train]
        y_val = Y_XGB_All.iloc[idx_val]

        list_X_val_concat.append(X_val)
        list_Y_val_concat.append(y_val)

        dval = xgb.DMatrix(X_val, label=y_val)

        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dval = xgb.DMatrix(X_val, label=y_val)

        xgb_model = xgb.train(XGB_best, dtrain, num_boost_round=num_boost_round)
        pred_val = xgb_model.predict(dval)
        list_pred.append(pred_val)

        explainer_split = shap.TreeExplainer(xgb_model, data=X_tr)  # ,
        shap_values_split = explainer_split.shap_values(X_val)
        list_shap_values.append(shap_values_split)
        list_val_sets.append(idx_val)
        list_parity_NMID.append(X_val_NMID)

    val_set = list_val_sets[0]
    shap_values_concat = np.array(list_shap_values[0])
    shap_value_nmid = np.array(list_parity_NMID[0])
    X_XGB_VAL_set = np.array(list_X_val_concat[0])
    for i in range(1, len(list_val_sets)):
        val_set = np.concatenate((val_set, list_val_sets[i]), axis=0)
        shap_values_concat = np.concatenate(
            (shap_values_concat, np.array(list_shap_values[i])), axis=0
        )
        shap_value_nmid = np.concatenate((shap_value_nmid, np.array(list_parity_NMID[i])), axis=0)
        X_XGB_VAL_set = np.concatenate((X_XGB_VAL_set, np.array(list_X_val_concat[i])), axis=0)

    json.dump(
        {
            "ID": shap_value_nmid.tolist(),
            "shap_value_concat": shap_values_concat.tolist(),
            "idx_split": val_set.tolist(),
            "latent_vars": X_XGB_VAL_set.tolist(),
        },
        open(f"{OS_path}/shap_nmid.json", "w"),
        indent=2,
    )
    combined_df = pd.DataFrame(
        np.concatenate(
            (
                shap_value_nmid[:, np.newaxis,].astype(int),
                val_set[:, np.newaxis,].astype(int),
                X_XGB_VAL_set,
                shap_values_concat,
            ),
            axis=1,
        )
    )
    n_vars = int((combined_df.columns.shape[0] - 2) / 2)
    combined_df.columns = (
        ["NMID"]
        + ["split_idx"]
        + [f"parameter-{i}" for i in range(n_vars)]
        + [f"shap-{i}" for i in range(n_vars)]
    )
    for var in range(n_vars):
        combined_df = combined_df.sort_values(by=[f"parameter-{var}"])
        combined_df.to_csv(f"{OS_path}/combined_results_sortvar-{var}.csv", index=False)

    plot_path = f"{cfg.DATA.OUTPUT_DIR}/{cfg.TEST.RUN_ID}/e{cfg.TEST.EPOCH_TO_TEST}/plots/SHAP-covars-{cfg.TEST.OS.USE_COVARS}_cutoff-{cfg.TEST.OS.CUTOFF_TO_USE}_varsdropped_{cfg.TEST.OS.DROP_VARS}"

    utils.make_dir(plot_path)

    plt.rc("xtick", labelsize=60)
    plt.rc("ytick", labelsize=60)
    font = {"weight": "bold", "size": 60}
    plt.rc("font", **font)

    plt.rc("legend", fontsize=20)
    plt.rc("axes", titlesize=60)
    plt.rc("axes", labelsize=60)

    topNum = 25
    X_val_concat = pd.DataFrame(X_XGB_All.iloc[val_set], columns=X_XGB_All.columns)
    X_val_concat.to_csv(f"{OS_path}/test_val_csv.csv")
    (f, a) = plt.subplots(1, 1, figsize=(18, 8))

    shap.summary_plot(
        shap_values_concat, X_val_concat, show=False, plot_size=(9, 9), max_display=topNum, 
    )
    sum_fig, sum_ax = plt.gcf(), plt.gca()
    for item in ([sum_ax.title, sum_ax.xaxis.label, sum_ax.yaxis.label] +
             sum_ax.get_xticklabels() + sum_ax.get_yticklabels()):
            
        item.set_fontsize(20)
    sum_fig.axes[-1].tick_params(labelsize=20) # color bar access
    sum_fig.axes[-1].set_ylabel("Feature Value", size=20)

    print(plot_path)
    plt.savefig(
        f"{plot_path}/tableSHAP_SummaryPlot_TDNODE_dropped-{cfg.TEST.OS.DROP_VARS}.png",
        dpi=400,
    )
    if cfg.TEST.OS.VISUAL:
        for var in range(1, latent_dim + 1):
            try:
                fig = plt.figure(figsize=(10, 10))
                plt.rc("font", size=28)

                shap.dependence_plot(
                    f"Parameter-{var}",
                    shap_values_concat,
                    X_val_concat,
                    interaction_index=f"parameter-{var}",
                    show=False,
                    title=f"SHAP Dependence Plot",
                )
                plt.tight_layout()
                plt.xlabel(f"p, Variable {var}")
                plt.ylabel("SHAP Value")
                plt.savefig(
                    f"{plot_path}/SHAP_latent{var}_TDNODE_dropped-{cfg.TEST.OS.DROP_VARS}.png",
                    dpi=400,
                )
            except ValueError:
                try:
                    print(f"Plotting PCA ")
                    shap.dependence_plot(
                        f"pca-{var}",
                        shap_values_concat,
                        X_val_concat,
                        interaction_index=f"pca-{var}",
                        show=False,
                        title=f"SHAP Dependence Plot",
                    )
                    plt.tight_layout()
                    plt.xlabel(f"p, {var}st Principal Component")
                    plt.ylabel("SHAP Value")
                    plt.savefig(
                        f"{plot_path}/SHAP_PCA-{var}_TDNODE_dropped-{cfg.TEST.OS.DROP_VARS}.png",
                        dpi=400,
                    )
                except ValueError:
                    continue
                continue
        plt.close()
    else:
        for var in range(1, latent_dim + 1):
            try:
                fig = plt.figure(figsize=(12, 10))
                plt.rc("font", size=32)

                shap.dependence_plot(
                    f"Parameter-{var}",
                    shap_values_concat,
                    X_val_concat,
                    interaction_index=f"parameter-{var}",
                    show=False,
                    title="",
                )
                plt.tight_layout()
                plt.xlabel(f"p, Variable {var}")
                plt.ylabel("SHAP Value")
                plt.savefig(
                    f"{plot_path}/SHAP_latent{var}_TDNODE_dropped-{cfg.TEST.OS.DROP_VARS}.png",
                    dpi=400,
                )
            except ValueError:
                try:
                    fig = plt.figure(figsize=(12, 10))
                    plt.rcParams['axes.facecolor'] = '#D3D3D3'
                    fig.patch.set_alpha(0.0)

                    plt.rc("font", size=32)
                    print(f"Plotting PCA ")
                    shap.dependence_plot(
                        f"pca-{var}",
                        shap_values_concat,
                        X_val_concat,
                        interaction_index=f"pca-{var}",
                        show=False,
                        title="",
                    )
                    plt.tight_layout()
                    plt.xlabel(f"PCA, Axis {var}", labelpad=25)
                    plt.ylabel("SHAP Value", labelpad=25)
                    plt.xticks([])
                    plt.yticks([])
                    plt.savefig(
                        f"{plot_path}/SHAP_PCA-{var}_TDNODE_dropped-{cfg.TEST.OS.DROP_VARS}_F4.png",
                        dpi=400,
                    )

                except ValueError:
                    continue
                continue
        plt.close()



def perform_XGBoost(
    cfg: Prodict,
    X_XGB_Train: pd.DataFrame,
    Y_XGB_Train: pd.DataFrame,
    X_XGB_Test: pd.DataFrame,
    Y_XGB_Test: pd.DataFrame,
    X_XGB_Train_NMID: pd.DataFrame,
    X_XGB_Test_NMID: pd.DataFrame,
    eventQ_Train: np.ndarray,
    eventQ_Test: np.ndarray,
    OS_path: str,
    kf: KFold,
    num_boost_round: int,
) -> xgb.Booster:
    """Performs XGBoost with a previously defined set of parameters.

    Parameters
    ----------
    cfg : Prodict
        A configuration file specifying the parameters of the evaluation.
    X_XGB_Train : pd.DataFrame
        A DataFrame containing the input features to be used in XGBoost during 5-fold
        cross-validation.
    Y_XGB_Train : pd.DataFrame
        A DataFrame containing the output feature to be used in XGBoost during 5-fold
        cross-validation.
    X_XGB_Test : pd.DataFrame
        A DataFrame containing the input features to be used in XGBoost during model testing.
    Y_XGB_Test : pd.DataFrame
        A DataFrame containing the output feature to be used in XGBoost during model testing.
    X_XGB_Train_NMID: pd.DataFrame
        A DataFrame containing the input patient NMIDS for the training set.
    X_XGB_Test_NMID: pd.DataFrame
        A DataFrame containing the input patient NMIDs for the test set.
    eventQ_Train : np.ndarray
        An array containing the censorship status of each patient in the training set
    eventQ_Test : np.ndarray
        An array containing the censorship status of each patient in the test set
    OS_path : str
        A path that references the location of the survival analysis directory, where results and
        cached data will be stored.
    kf: KFold
        A helper class that assists in data splitting for 5-fold cross-validation.
    num_boost_round: int
        

    Returns
    -------
    xgb_best: xgb.Booster
        The XGB model trained during 5-fold cross-validation that exhibited the predictivity of
        patient OS.
    """
    print("Performing XGBoost")
    XGB_best = {
        "eta": 0.011577762057175561,
        "max_depth": 5,
        "min_child_weight": 0.021182092600431864,
        "reg_alpha": 0.0014102118562630659,
        "reg_lambda": 3.415200277788344,
        "subsample": 0.8493701242120658,
        "objective": "survival:cox",
    }
    num_boost_round = 1000

    val_scores_TDNODE = []
    xgb_model_list = []
    train_index_list = []
    validation_index_list = []

    results_dict = {"c-indices": [], "5-fold_score": 0, "5-fold-score_std": 0, "overall_score": 0}
    overall_scores = []
    count = 1

    for (train_index, val_index) in kf.split(X_XGB_Train):

        (X_tr, X_val) = (X_XGB_Train.iloc[train_index], X_XGB_Train.iloc[val_index])
        (y_tr, y_val) = (Y_XGB_Train.iloc[train_index], Y_XGB_Train.iloc[val_index])

        X_tr.to_csv(f"{OS_path}/fold-{count}_X_train.csv")
        X_val.to_csv(f"{OS_path}/fold-{count}_X_val.csv")
        y_tr.to_csv(f"{OS_path}/fold-{count}_Y_tr.csv")
        y_val.to_csv(f"{OS_path}/fold-{count}_Y_val.csv")

        eventQ_val = eventQ_Train[val_index]  # event indicator

        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dval = xgb.DMatrix(X_val, label=y_val)

        xgb_model = xgb.train(XGB_best, dtrain, num_boost_round=num_boost_round)

        predictions = xgb_model.predict(dval)
        overall_results_df = pd.DataFrame(
            {
                "NMID": X_XGB_Train_NMID[val_index].values,
                "y_val": y_val,
                "-predictions": predictions,
                "event": eventQ_val,
            }
        ).sort_values(by=["y_val"])
        overall_results_df = pd.concat([overall_results_df, X_val], axis=1).to_csv(
            f"{OS_path}/overall_results-valfold-{count}.csv"
        )

        c_index_TDNODE = concordance_index(abs(y_val), -predictions, eventQ_val)
        val_scores_TDNODE.append(c_index_TDNODE)

        xgb_model_list.append(xgb_model)

        train_index_list.append(train_index)
        validation_index_list.append(val_index)

        with open(f"{OS_path}/NODE_xgb_cv" + str(count) + ".pkl", "wb") as f:
            pickle.dump(xgb_model_list, f)

        print("c_index_TDNODE = ", round(c_index_TDNODE, 2))
        overall_scores.append(c_index_TDNODE)
        count += 1

    (X_tr, X_test) = (X_XGB_Train, X_XGB_Test)
    (y_tr, y_test) = (Y_XGB_Train, Y_XGB_Test)

    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dtest = xgb.DMatrix(X_test, label=y_test)
    xgb_model = xgb.train(XGB_best, dtrain, num_boost_round=num_boost_round)
    predictions = xgb_model.predict(dtest)
    import json

    json.dump(json.loads(xgb_model.save_config()), open(f"{OS_path}/XGBoost_parameters.json", "w"), indent=2)
    c_index_TDNODE_test = concordance_index(abs(y_test), -predictions, eventQ_Test)
    print(f"c-index on test set: {round(c_index_TDNODE_test, 2)}")

    overall_results_df = pd.DataFrame(
        {
            "NMID": X_XGB_Test_NMID.values,
            "y_test": y_test,
            "-predictions": predictions,
            "event": eventQ_Test,
        }
    )
    overall_results_df = (
        pd.concat([overall_results_df, X_test], axis=1)
        .sort_values(by=["y_test"])
        .to_csv(f"{OS_path}/overall_results-test.csv")
    )

    results_dict["5-fold_score"] = round(np.mean(val_scores_TDNODE), 4)
    results_dict["5-fold-score_std"] = round(np.std(val_scores_TDNODE), 4)
    results_dict["overall_score"] = round(c_index_TDNODE_test, 4)
    results_dict["c-indices"] = overall_scores
    json.dump(
        results_dict,
        open(
            f"{OS_path}/metrics_dropped-{cfg.TEST.OS.DROP_VARS}_cutoff-{cfg.TEST.OS.CUTOFF_TO_USE}-cutoff_withno_covars.json",
            "w",
        ),
        indent=2,
    )
    return XGB_best


def load_data(cfg: Prodict) -> Tuple[pd.DataFrame, int, str, str,]:
    """Loads necessary CSV files in preparation for survival analysis.

    Parameters
    ----------
    cfg : Prodict
        A configuration file specifying the parameters of the evaluation.

    Returns
    -------
    TGIOS_Data_NMID: pd.DataFrame
        A csv file containing patient OS survival outcomes and (optionally) baseline covariates.
    latent_dim: int
        The dimensionality of the parameter encoding, obtained automatically from model_cfg.
    run_path: str
        A path that references the location of the selected TDNODE model instantiation and epoch.
    OS_path: str
        A path that references the location of the survival analysis directory, where results and
        cached data will be stored.
    """
    print("Loading Data.")
    run_path = f"{cfg.DATA.OUTPUT_DIR}/{cfg.TEST.RUN_ID}/e{cfg.TEST.EPOCH_TO_TEST}"
    OS_path = f"{run_path}/tables/OS_covars-{cfg.TEST.OS.USE_COVARS}_cutoff-{cfg.DATA.CUTOFF}_varsdropeed_{cfg.TEST.OS.DROP_VARS}"
    utils.make_dir(OS_path)

    model_cfg = json.load(
        open(f"{cfg.DATA.OUTPUT_DIR}/{cfg.TEST.RUN_ID}/{cfg.MODEL.PARAMETERS.PARAMETER_PATH}", "r")
    )
    latent_dim = model_cfg["ODE_SOLVER"]["LATENT_DIM"]

    TGIOS_Data = pd.read_csv(f"{cfg.DATA.ROOT_DIR}/{cfg.DATA.TGIOS_DATA}")
    TGIOS_Data.to_csv(f"{OS_path}/{cfg.DATA.TGIOS_DATA}")

    NMID_PTNM = pd.read_csv(f"{cfg.DATA.ROOT_DIR}/{cfg.DATA.PTNM_DATA}")
    NMID_PTNM = NMID_PTNM[["NMID", "PTNM", "ARM"]].drop_duplicates()
    NMID_PTNM.to_csv(f"{OS_path}/{cfg.DATA.PTNM_DATA}")

    TGIOS_Data_NMID = TGIOS_Data.merge(
        NMID_PTNM, on="PTNM", how="left"
    )
    TGIOS_Data_NMID.to_csv(f"{OS_path}/merged_data_input.csv")

    TGIOS_Data_NMID = TGIOS_Data_NMID.drop(columns=["STUD", "PTNM"])[
        [
            "NMID",
            "OS",
            "CNSRO",
            "ARM_x",
            "CRP",
            "BECOG",
            "HGB",
            "ALBU",
            "BNLR",
            "LIVER",
            "YSD",
            "TPRO",
            "METSITES",
            "NEU",
            "LDH",
        ]
    ].rename(columns={"ARM_x": "ARM"})
    if not cfg.TEST.OS.USE_COVARS:  # whether to use covariates in the calculation
        TGIOS_Data_NMID = TGIOS_Data_NMID[["NMID", "OS", "CNSRO", "ARM"]]
    return (TGIOS_Data_NMID, latent_dim, run_path, OS_path)


def preprocess_data(
    cfg: Prodict, run_path: str, OS_path: str, TGIOS_Data_NMID: pd.DataFrame
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
]:
    """Prepares data for XGBoost modeling.

    Parameters
    ----------
    cfg : Prodict
        A configuration file specifying the parameters of the evaluation.
    run_path : str
        A path that references the location of the selected TDNODE model instantiation and epoch.
    OS_path : str
        A path that references the location of the survival analysis directory, where results and
        cached data will be stored.
    TGIOS_Data_NMID : pd.DataFrame
        A csv file containing patient OS survival outcomes and (optionally) baseline covariates.

    Returns
    -------
    X_XGB_Train: pd.DataFrame
        A DataFrame containing the input features to be used in XGBoost during 5-fold
        cross-validation.
    Y_XGB_Train: pd.DataFrame
        A DataFrme containing the output feature to be used in XGBoost during 5-fold
        cross-validation.
    X_XGB_Test: pd.DataFrame
        A DataFrame containing the input features to be used in XGBoost during model testing.
    Y_XGB_Test: pd.DataFrame
        A DataFrame containing the output feature to be used in XGBoost during model testing.
    X_XGB_Train_NMID: pd.DataFrame
        A DataFrame containing the NMIDs for the training set in the analysis.
    X_XGB_Test_NMID: pd.DataFrame
        A DataFrame containing the NMIDs for the test set in the analysis.
    eventQ_Train: np.ndarray
        An array containing the censorship status of each patient in the training set.
    eventQ-Test: np.ndarray
        An array containing the censorship status of each patient in the test set.
    """
    print("Preprocessing Data.")
    TDNODE_OS_Data_Train = pd.read_csv(
        f"{run_path}/tables/encodings/epoch-{cfg.TEST.EPOCH_TO_TEST}_cutoff-{cfg.TEST.OS.CUTOFF_TO_USE}_cohort-train_encodings_var.csv",
        na_values=[".", ""],
        sep=",",
    )
    TDNODE_OS_Data_Train = TDNODE_OS_Data_Train.rename(columns={"nmid": "NMID"}, inplace=False)

    TDNODE_OS_Data_Train = TDNODE_OS_Data_Train.drop(cfg.TEST.OS.DROP_VARS, axis=1)

    TDNODE_OS_Data_Train_use = copy.deepcopy(TDNODE_OS_Data_Train)
    TDNODE_OS_Data_Train_use = TDNODE_OS_Data_Train.merge(
        TGIOS_Data_NMID, on="NMID", how="left"
    )
    TDNODE_OS_Data_Train_use.to_csv(f"{OS_path}/TDNODE_OS_Data_Train_use.csv")

    X_XGB_Train_NMID = TDNODE_OS_Data_Train_use["NMID"]
    X_XGB_Train = TDNODE_OS_Data_Train_use.drop(columns=["NMID", "OS", "CNSRO"])
    X_XGB_Train.to_csv(f"{OS_path}/X_XGB_Train_before_training_withnovars.csv")

    Y_XGB_Train = TDNODE_OS_Data_Train_use["OS"]
    Y_XGB_Train.to_csv(f"{OS_path}/Y_XGB_Train.csv")

    TDNODE_OS_Data_Test = pd.read_csv(
        f"{run_path}/tables/encodings/epoch-{cfg.TEST.EPOCH_TO_TEST}_cutoff-{cfg.TEST.OS.CUTOFF_TO_USE}_cohort-test_encodings_var.csv",
        na_values=[".", ""],
        sep=",",
    )
    TDNODE_OS_Data_Test = TDNODE_OS_Data_Test.rename(columns={"nmid": "NMID"}, inplace=False)

    TDNODE_OS_Data_Test = TDNODE_OS_Data_Test.drop(cfg.TEST.OS.DROP_VARS, axis=1)

    TDNODE_OS_Data_Test_use = copy.deepcopy(TDNODE_OS_Data_Test)
    TDNODE_OS_Data_Test_use = TDNODE_OS_Data_Test_use.merge(
        TGIOS_Data_NMID, on="NMID", how="left"
    )
    TDNODE_OS_Data_Test_use.to_csv(f"{OS_path}/TDNODE_OS_Data_Test_use.csv")
    X_XGB_Test_NMID = TDNODE_OS_Data_Test_use["NMID"]
    X_XGB_Test = TDNODE_OS_Data_Test_use.drop(columns=["NMID", "OS", "CNSRO"])
    Y_XGB_Test = TDNODE_OS_Data_Test_use["OS"]
    Y_XGB_Test.to_csv(f"{OS_path}/Y_XGB_Test.csv")

    eventQ_Train = np.where(TDNODE_OS_Data_Train_use["CNSRO"] == 0, 1, 0)
    eventQ_Test = np.where(TDNODE_OS_Data_Test_use["CNSRO"] == 0, 1, 0)

    return (
        X_XGB_Train,
        Y_XGB_Train,
        X_XGB_Test,
        Y_XGB_Test,
        X_XGB_Train_NMID,
        X_XGB_Test_NMID,
        eventQ_Train,
        eventQ_Test,
    )
