# Explainable deep learning for tumor dynamic modeling and overall survival prediction using Neural-ODE

This repository provides an implementation of Tumor Dynamic Neural-ODE (TDNODE), a pharmacology-informed neural network that enables model discovery from longitudinal tumor size data. This implementation of TDNODE provides a standardized workflow for the following:

1) Fitting models to a series of observed sum-of-longest diameter (SLD) measurements.
2) Generating continuous predictions of tumor dynamics for estimation and extrapolation.
3) A series of analysis and visualization utilities for assessing TDNODE's predictive capabilities:
    a) Bootstrapped calculation of root mean-squared error (RMSE) and R2 scores using TDNODE-generated predictions and SLD labels. Optionally enable data subsetting with respect to treatment arm. 
    b) Comparative visualizations of TDNODE's predictions and observed SLD data.
    c) Residual visualizations of TDNODE's predictions and observed SLD data.
    d) Visualization of individual patient TDNODE predictions and observed SLD data.
    e) Prediction and evaluation of patient Overall Survival (OS) using TDNODE's generated patient-specific kinetic rate parameters and XGBoost-ML.
    f) SHAP analysis depicting the contribution of kinetic rate parameters and  (optionally) baseline covariates to OS prediction.
    g) Visualization of the effect of composite kinetic rate parameter changes on tumor dynamic predictions for individual patients. 
4) Synthetic data containing de-identified clinical trial data for user experimentation.

This codebase is implemented in Pytorch and contains supporting GPU functionality. 


## Installation

To get started, first clone the repository: 

`git clone https://github.com/jameslu01/TDNODE`

`cd TDNODE`

Then, using the provided `environment.yml` file, create an Anaconda environment that contains all necessary packages for model training and prediction. Ensure that Anaconda is installed on your machine in order to proceed:

`conda env create -f environment.yml`

Finally, activate the virtual environment via the following: 

`conda activate TDNODE`


## Example data

We provide a synthetic dataset that contains de-identified SLD time series from patients in different treatment arms. This dataset can be found in `assets/data`. To experiment with your own longitudinal data, please provide a `.csv` file that contains the folllowing schema:

1) NMID: a unique integer that corresponds to the patient from which the measurement was collected.
1) SLD: tumor sum of longest diameter measurement, in mm.
2) TIME: the time at which the corresponding SLD feature was observed.
3) ARM: the treatment arm in which the patient was enrolled.


## Basic usage

This repository provides functionality for either training or evaluating an instantiation of TDNODE

### Training TDNODE

TDNODE uses a single configuration file for defining the parameters used for training. All varialbes fall into the following scopes: `NAME`, `NOTE`, `DATA`, `TRAIN`, `TEST`, `MODEL`. This configuration file can be found in `experiments/SLD/configs`. We define the variables below. 

## Training 

TDNODE uses a single configuration file for defining the parameters used for training. All varialbes fall into the following scopes: `NAME`, `NOTE`, `DATA`, `TRAIN`, `TEST`, `MODEL`. This configuration file can be found in `experiments/SLD/configs`. We define the variables below. 

### NAME

`NAME: str`: The name of the model. Determines the name of the run directory located in `DATA/OUTPUT_DIR`

### NOTE

`NOTE: str`: Notes for a particular run. This field can be referred to in `{DATA}/{OUTPUT_DIR}/run_config.json` for future reference.

### DATA

`ROOT_DIR: str`: The root directory of the data used for model training. Place any additional datasets here.

`TGI_DATA: str`: The path of the dataset used for training and evaluating the model.

`OUTPUT_DIR: str`: The directory to store model checkpoints and analysis. In this directory, the model run will be saved with a timestamp (at initiation of model training) that is prepended to `NAME`. This value is considered the `RUN_ID` of the model and is used as an identifier when performing model evaluation. The directory contains the cached model dimensions, training parameters, epoch checkpoints, and summarization assets such as a loss vs. epoch curve and a csv file containing the running performance of the model on the training and validation set. 

`SPLIT: float [0.0, 1.0]`: The train/test split proportion: specifically defines the proportion of subjects to be assigned to the training set.

`SEED: int`: The seed used to split the dataset and perform other functions that require randomness to execute. Leave as constant to ensure deterministic results.

`AUGMENTATION: bool`: Indicator of whether to perform pointwise augmentation on the training set. If `True`, subsamples patient data with pre-observation window tumor size measurements and (optionally) post-treatment tumor size measurements. If `False`, does not augment the training set.

Example: Patient A has the following measurements: [[0, 120], [10, 35], [20, 45], [40, 30]]. With a 24 week observation window, the input SLD tensor used is the follwing: [[0, 120], [10, 35], [20, 45]]. With augmentation, _additional_ SLD tensors are generated with the following two truncated time-series measurements: [[0, 120], [10, 35]], & [[0, 120]]. 

`TRAIN_ALL: bool`: Indicator for whether to subsample post-treatment tumor size measurements while augmenting the training set. In the above example, the SLD tensor [[0, 120], [10, 35], [20, 45], [40, 30]] would be included for training in addition to [[0, 120], [10, 35]], & [[0, 120]].

`CUTOFF: float`: The observation window to use during model training and evaluation. During training, sets the maximum time measurement at which SLD measurements can be included in the input tensor. During evaluation, evaluates TDNODE's extrapolation capabilities at time points _after_ the observation window. This value is fixed for all subjets in the dataset.

`NORM_METHOD: str`: Normalization method for patient observation times: can be one of 'cohort' or 'patient'. If 'cohort', performs Z-score nromalization on the times series data using the mean and standard deviation of all times. If 'patient', performs patient-level normalization of time using the last observed measurement as a scale factor. The last observed measurement for a given patient is derived as the measurement with the observation time that is closest to but does not exceed the `CUTOFF` parameter.

### TRAIN

`LR: float`: Model learning rate. Adjust to experiment with the model's rate of convergence. 

`L2: float`: L2 weight decay. Adjusts the degree of regularization of the model to address possible issues of variance or bias. 

`NUM_EPOCH: int`: The number of epochs used to train the model.

`BATCH_SIZE: int [1, num_patients(training set)]`: The number of samples to process per iteration. 

`PAD_VALUE: float`: The pad value used when creating sample batches. Applies to both initial condition tensor and SLD tensor. See IMPLEMENTATION.pdf for details related to TDNODE's custom batching operation. 

`USE_COLLATE_FN: bool [True, False]`: Indicator for whether to use the custom collate function to create padded batches of samples. Keep as `True` to avoid runtime errors, as setting this parameter to `False` uses the default Pytorch collate function, which strictly assumes equal number of patient SLD measurements.

`USE_PRETRAINED: bool`: Indicator for whether to use a pre-trained model as the starting point for evlauation. If `False`, begins training TDNODE using a new set of random weights. If `True`, uses `MODEL.FROM_PRETRAINED_ID` and `MODEL.FROM_PRETRAINED_EPOCH_START` to load model. TDNODE will also load the previous model's parameters and make a new results directory using `cfg.NAME`.

### TEST

`EVAL_ALL: bool [True, False]`: Indicator for whether to evaluate predictions at all times, instead of the standard procedure to evaluate predictions at times after the `CUTOFF`. If `True`, evaluates predictions at all time points for each patient. If `False`, evaluates predictions where `t > DATA.CUTOFF` for the test set. Does not apply to the training set, where all measurements are evaluated regardless of this parameter setting

### MODEL

`NAME: str [TDNODE]`: Name of the model to use. References model architecture specified in `src/model/__init__.py`. 

`TOL: float`: NODE-specific hyperparamter used in the `torchdiffeq` library. Refers to the "tolerance for convergence of the Adams-Moulton corrector" link: https://github.com/rtqichen/torchdiffeq/blob/master/FURTHER_DOCUMENTATION.md

`FROM_PRETRAINED_ID: str`: The `RUN_ID` of the desired model to load as the starting state for a new training run. 

`FROM_PRETRAINED_EPOCH_START: int`: The epoch number from which to load the pretrained model.

`PARAMETERS: Dict[str, dynamic]`: Model architecture dimenstions. Cached in `{OUTPUT_DIR}/{NAME}/model_config.json` for reference during analysis. 

### Training TDNODE

Once your parameters are set, run the following:

`sh experiments/SLD/scripts/train.sh`

Real-time loss curves and R2 scores will be plotted as the model trains. Track these plots to assess convergence. 

### Evaluation

We have implemented several analysis functions to assess TDNODE's capabilities in predicting both tumor dynamics and overall survival (OS). The configuration file for analysis can be found in `experiments/analysis/compound/compound.yaml`. Descriptions for each parameter can be referenced below. 

To control which analysis to perform, modify the list of function names in `ANALYSIS.FUNCTIONS`. To modify the observation windows used, modify `ANALYSIS.WINDOWS`.

Function names: ["cumulative", "boostrapping", "individual", "encodings", "OS", "feature_dependence", "loss_curve", "log_residual"]

### DATA

`ROOT_DIR: str`: The root directory of the data used for model training. Place any additional datasets here.

`TGI_DATA: str`: The path of the dataset used for training and evaluating the model.

`OUTPUT_DIR: str`: The directory to store model checkpoints and analysis. In this directory, the model run will be saved with a timestamp (at initiation of model training) that is prepended to `NAME`. This value is considered the `RUN_ID` of the model and is used as an identifier when performing model evaluation. The directory contains the cached model dimensions, training parameters, epoch checkpoints, and summarization assets such as a loss vs. epoch curve and a csv file containing the running performance of the model on the training and validation set. 

`TGIOS_DATA: str`: The name of the TGI-OS file to use for any type of `OS` analysis. This file should contain a column names `OS` that contains OS data for each patient

`PTNM_DATA: str`: The name of the IMPower dataset csv file to load during survival analysis.

`SPLIT: float [0.0, 1.0]`: The train/test split proportion (i.e. the proportion of subjects to assign to the training set)

`SEED: int`: The seed used to split the dataset and perform other functions that require randomness to execute. Leave as constant to ensure deterministic results.

`AUGMENTATION: bool`: Indicator of whether to perform pointwise augmentation on the training set. If `True`, subsamples patient data with pre-observation window tumor size measurements and (optionally) post-treatment tumor size measurements. If `False`, does not augment the training set.

Example: Patient A has the following measurements: [[0, 120], [10, 35], [20, 45], [40, 30]]. With a 24 week observation window, the input SLD tensor used is the follwing: [[0, 120], [10, 35], [20, 45]]. With augmentation, _additional_ SLD tensors are generated with the following two truncated time-series measurements: [[0, 120], [10, 35]], & [[0, 120]]. 

`TRAIN_ALL: bool`: Indicator for whether to subsample post-treatment tumor size measurements while augmenting the training set. In the above example, the SLD tensor [[0, 120], [10, 35], [20, 45], [40, 30]] would be included for training in addition to [[0, 120], [10, 35]], & [[0, 120]].

`CUTOFF: float`: The observation window to use during model training and evaluation. During training, sets the maximum time measurement at which SLD measurements can be included in the input tensor. During evaluation, evaluates TDNODE's extrapolation capabilities at time points _after_ the observation window. This value is fixed for all subjets in the dataset.

`NORM_METHOD: str`: Normalization method for patient observation times: can be one of 'cohort' or 'patient'. If 'cohort', performs Z-score nromalization on the times series data using the mean and standard deviation of all times. If 'patient', performs patient-level normalization of time using the last observed measurement as a scale factor. The last observed measurement for a given patient is derived as the measurement with the observation time that is closest to but does not exceed the `CUTOFF` parameter.

`COHORT: str [train, test]`: The cohort on which to evaluate TDNODE. 

`ARM: Union[str, int] [all, 1, 2, 3]`: The treatment arm to evaluate. Use `all` to evaluate all treatment arms. 

### TEST

`RUN_ID: str`: The directory name of the model run to evaluate. Can be found in `DATA.OUTPUT_DIR`. 

`EPOCH_TO_TEST: int`. The epoch number to evaluate. Loads the saved model with this epoch ID for evaluation.

`EVAL_ALL: bool [True, False]`: Indicator for whether to evaluate predictions at all times, instead of the standard procedure to evaluate predictions at times after the observation window. If `True`, evaluates predictions at all time points for each patient. If `False`, evaluates predictions where `t > DATA.CUTOFF` for the test set. Does not apply to the training set.

### CUMULATIVE

Name: cumulative.

Function(s): Generates scatter plots comparing predicted SLD and observed SLD data, obtains csv file of prediction-observation pairs for all patients in selected cohort.

`CUMULATIVE_PLOT_TITLE: str`: The title to use when generating scatter plots displaying the correlations between TDNODE predictions and obvervations. 

### BOOTSTRAP

Name: bootstrapping

Function: generates bootstrapped mean and median RMSE and R2 values with standard deviations and median absolute deviation. More specifically, metrics are derived from `NUM_SAMPLE` distrubution of RMSE and R2 values each calculated from a random sample of individual metrics collected with replacement.

`NUM_SAMPLE: int`: The number of bootstrap iterations to perform. For each iteration, _n_ samples are picked from a list of prediction-observation pairs with replacement, where _n_ is the number of prediction-observation pairs in the cohort. 

### ARM

Name: Arm

Function: Same functionality as `CUMULATIVE`, but specifically for the selected arm.

`PTID_PATH`: The path to a .json file specifying which patients belong to which treatment arm. Auto-generated upon running the 'cumulative' analysis function.
 
### INDIVIDUAL

Name: individual 

Function: generates plots comparing individual patients' TDNODE predictions and discrete observed SLD measurements

`NUM_PRED_POINTS: int`: The number of predictions to generate. Range is [0, last observed time measurement]. 

`PATIENT_IDS: List[int]`: The list of patient IDS to plot.

`PATIENT_COLORS: List[str]`: The list of colors to assign to each patient ID. Shape must match `PATIENT_IDS`.

### ENCODINGS

Name: encodings

Function: generates initial condition and kinetic rate parameters for each patient given a trained model. 

`CUTOFF_TO_SET: float`: Obvervation window that constrains data observable by TDNODE when generating the predictions.

### OS

Name: OS

Function: Performs OS analysis and SHAP analysis using generated encodings. 

`CUTOFF_TO_USE: float`: Identifier for encoding file to use when calculating OS. 

`LATENT_DIM: int`: The shape of the latent encoder, should be same as `MODEL.PARAMETERS.ODE_FUNC.LATENT_DIM` to avoid error.

`DROP_VARS: List[str]`: The list of variables to exclude when performing the analyses. Format: [latent-x, IC-x]. Raises error if either the variable is not found in the list of available variables, or if all variables are excluded.

`USE_COVARS: bool`: Indicator for whether to use baseline covariates in the analysis. If `False`, only uses kinetic rate metrics to predict OS. If `True`, uses baseline covariates provided in the `TGIOS_DATA` entry.

### FEATURE DEPENDENCE

Name: feature_dependence

Function: Plots invididual patient predictions with variations for either a single or mulitple encoding variables. Allows for observation of possible connection between OS prediction and tumor dynamic prediction via SHAP analysis of kinetic rate metrics.

`VARIABLE_IDX: int [0, np.sum(latent.shape, IC.shape)]`: The variable index to perturb.

`NUMBER_PERTURBED: int`: The number of perturbations to make, including the original trained values.

`RANGE: List[float]`: The upper and lower relative bounds of the perturbation. The first entry in the list represents the lower bound, while the second entry represents the upper bound. 

`MULTIVAR: bool [True, False]`: Indicator for whether to perturb multiple encoding variables at a time. If `True`, allows for perturbation of multiple encoding variables at a time. If `False`, refers to `VARIABLE_IDX` for the encoding index to perturb.

`MULTIVAR_IDX: List[int]`: A list of 1-indexed encoding variables to perturb. Note that the index of this variable will be used in `MULTIVAR_DIRECTION` and `MULTIVAR_MAGNITUDE`. Only used when `MULTIVAR` is set to `True`.

`MULTIVAR_DIRECTION: List[str]`: a list of directions with which to perturb the list of encoding variabls. If `up`, perturbs the encoding in the direction specified by `MULTIVAR_MAGNITUDE`. If set to `down`, the ranges are flipped such that the upper bound is the first entry of `RANGE` while the lower bound is the second entry of `RANGE`.

`PATIENT_IDS: List[int]`: The list of patients to plot feature dependence.

### LOG RESIDUAL

Name: log_residual

Function: Plots the Logarithm of the residuals between all cohort predictions and all cohort measurements. Also plots LOWESS smoothing curve with confidence interval to identify the direction, if any, of systemic bias in the TDNODE predictions.

`COLOR: str`: The color of the plot.

`CONFIDENCE_INTERVAL: float`: The confidence interval of the LOWESS smoothes curve prediction to set. A higher confidence interval tends to leave a wider spread of possible residual predictions.

### MODEL

`NAME: str [Baseline TDNODE]`: Name of the model to use. References model architecture specified in `src/model/__init__.py`. Other model configurations are currently deprecated. 

`TOL: float`: NODE-specific hyperparamter used in the `torchdiffeq` library. Refers to the "tolerance for convergence of the Adams-Moulton corrector" link: https://github.com/rtqichen/torchdiffeq/blob/master/FURTHER_DOCUMENTATION.md

`PARAMETERS.PARAMETER_PATH: str`: The name of the JSON file that contains the model dimensions. Leave as-is to prevent errors in model loading. 

### Running the Analysis

Once your functions and their corresponding factors are delineated, run the analysis with the following:

`sh experiments/analysis/compound/compound.sh`

### Running all Analysis

If automated analysis is desired, simply adjust the following parameter:

`cfg.ANALYSIS.FUNCTIONS: all`

## Reference
Refer to the following for the paper that this code supports: 

```
@article{laurie2023explainable,
  title={Explainable deep learning for tumor dynamic modeling and overall survival prediction using Neural-ODE},
  author={Laurie, Mark and Lu, James},
  journal={npj Systems Biology and Applications},
  volume={9},
  number={1},
  pages={58},
  year={2023},
  publisher={Nature Publishing Group UK London}
}
```

## Contact
Any questions or issues with the code can be referred to either James Lu (lu.james@gene.com) or Mark Laurie (markl21@stanford.edu). 
