import argparse

parser = argparse.ArgumentParser(
    "Explainable Deep Learning for Tumor Dynamic Prediction and Overall Survival using Neural ODE"
)

parser.add_argument(
    "--config",
    type=str,
    default="assets/configs/config.json",
    help="paremeter file that specifies TDNODE configuration",
)

args = parser.parse_args()
