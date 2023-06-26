import argparse
import os
import mlflow
import mlflow.sklearn

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

from housing_project.nonstandardcode import fetch_housing_data
from housing_project.nonstandardcode import load_housing_data

HOUSING_PATH = (
    "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/"
)

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

# fetch_housing_data(HOUSING_URL, HOUSING_PATH)

housing = load_housing_data(HOUSING_PATH)

housing["income_cat"] = pd.cut(
    housing["median_income"],
    bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
    labels=[1, 2, 3, 4, 5],
)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

parser = argparse.ArgumentParser(
    description="Download and create training and validation datasets"
)

parser.add_argument(
    "--train_test_loc",
    metavar="train_test_loc",
    type=str,
    help="Enter data loc",
    default="split_data_raw",
)
args = parser.parse_args()
train_test_loc = args.train_test_loc
SPLIT_PATH = os.path.join("datasets", train_test_loc)
os.makedirs(SPLIT_PATH, exist_ok=True)
train_set.to_csv(os.path.join("datasets", train_test_loc, "train_raw.csv"))
test_set.to_csv(os.path.join("datasets", train_test_loc, "test_raw.csv"))
strat_train_set.to_csv(os.path.join("datasets", train_test_loc, "strat_train_set.csv"))
strat_test_set.to_csv(os.path.join("datasets", train_test_loc, "strat_test_set.csv"))

mlflow.log_artifact("datasets/housing/housing.csv")
