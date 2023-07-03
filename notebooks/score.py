import argparse
import logging
import os
import pickle
import mlflow
import mlflow.sklearn

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

parser = argparse.ArgumentParser(description="To score the model")

parser.add_argument(
    "--train_test_loc",
    metavar="train_test_loc",
    type=str,
    default="split_data_raw",
    help="Enter data loc (default: %(default)s)",
    # required=False,
)

parser.add_argument(
    "--model_loc",
    metavar="model_loc",
    type=str,
    default="models",
    help="Enter model loc (default: %(default)s)",
    # required=False,
)

parser.add_argument(
    "--log_loc",
    metavar="log_loc",
    type=str,
    default="logs",
    help="Enter log loc (default: %(default)s)",
    # required=False,
)

args = parser.parse_args()

train_test_loc = args.train_test_loc
model_loc = args.model_loc
log_loc = args.log_loc

LOG_PATH = os.path.join(log_loc)
os.makedirs(LOG_PATH, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s:%(name)s:%(message)s")

file_handler = logging.FileHandler(os.path.join(log_loc, "test.log"))
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)


# loading datasets
X_test_prepared = pd.read_csv(
    os.path.join("datasets", train_test_loc, "X_test_prepared.csv")
)
X_test_prepared.drop(columns=X_test_prepared.columns[0], axis=1, inplace=True)
y_test = pd.read_csv(os.path.join("datasets", train_test_loc, "y_test.csv"))
y_test.drop(columns=y_test.columns[0], axis=1, inplace=True)
housing_prepared = pd.read_csv(
    os.path.join("datasets", train_test_loc, "housing_prepared.csv")
)
housing_prepared.drop(columns=housing_prepared.columns[0], axis=1, inplace=True)
housing_labels = pd.read_csv(
    os.path.join("datasets", train_test_loc, "housing_labels.csv")
)
housing_labels.drop(columns=housing_labels.columns[0], axis=1, inplace=True)

# load the model from disk
LR_PATH = os.path.join(model_loc, "LinearRegression.sav")
lin_reg = pickle.load(open(LR_PATH, "rb"))
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_mae = mean_absolute_error(housing_labels, housing_predictions)

logger.debug("Linear Regression")
logger.debug("MSE = {}".format(lin_mse))
logger.debug("RMSE = {}".format(lin_rmse))
logger.debug("MAE = {}".format(lin_mae))

# load the model from disk
DT_PATH = os.path.join(model_loc, "DecisionTreeRegressor.sav")
tree_reg = pickle.load(open(DT_PATH, "rb"))
housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)

logger.debug("Decision Tree Regressor")
logger.debug("MSE = {}".format(tree_mse))
logger.debug("RMSE = {}".format(tree_rmse))

# load the model from disk
RFRSCV_PATH = os.path.join(model_loc, "RandomForestRegressorRSCV.sav")
rnd_search = pickle.load(open(RFRSCV_PATH, "rb"))
cvres = rnd_search.cv_results_

logger.debug("Random Forest Regressor RSCV")
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    logger.debug(np.sqrt(-mean_score), params)

# load the model from disk
RFGSCV_PATH = os.path.join(model_loc, "RandomForestRegressorGSCV.sav")
grid_search = pickle.load(open(RFGSCV_PATH, "rb"))

grid_search.best_params_
cvres = grid_search.cv_results_

logger.debug("Random Forest Regressor GSCV")
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    logger.debug(np.sqrt(-mean_score), params)

# load the model from disk
FM_PATH = os.path.join(model_loc, "FinalModel.sav")
final_model = pickle.load(open(FM_PATH, "rb"))
final_predictions = final_model.predict(X_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)

logger.debug("Final Model")
logger.debug("MSE = {}".format(final_mse))
logger.debug("RMSE = {}".format(final_rmse))

mlflow.log_param("log_path", LOG_PATH)
mlflow.log_metric("mse", final_mse)
mlflow.log_metric("rmse", final_rmse)
# mlflow.log_metric("mae", final_mae)
