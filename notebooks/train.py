import argparse
import os
import pickle
import mlflow
import mlflow.sklearn

import numpy as np
import pandas as pd
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedShuffleSplit,
    train_test_split,
)
from sklearn.tree import DecisionTreeRegressor

from housing_project.nonstandardcode import income_cat_proportions, load_housing_data

parser = argparse.ArgumentParser(description="To train the model")

parser.add_argument(
    "--train_test_loc",
    metavar="train_test_loc",
    type=str,
    help="Enter data loc",
    default="split_data_raw",
)
parser.add_argument(
    "--model_loc",
    metavar="model_loc",
    type=str,
    help="Enter model loc",
    default="models",
)
args = parser.parse_args()
train_test_loc = args.train_test_loc
model_loc = args.model_loc

HOUSING_PATH = "datasets\housing"
print(HOUSING_PATH)
print(os.path.join("datasets", train_test_loc, "train_raw.csv"))
housing = load_housing_data(HOUSING_PATH)
train_set = pd.read_csv(os.path.join("datasets", train_test_loc, "train_raw.csv"))
test_set = pd.read_csv(os.path.join("datasets", train_test_loc, "test_raw.csv"))


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

compare_props = pd.DataFrame(
    {
        "Overall": income_cat_proportions(housing),
        "Stratified": income_cat_proportions(strat_test_set),
        "Random": income_cat_proportions(test_set),
    }
).sort_index()
compare_props["Rand. %error"] = (
    100 * compare_props["Random"] / compare_props["Overall"] - 100
)
compare_props["Strat. %error"] = (
    100 * compare_props["Stratified"] / compare_props["Overall"] - 100
)

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

housing = strat_train_set.copy()
housing.plot(kind="scatter", x="longitude", y="latitude")
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)

corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)
housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["population_per_household"] = housing["population"] / housing["households"]

housing = strat_train_set.drop(
    "median_house_value", axis=1
)  # drop labels for training set
housing_labels = strat_train_set["median_house_value"].copy()


imputer = SimpleImputer(strategy="median")

housing_num = housing.drop("ocean_proximity", axis=1)

imputer.fit(housing_num)
X = imputer.transform(housing_num)

housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing.index)
housing_tr["rooms_per_household"] = housing_tr["total_rooms"] / housing_tr["households"]
housing_tr["bedrooms_per_room"] = (
    housing_tr["total_bedrooms"] / housing_tr["total_rooms"]
)
housing_tr["population_per_household"] = (
    housing_tr["population"] / housing_tr["households"]
)

housing_cat = housing[["ocean_proximity"]]
housing_prepared = housing_tr.join(pd.get_dummies(housing_cat, drop_first=True))


# lin_reg = LinearRegression()
# lin_reg.fit(housing_prepared, housing_labels)
MODEL_PATH = os.path.join(model_loc)
os.makedirs(MODEL_PATH, exist_ok=True)
# filename = 'LinearRegression.sav'
# LR_DIR = os.path.join(model_loc, filename)
# pickle.dump(lin_reg, open(LR_DIR, 'wb'))


tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(housing_prepared, housing_labels)
filename = "DecisionTreeRegressor.sav"
TR_DIR = os.path.join(model_loc, filename)
pickle.dump(tree_reg, open(TR_DIR, "wb"))


param_distribs = {
    "n_estimators": randint(low=1, high=200),
    "max_features": randint(low=1, high=8),
}

forest_reg = RandomForestRegressor(random_state=42)
rnd_search = RandomizedSearchCV(
    forest_reg,
    param_distributions=param_distribs,
    n_iter=10,
    cv=5,
    scoring="neg_mean_squared_error",
    random_state=42,
)
rnd_search.fit(housing_prepared, housing_labels)
filename = "RandomForestRegressorRSCV.sav"
RFRSCV_DIR = os.path.join(model_loc, filename)
pickle.dump(rnd_search, open(RFRSCV_DIR, "wb"))


param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {"bootstrap": [False], "n_estimators": [3, 10], "max_features": [2, 3, 4]},
]

forest_reg = RandomForestRegressor(random_state=42)
# train across 5 folds, that's a total of (12+6)*5=90 rounds of training
grid_search = GridSearchCV(
    forest_reg,
    param_grid,
    cv=5,
    scoring="neg_mean_squared_error",
    return_train_score=True,
)
grid_search.fit(housing_prepared, housing_labels)

grid_search.best_params_
filename = "RandomForestRegressorGSCV.sav"
RFGSCV_DIR = os.path.join(model_loc, filename)
pickle.dump(grid_search, open(RFGSCV_DIR, "wb"))

feature_importances = grid_search.best_estimator_.feature_importances_
sorted(zip(feature_importances, housing_prepared.columns), reverse=True)


final_model = grid_search.best_estimator_

X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

X_test_num = X_test.drop("ocean_proximity", axis=1)
X_test_prepared = imputer.transform(X_test_num)
X_test_prepared = pd.DataFrame(
    X_test_prepared, columns=X_test_num.columns, index=X_test.index
)
X_test_prepared["rooms_per_household"] = (
    X_test_prepared["total_rooms"] / X_test_prepared["households"]
)
X_test_prepared["bedrooms_per_room"] = (
    X_test_prepared["total_bedrooms"] / X_test_prepared["total_rooms"]
)
X_test_prepared["population_per_household"] = (
    X_test_prepared["population"] / X_test_prepared["households"]
)

X_test_cat = X_test[["ocean_proximity"]]
X_test_prepared = X_test_prepared.join(pd.get_dummies(X_test_cat, drop_first=True))

X_test_prepared.to_csv(os.path.join("datasets", train_test_loc, "X_test_prepared.csv"))
y_test.to_csv(os.path.join("datasets", train_test_loc, "y_test.csv"))
housing_prepared.to_csv(
    os.path.join("datasets", train_test_loc, "housing_prepared.csv")
)
housing_labels.to_csv(os.path.join("datasets", train_test_loc, "housing_labels.csv"))

filename = "FinalModel.sav"
FINAL_DIR = os.path.join(model_loc, filename)
pickle.dump(final_model, open(FINAL_DIR, "wb"))

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
filename = "LinearRegression.sav"
LR_DIR = os.path.join(model_loc, filename)
pickle.dump(lin_reg, open(LR_DIR, "wb"))

mlflow.log_param("PATH", HOUSING_PATH)
mlflow.log_param("model_path", MODEL_PATH)
