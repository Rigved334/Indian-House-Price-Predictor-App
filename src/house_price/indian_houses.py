import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from scipy import stats
import joblib

housing = pd.read_csv("indian_house_prices.csv")

housing["area_cat"] = pd.cut(housing["Area of the house(excluding basement)"], bins=[0, 750, 1500, 2250, 3000, np.inf], labels=[1, 2, 3, 4, 5])

housing["house_age"] = 2026 - housing["Built Year"]
housing["years_since_renovation"] = (2026 - housing["Renovation Year"])
housing["bedroom_density"] = (housing["number of bedrooms"] / housing["living area"])
housing["bathroom_density"] = (housing["number of bathrooms"] / housing["living area"])

splitter = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
strat_splits = []
for train_index, test_index in splitter.split(housing, housing["area_cat"]):
    strat_train_set_n = housing.iloc[train_index].copy()
    strat_test_set_n = housing.iloc[test_index].copy()
    strat_splits.append([strat_train_set_n, strat_test_set_n])

strat_train_set, strat_test_set = strat_splits[0]

for set_ in (strat_train_set, strat_test_set):
    set_.drop("area_cat", axis=1, inplace=True)

housing = strat_train_set.copy()

corr_matrix = housing.corr()

housing_labels = housing["Price"].copy()
housing = housing.drop(columns=["id", "Date", "Price"], axis=1)

num_pipeline = make_pipeline(StandardScaler())
housing_prepared = num_pipeline.fit_transform(housing)

num_attribs = list(housing.columns)
preprocessing = ColumnTransformer([("num", num_pipeline, num_attribs)])

preprocessing = make_column_transformer((num_pipeline, make_column_selector(dtype_include=np.number)))
housing_prepared = preprocessing.fit_transform(housing)

full_pipeline = Pipeline([("preprocessing", preprocessing), ("random_forest", RandomForestRegressor(n_estimators=100, random_state=42))])
param_grid = [{"random_forest__n_estimators": [100, 200], "random_forest__max_features": [4, 6, 8]}, 
              {"random_forest__n_estimators": [200, 300], "random_forest__max_features": [6, 8, 10]}]
grid_search = GridSearchCV(full_pipeline, param_grid, cv=3, scoring="neg_root_mean_squared_error")
grid_search.fit(housing, housing_labels)

cv_res = pd.DataFrame(grid_search.cv_results_)
cv_res.sort_values(by="mean_test_score", ascending=False, inplace=True)

param_dist = {"random_forest__n_estimators": randint(low=3, high=50), "random_forest__max_features": randint(low=2, high=20)}
rnd_search = RandomizedSearchCV(full_pipeline, param_distributions=param_dist, n_iter=10, cv=3, scoring="neg_root_mean_squared_error", random_state=42)
rnd_search.fit(housing, housing_labels)

final_model = rnd_search.best_estimator_
feature_importances = final_model["random_forest"].feature_importances_

feature_names = housing.columns

importance_df = pd.DataFrame({"feature": feature_names, "importance": feature_importances})

X_test = strat_test_set.drop("Price", axis=1)
y_test = strat_test_set["Price"].copy()
final_predictions = final_model.predict(X_test)
final_rmse = root_mean_squared_error(y_test, final_predictions)

def rmse(squared_errors):
    return np.sqrt(np.mean(squared_errors))

confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
boot_result = stats.bootstrap([squared_errors], rmse, confidence_level=confidence, random_state=42)
rmse_lower, rmse_upper = boot_result.confidence_interval

joblib.dump(final_model, "Indian_house_prices.pkl", compress=3)

new_house = pd.DataFrame([{
    "number of bedrooms": 3,
    "number of bathrooms": 2.0,
    "living area": 1800,
    "lot area": 5000,
    "number of floors": 2.0,
    "waterfront present": 0,
    "number of views": 2,
    "condition of the house": 4,
    "grade of the house": 8,
    "Area of the house(excluding basement)": 1800,
    "Area of the basement": 0,
    "Built Year": 2015,
    "Renovation Year": 0,
    "Postal Code": 452001,
    "Lattitude": 22.7196,
    "Longitude": 75.8577,
    "living_area_renov": 1900,
    "lot_area_renov": 5200,
    "Number of schools nearby": 3,
    "Distance from the airport": 15,

    # Engineered features
    "house_age": 2026 - 2015,
    "years_since_renovation": 2026 - 0,
    "bedroom_density": 3 / 1800,
    "bathroom_density": 2 / 1800
}])

predicted_price = final_model.predict(new_house)

print("Predicted Price:", predicted_price[0], "Rupees")