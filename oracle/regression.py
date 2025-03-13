# src/backend_new/core/oracle/regression.py
import copy
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from .base import BaseOracle

class RegressionOracle(BaseOracle):
    def train(self, data: pd.DataFrame, target_col: str) -> float:
        dataset = copy.deepcopy(data)
        columns = list(dataset.columns)
        columns.remove(target_col)

        for col in columns:
            dataset[col] = dataset[col].fillna(0)
            dataset[col] = dataset[col].replace(np.nan, 0)
            if dataset.dtypes[col] == 'object':
                dataset[col] = dataset[col].astype('category')
                dataset[col] = dataset[col].cat.codes

        mae = 0
        for seed in [42]:
            X = dataset[columns]
            y = dataset[target_col]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

            clf = RandomForestRegressor(random_state=0)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            mae += mean_absolute_error(y_test, y_pred)

        return mae