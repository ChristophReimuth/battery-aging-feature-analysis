# Grid Search Funktion für Random Forest Regressor

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score

from sklearn.inspection import permutation_importance

def run_rf_gridsearch(
    df,
    feature_columns,
    target_column="SOH"
):

    data = df[feature_columns + [target_column]].dropna()

    X = data[feature_columns]
    y = data[target_column]

    # Train/Test Split
    split_idx = int(len(data) * 0.8)

    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]

    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    # Cross Validation
    tscv = TimeSeriesSplit(n_splits=5)

    rf = RandomForestRegressor(
        random_state=42,
        n_jobs=-1
    )

    param_grid = {
        "n_estimators": [100, 200, 500],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt", 0.8]
    }

    grid = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=tscv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        verbose=0,
        refit=True
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    pred = best_model.predict(X_test)

    return {
        "MAE": mean_absolute_error(y_test, pred),
        "R2": r2_score(y_test, pred),
        "best_params": grid.best_params_,
        "best_model": best_model,
        "X_test": X_test,
        "y_test": y_test
    }





def permutation_report(model, X_test, y_test, n_repeats=20, random_state=42):

    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1
    )

    importance_df = (
        pd.DataFrame({
            "feature": X_test.columns,
            "importance": result.importances_mean,
            "std": result.importances_std
        })
        .sort_values(
            "importance",
            ascending=False
        )
    .reset_index(drop=True)
    )
    
    return importance_df