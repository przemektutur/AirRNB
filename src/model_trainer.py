"""Model trainer class."""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from typing import Any


class ModelTrainer:
    """Class ModelTrainer."""

    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_type: str = "xgb",
    ) -> None:
        """
        Initialize the ModelTrainer with training data and create 
        an XGBoost model.

        Parameters
        ----------
        X: pd.DataFrame - the input features for training the model.
        y: pd.Series - the target variable for training the model.
        model_type: str - the type of model to use ('xgb' for XGBoost or 'rf'
            for Random Forest).

        Returns
        -------
        None
        """
        self.X = X
        self.y = y
        if model_type == "xgb":
            self.model: XGBRegressor = xgb.XGBRegressor(
                n_estimators=200,
                verbosity=1
            )
        elif model_type == "rf":
            self.model: RandomForestRegressor(
                n_estimators=100,
                random_state=42,
            )
        else:
            raise ValueError("Unsupported model type. Use 'xgb' or 'rf'.")

    def train_model(self) -> Any:
        """
        Train the XGBoost model with the provided training data.

        The method takes no arguments.

        Returns
        -------
        Any trained machine learning model (XGBRegressor, RandomForest so far).
        """
        self.model.fit(self.X, self.y)
        return self.model
