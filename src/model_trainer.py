"""Model trainer class."""
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBRegressor


class ModelTrainer:
    """Class ModelTrainer."""

    def __init__(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Initialize the ModelTrainer with training data and create 
        an XGBoost model.

        Parameters
        ----------
        X: pd.DataFrame - the input features for training the model.
        y: pd.Series - the target variable for training the model.

        Returns
        -------
        None
        """
        self.X = X
        self.y = y
        self.model: XGBRegressor = xgb.XGBRegressor(
            n_estimators=200,
            verbosity=1
        )

    def train_model(self) -> XGBRegressor:
        """
        Train the XGBoost model with the provided training data.

        The method takes no arguments.

        Returns
        -------
        XGBRegressor - the trained XGBoost model.
        """
        self.model.fit(self.X, self.y)
        return self.model
