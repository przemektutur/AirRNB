"""Class Predictor."""
from typing import Any

import pandas as pd
import numpy as np


class Predictor:
    """Class Predictor."""

    def __init__(self, model: Any) -> None:
        """
        Initialize the Predictor with a model.

        Parameters
        ----------
        model : Any
            The predictive model to use for making predictions. 
            The type is kept generic (`Any`) since different libraries 
            have different model types, and they don"t share a base class.

        Returns:
        --------
        None
        """
        self.model = model

    def predict(self, input_df: pd.DataFrame) -> np.ndarray:
        """
        Make a prediction using the provided model.

        Parameters
        ----------
        input_df : pd.DataFrame
            A pandas DataFrame containing the input features for making 
            the prediction.

        Returns
        -------
        np.ndarray
            The prediction made by the model. The exact shape and contents 
            of this array will depend on the model"s type and the input data.
        """
        prediction = self.model.predict(input_df)

        return prediction
