"""Class GUI."""
from typing import (
    Dict,
    Optional,
)
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

from .data_preprocessor import DataPreprocessor
from .model_trainer import ModelTrainer
from .predictor import Predictor
from .utilities import Utilities


class GUI:
     """
    A GUI application for predicting rental prices based on various features.

    Attributes:
    -----------
    root (tk.Tk): The root Tkinter window.
    data_paths (Dict[str, str]): Dictionary mapping city names to paths.
    utilities (Utilities): Instance of the Utilities class operations.
    selected_data_var (tk.StringVar): Tkinter string variable.
    input_entries (Dict[str, tk.Entry]): Dictionary of Tkinter entry widgets.
    preprocessed_data (Dict[str, pd.DataFrame]): Preprocessed datasets.
    models (Dict[str, XGBRegressor]): Trained models for each city.
    """

    def __init__(self, root: tk.Tk, data_paths: Dict[str, str]) -> None:
                """
        Initializes the GUI application.

        Parameters:
        -----------
        root (tk.Tk): The root Tkinter window.
        data_paths (Dict[str, str]): A dictionary mapping city names to
            dataset paths.

        Returns:
        --------
        None
        """
        self.root = root
        self.data_paths = data_paths
        self.utilities = Utilities()
        self.selected_data_var = tk.StringVar(self.root)
        self.input_entries = {}
        self.preprocessed_data: dict[str, pd.DataFrame] = {}
        self.models: dict[str, XGBRegressor] = {}

        # Initiate methods
        self.setup_ui()
        self.preprocess_data()
        self.plot_data()

    def setup_ui(self) -> None:
        """
        Sets up the user interface for the application.

        This method initializes the main window and its widgets, including
        menus, labels, entries, and buttons for interacting with the
        application.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.root.title("Predictor")
        self.root.geometry("300x450")

        # Menu
        help_menu: tk.Menu = tk.Menu(self.root)
        self.root.config(menu=help_menu)

        # Create the "Legend" submenu
        legend_submenu: tk.Menu = tk.Menu(help_menu, tearoff=0)
        help_menu.add_cascade(label="Legend", menu=legend_submenu)
        legend_submenu.add_command(label="Help", command=self.show_about)

        # Create the "Team" submenu
        team_submenu: tk.Menu = tk.Menu(help_menu, tearoff=0)
        help_menu.add_cascade(label="Team", menu=team_submenu)
        team_submenu.add_command(label="About", command=self.show_team)

        # Dropdown menu for selecting dataset
        data_dropdown_label: tk.Label = tk.Label(
            self.root,
            text="Select City:"
        )
        data_dropdown_label.pack(pady=5)

        self.data_dropdown: ttk.Combobox = ttk.Combobox(
            self.root,
            textvariable=self.selected_data_var,
            values=list(self.data_paths.keys())
        )
        self.data_dropdown.pack(pady=5)

        # Labels and Entry Widgets for Selected Features
        selected_features: list = [
            "accommodates",
            "room_type_encoded",
            "bathrooms",
            "address",
        ]
        for feature in selected_features:
            label: tk.Label = tk.Label(
                self.root,
                text=feature.capitalize() + ":"
            )
            label.pack(pady=5)
            entry: tk.Entry = tk.Entry(self.root)
            entry.pack(pady=5)
            self.input_entries[feature] = entry

        # Prediction Button
        predict_button: tk.Button = tk.Button(
            self.root,
            text="Predict Price",
            command=self.on_predict_button_click
        )
        predict_button.pack(pady=10)

        # Prediction Result Label
        self.prediction_label: tk.Label = tk.Label(self.root, text="")
        self.prediction_label.pack(pady=10)

    def preprocess_data(self) -> None:
        """
        Preprocesses the data for each city specified in `data_paths`.

        It loads the data from the provided paths, applies preprocessing steps,
        and stores the preprocessed data in `self.preprocessed_data`. It also
        initiates model training for each city"s dataset.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        for city, path in self.data_paths.items():
            df = pd.read_csv(path)
            preprocessor = DataPreprocessor(df)
            preprocessor.preprocess()
            self.preprocessed_data[city] = preprocessor.get_data()
            # Train model for each city
            self.train_model(city)

    def train_model(self, city: str) -> None:
        """
        Trains a machine learning model for the specified city.

        Extracts the preprocessed dataset for the city from
        `self.preprocessed_data`, identifies features and target
        variable, trains the model using the ModelTrainer class,
        and stores the trained model in `self.models`.

        Parameters
        ----------
        city : str The name of the city for which to train the model.

        Returns
        -------
        None
        """
        # Extract the preprocessed dataset for the specified city
        data: pd.DataFrame = self.preprocessed_data[city]

        data["has_availability"] = data["has_availability"].map(
            {True: 1, False: 0}
        )

        features: list[str] = [
            "latitude",
            "longitude",
            "accommodates",
            "room_type_encoded",
            "bathrooms",
            "review_scores_rating",
            "availability_365",
            "has_availability",
            "beds",
            "neighbourhood_cleansed_encoded",
            "availability_30",
            "review_scores_cleanliness",
            "reviews_per_month",
            "calculated_host_listings_count_entire_homes",
            "number_of_reviews_ltm",
            "availability_90"
        ]
        target: str = "price"

        data[target] = data[target].replace([np.inf, -np.inf], np.nan)
        data.dropna(subset=[target], inplace=True)

        # Splitting the data into features (X) and target (y)
        X: pd.DataFrame = data[features]
        y: pd.Series = data[target]
        # Initializing the ModelTrainer with the dataset
        trainer: ModelTrainer = ModelTrainer(X, y)
        # Training the model
        model: XGBRegressor = trainer.train_model()
        # Storing the trained model for later use
        self.models[city] = model

    def predict_price(self) -> None:
        """
        Predicts the rental price based on user inputs and the selected city.

        Gathers input features from the GUI, converts them to the correct
        format, and ensures that the feature order matches exactly what
        was used during training.

        Parameters:
        -----------
        None

        Returns:
        --------
        None
        """
        city = self.selected_data_var.get()
        if city not in self.models:
            messagebox.showerror("Error", "Model for the city is not available.")
            return

        feature_order = [
            "latitude",
            "longitude",
            "accommodates",
            "room_type_encoded",
            "bathrooms",
            "review_scores_rating",
            "availability_365",
            "has_availability",
            "beds",
            "neighbourhood_cleansed_encoded",
            "availability_30",
            "review_scores_cleanliness",
            "reviews_per_month",
            "calculated_host_listings_count_entire_homes",
            "number_of_reviews_ltm",
            "availability_90"
        ]

        # Prepare a dictionary with the current input values or their defaults
        input_values = {feature: self.get_feature_value(
            feature, city
        ) for feature in feature_order}

        # Convert dictionary to DataFrame with columns in the specified order
        input_df = pd.DataFrame([input_values], columns=feature_order)

        try:
            model = self.models[city]
            prediction = model.predict(input_df)
            predicted_price = prediction[0]
            text = f"Predicted Price: ${predicted_price:.2f}"
            self.prediction_label.config(text=text)
        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))

    def get_feature_value(self, feature: str, city: str) -> Optional[float]:
        """
        Returns the input value for a given feature. If not provided, returns
        the mean value from the dataset.

        Parameters:
        -----------
        feature (str): The feature for which to get the value.
        city (str): The selected city for which to predict the price.

        Returns:
        --------
        Optional[float]: The input value or mean value for the feature.
        """
        if feature in ["latitude", "longitude"]:
            # Handling "address" conversion to "latitude" and "longitude"
            address = (
                self.input_entries.get("address").get()
                if "address" in self.input_entries
                else None
            )
            coordinates = (
                Utilities.get_coordinates(address)
                if address
                else None
            )
            if coordinates:
                index = 0 if feature == "latitude" else 1
                return coordinates[index]
            else:
                return None
        elif feature in self.input_entries:
            # Return user-provided value, converting to float
            return float(self.input_entries[feature].get())
        else:
            # Return mean value from the preprocessed data for the city
            return self.preprocessed_data[city][feature].mean()

    def on_predict_button_click(self) -> None:
        """
        Handles the "Predict Price" button click event.
        Validates user inputs, calls the `predict_price` method to perform the
        prediction if inputs are valid, and handles any exceptions that occur
        during prediction.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        is_valid_input: bool = True
        for feature, entry in self.input_entries.items():
            if not entry.get().strip():
                msg = f"{feature.capitalize()} is required."
                tk.messagebox.showerror("Input Error", msg)
                is_valid_input = False
                break
        if is_valid_input:
            try:
                self.predict_price()
            except Exception as e:
                # Handle any exceptions that occur during prediction
                tk.messagebox.showerror("Prediction Error", str(e))

    def show_about(self) -> None:
        """
        Displays an informational dialog about the application.
        Shows details about how to use the application, the meaning of
        different room types, and other relevant information to the user.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        about_text: str = (
            "Application predicts rental prices based on features like "
            "location, room type, accommodations, and more. \n\n"
            "Room type encoded:\n"
            "3 - Shared room\n"
            "2 - Private room\n"
            "1 - Hotel room\n"
            "0 - Entire home/apt\n\n"
            "Select a city, input the form, and click \"Predict Price\" "
            "to get the estimated rental price."
        )
        messagebox.showinfo("About", about_text)

    def plot_data(self) -> None:
        """
        Plots the data for up to three cities to visualize geographical price
        distribution.

        Parameters:
        -----------
        None

        Returns:
        --------
        None
        """
        cities = list(self.preprocessed_data.keys())[:3]
        if len(cities) == 3:
            data = [self.preprocessed_data[city] for city in cities]
            titles = [f"Dane dla {city}" for city in cities]

            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            for ax, d, title in zip(axs, data, titles):
                scatter = ax.scatter(
                    d['longitude'],
                    d['latitude'],
                    c=d['price'],
                    cmap='viridis',
                    vmax=200,
                    vmin=20
                )
                ax.set_title(title)
                ax.set_xlabel('Długość geograficzna')
                ax.set_ylabel('Szerokość geograficzna')
                plt.colorbar(scatter, ax=ax, label='Cena')
            plt.tight_layout()
            plt.show()


    def show_team(self) -> None:
        """
        Give information about developers team.

        Parameters:
        -----------
        None

        Returns:
        --------
        None
        """
        messagebox.showinfo("Team:", "Przemyslaw Tutur")

    @staticmethod
    def run_gui(data_paths):
        """
        Initializes and runs the GUI application.

        Parameters:
        -----------
        data_paths (Dict[str, str]): Dict mapping city names to dataset paths.

        Returns:
        --------
        None
        """
        root = tk.Tk()
        app = GUI(root, data_paths)
        root.mainloop()
