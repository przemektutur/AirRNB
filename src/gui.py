"""Class GUI."""
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import pandas as pd
import numpy as np
from xgboost import XGBRegressor

from .data_preprocessor import DataPreprocessor
from .model_trainer import ModelTrainer
from .predictor import Predictor
from .utilities import Utilities


class GUI:
    """Class GUI to handle graphical user interface."""

    def __init__(self, root: tk.Tk, data_paths: dict):
        self.root = root
        self.data_paths = data_paths
        self.utilities = Utilities()
        self.preprocessed_data = {}
        self.models = {}
        self.setup_ui()
        self.preprocess_data()

    def setup_ui(self) -> None:
        self.root.title("Predictor")
        self.root.geometry("300x450")

        # Menu
        help_menu: tk.Menu = tk.Menu(self.root)
        self.root.config(menu=help_menu)

        # Create the 'Legend' submenu
        legend_submenu: tk.Menu = tk.Menu(help_menu, tearoff=0)
        help_menu.add_cascade(label="Legend", menu=legend_submenu)
        legend_submenu.add_command(label="Help", command=self.show_about)

        # Create the 'Team' submenu
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
            'accommodates', 
            'room_type_encoded', 
            'bathrooms', 
            'address',
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

    def preprocess_data(self):
        for city, path in self.data_paths.items():
            df = pd.read_csv(path)
            preprocessor = DataPreprocessor(df)
            preprocessor.preprocess()
            self.preprocessed_data[city] = preprocessor.get_data()
            # Train model for each city
            self.train_model(city)

    def train_model(self, city: str) -> None:
        """
        Trains a model for a specific city using preprocessed data.

        Parameters
        ----------
        city : str
            The name of the city for which to train the model.
        """
        # Extract the preprocessed dataset for the specified city
        data: pd.DataFrame = self.preprocessed_data[city]
        
        features: list[str] = [
            'latitude',
            'longitude',
            'accommodates',
            'room_type_encoded',
            'bathrooms',
            'review_scores_rating',
            'availability_365',
            'has_availability',
            'beds',
            'neighbourhood_cleansed_encoded',
            'availability_30',
            'review_scores_cleanliness',
            'reviews_per_month',
            'calculated_host_listings_count_entire_homes',
            'number_of_reviews_ltm',
            'availability_90'
        ]  
        target: str = 'price'  
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
        """
        city: str = self.selected_data_var.get()
        if city not in self.models:
            messagebox.showerror(
                "Error",
                "Model for the city is not available."
            )
            return
        
        # Gather input features from the GUI
        input_features: dict = {
            feature: entry.get()
            for feature, entry in self.input_entries.items()
        }
        
        # Convert input features to DataFrame for prediction
        try:
            input_df: pd.DataFrame = pd.DataFrame([input_features])
            input_df = input_df.apply(pd.to_numeric, errors='coerce')
        except Exception as e:
            messagebox.showerror("Error", f"Invalid input: {e}")
            return
        
        # Use the Predictor class for prediction
        model = self.models[city]
        predictor: Predictor = Predictor(model)
        prediction: np.ndarray = predictor.predict(input_df)
        
        predicted_price: float = prediction[0] if prediction.size > 0 else 0
        text = (f"Predicted Price: ${predicted_price:.2f}")
        self.prediction_label.config(text=text)


    def on_predict_button_click(self) -> None:
        """
        Handles the event triggered by clicking the 'Predict Price' button.
        This method validates the inputs and calls the predict_price method
        to perform the prediction.
        """
        is_valid_input: bool = True
        for feature, entry in self.input_entries.items():
            if not entry.get().strip():
                msg = f"{feature.capitalize()} is required."
                tk.messagebox.showerror("Input Error", msh)
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
        """
        about_text: str = (
            "Application predicts rental prices based on features like "
            "location, room type, accommodations, and more. \n\n"
            "Room type encoded:\n"
            "3 - Shared room\n"
            "2 - Private room\n"
            "1 - Hotel room\n"
            "0 - Entire home/apt\n\n"
            "Select a city, input the required features, and click 'Predict Price' "
            "to get the estimated rental price."
        )
        messagebox.showinfo("About", about_text)

    @staticmethod
    def run_gui(data_paths):
        root = tk.Tk()
        app = GUI(root, data_paths)
        root.mainloop()
