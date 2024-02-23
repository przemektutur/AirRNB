"""Class GUI."""
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import pandas as pd
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

    def setup_ui(self):
        # Setup UI elements here (similar to the existing setup in main.py)
        pass

    def preprocess_data(self):
        for city, path in self.data_paths.items():
            df = pd.read_csv(path)
            preprocessor = DataPreprocessor(df)
            preprocessor.preprocess()
            self.preprocessed_data[city] = preprocessor.get_data()
            # Train model for each city
            self.train_model(city)

    def train_model(self, city):
        data = self.preprocessed_data[city]
        # Extract features and target variable, then train model
        # Code here to train the model using ModelTrainer
        pass

    def predict_price(self):
        # Use Predictor class to predict price
        pass

    def on_predict_button_click(self):
        # Event handler for predict button click
        pass

    def show_about(self):
        # Show about dialog
        pass

    def show_team(self):
        # Show team info dialog
        pass

    @staticmethod
    def run_gui(data_paths):
        root = tk.Tk()
        app = GUI(root, data_paths)
        root.mainloop()
