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
