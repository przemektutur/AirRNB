"""Main module."""
from src.gui import GUI


data_paths = {
    "Barcelona": "http://data.insideairbnb.com/spain/catalonia/barcelona/2023-12-13/data/listings.csv.gz",
    "London": "http://data.insideairbnb.com/united-kingdom/england/london/2023-12-10/data/listings.csv.gz",
    "Athens": "http://data.insideairbnb.com/greece/attica/athens/2023-12-25/data/listings.csv.gz"
}

if __name__ == "__main__":
    GUI.run_gui(data_paths)
