"""Main module."""
from src.gui import GUI


prefix = "http://data.insideairbnb.com/"
data_paths = {
    "Barcelona": (
        prefix + "spain/catalonia/barcelona/2023-12-13/"
        "data/listings.csv.gz"
    ),
    "London": (
        prefix + "united-kingdom/england/london/2023-12-10/"
        "data/listings.csv.gz"
    ),
    "Athens": (
        prefix + "greece/attica/athens/2023-12-25/"
        "data/listings.csv.gz"
    )
}

if __name__ == "__main__":
    GUI.run_gui(data_paths)
