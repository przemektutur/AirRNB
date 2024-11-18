"""Main module."""
from src.gui import GUI
from src.data_downloader import DataDownloader

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
    # Class initialization 'DataDownloader' and data download
    downloader = DataDownloader(data_paths)
    downloaded_data = downloader.download_data()

    # Check and write data to temporary files (if needed)
    for city, content in downloaded_data.items():
        if content:
            with open(f"data/{city.lower()}.csv", "wb") as f:
                f.write(content)
        else:
            print(f"Unable to download and save data for {city}.")

    # Run gui with data paths
    GUI.run_gui(data_paths)

