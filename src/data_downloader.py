"""Module for downloading data using multithreading."""
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any


class DataDownloader:
    """Class for downloading data from URLs using multithreading."""

    def __init__(self, urls: Dict[str, str]) -> None:
        """
        Initializes the DataDownloader with the given URLs.

        Parameters
        ----------
        urls : Dict[str, str] - A dictionary where keys are data labels (e.g.
                                and values are URLs pointing to the data files.
        """
        self.urls = urls

    def download_data(self) -> Dict[str, Any]:
        """
        Downloads data from the provided URLs using multithreading.

        Returns
        -------
        Dict[str, Any] - A dictionary where keys are data labels and values are
                         the content of the downloaded data.
        """
        results = {}

        def fetch_data(label: str, url: str) -> Dict[str, Any]:
            """Fetch data from a single URL."""
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                return {label: response.content}
            except requests.RequestException as e:
                print(f"Failed to download {label} from {url}: {e}")
                return {label: None}

        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_label = {
                executor.submit(fetch_data, label, url): label
                for label, url in self.urls.items()
            }

            for future in as_completed(future_to_label):
                label = future_to_label[future]
                try:
                    result = future.result()
                    results.update(result)
                except Exception as exc:
                    print(f"{label} generated an exception: {exc}")

        return results

# Usage
if __name__ == "__main__":
    urls = {
        "London": (
            "http://data.insideairbnb.com/united-kingdom/england/london/"
            "2023-12-10/data/listings.csv.gz"
        ),
        "Athens": (
            "http://data.insideairbnb.com/greece/attica/athens/"
            "2023-12-25/data/listings.csv.gz"
        ),
        "Barcelona": (
            "http://data.insideairbnb.com/spain/catalonia/barcelona/"
            "2023-12-13/data/listings.csv.gz"
        ),
    }
    downloader = DataDownloader(urls)
    data = downloader.download_data()

    for city, content in data.items():
        if content:
            print(f"{city} data downloaded successfully.")
        else:
            print(f"Failed to download data for {city}.")
