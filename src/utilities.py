"""Utilities for program."""
from typing import Optional, Tuple

from geopy.geocoders import Nominatim


class Utilities:
    """Utilities class."""

    @staticmethod
    def get_coordinates(address: str) -> Optional[Tuple[float, float]]:
        """Get all child Network Blocks of given parent ID.

        Parameters
        ----------
        address: API Client connector

        Returns
        -------
        Optional[Tuple[float, float]] - a tuple containing the latitude and 
            longitude of the given address, or None if the address cannot
            be geolocated.
        """

        geolocator = Nominatim(user_agent="geoapiExercises")
        location = geolocator.geocode(address)
        
        if location:
            return (location.latitude, location.longitude)
        else:
            return None
