import requests

from config import Credentials
credentials = Credentials()

API_KEY = credentials.gmap_key
GMAPS_URI = 'https://maps.googleapis.com/maps/api/geocode/json'

class GoogleMaps:
    def __init__(self):
        pass

    def location_from_address(self, zipcode):
        parameters = {
            'address': zipcode,
            'key': API_KEY
        }
        try:
            return requests.get(GMAPS_URI, params=parameters, timeout=2).json()
        except:
            print('ERROR: Unable to communicate with Google API')
            return None
