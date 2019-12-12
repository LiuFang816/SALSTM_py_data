import requests

RADAR_URL = 'http://public-api.adsbexchange.com/VirtualRadar/AircraftList.json'

class VirtualRadar:
    def get_radar(self, lat, lon, dist_radius, max_alt):
        parameters = {
            'lat': lat,
            'lng': lon,
            'fDstU': dist_radius, # Distance in kilometers
            'fAltU': max_alt # Altitude in feet
        }
        try:
            return requests.get(RADAR_URL, params=parameters, timeout=2).json()
        except:
            print('ERROR: Unable to communicate with Radar API')
            return None
