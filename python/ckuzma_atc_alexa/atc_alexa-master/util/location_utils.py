class LocationUtils:
    def parse_useful_bits(self, location_blob):
        """
        The Google Maps Geocoding API gives us a lot of information, only
        some of which is actually useful. Unfortunately it's also nested,
        which means any extractions will look really messy. This basically
        just gives us a nice little object to work with instead.
        """
        if len(location_blob['results']) == 0:
            return None
        else:
            useful_bits = {
                'name': location_blob['results'][0]['address_components'][0]['long_name'],
                'lat': location_blob['results'][0]['geometry']['location']['lat'],
                'lon': location_blob['results'][0]['geometry']['location']['lng']
            }
            return useful_bits
