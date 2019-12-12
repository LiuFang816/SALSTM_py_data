import json

from api.google_maps import GoogleMaps
from api.virtual_radar import VirtualRadar
from util.location_utils import LocationUtils
from util.object_parsing import ObjectParsing
from util.radar_interpreter import RadarInterpreter

DISTANCE_RADIUS = 20, # Distance radius (2-dimensional) max
MAX_ALTITUDE = 40000 # Maximum Altitude

class AirTrafficControl:
    def __init__(self):
        self.virtual_radar = VirtualRadar()
        self.google_maps = GoogleMaps()
        self.location_utils = LocationUtils()
        self.radar_interpreter = RadarInterpreter()
        self.response_builder = ResponseBuilder()

    def _get_location(self, user_location_string):
        """
        Returns a nice little object that has the latitude and
        longitude corresponding to the location from the user.
        """
        location = self.google_maps.location_from_address(user_location_string)
        if location == None:
            return None
        else:
            location = self.location_utils.parse_useful_bits(location)
            return location

    def _get_aircraft(self, user_location_string, debug=False):
        """
        Returns a list of all aircraft found for that location, or
        a null value if none are found or if an IO error.
        """
        location = self._get_location(user_location_string)
        if location == None:
            print('DEBUG: Unable to get user location (' + user_location_string + ')')
            return None
        radar_results = self.virtual_radar.get_radar(location['lat'], location['lon'], DISTANCE_RADIUS, MAX_ALTITUDE)
        if debug is True:
            print(json.dumps(radar_results, indent=2))
        if radar_results == None:
            print('DEBUG: Unable to get radar scan from API (' + user_location_string + ')')
            return None
        aircraft = self.radar_interpreter.extract_aircraft(radar_results['acList'])
        if len(aircraft) == 0:
            print('DEBUG: No aircraft overhead (' + location['name'] + ')')
            return None
        return aircraft

    def aircraft_count(self, user_location_string):
        """
        Counts the number of aircraft overhead. Returns a string.
        """
        aircraft = self._get_aircraft(user_location_string)
        if not aircraft:
            return self.response_builder.craft_result_response(None, user_location_string)
        if len(aircraft) == 1:
            return 'There is 1 aircraft over ' + user_location_string + '.'
        else:
            return 'There are ' + str(len(aircraft)) + ' aircraft over ' + user_location_string + '.'

    def aircraft_count_specific(self, user_location_string):
        """
        Note: This is kind of a messy function, but it's easier to read this way.

        Counts the number of each type of aircraft overhead. Returns a string.
        """
        aircraft = self._get_aircraft(user_location_string)
        if not aircraft:
            return self.response_builder.craft_result_response(None, user_location_string)
        landplanes = 0
        seaplanes = 0
        amphibians = 0
        helicopters = 0
        gyrocopters = 0
        tiltwings = 0
        for craft in aircraft:
            if craft['craft_type'] == 1:
                landplanes+=1
            if craft['craft_type'] == 2:
                seaplanes+=1
            if craft['craft_type'] == 3:
                amphibians+=1
            if craft['craft_type'] == 4:
                helicopters+=1
            if craft['craft_type'] == 5:
                gyrocopters+=1
            if craft['craft_type'] == 6:
                tiltwings+=1
        response_string = 'There are '
        if landplanes > 0:
            response_string += str(landplanes) + ' airplanes, '
        if seaplanes > 0:
            response_string += str(seaplanes) + ' seaplanes, '
        if amphibians > 0:
            response_string += str(amphibians) + ' amphibians, '
        if helicopters > 0:
            response_string += str(helicopters) + ' helicopters, '
        if gyrocopters > 0:
            response_string += str(gyrocopters) + ' gyrocopters, '
        if tiltwings > 0:
            response_string += str(tiltwings) + ' tilting-wing aircraft '
        if response_string == 'There are ':
            response_string += 'no recognizable aircraft '
        response_string += 'over ' + user_location_string + '.'
        return response_string

    def aircraft_of_type(self, user_location_string, desired_type):
        """
        1 = Landplane
        2 = Seaplane
        3 = Amphibian
        4 = Helicopter
        5 = Gyrocopter
        6 = Tiltwing
        7 = GroundVehicle
        8 = Tower

        Returns a string.
        """
        aircraft = self._get_aircraft(user_location_string)
        if not aircraft:
            return self.response_builder.craft_result_response(None, user_location_string)
        single_type = {'altitude': 0}
        type_count = 0
        for craft in aircraft:
            if craft['craft_type'] == desired_type:
                if craft['altitude'] > single_type['altitude']:
                    single_type = craft
                type_count+=1
        if type_count == 0:
            return 'There are none over ' + user_location_string + ' right now.'
        if type_count == 1:
            return self.response_builder.craft_result_response(single_type, user_location_string)
        else:
            return 'There are ' + str(type_count) + ' over ' + user_location_string + '.'

    def highest_aircraft(self, user_location_string):
        """
        Gets information about the highest aircraft on radar. Returns a string.
        """
        aircraft = self._get_aircraft(user_location_string)
        if not aircraft:
            return self.response_builder.craft_result_response(None, user_location_string)
        highest_craft = {'altitude': 0}
        for craft in aircraft:
            if craft['altitude'] > highest_craft['altitude']:
                highest_craft = craft
        return self.response_builder.craft_result_response(highest_craft, user_location_string)

    def lowest_aircraft(self, user_location_string):
        """
        Gets information about the lowest aircraft on radar. Returns a string.
        """
        aircraft = self._get_aircraft(user_location_string)
        if not aircraft:
            return self.response_builder.craft_result_response(None, user_location_string)
        lowest_craft = {'altitude': 500000} # Outer space begins at 264,000 feet
        for craft in aircraft:
            if craft['altitude'] < lowest_craft['altitude'] and craft['altitude'] > 0:
                lowest_craft = craft
        return self.response_builder.craft_result_response(lowest_craft, user_location_string)


class ResponseBuilder:
    def __init__(self):
        self.object_parsing = ObjectParsing()

    def craft_result_response(self, aircraft, location):
        """
        This is where we craft our nice natural-language response
        including all of the information we have been able to get.
        """
        if not aircraft:
            return 'There are currently no aircraft on radar near ' + location + '.'
        else:
            response_text = 'There is a'
            if self.object_parsing.get_param(aircraft, 'operator'):
                response_text += ' '
                response_text += aircraft['operator']
            if self.object_parsing.get_param(aircraft, 'manufacturer'):
                response_text += ' '
                response_text += aircraft['manufacturer']
            if self.object_parsing.get_param(aircraft, 'model'):
                response_text += ' '
                response_text += aircraft['model']
            if response_text == 'There is a': # If we got no aircraft information
                response_text = 'There is an aircraft with no public information'
            response_text += ' at '
            response_text += str(aircraft['altitude'])
            response_text += ' feet.'
            return response_text

if __name__ == '__main__':
    print('\n\n\tRunning demonstration...\n')

    atc = AirTrafficControl()
    print(atc.lowest_aircraft('New York'))

    print('\n\tDone running demonstration!\n\n')
