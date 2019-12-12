from util.object_parsing import ObjectParsing

class RadarInterpreter:
    def __init__(self):
        self.object_parsing = ObjectParsing()

    def extract_aircraft(self, raw_aircrafts):
        """
        We're only grabbing information that's useful (for now) but
        there is more that we could get and add some fun logic to in
        the future. Such as:

            Year    When the aircraft was built
            Reg     Tail number on the aircraft
            GAlt    Altitude at sea level
            Engines Number of engines on the aircraft
        """
        aircraft_list = []
        for aircraft in raw_aircrafts:
            aircraft_info = {
                'manufacturer': self.object_parsing.get_param(aircraft, 'Man'),
                'model': self.object_parsing.get_param(aircraft, 'Type'),
                'craft_type': self.object_parsing.get_param(aircraft, 'Species'),
                'operator': self.object_parsing.get_param(aircraft, 'Op'),
                'flight_num': self.object_parsing.get_param(aircraft, 'Call'),
                'from': self.object_parsing.get_param(aircraft, 'From'),
                'to': self.object_parsing.get_param(aircraft, 'To'),
                'altitude': self.object_parsing.get_param(aircraft, 'Alt')
            }
            if aircraft_info['operator']: # Have to clean this up a bit
                operator = aircraft_info['operator']
                operator = operator.split(' ')
                aircraft_info['operator'] = ' '.join(operator[0:2]).lower()
            aircraft_list.append(aircraft_info)
        return aircraft_list
