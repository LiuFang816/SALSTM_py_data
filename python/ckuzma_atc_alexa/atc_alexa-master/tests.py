from atc import AirTrafficControl

atc_control = AirTrafficControl()

single_location = ['Denver']
locations = [
    'portland',
    'denver',
    'new york',
    'barcelona',
    'cleveland',
    'narnia',
    'this shouldn\'t have a match',
]

# print('\n\t----[ TEST: Print the radar stack ]----')
# for entry in single_location:
#     print(entry)
#     atc_control._get_aircraft(entry, debug=True)
#     print('')

# print('\n\t----[ TEST: Lowest aircraft for location ]----')
# for entry in locations:
#     print(entry)
#     print(atc_control.lowest_aircraft(entry))
#     print('')

# print('\n\t----[ TEST: Highest aircraft for location ]----')
# for entry in locations:
#     print(entry)
#     print(atc_control.highest_aircraft(entry))
#     print('')

# print('\n\t----[ TEST: Number of aircraft for location ]----')
# for entry in locations:
#     print(entry)
#     print(atc_control.aircraft_count(entry))
#     print('')

# print('\n\t----[ TEST: Get number of each type of aircraft ]----')
# for entry in single_location:
#     print(entry)
#     print(atc_control.aircraft_count_specific(entry))
#     print('')

# print('\n\t----[ TEST: Get specific type of aircraft ] ----')
# print('--- Land Planes')
# for entry in locations:
#     print(entry)
#     print(atc_control.aircraft_of_type(entry, 1))
#     print('')
# print('--- Sea Planes')
# for entry in locations:
#     print(entry)
#     print(atc_control.aircraft_of_type(entry, 2))
#     print('')
# print('--- Helicopters')
# for entry in locations:
#     print(entry)
#     print(atc_control.aircraft_of_type(entry, 4))
#     print('')
