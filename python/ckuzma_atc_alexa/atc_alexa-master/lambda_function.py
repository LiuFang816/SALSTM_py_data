from __future__ import print_function

## Import our classes...
from atc import AirTrafficControl
from config import Credentials

## Initialize our classes...
atc_control = AirTrafficControl()
credentials = Credentials()

# --------------- Helpers that build all of the responses ----------------------

def build_speechlet_response(title, output, reprompt_text, should_end_session):
    return {
        'outputSpeech': {
            'type': 'PlainText',
            'text': output
        },
        'card': {
            'type': 'Simple',
            'title': title,
            'content': output
        },
        'reprompt': {
            'outputSpeech': {
                'type': 'PlainText',
                'text': reprompt_text
            }
        },
        'shouldEndSession': should_end_session
    }


def build_response(session_attributes, speechlet_response):
    return {
        'version': '1.0',
        'sessionAttributes': session_attributes,
        'response': speechlet_response
    }


# --------------- Functions that control the skill's behavior ------------------

def get_welcome_response():
    """ If we wanted to initialize the session to have some attributes we could
    add those here
    """

    session_attributes = {}
    card_title = "Welcome"
    speech_output = "Air Traffic Control. " \
                    "You can ask me what is flying over any location."
    reprompt_text = "Please try again. You can say: " \
                    "Tell me what is flying above New York."
    should_end_session = False
    return build_response(session_attributes, build_speechlet_response(
        card_title, speech_output, reprompt_text, should_end_session))


def handle_session_end_request():
    card_title = "Goodbye"
    speech_output = "Goodbye from Air Traffic Control!"
    # Setting this to true ends the session and exits the skill.
    should_end_session = True
    return build_response({}, build_speechlet_response(
        card_title, speech_output, None, should_end_session))


def get_radar(intent, session):
    card_title = "Get Radar"
    session_attributes = {}
    if 'Location' in intent['slots'] and 'value' in intent['slots']['Location']:
        user_location = intent['slots']['Location']['value']
        speech_output = get_intent_speech(intent, user_location)
        reprompt_text = "You can make a request, or say \"quit\" or \"cancel\" to exit."
        should_end_session = True
    else:
        speech_output = "You did not give a valid location. " \
                        "Please try again."
        reprompt_text = "I did not recognize a location. Try something like: " \
                        "What is flying over New York."
        should_end_session = False
    return build_response(session_attributes, build_speechlet_response(
        card_title, speech_output, reprompt_text, should_end_session))

def get_intent_speech(intent, location):
    if intent['name'] == 'AircraftCountIntent':
        return atc_control.aircraft_count(location)
    if intent['name'] == 'AircraftCountSpecificIntent':
        return atc_control.aircraft_count_specific(location)
    if intent['name'] == 'AircraftOfTypeIntent':
        craft_type = None
        if intent['slots']['CraftType']['value'] == 'airplanes':
            craft_type = 1
        if intent['slots']['CraftType']['value'] == 'seaplanes':
            craft_type = 2
        if intent['slots']['CraftType']['value'] == 'amphibians':
            craft_type = 3
        if intent['slots']['CraftType']['value'] == 'helicopters':
            craft_type = 4
        if intent['slots']['CraftType']['value'] == 'gyrocopters':
            craft_type = 5
        if intent['slots']['CraftType']['value'] == 'tiltwings':
            craft_type = 6
        if intent['slots']['CraftType']['value'] == 'ground vehicles':
            craft_type = 7
        if intent['slots']['CraftType']['value'] == 'towers':
            craft_type = 8
        if craft_type is not None:
            return atc_control.aircraft_of_type(location, craft_type)
        return 'I did not recognize the type of aircraft you were asking about, please try again.'
    if intent['name'] == 'HighestAircraftIntent':
        return atc_control.highest_aircraft(location)
    if intent['name'] == 'LowestAircraftIntent':
        return atc_control.lowest_aircraft(location)


# --------------- Events ------------------

def on_session_started(session_started_request, session):
    """ Called when the session starts """

    print("on_session_started requestId=" + session_started_request['requestId']
          + ", sessionId=" + session['sessionId'])


def on_launch(launch_request, session):
    """ Called when the user launches the skill without specifying what they
    want
    """

    print("on_launch requestId=" + launch_request['requestId'] +
          ", sessionId=" + session['sessionId'])
    # Dispatch to your skill's launch
    return get_welcome_response()


def on_intent(intent_request, session):
    """ Called when the user specifies an intent for this skill """

    print("on_intent requestId=" + intent_request['requestId'] +
          ", sessionId=" + session['sessionId'])

    intent = intent_request['intent']
    intent_name = intent_request['intent']['name']

    # Dispatch to your skill's intent handlers
    custom_intents = [    'AircraftCountIntent',
                        'AircraftCountSpecificIntent',
                        'AircraftOfTypeIntent',
                        'HighestAircraftIntent',
                        'LowestAircraftIntent']
    if intent_name in custom_intents:
        return get_radar(intent, session)
    elif intent_name == "AMAZON.HelpIntent":
        return get_welcome_response()
    elif intent_name == "AMAZON.CancelIntent" or intent_name == "AMAZON.StopIntent":
        return handle_session_end_request()
    else:
        raise ValueError("Invalid intent")


def on_session_ended(session_ended_request, session):
    """ Called when the user ends the session.

    Is not called when the skill returns should_end_session=true
    """
    print("on_session_ended requestId=" + session_ended_request['requestId'] +
          ", sessionId=" + session['sessionId'])


# --------------- Main handler ------------------

def lambda_handler(event, context):
    """ Route the incoming request based on type (LaunchRequest, IntentRequest,
    etc.) The JSON body of the request is provided in the event parameter.
    """
    print("event.session.application.applicationId=" +
          event['session']['application']['applicationId'])

    """
    Uncomment this if statement and populate with your skill's application ID to
    prevent someone else from configuring a skill that sends requests to this
    function.
    """
    if (event['session']['application']['applicationId'] !=
            credentials.alexa_id):
        raise ValueError("Invalid Application ID")

    if event['session']['new']:
        on_session_started({'requestId': event['request']['requestId']},
                           event['session'])

    if event['request']['type'] == "LaunchRequest":
        return on_launch(event['request'], event['session'])
    elif event['request']['type'] == "IntentRequest":
        return on_intent(event['request'], event['session'])
    elif event['request']['type'] == "SessionEndedRequest":
        return on_session_ended(event['request'], event['session'])
