# From http://flask.pocoo.org/snippets/8/
from functools import wraps
from flask import request, Response
from elasticpypi.config import config


def check_auth(username, password):
    """This function is called to check if a username /
    password combination is valid.
    """
    return username == config['username'] and password == config['password']


def authenticate():
    """Sends a 401 response that enables basic auth"""
    return Response(
        'Could not verify your access level for that URL.\n'
        'You have to login with proper credentials', 401, {'WWW-Authenticate': 'Basic realm="Login Required"'}
    )


def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return authenticate()
        return f(*args, **kwargs)
    return decorated
