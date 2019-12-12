"""
This is a simple cheatsheet webapp.
"""
import os

from flask import Flask, send_from_directory
from flask_sslify import SSLify

DIR = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.join(DIR, 'docs', '_build', 'html')

def find_key(token):
    if token == os.environ.get("ACME_TOKEN"):
        return os.environ.get("ACME_KEY")
    for k, v in os.environ.items():
        if v == token and k.startswith("ACME_TOKEN_"):
            n = k.replace("ACME_TOKEN_", "")
            return os.environ.get("ACME_KEY_{}".format(n))


app = Flask(__name__)

if 'DYNO' in os.environ:
    sslify = SSLify(app, skips=['.well-known'])


@app.route('/<path:path>')
def static_proxy(path):
    """Static files proxy"""
    return send_from_directory(ROOT, path)


@app.route('/')
def index_redirection():
    """Redirecting index file"""
    return send_from_directory(ROOT, 'index.html')


@app.route("/.well-known/acme-challenge/<token>")
def acme(token):
    key = find_key(token)
    if key is None: abort(404)
    return key


if __name__ == "__main__":
    app.run(debug=True)
