# -*- coding: utf-8 -*-

import requests

from .exceptions import *

def get(url):
    try:
        r = requests.get(url)
        if r.ok:
            encoding = requests.utils.get_encodings_from_content(r.text)
            r.encoding = encoding[0] if encoding else requests.utils.get_encoding_from_headers(r.headers)
            return r.text
        else:
            raise RequestException(r.status_code)
    except requests.RequestException as e:
        raise RequestException(e)
