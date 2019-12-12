#!/usr/bin/env python2

import pymacaroons
import sys

if len(sys.argv) == 1:
    sys.stderr.write("usage: %s macaroon [key]\n" % (sys.argv[0],))
    sys.exit(1)

macaroon_string = sys.argv[1]
key = sys.argv[2] if len(sys.argv) > 2 else None

macaroon = pymacaroons.Macaroon.deserialize(macaroon_string)
print macaroon.inspect()

print ""

verifier = pymacaroons.Verifier()
verifier.satisfy_general(lambda c: True)
try:
    verifier.verify(macaroon, key)
    print "Signature is correct"
except Exception as e:
    print e.message
