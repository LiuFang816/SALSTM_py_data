'''
Created on Nov 27, 2011

@author: ppa
'''
import unittest
from ultrafinance.dam.googleDAM import GoogleDAM

class testGoogleDam(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testReadQuotes(self):
        dam = GoogleDAM()
        dam.setSymbol('NASDAQ:EBAY')
        data = dam.readQuotes('20131101', '20131110')
        print([str(q) for q in data])
        self.assertNotEqual(0, len(data))

    def testReadTicks(self):
        dam = GoogleDAM()
        dam.setSymbol('EBAY')
        data = dam.readTicks('20111120', '20111201')
        print(data)
        self.assertNotEqual(0, len(data))

    def testReadFundamental(self):
        dam = GoogleDAM()
        dam.setSymbol('EBAY')
        keyTimeValueDict = dam.readFundamental()
        print(keyTimeValueDict)
        self.assertNotEqual(0, len(keyTimeValueDict))
