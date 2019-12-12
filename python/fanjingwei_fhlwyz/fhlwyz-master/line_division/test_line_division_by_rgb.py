#encoding: utf-8

import unittest

from line_division_by_rgb import *

class TestLineDivisionByRgb(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None

    def  testGetRGBHistogram(self):
        im = Image.new("RGB",(48,48),(255,0,0))
        r,g,b = getRGBHistogram(im)
        self.assertEqual(48*48,r[255])
        self.assertEqual(48*48,g[0])
        self.assertEqual(48*48,b[0])

    def testRealPicture(self):
        im = Image.open("3.png")
        r,g,b = getRGBHistogram(im)
        rThreshold = getThreshold(r)
        gThreshold = getThreshold(g)
        bThreshold = getThreshold(b)
        drawHistogram(r,rThreshold)
        drawHistogram(g,gThreshold)
        drawHistogram(b,bThreshold)

    


if __name__ =='__main__':  
    unittest.main()  
