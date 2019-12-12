#encoding: utf-8

import unittest
from create_mnist_image import *

class TestCreateMnistImage(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None

    def testCreateNoneColorCh(self):
        im = Image.new("1", (48, 48), (255))
        createNoneColorCh(im, u"冯", "FZSTK.TTF", 32, 5, 5)
        im.save("testData/current.png")
        currentIm = Image.open("testData/current.png","r")
        exceptIm = Image.open("testData/picture.png","r")
        width,height = currentIm.size
        exceptWidth,exceptHeight = exceptIm.size
        self.assertEqual(width,exceptWidth)
        self.assertEqual(height,exceptHeight)
        currentArray = currentIm.load()
        exceptArray = exceptIm.load()
        x = 0
        y = 0
        while y < height:
            x = 0
            while x < width:
                self.assertEqual(exceptArray[x,y],currentArray[x,y])
                x+=1
            y+=1

if __name__ =='__main__':  
    unittest.main()  