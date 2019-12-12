#encoding: utf-8

import unittest
from remove_word_from_image import *

class TestRemoveWordFromImage(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None

    def testGetEveryValue(self):
        self.assertEqual([1,2,3,4,5,6,7,8],GetEveryValue("1,2,3,4,5,6,7,8"))

    def testGetAllBg(self):
        fileList = GetAllBg("testbg")
        self.assertEqual(500,len(fileList))
        self.assertEqual(("0001.BMP","0001.txt"),fileList[0])
        self.assertEqual(("image_0150.jpg","image_0150.txt"),fileList[499])

    def testGetRectangles(self):
    	rectangles = GetRectangles("testData/0023.txt")
    	self.assertEqual(5,len(rectangles))
    	self.assertEqual((46,10),rectangles[0].getLeftTop())
    	self.assertEqual(47,rectangles[0].getWidth())
    	self.assertEqual(13,rectangles[0].getHeight())

    	self.assertEqual((134,55),rectangles[len(rectangles)-1].getLeftTop())
    	self.assertEqual(277,rectangles[len(rectangles)-1].getWidth())
    	self.assertEqual(30,rectangles[len(rectangles)-1].getHeight())

    def testRemoveChFromImage(self):
        rectangles = GetRectangles("testData/0023.txt")
        im = image("testData/0023.BMP")
        newImage = im.removeWordFromImage(rectangles)
        newImage.save("testData/current.BMP")

        currentIm = Image.open("testData/current.BMP","r")
        exceptIm = Image.open("testData/except.BMP","r")
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