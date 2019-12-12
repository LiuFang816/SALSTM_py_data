#encoding: utf-8
from PIL import Image, ImageFont, ImageDraw
import os

def GetRectangles(fileName):
	f = open(fileName,"r")
	lines = f.readlines()
	rectangles = []
	for line in lines:
		if -1 == line.find(","):
			continue
		rectangles.append(Rectangle(line))
	return rectangles

def GetEveryValue(line):
	values = []
	endPos = line.find(",")
	while -1 != endPos:
		values.append(int(line[0:endPos]))
		line = line[endPos+1:len(line)]
		endPos = line.find(",")
	values.append(int(line))
	return values

def GetAllBg(fileDir):
    fileList = []
    for parent,dirnames,filenames in os.walk(fileDir):
        for filename in filenames:
            if -1 != filename.find(".BMP") or -1 != filename.find(".jpg"):
                txtName = filename[0:filename.find(".")] + ".txt"
                fileList.append( (filename,txtName) )
    return fileList

class Rectangle:
	def __init__(self, line):
		values = GetEveryValue(line)
		self.leftTopX = values[0]
		self.leftTopY = values[1]
		self.rightTopX = values[2]
		self.rightTopY = values[3]
		self.rightBottomX = values[4]
		self.rightBottomY = values[5]
		self.leftBottomX = values[6]
		self.leftBottomY = values[7]

		self.width = self.rightBottomX - self.leftBottomX
		self.height = self.leftBottomY - self.leftTopY 

	def getLeftTop(self):
		return self.leftTopX, self.leftTopY

	def getWidth(self):
		return self.width

	def getHeight(self):
		return self.height

class image:
	def __init__(self, fileName):
		self.im = Image.open(fileName, 'r')

	def removeWordFromImage(self, rectangles):
		for rectangle in rectangles:
			self.fillRectangle(rectangle)
		return self.im

	def fillRectangle(self,rectangle):
		x,y = rectangle.getLeftTop()
		if x < 0:
			print "image have wrong data"
			x = 0
		if y < 0:
			print "image have wrong data"
			y = 0
		maxX,maxY = self.im.size
		if x > maxX:
			x = maxX
		if y > maxY:
			y = maxY
		width = rectangle.getWidth()
		height = rectangle.getHeight()
		imageValues = self.im.load()
		fill = imageValues[x,y]
		fillX = x
		fillY = y
		while fillY < y + height:
			fillX = x
			while fillX < x + width:
				try:
				    self.im.putpixel((fillX,fillY),fill)
				except:
					if fillY >= maxY:
						self.im.putpixel((fillX,maxY-1),fill)
					print "image have wrong data"
				fillX+=1
			fillY+=1
		return

if __name__ =='__main__':
	path = "testbg/"
	outPath = "realBg/"
	fileList = GetAllBg(path)
	for files in fileList:
		print files
		rectangles = GetRectangles(path+files[1])
		im = image(path+files[0])
		newImage = im.removeWordFromImage(rectangles)
		newImage.save(outPath+files[0])