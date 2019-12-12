#encoding: utf-8
import sys,getopt
import string
import os

from PIL import Image,ImageDraw

def getPeakIndex(l,endPos=None):
	last = 0
	current = 0
	next = 0
	maxV = max(l)
	if endPos != None:
		rangeNum = endPos
	else:
		rangeNum = len(l)-1
	count=0
	while True:
		for i in range(rangeNum):
			current = l[i]
			next = l[i+1]
			if current > last and current > next and maxV/current < 10+multiply*count:
				return i
			last = current
		count+=1

def getThreshold(pixes):
	pixes.reverse()
	endPos = len(pixes) - getPeakIndex(pixes) - 1
	pixes.reverse()
	startPos = getPeakIndex(pixes,endPos)
	if pixes[startPos] > pixes[endPos]:
		endPos = startPos
		startPos = getPeakIndex(pixes,endPos)
	print(startPos,endPos)
	newPixes = pixes[startPos:endPos]
	return pixes.index(min(newPixes))

def drawHistogram(data,threshold):
	maxPix = max(pixes)
	histogram = Image.new('RGB',(256,256),(255,255,255))
	draw = ImageDraw.Draw(histogram)
	for i in range(len(pixes)):
		current = pixes[i]*200/maxPix
		source = (i, 255)
		target = (i, 255-current)
		if threshold == i:
			draw.line([source,target],(255,0,0))
		else:
			draw.line([source,target],(0,0,255))
	histogram.show()
	return

def binaryzation(im,threshold):
	x = 0
	y = 0
	maxX,maxY = im.size
	data = im.load()
	lines = {}
	while y < maxY:
		x = 0
		count = 0
		while x < maxX:
			#0-black 255-write
			if colorRange != None:
				if data[x,y] < threshold and data[x,y] > threshold - colorRange:
					im.putpixel((x,y),0)
					count+=1
				else:
					im.putpixel((x,y),255)
			elif data[x,y] < threshold:
				im.putpixel((x,y),0)
				count+=1
			else:
				im.putpixel((x,y),255)
			x+=1
		#print(maxX,count)
		if count == 0:
			count = 1
		if maxX/count <= 20:
			lines[y] = 1
		else:
			lines[y] = 0
		y+=1
	return lines

def ifDirtThenRemove(x,y,size,im,data):
	currentX = x
	currentY = y
	totalPixNum = size*size
	blackNum = 0
	while currentY < y + size:
		currentX = x
		while currentX < x + size:
			if data[currentX,currentY] == 0:
				blackNum+=1
			currentX+=1
		currentY+=1

	if blackNum*100/totalPixNum > 90:
		currentX = x
		currentY = y
		while currentY < y + size:
			blackNum = 0
			currentX = x
			while currentX < x + size:
				im.putpixel((currentX,currentY),255)
				currentX+=1
			currentY+=1

def calcContinuous(data,maxX,maxY):
	xContinuous = {}
	yContinuous = {}
	x = 0
	y = 0
	startFlag = False
	continuous = 0
	while y < maxY:
		x = 0
		while x < maxX:
			if data[x,y] == 0 and startFlag == True and x!=maxX-1:
				continuous+=1
			elif data[x,y] == 0 and startFlag == False:
				startFlag = True
				continuous = 1
			elif (data[x,y] == 255 and startFlag == True) or (x==maxX-1 and startFlag == True):
				if True == xContinuous.has_key(continuous):
					xContinuous[continuous]+=1
				else:
					xContinuous[continuous]=1
				startFlag = False
				continuous = 0
			else:
				startFlag = False
				continuous = 0
			x+=1
		y+=1

	x = 0
	y = 0
	startFlag = False
	continuous = 0
	while x < maxX:
		y = 0
		while y < maxY:
			if data[x,y] == 0 and startFlag == True and y!=maxY-1:
				continuous+=1
			elif data[x,y] == 0 and startFlag == False:
				startFlag = True
				continuous = 1
			elif (data[x,y] == 255 and startFlag == True) or (y==maxY-1 and startFlag == True):
				if True == yContinuous.has_key(continuous):
					yContinuous[continuous]+=1
				else:
					yContinuous[continuous]=1
				startFlag = False
				continuous = 0
			else:
				startFlag = False
				continuous = 0
			y+=1
		x+=1
	return xContinuous,yContinuous

def removeByContinuous(continuousX,continuousY,im,data):
	x = 0
	y = 0
	maxX,maxY = im.size
	startFlag = False
	continuous = 0
	while y < maxY:
		x = 0
		while x < maxX:
			if data[x,y] == 0 and startFlag == True and x!=maxX-1:
				continuous+=1
			elif data[x,y] == 0 and startFlag == False:
				startFlag = True
				continuous = 1
			elif (data[x,y] == 255 and startFlag == True) or (x==maxX-1 and startFlag == True):
				if continuous > continuousX:
					current = x
					while current > x - continuous + 1:
						try:
							im.putpixel((current,y),255)
						except:
							print(current,x,continuous,y)
						current-=1
				startFlag = False
				continuous = 0
			else:
				startFlag = False
				continuous = 0
			x+=1
		y+=1

	x = 0
	y = 0
	startFlag = False
	continuous = 0
	while x < maxX:
		y = 0
		while y < maxY:
			if data[x,y] == 0 and startFlag == True and y!=maxY-1:
				continuous+=1
			elif data[x,y] == 0 and startFlag == False:
				startFlag = True
				continuous = 1
			elif (data[x,y] == 255 and startFlag == True) or (y==maxY-1 and startFlag == True):
				if continuous > continuousX:
					while current > y - continuous + 1:
							im.putpixel((x,current),255)
							current-=1
				startFlag = False
				continuous = 0
			else:
				startFlag = False
				continuous = 0
			y+=1
		x+=1

def calcContinuousThreshold(continuous):
	return 1

def removeDirtArea(im):
	maxX,maxY = im.size
	x = 0
	y = 0
	data = im.load()
	xContinuous,yContinuous = calcContinuous(data,maxX,maxY)
	#print(xContinuous)
	#print(yContinuous)
	continuousX = calcContinuousThreshold(xContinuous)
	continuousY = calcContinuousThreshold(xContinuous)
	removeByContinuous(continuousX,continuousY,im,data)

def divideLines(orgIm,lines):
	maxX,maxY = orgIm.size
	lastY = None
	lastFlag = None
	for (y,flag) in lines.items():
		x = 0
		if None == lastY:
			#print("first line")
			lastY = y
			lastFlag = flag
			continue
		elif lastFlag == 0 and flag == 1 :
			#print("find division line")
			lIndex = lastY-5
			if lIndex < 0:
				lIndex = 0
			while x < maxX:
				orgIm.putpixel((x,lIndex),(255,0,0))
				x+=1
			lastY = y
			lastFlag = flag
			continue
		elif lastFlag == 1 and flag == 0 :
			#print("find division line down")
			lIndex = lastY+5
			if lIndex > maxY:
				lIndex = maxY-1
			while x < maxX:
				orgIm.putpixel((x,lIndex),(0,0,255))
				x+=1
			lastY = y
			lastFlag = flag
			continue
		else:
			#print("normal line")
			lastY = y
			lastFlag = flag
			continue

def isCh(data,startX,endX,startY,endY):
	i = startX
	j = startY
	blackCount = 0
	totalCount = areaSize*areaSize
	while i < endX:
		j = startY
		while j < endY:
			try:
				if blackColor == data[i,j]:
					blackCount+=1
			except:
				print(i,j,startX,endX,startY,endY)
			j+=1
		i+=1
	if blackCount*100/totalCount > 50:
		return True
	else:
		return False

def hasAtLeastOneCh(data,maxX,maxY,y):
	if y < areaSize:
		startY = 0
	else:
		startY = y - areaSize
	if y + areaSize > maxY :
		endY = maxY
	else:
		endY = y + areaSize
	currentX = 0
	currentY = startY
	x = areaSize
	while x < maxX - areaSize:
		if True == isCh(data,x-areaSize,x+areaSize,startY,endY):
			return True
		x+=areaSize*2
	return False

def divide(im,orgIm):
	x = 0
	y = 0
	maxX,maxY = im.size
	data = im.load()
	lines = []
	while y < maxY:
		if True == hasAtLeastOneCh(data,maxX,maxY,y):
			lines.append(hasCh)
		else:
			lines.append(notHasCh)
		y+=1

	draw = ImageDraw.Draw(orgIm)
	y = 5
	isStart = False
	while y < maxY-5:
		if lines[y] == hasCh and lines[y-1] == notHasCh and isStart == False:
			isStart = True
			source = (0, y-1)
			target = (maxX, y-1)
			draw.line([source,target],(255,0,0))
		elif lines[y-1] == hasCh and lines[y] == notHasCh and isStart == True:
			isStart = False
			source = (0, y+1)
			target = (maxX, y+1)
			draw.line([source,target],(0,0,255))
		y+=1



if __name__ =='__main__': 
    opts,args = getopt.getopt(sys.argv[1:],"hi:")

    for op,arg in opts:
        if "-h" == op:
            print("参数列表：")
            print("-h:显示帮助")
            print("-i:设置需要分割的图片，必选项，例:-i 1.png")
            sys.exit(0)
        else:
            if "-i" == op:
                image = arg

    try:
        image
    except NameError:
        print("参数列表：")
        print("-h:显示帮助")
        print("-i:设置需要分割的图片，必选项，例:-i 1.png")
        sys.exit(0)

multiply = 10
colorRange = None
areaSize = 4
notHasCh = 0
hasCh = 1
blackColor = 0
whiteColor = 255

orgIm = Image.open(image,"r")
im = orgIm.convert("L")
pixes = im.histogram()
threshold = getThreshold(pixes)
print(threshold)
drawHistogram(pixes,threshold)
lines = binaryzation(im,threshold)
#divideLines(orgIm,lines)
removeDirtArea(im)
divide(im,orgIm)

orgIm.show()
#orgIm.convert("L").show()
im.show()