#encoding: utf-8
import sys,getopt
import string
import os

from PIL import Image,ImageDraw

def getRGBHistogram(im):
    rPixes = []
    gPixes = []
    bPixes = []
    pixes = im.histogram()
    for i in range(256):
        rPixes.append(pixes[i])
        gPixes.append(pixes[i+256])
        bPixes.append(pixes[i+256+256])
    return rPixes,gPixes,bPixes

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
        if count > 1000:
            return 0

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
    maxPix = max(data)
    histogram = Image.new('RGB',(256,256),(255,255,255))
    draw = ImageDraw.Draw(histogram)
    for i in range(len(data)):
        current = data[i]*200/maxPix
        source = (i, 255)
        target = (i, 255-current)
        if threshold == i:
            draw.line([source,target],(255,0,0))
        else:
            draw.line([source,target],(0,0,255))
    histogram.show()
    return

def removeEdge(im):
    x = 0
    y = 0
    maxX,maxY = im.size
    data = im.load()
    while y < maxY:
        x = 0
        while x < maxX:
            if x < 10 or x > maxX-10:
                im.putpixel((x,y),(255,255,255))
            x+=1
        y+=1

def avgBinaryzation(im,threshold):
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
                if data[x,y][0] < threshold and data[x,y][0] > threshold - colorRange and data[x,y][1] < threshold and data[x,y][1] > threshold - colorRange and data[x,y][2] < threshold and data[x,y][2] > threshold - colorRange:
                    im.putpixel((x,y),(0,0,0))
                    count+=1
                else:
                    im.putpixel((x,y),(255,255,255))
            elif data[x,y][0] < threshold and data[x,y][1] < threshold and data[x,y][2] < threshold:
                im.putpixel((x,y),(0,0,0))
                count+=1
            else:
                im.putpixel((x,y),(255,255,255))
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

def binaryzation(im,rThreshold,gThreshold,bThreshold):
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
                if data[x,y][0] < rThreshold and data[x,y][0] > rThreshold - colorRange and data[x,y][1] < gThreshold and data[x,y][1] > gThreshold - colorRange and data[x,y][2] < bThreshold and data[x,y][2] > bThreshold - colorRange:
                    im.putpixel((x,y),(0,0,0))
                    count+=1
                else:
                    im.putpixel((x,y),(255,255,255))
            elif data[x,y][0] < rThreshold and data[x,y][1] < gThreshold and data[x,y][2] < bThreshold:
                im.putpixel((x,y),(0,0,0))
                count+=1
            else:
                im.putpixel((x,y),(255,255,255))
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

multiply = 10
colorRange = None

def getAvgThreshold(l):
    l.sort()
    if abs(l[0]-l[1]) > abs(l[2]-l[1]):
        return (l[1]+l[2])/2
    else:
        return (l[0]+l[1])/2
    

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

    orgIm = Image.open(image,"r")
    im = orgIm.crop()
    #avgIm = orgIm.crop()
    r,g,b = getRGBHistogram(im)
    rThreshold = getThreshold(r)
    gThreshold = getThreshold(g)
    bThreshold = getThreshold(b)
    #drawHistogram(r,rThreshold)
    #drawHistogram(g,gThreshold)
    #drawHistogram(b,bThreshold)
    #lines = binaryzation(im,rThreshold,gThreshold,bThreshold)
    #divideLines(orgIm, lines)
    
    #lines = avgBinaryzation(im,sum([rThreshold,gThreshold,bThreshold])/3)
    #removeEdge(im)
    threshold = getAvgThreshold([rThreshold,gThreshold,bThreshold])
    lines = avgBinaryzation(im,threshold)
    #lines = avgBinaryzation(im,max([rThreshold,gThreshold,bThreshold]))
    divideLines(orgIm, lines)
    

    orgIm.show()
    im.show()
    #avgIm.show()
    
    

