#coding=utf-8

def divisionBy(line,separator):
	parts = []
	endPos = line.find(separator)
	while -1 != endPos:
		parts.append(line[0:endPos])
		line = line[endPos+1:len(line)]
		endPos = line.find(separator)
	parts.append(line)
	return parts

class Line:
	def __init__(self):
		self.chs = []
		self.LeftTopY = 0
		return

	def appendHeadLineChs(self, chs):
		minYIndex = 0
		current = 0
		minY = chs[current].getLeftTopY()
		while current < len(chs):
			if chs[current].getLeftTopY() < minY:
				minY = chs[current].getLeftTopY()
				minYIndex = current
			current+=1

		self.LeftTopY = minY
		while minYIndex < len(chs):
			if minY == chs[minYIndex].getLeftTopY():
				self.chs.append(chs.pop(minYIndex))
			else:
				break
		return chs

	def getLeftTopY(self):
		return self.LeftTopY

class ChInfo:
	def __init__(self, string):
		self.string = string
		self.id, self.chType, self.leftTopX, self.leftTopY, \
		self.rightBottomX, self.rightBottomY, self.chId \
		= divisionBy(string, "_")

	def isSameCh(self, other):
		if self.chType == other.chType and self.chId == other.chId:
			return True
		else:
			return False

	def isSamePlace(self, other):
		if True == self.isInRange(other.leftTopX,other.leftTopY):
			return True
		if True == self.isInRange(other.rightBottomX,other.leftTopY):
			return True
		if True == self.isInRange(other.rightBottomX,other.rightBottomY):
			return True
		if True == self.isInRange(other.leftTopX,other.rightBottomY):
			return True
		return False

	def isInRange(self,x,y):
		if x >= self.leftTopX and x <= self.rightBottomX and y >= self.leftTopY and y <= self.rightBottomY:
			return True
		else:
			return False

	def getLeftTopY(self):
		return int(self.leftTopY)

def loadResult(fileName):
	f = open(fileName, "r")
	lines = f.readlines()
	f.close()
	records = []
	for line in lines:
		chs = []
		parts = divisionBy(line, ",")
		for part in parts:
			chs.append(ChInfo(part))
		records.append(chs)
	return records

def loadLabels(fileName):
	f = open(fileName, "r")
	lines = f.readlines()
	f.close()
	records = []
	for line in lines:
		chs = []
		parts = divisionBy(line, ",")
		pictureName = parts[0]
		parts = parts[1:len(parts)]
		for part in parts:
			chs.append(ChInfo(part))
		records.append((pictureName,chs))
	return records

def createLines(chs):
	lines = []
	while len(chs) > 0:
		line = Line()
		chs = line.appendHeadLineChs(chs)
		lines.append(line)
	return lines

def eval(labels,result):
	return None
