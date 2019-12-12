from PIL import Image, ImageFont, ImageDraw
import os
import struct
import random

def getBackGroud():
	id = random.randint(1,63)
	im = Image.open(str(id)+".jpg",'r')
	maxX,maxY = im.size
	xEnd = maxX-48
	yEnd = maxY-48
	xStart = random.randint(0,xEnd)
	yStart = random.randint(0,yEnd)
	return im.crop((xStart,yStart,xStart+48,yStart+48))

def createOne48x48Ch(ftrain, flable, frgbtrain, frgblable, ch, chlable, num, font_size):
	im = Image.new("1", (48, 48), (255))
	imRGB = getBackGroud()
	
	dr = ImageDraw.Draw(im)
	drRGB = ImageDraw.Draw(imRGB)
	
	styleId = random.randint(0,styleNum-1)
	font = ImageFont.truetype(os.path.join("fonts", styles[styleId]), font_size)
	#32大小时可以选择生成的x，y范围：x=0~15(0~48-32)，y=-8~11(-32/4-48-32-32/4)
	xp = random.randint(0,48-font_size) #随机x
	yp = random.randint(-font_size/4,48-font_size-font_size/4) #随机y
	#xp = 4 #固定 x
	#yp = -2 #固定 y
	
	dr.text((xp, yp), ch, font=font, fill="#000000")
	drRGB.text((xp, yp), ch, font=font, fill="#000000")
	#im.show()
	numstr = str(num)
	pad = len("10000") - len(numstr) 
	realNumStr = ""
	while pad > 0:
		realNumStr += "0"
		pad-=1
	realNumStr += numstr
	
	#im.save("picture/"+realNumStr+"_"+ch+".png")
	#imRGB.save("picture/"+realNumStr+"_"+ch+"RGB.png")
	
	image_array = im.load()
	array_rgb = imRGB.load()
	
	x=0
	y=0
	while y<48:
		x=0
		while x<48:
			ftrain.write(struct.pack("B",image_array[x,y]))
			if 255 == image_array[x,y]:
				frgblable.write(struct.pack("B",1))
				frgblable.write(struct.pack("B",0))
			else:
				frgblable.write(struct.pack("B",0))
				frgblable.write(struct.pack("B",1))
			x+=1
		y+=1
	flable.write(struct.pack("B",chlable))
	
	x=0
	y=0
	while y<48:
		x=0
		while x<48:
			frgbtrain.write(struct.pack("B",array_rgb[x,y][0]))
			frgbtrain.write(struct.pack("B",array_rgb[x,y][1]))
			frgbtrain.write(struct.pack("B",array_rgb[x,y][2]))
			x+=1
		y+=1
	return
		
def create48x48Data(text,chNum,imagePerCh,dataFileName,dataLableName):
	chNumCurrentCount = []
	choice = []
	i = 0
	while i < chNum:
		chNumCurrentCount.append(0)
		choice.append(i)
		i+=1
		
	frgbtrain = open("rgb_"+dataFileName,"wb")
	frgbtrain.write(struct.pack("!I",2051)) #magic num
	frgbtrain.write(struct.pack("!I",imagePerCh*chNum)) #image num
	frgbtrain.write(struct.pack("!I",48)) #h
	frgbtrain.write(struct.pack("!I",48)) #w
	frgbtrain.write(struct.pack("!I",3)) #colord
	
	frgblable = open("rgb_"+dataLableName,"wb")
	frgblable.write(struct.pack("!I",2051)) #magic num
	frgblable.write(struct.pack("!I",imagePerCh*chNum)) #image num
	frgblable.write(struct.pack("!I",48)) #h
	frgblable.write(struct.pack("!I",48)) #w
	frgblable.write(struct.pack("!I",2)) #colord
	
	ftrain = open(dataFileName,"wb")
	#需要字节序转换，加上'!'
	ftrain.write(struct.pack("!I",2051)) #magic num
	ftrain.write(struct.pack("!I",imagePerCh*chNum)) #image num
	ftrain.write(struct.pack("!I",48)) #h
	ftrain.write(struct.pack("!I",48)) #w
	ftrain.write(struct.pack("!I",1)) #colord

	flable = open(dataLableName,"wb")
	flable.write(struct.pack("!I",2049)) #magic num
	flable.write(struct.pack("!I",imagePerCh*chNum)) #image num
	count = 0
	totale = chNum*imagePerCh
	while count < totale:
		#size = random.randint(32,40) #随机字大小
		size = 40 #固定字体
		textlable = random.choice(choice)
		if textlable > 100:
			print "error:",choice
		chNumCurrentCount[textlable] += 1
		if chNumCurrentCount[textlable] == imagePerCh:
			choice.remove(textlable)
		createOne48x48Ch(ftrain,flable,frgbtrain,frgblable,text[textlable],textlable,count,size)
		count+=1
		if count % 100 == 0:
			print str(count)

	ftrain.close()
	flable.close()
	frgbtrain.close()
	frgblable.close()
		
#text = u"赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜戚邹喻柏水窦章云苏潘葛奚范彭郎鲁韦昌马苗凤花方俞任袁柳酆鲍史唐费廉岑薛雷贺倪汤滕殷罗毕郝邬安常乐于时傅皮卞齐康伍余元卜顾孟平黄和穆萧"
text = u"赵钱孙李周吴郑王冯陈"
styles = ["FZSTK.TTF", #方正舒体
"FZYTK.TTF", #方正姚体
"simfang.ttf", #仿宋
"simhei.ttf", #黑体
"STCAIYUN.TTF", #华文彩云
"STFANGSO.TTF", #华文仿宋
"STXINGKA.TTF", #华文行楷
"STHUPO.TTF", #华文琥珀
"STKAITI.TTF", #华文楷体
"STLITI.TTF", #华文隶书
"STSONG.TTF", #华文宋体
"STXIHEI.TTF", #华文细黑
"STXINWEI.TTF", #华文新魏
"STZHONGS.TTF", #华文中宋
"simkai.ttf", #楷体
"SIMLI.TTF", #隶书
"simsun.ttc", #宋体
"SIMYOU.TTF", #圆幼
] 
styleNum = len(styles)
chNum = len(text)
imagePerCh = 100

create48x48Data(text,chNum,imagePerCh,"t10k-images.idx3-ubyte","t10k-labels.idx1-ubyte")
create48x48Data(text,chNum,imagePerCh*10,"train-images.idx3-ubyte","train-labels.idx1-ubyte")





