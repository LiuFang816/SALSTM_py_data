#encoding: utf-8

import unittest
from ocr_tester import *
from eval_result import *

class TestCreateMnistImage(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None

    def testDrawChs(self):
        im = Image.new("L", (48, 48), (255))
        drawChs(im, u"冯", "FZSTK.TTF", "#000000", 32, 5, 5)
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

    def testTransformToHex(self):
        im = Image.open("testData/picture.png","r")
        except_array = im.load()
        exceptHexFile = open("./testData/picture.idx3-ubyte","wb")        
        x=0
        y=0
        while y<48:
            x=0
            while x<48:
                exceptHexFile.write(struct.pack("B",except_array[x,y]))
                x+=1
            y+=1

        currentHexFile = open("./testData/current.idx3-ubyte","wb")
        currentHexFile.write(transformToHex(im))

        exceptHexFile.close()
        currentHexFile.close()

        currentHexFile = open("./testData/current.idx3-ubyte","rb")
        currentData = currentHexFile.read()
        currentHexFile.close()
        exceptHexFile = open("./testData/picture.idx3-ubyte","rb")
        exceptData = exceptHexFile.read()
        exceptHexFile.close()

        self.assertEqual(2304,len(exceptData))
        self.assertEqual(len(exceptData),len(currentData))
        maxLen = len(exceptData)
        current = 0
        while current < maxLen:
            self.assertEqual(exceptData[current],currentData[current])
            current+=1

    def testDrawChsAndCreateLable(self):
        im = Image.new("L", (480, 48), (255))
        lables = drawChsAndCreateLable(im, (u"啊阿1埃A", "FZSTK.TTF", "#000000", 32, 5, 5))
        im.save("testData/current.png")
        self.assertEqual(5,len(lables))
        self.assertEqual("0_zh_5_5_37_34_0",lables[0])
        self.assertEqual("1_zh_37_5_69_36_1",lables[1])
        self.assertEqual("2_en_69_5_80_34_1",lables[2])
        self.assertEqual("3_zh_80_5_112_37_2",lables[3])
        self.assertEqual("4_en_112_5_133_34_10",lables[4])

    def testCreateOneLine(self):
        im = Image.open("testData/0023.BMP",'r')
        lables = createOneLine(im,"testData/data1.png",(u"啊阿1埃A", "FZSTK.TTF", "#000000", 32, 5, 5))
        self.assertEqual(5,len(lables))
        self.assertEqual("0_zh_5_5_37_34_0",lables[0])
        self.assertEqual("1_zh_37_5_69_36_1",lables[1])
        self.assertEqual("2_en_69_5_80_34_1",lables[2])
        self.assertEqual("3_zh_80_5_112_37_2",lables[3])
        self.assertEqual("4_en_112_5_133_34_10",lables[4])

    def testGetAllBg(self):
        files = getAllBg("./realBg")
        self.assertEqual(418,len(files))

    def testCreateOneData(self):
        print(createOneData("./testData/randomPicture.png"))

    def testLoadLabels(self):
        records = loadLabels("./testData/labels.txt")
        self.assertEqual(10,len(records))
        self.assertEqual("./testData/0.png", records[0][0])
        chs = records[0][1]
        self.assertEqual(2,len(chs))

    def testLoadResult(self):
        records = loadResult("./testData/result.txt")
        self.assertEqual(10,len(records))
        self.assertEqual(9, records[0][0].getLeftTopY())
        chs = records[0]
        self.assertEqual(2,len(chs))

    def testDivisionBy(self):
        line = "./testData/1.png,0_zh_289_45_302_61_3008,1_en_302_45_312_59_33,2_en_312_45_319_59_61,3_en_319_45_329_62_26,4_en_329_45_337_59_19,0_en_234_335_241_347_40,1_en_241_335_248_349_52,2_zh_248_335_261_348_3330,3_zh_261_335_274_348_2921,4_en_274_335_281_347_10,0_zh_42_92_58_107_1625,1_zh_58_92_74_107_1992,2_en_74_92_82_106_41,3_en_82_92_90_106_38,4_zh_90_92_106_107_2353,5_en_106_92_114_108_52,6_en_114_92_122_106_28"
        parts = divisionBy(line, ",")
        self.assertEqual(18,len(parts))

    def testCreateLines(self):
        records = loadLabels("./testData/labels.txt")
        chs = records[1][1]
        self.assertEqual("./testData/1.png", records[1][0])
        lines = createLines(chs)
        self.assertEqual(3,len(lines))
        self.assertEqual(45,lines[0].getLeftTopY())
        self.assertEqual(92,lines[1].getLeftTopY())
        self.assertEqual(335,lines[2].getLeftTopY())

    def testChInfo(self):
        exceptCh = ChInfo("0_zh_289_45_302_61_3008")
        ch = ChInfo("0_zh_289_45_302_61_3008")
        self.assertTrue(ch.isSamePlace(exceptCh))

        chRightTopInRange = ChInfo("0_zh_288_46_300_62_3008")
        self.assertTrue(ch.isSamePlace(chRightTopInRange))

        chRightBottomInRange = ChInfo("0_zh_288_44_300_60_3008")
        self.assertTrue(ch.isSamePlace(chRightBottomInRange))

        chLeftBottomInRange = ChInfo("0_zh_300_44_305_60_3008")
        self.assertTrue(ch.isSamePlace(chLeftBottomInRange))

        chNotInRange = ChInfo("0_zh_285_45_287_67_3008")
        self.assertFalse(ch.isSamePlace(chNotInRange))

        self.assertTrue(ch.isSameCh(chNotInRange))

        chNotSame = ChInfo("0_en_285_45_287_67_3008")
        self.assertFalse(ch.isSameCh(chNotSame))

if __name__ =='__main__':
    unittest.main()