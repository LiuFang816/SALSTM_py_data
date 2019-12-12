'''
Created on Dec 18, 2010

@author: ppa
'''
import sys
from bs4 import BeautifulSoup
from datetime import datetime
from datetime import timedelta
from datetime import date
import time
from time import gmtime, strftime

import logging
LOG = logging.getLogger()

googCSVDateformat = "%d-%b-%y"

def importClass(path, moduleName, className = None):
    ''' dynamically import class '''
    if not className:
        className = capitalize(moduleName)
    sys.path.append(path)

    mod = __import__(moduleName)
    return getattr(mod, className)

def capitalize(inputString):
    ''' capitalize first letter '''
    if not inputString:
        return inputString
    elif 1 == len(inputString):
        return inputString[0].upper()
    else:
        return inputString[0].upper() + inputString[1:]

def deCapitalize(inputString):
    ''' capitalize first letter '''
    if not inputString:
        return inputString
    elif 1 == len(inputString):
        return inputString[0].lower()
    else:
        return inputString[0].lower() + inputString[1:]

def splitByComma(inputString):
    ''' split string by comma '''
    return [name.strip() for name in inputString.split(',')]

def convertGoogCSVDate(googCSVDate):
    ''' convert date 25-Jul-2010 to 20100725'''
    d = str(datetime.strptime(googCSVDate, googCSVDateformat).date())
    return d.replace("-", "")

def findPatthen(page, pList):
    datas = [BeautifulSoup(page)]
    for key, pattern in pList:
        newDatas = findPattern(datas, key, pattern)

        datas = newDatas
        if not datas:
            break

    return datas

def findPattern(datas, key, pattern):
    newDatas = []
    for data in datas:
        if 'id' == key:
            newDatas.extend(data.findAll(id = pattern, recursive = True))
        if 'text' == key:
            newDatas.extend(data.findAll(text = pattern, recursive = True))

    return newDatas
def string2EpochTime(stingTime, format = '%Y%m%d'):
    ''' convert string time to epoch time '''
    return int(time.mktime(datetime.strptime(stingTime, '%Y%m%d').timetuple()))

def string2datetime(stringTime, format = '%Y%m%d'):
    ''' convert string time to epoch time'''
    return datetime.strptime(stringTime, '%Y%m%d')


def splitListEqually(inputList, chunks):
    return [inputList[i : i + chunks] for i in range(0, len(inputList), chunks)]

def splitDictEqually(inputDict, chunks):
    '''Splits dict by keys. Returns a list of dictionaries.
    from http://enginepewpew.blogspot.com/2012/03/splitting-dictionary-into-equal-chunks.html
    '''
    return_list = [dict() for idx in xrange(chunks)]
    idx = 0
    for k,v in inputDict.iteritems():
        return_list[idx][k] = v
        if idx < chunks-1:  # indexes start at 0
            idx += 1
        else:
            idx = 0
    return return_list

def getDateString(numDaysBefore):
    ''' return string represent date of n days ago '''
    t = date.today() - timedelta(days=numDaysBefore)
    return t.strftime("%Y%m%d")
