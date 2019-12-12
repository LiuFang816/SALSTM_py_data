#coding=utf8
import random

import config
from client.ExcelClient import ExcelClient

def get_stored_cases(storageDir):
    # case_num, case_id, project
    ec = ExcelClient(storageDir, sourceIndex = (0,2,3))
    r = {}
    while 1:
        userInfo = ec.getData()
        if userInfo is None: break
        r[userInfo[1]] = [userInfo[0], userInfo[1]]
    return r

def makeuploadfile():
    print 'Getting the %s file.'%config.metaStorage
    caseDict = get_stored_cases(config.caseStorage)
    unfoundList = []

    print 'Please wait for about half a minute'
    oec = ExcelClient(outputDir = config.uploadFile, outputHeader = ['case_num', 'case_id', 'date', 'time', 'description'])
    # to, filename, created_on, case_id, subject
    iec = ExcelClient(config.metaStorage, sourceIndex = (0,4,5,7,6))
    while 1:
        mailInfo = iec.getData()
        if mailInfo is None: break
        outputList = []
        for key, value in caseDict.items():
            if '' != mailInfo[3] == key:
                def clear_colon(s):
                    return s.encode('utf8').split(':')[-1].split('：')[-1].replace(' ', '').decode('utf8')
                outputList = sum([outputList, [value[0], value[1], mailInfo[2].replace('/','-').split(' ')[0], str(random.randint(4,7)/10.0),
                    u'与{0}就{2}进行沟通，并发送{1}给{0}。'.format(mailInfo[0], clear_colon(mailInfo[1]), clear_colon(mailInfo[4]))]
                    ], [])
        if not outputList: unfoundList.append(str(mailInfo[3]))#print('Case on {} not found'.format(mailInfo[2]))
        oec.storeData(outputList)
    if unfoundList: print 'Cases not found: ' + ', '.join(list(set(unfoundList)))
    print 'Output succeeded! Please check %s before run the upload.py'%config.uploadFile

if __name__ == '__main__':
    makeuploadfile()
