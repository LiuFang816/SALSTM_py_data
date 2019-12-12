from client.ExcelClient import ExcelClient
from client.OutlookAttachViewClient import xml_solve
import config

def get_stored_username(storageDir):
    ec = ExcelClient(storageDir, sourceIndex = (0,1,2))
    r = {}
    while 1:
        userInfo = ec.getData()
        if userInfo is None: break
        r[userInfo[0]] = [userInfo[1], userInfo[2]]
    return r

def producexml():
    header = ['to', 'to_email', 'from', 'from_email', 'filename', 'created_on', 'subject']
    ec = ExcelClient(outputDir = config.metaStorage, outputHeader = sum([header, ['case_id']], []))
    print 'Please wait for about half a minute'
    nameList = get_stored_username(config.userNameStorage)
    for info in xml_solve(config.xmlInput):
        infoList = [info[key] for key in header]
        if nameList.has_key(infoList[1].split(',')[0]):
            infoList[0] = nameList[infoList[1].split(',')[0]][0]
            infoList.append(nameList[infoList[1].split(',')[0]][1])
        if nameList.has_key(infoList[3]): infoList[2] = nameList[infoList[3]][0]
        ec.storeData(infoList)
    print 'Output succeeded!'
    print 'Please check %s before run the makeuploadfile.py'%config.metaStorage
    print 'You need to fill in the blank case_id'

if __name__ == '__main__':
    producexml()
