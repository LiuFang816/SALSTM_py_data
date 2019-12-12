#coding=utf8
import requests, os

class LogClient:
    def __init__(self, userID, password, baseUrl):
        self.userID = userID
        self.password = password
        self.baseUrl = baseUrl
        self.s = requests.Session()
        while not 'userstatus' in self.login(): print 'Try Again'
        print 'Login Succeed'
    def login(self):
        r = self.s.get(self.baseUrl + '/count.asp', stream = True)
        with open('count.jpg', 'wb') as f: f.write(r.content)
        os.startfile('count.jpg')
        mofei = raw_input('mofei: ')
        payloads = {
            'userID': self.userID,
            'password': self.password,
            'mofei': mofei, }
        headers = { 'Content-Type': 'application/x-www-form-urlencoded', }
        r = self.s.post(self.baseUrl + '/loginResult.asp',
            data = payloads, headers = headers)
        return r.url
    def upload_log(self, clientId, caseId, date, description, hours = 0):
        try:
            payloads = {
                'RegisterType': 'NEW',
                'wl_category' : '0',
                'wl_client_id' : clientId,
                'wl_case_id' : caseId,
                'wl_empl_id' : '111',
                'wl_work_type': '01',
                'wl_date': date,
                'wl_own_hours': hours,
                'wl_start_date': '09:00',
                'wl_description': description.encode('gbk'),}
            headers = { 'Content-Type': 'application/x-www-form-urlencoded', }
            r = self.s.post(self.baseUrl + '/worklog/WorklogSave.asp', data = payloads, headers = headers)
            return True if 'document.frmWorklog.submit' in r.text else False
        except:
            return False

if __name__ == '__main__':
    lc = LogClient()
    r = lc.upload_log('0210340', '021G20110002', '2016-2-3', u'测试', 0)
