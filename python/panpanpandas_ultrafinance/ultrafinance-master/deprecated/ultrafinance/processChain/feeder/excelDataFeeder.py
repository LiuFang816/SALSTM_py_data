'''
Created on Jan 30, 2010

@author: ppa
'''
from ultrafinance.lib.excelLib import ExcelLib
from ultrafinance.processChain.baseModule import BaseModule
from ultrafinance.lib.dataType import DateValueType
from os.path import join

import logging
LOG = logging.getLogger(__name__)

class ExcelDataFeeder(BaseModule):
    '''
    feeder that get stock, hoursing and interest rate from excel
    '''
    def __init__(self):
        ''' Constructor '''
        super(ExcelDataFeeder, self).__init__()
        self.stockData = []
        self.hoursingData = []
        self.interestData = []

    def execute(self, input):
        ''' preparing data'''
        with ExcelLib(join('dataSource', 'hoursing_interestRate.xls'), 0) as excel:
            year = excel.readCol(0, 7, 127)
            hoursing = excel.readCol(1, 7, 127)
            interest = excel.readCol(5, 7, 127)
            for i in range(len(year)):
                self.hoursingData.append(DateValueType(str(int(year[i])), hoursing[i]))
                self.interestData.append(DateValueType(str(int(year[i])), interest[i]))

        with ExcelLib(join('dataSource', 'longTerm_1871.xls'), 0) as excel:
            year = excel.readCol(0, 8, 147)
            stock = excel.readCol(1, 8, 147)
            for i in range(len(year)):
                self.stockData.append(DateValueType(str(int(year[i])), stock[i]))

        ret = {'stock': self.stockData, 'hoursing': self.hoursingData, 'interest': self.interestData}
        return ret

if __name__ == '__main__':
    feeder = ExcelDataFeeder()
    feeder.execute("")
    print feeder.hoursingData
    print feeder.interestData
    print feeder.stockData