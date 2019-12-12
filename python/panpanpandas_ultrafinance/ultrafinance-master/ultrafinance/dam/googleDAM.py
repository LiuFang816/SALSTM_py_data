'''
Created on Nov 9, 2011

@author: ppa
'''
from ultrafinance.dam.baseDAM import BaseDAM
from ultrafinance.dam.googleFinance import GoogleFinance

import logging
LOG = logging.getLogger()

class GoogleDAM(BaseDAM):
    ''' Google DAO '''

    def __init__(self):
        ''' constructor '''
        super(GoogleDAM, self).__init__()
        self.__gf = GoogleFinance()

    def readQuotes(self, start, end):
        ''' read quotes from google Financial'''
        if self.symbol is None:
            LOG.debug('Symbol is None')
            return []

        return self.__gf.getQuotes(self.symbol, start, end)

    def readTicks(self, start, end):
        ''' read ticks from google Financial'''
        if self.symbol is None:
            LOG.debug('Symbol is None')
            return []

        return self.__gf.getTicks(self.symbol, start, end)

    def readFundamental(self):
        ''' read fundamental '''
        if self.symbol is None:
            LOG.debug('Symbol is None')
            return {}

        return self.__gf.getFinancials(self.symbol)
