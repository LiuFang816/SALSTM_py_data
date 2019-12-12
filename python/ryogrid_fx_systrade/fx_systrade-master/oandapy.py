import json
import requests
#from .exceptions import BadEnvironment, OandaError

""" OANDA API wrapper for OANDA's REST API """

""" EndpointsMixin provides a mixin for the API instance
Parameters that need to be embedded in the API url just need to be passed as a
keyword argument.
E.g. oandapy_instance.get_instruments(instruments="EUR_USD")
"""


class EndpointsMixin(object):

    """Rates"""

    def get_instruments(self, account_id, **params):
        """ Get an instrument list
        Docs: http://developer.oanda.com/rest-live/rates
        """
        params['accountId'] = account_id
        endpoint = 'v1/instruments'
        return self.request(endpoint, params=params)

    def get_prices(self, **params):
        """ Get current prices
        Docs: http://developer.oanda.com/rest-live/rates
        """
        endpoint = 'v1/prices'
        return self.request(endpoint, params=params)

    def get_history(self, **params):
        """ Retrieve instrument history
        Docs: http://developer.oanda.com/rest-live/rates
        """
        endpoint = 'v1/candles'
        return self.request(endpoint, params=params)

    """Accounts"""

    def create_account(self, **params):
        """ Create an account. Valid only in sandbox.
        Docs: http://developer.oanda.com/rest-live/accounts
        """
        endpoint = 'v1/accounts'
        return self.request(endpoint, "POST", params=params)

    def get_accounts(self, **params):
        """ Get accounts for a user.
        Docs: http://developer.oanda.com/rest-live/accounts
        """
        endpoint = 'v1/accounts'
        return self.request(endpoint, params=params)

    def get_account(self, account_id, **params):
        """ Get account information
        Docs: http://developer.oanda.com/rest-live/accounts
        """
        endpoint = 'v1/accounts/%s' % (account_id)
        return self.request(endpoint, params=params)

    """Orders"""

    def get_orders(self, account_id, **params):
        """ Get orders for an account
        Docs: http://developer.oanda.com/rest-live/orders
        """
        endpoint = 'v1/accounts/%s/orders' % (account_id)
        return self.request(endpoint, params=params)

    def create_order(self, account_id, **params):
        """ Create a new order
        Docs: http://developer.oanda.com/rest-live/orders
        """
        endpoint = 'v1/accounts/%s/orders' % (account_id)
        return self.request(endpoint, "POST", params=params)

    def get_order(self, account_id, order_id, **params):
        """ Get information for an order
        Docs: http://developer.oanda.com/rest-live/orders
        """
        endpoint = 'v1/accounts/%s/orders/%s' % (account_id, order_id)
        return self.request(endpoint, params=params)

    def modify_order(self, account_id, order_id, **params):
        """ Modify an existing order
        Docs: http://developer.oanda.com/rest-live/orders
        """
        endpoint = 'v1/accounts/%s/orders/%s' % (account_id, order_id)
        return self.request(endpoint, "PATCH", params=params)

    def close_order(self, account_id, order_id, **params):
        """ Close an order
        Docs: http://developer.oanda.com/rest-live/orders
        """
        endpoint = 'v1/accounts/%s/orders/%s' % (account_id, order_id)
        return self.request(endpoint, "DELETE", params=params)

    """Trades"""

    def get_trades(self, account_id, **params):
        """ Get a list of open trades
        Docs: http://developer.oanda.com/rest-live/trades
        """
        endpoint = 'v1/accounts/%s/trades' % (account_id)
        return self.request(endpoint, params=params)

    def get_trade(self, account_id, trade_id, **params):
        """ Get information on a specific trade
        Docs: http://developer.oanda.com/rest-live/trades
        """
        endpoint = 'v1/accounts/%s/trades/%s' % (account_id, trade_id)
        return self.request(endpoint, params=params)

    def modify_trade(self, account_id, trade_id, **params):
        """ Modify an existing trade
        Docs: http://developer.oanda.com/rest-live/trades
        """
        endpoint = 'v1/accounts/%s/trades/%s' % (account_id, trade_id)
        return self.request(endpoint, "PATCH", params=params)

    def close_trade(self, account_id, trade_id, **params):
        """ Close an open trade
        Docs: http://developer.oanda.com/rest-live/trades
        """
        endpoint = 'v1/accounts/%s/trades/%s' % (account_id, trade_id)
        return self.request(endpoint, "DELETE", params=params)

    """Positions"""

    def get_positions(self, account_id, **params):
        """ Get a list of all open positions
        Docs: http://developer.oanda.com/rest-live/positions
        """
        endpoint = 'v1/accounts/%s/positions' % (account_id)
        return self.request(endpoint, params=params)

    def get_position(self, account_id, instrument, **params):
        """ Get the position for an instrument
        Docs: http://developer.oanda.com/rest-live/positions
        """
        endpoint = 'v1/accounts/%s/positions/%s' % (account_id, instrument)
        return self.request(endpoint, params=params)

    def close_position(self, account_id, instrument, **params):
        """ Close an existing position
        Docs: http://developer.oanda.com/rest-live/positions
        """
        endpoint = 'v1/accounts/%s/positions/%s' % (account_id, instrument)
        return self.request(endpoint, "DELETE", params=params)

    """Transaction History"""

    def get_transaction_history(self, account_id, **params):
        """ Get transaction history
        Docs: http://developer.oanda.com/rest-live/transaction-history
        """
        endpoint = 'v1/accounts/%s/transactions' % (account_id)
        return self.request(endpoint, params=params)

    def get_transaction(self, account_id, transaction_id):
        """ Get information for a transaction
        Docs: http://developer.oanda.com/rest-live/transaction-history
        """
        endpoint = 'v1/accounts/%s/transactions/%s' % \
                   (account_id, transaction_id)
        return self.request(endpoint)

    """Forex Labs"""

    def get_eco_calendar(self, **params):
        """Returns up to 1 year of economic calendar info
        Docs: http://developer.oanda.com/rest-live/forex-labs/
        """
        endpoint = 'labs/v1/calendar'
        return self.request(endpoint, params=params)

    def get_historical_position_ratios(self, **params):
        """Returns up to 1 year of historical position ratios
        Docs: http://developer.oanda.com/rest-live/forex-labs/
        """
        endpoint = 'labs/v1/historical_position_ratios'
        return self.request(endpoint, params=params)

    def get_historical_spreads(self, **params):
        """Returns up to 1 year of spread information
        Docs: http://developer.oanda.com/rest-live/forex-labs/
        """
        endpoint = 'labs/v1/spreads'
        return self.request(endpoint, params=params)

    def get_commitments_of_traders(self, **params):
        """Returns up to 4 years of Commitments of Traders data from the CFTC
        Docs: http://developer.oanda.com/rest-live/forex-labs/
        """
        endpoint = 'labs/v1/commitments_of_traders'
        return self.request(endpoint, params=params)

    def get_orderbook(self, **params):
        """Returns up to 1 year of OANDA Order book data
        Docs: http://developer.oanda.com/rest-live/forex-labs/
        """
        endpoint = 'labs/v1/orderbook_data'
        return self.request(endpoint, params=params)


""" Provides functionality for access to core OANDA API calls """


class API(EndpointsMixin, object):
    def __init__(self,
                 environment="practice", access_token=None, headers=None):
        """Instantiates an instance of OandaPy's API wrapper
        :param environment: (optional) Provide the environment for oanda's
         REST api, either 'sandbox', 'practice', or 'live'. Default: practice
        :param access_token: (optional) Provide a valid access token if you
         have one. This is required if the environment is not sandbox.
        """

        if environment == 'sandbox':
            self.api_url = 'http://api-sandbox.oanda.com'
        elif environment == 'practice':
            self.api_url = 'https://api-fxpractice.oanda.com'
        elif environment == 'live':
            self.api_url = 'https://api-fxtrade.oanda.com'
        else:
            print("raise BadEnvironment(environment)")
#            raise BadEnvironment(environment)

        self.access_token = access_token
        self.client = requests.Session()

        # personal token authentication
        if self.access_token:
            self.client.headers['Authorization'] = 'Bearer '+self.access_token

        if headers:
            self.client.headers.update(headers)

    def request(self, endpoint, method='GET', params=None):
        """Returns dict of response from OANDA's open API
        :param endpoint: (required) OANDA API (e.g. v1/instruments)
        :type endpoint: string
        :param method: (optional) Method of accessing data, either GET or POST.
         (default GET)
        :type method: string
        :param params: (optional) Dict of parameters (if any) accepted the by
         OANDA API endpoint you are trying to access (default None)
        :type params: dict or None
        """

        url = '%s/%s' % (self.api_url, endpoint)

        method = method.lower()
        params = params or {}

        func = getattr(self.client, method)

        request_args = {}
        if method == 'get':
            request_args['params'] = params
        else:
            request_args['data'] = params

        try:
            response = func(url, **request_args)
        except requests.RequestException as e:
            print (str(e))
        content = response.content.decode('utf-8')

        content = json.loads(content)

        # error message
#        if response.status_code >= 400:
#            raise OandaError(content)

        return content
