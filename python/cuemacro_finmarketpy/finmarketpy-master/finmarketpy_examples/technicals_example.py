__author__ = 'saeedamen'


# loading data
import datetime

from chartpy import Chart, Style
from findatapy.market import Market, MarketDataGenerator, MarketDataRequest
from finmarketpy.economics import TechIndicator, TechParams

from findatapy.util.loggermanager import LoggerManager

logger = LoggerManager().getLogger(__name__)

chart = Chart(engine='matplotlib')

market = Market(market_data_generator=MarketDataGenerator())
tech_ind = TechIndicator()

# choose run_example = 0 for everything
# run_example = 1 - download S&P500 from Quandl, calculate ATR and plot

run_example = 0

###### fetch data from Quandl for BoE rate (using Bloomberg data)
if run_example == 1 or run_example == 0:

    # downloaded S&P500
    md_request = MarketDataRequest(
                start_date = "01 Jan 2000",                         # start date
                data_source = 'quandl',                             # use Quandl as data source
                tickers = ['S&P500'],
                fields = ['close', 'open', 'high', 'low'],          # which fields to download
                vendor_tickers = ['YAHOO/INDEX_GSPC'],              # ticker (Bloomberg)
                vendor_fields = ['close', 'open', 'high', 'low'],   # which Bloomberg fields to download
                cache_algo = 'internet_load_return')                # how to return data

    df = market.fetch_market(md_request)

    print(df)

    tech_params = TechParams()
    tech_params.atr_period = 14
    tech_ind.create_tech_ind(df, 'ATR', tech_params)

    style = Style()

    style.title = 'S&P500 ATR'
    style.scale_factor = 2
    style.file_output = "sp500.png"
    style.source = 'Quandl/Yahoo'

    df = tech_ind.get_techind()

    chart.plot(df, style=style)