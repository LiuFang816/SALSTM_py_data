# coding=utf-8

import numpy as np
from pandas import (DataFrame, date_range, Timestamp, Series,
                    to_datetime)

from pandas.util.testing import assert_frame_equal, assert_series_equal
import pandas.util.testing as tm

from .common import TestData


class TestFrameAsof(TestData, tm.TestCase):

    def setUp(self):
        self.N = N = 50
        rng = date_range('1/1/1990', periods=N, freq='53s')
        self.df = DataFrame({'A': np.arange(N), 'B': np.arange(N)},
                            index=rng)

    def test_basic(self):

        df = self.df.copy()
        df.loc[15:30, 'A'] = np.nan
        dates = date_range('1/1/1990', periods=self.N * 3,
                           freq='25s')

        result = df.asof(dates)
        self.assertTrue(result.notnull().all(1).all())
        lb = df.index[14]
        ub = df.index[30]

        dates = list(dates)
        result = df.asof(dates)
        self.assertTrue(result.notnull().all(1).all())

        mask = (result.index >= lb) & (result.index < ub)
        rs = result[mask]
        self.assertTrue((rs == 14).all(1).all())

    def test_subset(self):

        N = 10
        rng = date_range('1/1/1990', periods=N, freq='53s')
        df = DataFrame({'A': np.arange(N), 'B': np.arange(N)},
                       index=rng)
        df.loc[4:8, 'A'] = np.nan
        dates = date_range('1/1/1990', periods=N * 3,
                           freq='25s')

        # with a subset of A should be the same
        result = df.asof(dates, subset='A')
        expected = df.asof(dates)
        assert_frame_equal(result, expected)

        # same with A/B
        result = df.asof(dates, subset=['A', 'B'])
        expected = df.asof(dates)
        assert_frame_equal(result, expected)

        # B gives self.df.asof
        result = df.asof(dates, subset='B')
        expected = df.resample('25s', closed='right').ffill().reindex(dates)
        expected.iloc[20:] = 9

        assert_frame_equal(result, expected)

    def test_missing(self):
        # GH 15118
        # no match found - `where` value before earliest date in index
        N = 10
        rng = date_range('1/1/1990', periods=N, freq='53s')
        df = DataFrame({'A': np.arange(N), 'B': np.arange(N)},
                       index=rng)
        result = df.asof('1989-12-31')

        expected = Series(index=['A', 'B'], name=Timestamp('1989-12-31'))
        assert_series_equal(result, expected)

        result = df.asof(to_datetime(['1989-12-31']))
        expected = DataFrame(index=to_datetime(['1989-12-31']),
                             columns=['A', 'B'], dtype='float64')
        assert_frame_equal(result, expected)
