"""
Test various volatility estimates
"""

import os
import unittest

import pandas as pd

from mlfinpy.util.volatility import (
    get_daily_vol,
    get_garman_class_vol,
    get_parksinson_vol,
    get_yang_zhang_vol,
)


class TestVolatilityEstimators(unittest.TestCase):
    """
    Test various volatility estimates (YZ, GS, Parksinson)
    """

    def setUp(self):
        """
        Set the file path for the sample dollar bars data.
        """
        project_path = os.path.dirname(__file__)
        self.path = project_path + "/test_data/dollar_bar_sample.csv"
        self.trades_path = project_path + "/test_data/tick_data.csv"
        self.data = pd.read_csv(self.path, index_col="date_time", parse_dates=[0])
        self.data.index = pd.to_datetime(self.data.index)

    def test_volatility(self):
        """
        Test volatility estimators.
        """
        gm_vol = get_garman_class_vol(self.data.open, self.data.high, self.data.low, self.data.close, window=20)
        yz_vol = get_yang_zhang_vol(self.data.open, self.data.high, self.data.low, self.data.close, window=20)
        park_vol = get_parksinson_vol(self.data.high, self.data.low, window=20)

        self.assertEqual(self.data.shape[0], gm_vol.shape[0])
        self.assertEqual(self.data.shape[0], yz_vol.shape[0])
        self.assertEqual(self.data.shape[0], park_vol.shape[0])

        self.assertAlmostEqual(gm_vol.mean(), 0.001482, delta=1e-6)
        self.assertAlmostEqual(yz_vol.mean(), 0.00162001, delta=1e-6)
        self.assertAlmostEqual(park_vol.mean(), 0.00149997, delta=1e-6)

    def test_get_daily_vol_use_bars(self):
        """
        Test get_daily_vol's fork-only `use_bars=`/`adjust=` arguments: default
        behaviour (Snippet 3.1's closest-bar-at-least-1-calendar-day-earlier
        search) is unchanged, `use_bars=True` switches to plain
        `close.pct_change()` (a full-length series, vs. the default's
        calendar-search dropping some rows on this tick-level fixture).
        """
        default_vol = get_daily_vol(self.data.close, lookback=20)
        bars_vol = get_daily_vol(self.data.close, lookback=20, adjust=False, use_bars=True)

        self.assertEqual(960, default_vol.shape[0])
        self.assertAlmostEqual(default_vol.mean(), 0.0043749, delta=1e-6)

        self.assertEqual(self.data.shape[0], bars_vol.shape[0])
        self.assertAlmostEqual(bars_vol.mean(), 0.00156306, delta=1e-6)
