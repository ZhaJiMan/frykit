from unittest import TestCase

import numpy as np

from frykit.calc import binning2d, interp_nearest_2d, interp_nearest_dd


class TestCalc(TestCase):
    def test_interp_nearest_dd(self) -> None:
        points = [[0, 0], [1, 1]]
        values = [[1, 2], [3, 4]]
        xi = [[-1, -1], [0.1, 0.1], [0.9, 0.9], [10, 10]]
        desired = [[1, 1, 2, np.nan], [3, 3, 4, np.nan]]

        actual = interp_nearest_dd(points, values, xi, radius=2)
        np.testing.assert_array_equal(actual, desired)

    def test_interp_nearest_2d(self) -> None:
        x = y = [0, 1]
        xi = yi = [-1, 0.1, 0.9, 10]
        values = [[1, 2], [3, 4]]
        desired = [[1, 1, 2, np.nan], [3, 3, 4, np.nan]]

        actual = interp_nearest_2d(x, y, values, xi, yi, radius=2)
        np.testing.assert_array_equal(actual, desired)

    def test_binning2d(self) -> None:
        x = [0.1, 0.9, 1.5, 0.5]
        y = [0.1, 0.9, 1.5, 1.5]
        values = [[1, 2, 3, 4], [5, 6, 7, 8]]
        xbins = [0, 1, 2, 3]
        ybins = [2, 1, 0]  # reverse
        desired = [
            [
                [[4, 3, np.nan], [1.5, np.nan, np.nan]],
                [[4, 3, np.nan], [3, np.nan, np.nan]],
                [[4, 3, np.nan], [2, np.nan, np.nan]],
            ],
            [
                [[8, 7, np.nan], [5.5, np.nan, np.nan]],
                [[8, 7, np.nan], [11, np.nan, np.nan]],
                [[8, 7, np.nan], [6, np.nan, np.nan]],
            ],
        ]

        actual = binning2d(x, y, values, xbins, ybins, func=["mean", "sum", "max"])
        np.testing.assert_array_equal(actual, desired)
