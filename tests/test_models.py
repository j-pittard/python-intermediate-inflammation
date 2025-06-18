"""Tests for statistics functions within the Model layer."""

import numpy as np
import numpy.testing as npt
import pytest

from inflammation.models import *


@pytest.mark.parametrize(
        ('test', 'expected'), 
        [
        [[[1, 2, -30],[3, 4, 10], [5, -6, 11.5]],[5, 4, 11.5]],
        [[[1, 5, -30],[3, 4, 10], [5, -6, 15.5]],[5, 5, 15.5]],
        [[[1, 2, -7],[3, 4, 10], [5, -6, 0]],[5, 4, 10]]
        ],
        ids = ['test1_name', 'test2_name', 'test3name']
)
def test_daily_max(test, expected):
    test_input = np.array(test)
    npt.assert_array_equal(daily_max(test_input), np.array(expected))



@pytest.mark.parametrize(
       ('test', 'expected'),
       [
        [[[0, 0, 0],[0, 0, 0], [0, 0, 0]],[0, 0, 0]],
        [[[1, 1, 1],[1, 1, 1], [1, 1, 1]],[1, 1, 1]],

        ],
        ids = ['zero_test', 'ones_test']
)
def test_daily_mean(test, expected):
    """Test that mean function works for an array of zeros."""
    test_input = np.array(test)
    npt.assert_array_equal(daily_max(test_input), np.array(expected))


@pytest.mark.parametrize(
    "test, expected",
    [
        ([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[0.33, 0.67, 1], [0.67, 0.83, 1], [0.78, 0.89, 1]])
    ])
def test_patient_normalise(test, expected):
    """Test normalisation works for arrays of one and positive integers.
       Test with a relative and absolute tolerance of 0.01."""

    result = patient_normalise(np.array(test))
    npt.assert_allclose(result, np.array(expected), rtol=1e-2, atol=1e-2)


def test_daily_min_integers():
    """Test that min function works for an array."""

    test_input = np.array([[1, 2, -30],
                           [3, 4, 10],
                           [5, -6, 11.5]])
    test_result = np.array([1, -6, -30])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_min(test_input), test_result)

def test_daily_min_string():
    """Test for TypeError when passing strings"""

    with pytest.raises(TypeError):
        error_expected = daily_min([['Hello', 'there'], ['General', 'Kenobi']])
