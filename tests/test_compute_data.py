from unittest.mock import Mock
import os
import numpy.testing as npt
import numpy as np
import pytest
import math
from inflammation.models import analyse_data, compute_standard_deviation_by_day, CSVDataSource, JSONDataSource


def test_analyse_data_mock_source():
    '''Mock test'''
    data_source = Mock()
    mock_data = [np.array([[2,3,2],[1,6,3]]), np.array([[5,4,2],[4,7,3]])]
    data_source.method_to_mock.return_value = mock_data

    analyse_data(data_source)

@pytest.mark.parametrize('data,expected_output', [
    ([[[0, 1, 0], [0, 2, 0]]], [0, 0, 0]),
    ([[[0, 2, 0]], [[0, 1, 0]]], [0, math.sqrt(0.25), 0]),
    ([[[0, 1, 0], [0, 2, 0]], [[0, 1, 0], [0, 2, 0]]], [0, 0, 0])
],
ids=['Two patients in same file', 'Two patients in different files', 'Two identical patients in two different files'])
def test_compute_standard_deviation_by_day(data, expected_output):

    result = compute_standard_deviation_by_day(data)
    npt.assert_array_almost_equal(result, expected_output)


def test_analyse_data():
    '''Regression test'''
    path = os.path.join( os.getcwd(), "data")
    data_source = CSVDataSource(path)
    result = analyse_data(data_source)

    expected_output = np.array(
        [0.        , 0.22510286, 0.18157299, 0.1264423 , 0.9495481 ,
        0.27118211, 0.25104719, 0.22330897, 0.89680503, 0.21573875,
        1.24235548, 0.63042094, 1.57511696, 2.18850242, 0.3729574 ,
        0.69395538, 2.52365162, 0.3179312 , 1.22850657, 1.63149639,
        2.45861227, 1.55556052, 2.8214853 , 0.92117578, 0.76176979,
        2.18346188, 0.55368435, 1.78441632, 0.26549221, 1.43938417,
        0.78959769, 0.64913879, 1.16078544, 0.42417995, 0.36019114,
        0.80801707, 0.50323031, 0.47574665, 0.45197398, 0.22070227]
        )
    
    npt.assert_array_almost_equal(result['standard deviation by day'], expected_output)

