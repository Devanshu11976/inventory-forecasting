import pytest
import numpy as np
from evaluation.metrics import mae, rmse, mape, smape

def test_perfect_forecast():
    actual = [10, 20, 30, 40]
    predicted = [10, 20, 30, 40]
    assert mae(actual, predicted) == 0.0
    assert rmse(actual, predicted) == 0.0
    assert mape(actual, predicted) == 0.0
    assert smape(actual, predicted) == 0.0

def test_known_errors():
    actual = [10, 20, 30]
    predicted = [12, 18, 33]
    
    # MAE = (|10-12| + |20-18| + |30-33|)/3 = (2 + 2 + 3)/3 = 7/3 = 2.333
    assert mae(actual, predicted) == pytest.approx(7.0 / 3.0)
    
    # RMSE = sqrt((2^2 + 2^2 + 3^2)/3) = sqrt((4 + 4 + 9)/3) = sqrt(17/3) = 2.380
    assert rmse(actual, predicted) == pytest.approx(np.sqrt(17.0 / 3.0))
    
    # MAPE = (2/10 + 2/20 + 3/30)/3 * 100 = (0.2 + 0.1 + 0.1)/3 * 100 = 13.333%
    assert mape(actual, predicted) == pytest.approx(40.0 / 3.0)
    
    # sMAPE = (|10-12|/11 + |20-18|/19 + |30-33|/31.5)/3 * 100
    val1 = 2.0 / 11.0
    val2 = 2.0 / 19.0
    val3 = 3.0 / 31.5
    expected_smape = np.mean([val1, val2, val3]) * 100.0
    assert smape(actual, predicted) == pytest.approx(expected_smape)

def test_zero_actuals_mape():
    actual = [0, 10, 0, 20]
    predicted = [5, 12, 5, 18]
    # MAPE excludes zeros.
    # index 1: |10-12|/10 = 0.2
    # index 3: |20-18|/20 = 0.1
    # mean([0.2, 0.1]) * 100 = 15%
    assert mape(actual, predicted) == pytest.approx(15.0)

def test_all_zero_actuals_mape():
    actual = [0, 0, 0]
    predicted = [5, 5, 5]
    assert mape(actual, predicted) == 0.0

def test_all_zero_smape():
    actual = [0, 0, 0]
    predicted = [0, 0, 0]
    assert smape(actual, predicted) == 0.0

def test_empty_lists():
    assert mae([], []) == 0.0
    assert rmse([], []) == 0.0
    assert mape([], []) == 0.0
    assert smape([], []) == 0.0
