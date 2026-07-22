import pytest
import numpy as np
from decision.reorder_point import calculate_reorder_point, run_worked_example

def test_calculate_reorder_point():
    lead_time_predictions = [10, 12, 8, 15, 11]
    rmse = 4.0
    lead_time = 3
    z = 1.65
    
    # Forecasted demand over lead_time=3: 10 + 12 + 8 = 30
    # Safety Stock: 1.65 * 4.0 * sqrt(3) = 6.6 * 1.732 = 11.4315
    # Reorder Point: 30 + 11.4315 = 41.4315
    forecast, ss, rop = calculate_reorder_point(lead_time_predictions, rmse, lead_time, z)
    
    assert forecast == 30.0
    assert ss == pytest.approx(1.65 * 4.0 * np.sqrt(3))
    assert rop == pytest.approx(30.0 + 1.65 * 4.0 * np.sqrt(3))

def test_worked_example():
    res = run_worked_example()
    assert res['unit_cost'] == 50.00
    assert res['lead_time'] == 16
    assert res['z'] == 1.65
    assert res['rmse_baseline'] == 20.20
    assert res['rmse_xgboost'] == 11.11
    
    assert res['ss_baseline_units'] == pytest.approx(133.32)
    assert res['ss_xgboost_units'] == pytest.approx(73.326)
    assert res['reduction_units'] == pytest.approx(59.994)
    assert res['savings_dollars'] == pytest.approx(2999.70)
