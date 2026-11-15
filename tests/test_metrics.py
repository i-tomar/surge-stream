import numpy as np
from surgecast.training.metrics import clip_prediction, regression_scores


def test_clip_prediction_floors_at_zero():
    assert clip_prediction(np.array([-3.0, 1.5])).tolist() == [0.0, 1.5]


def test_regression_scores_known_values():
    y = np.array([0.0, 2.0, 4.0])
    yhat = np.array([0.0, 1.0, 1.0])
    scores = regression_scores(y, yhat)
    assert scores["n"] == 3
    assert scores["n_positive"] == 2
    assert scores["mae"] == np.mean([0, 1, 3])
    assert abs(scores["mae_y_gt_0"] - 2.0) < 1e-12
    assert abs(scores["mape_pct_y_gt_0"] - 62.5) < 1e-12
    assert abs(scores["rmse"] - np.sqrt(10 / 3)) < 1e-12
    assert abs(scores["wape_pct_y_gt_0"] - (4 / 6 * 100)) < 1e-12
