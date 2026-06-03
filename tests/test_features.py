import numpy as np
import pandas as pd

from rogii_wellbore.features import build_horizontal_features, canonicalize, prediction_mask


def test_prediction_mask_after_known_prefix() -> None:
    frame = pd.DataFrame(
        {
            "MD": [0, 1, 2, 3],
            "TVT": [10.0, 11.0, 12.0, 13.0],
            "TVT_input": [10.0, 11.0, np.nan, np.nan],
        }
    )
    horizontal = canonicalize(frame)
    assert prediction_mask(horizontal, require_target=True).tolist() == [False, False, True, True]


def test_build_horizontal_features_linear_prior() -> None:
    frame = pd.DataFrame(
        {
            "MD": [0.0, 1.0, 2.0, 3.0],
            "GR": [80.0, 82.0, 84.0, 85.0],
            "TVT_input": [10.0, 11.0, np.nan, np.nan],
        }
    )
    features = build_horizontal_features(frame)
    assert "linear_tvt_prior" in features
    assert features["linear_tvt_prior"].iloc[-1] > features["linear_tvt_prior"].iloc[1]
