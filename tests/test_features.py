import numpy as np
import pandas as pd

from rogii_wellbore.alignment import prefix_ncc_features, typewell_beam_features
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


def test_prefix_ncc_features_are_target_free_and_finite() -> None:
    frame = pd.DataFrame(
        {
            "GR": [1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0],
            "TVT_input": [10.0, 11.0, 12.0, 13.0, np.nan, np.nan, np.nan],
        }
    )
    ncc = prefix_ncc_features(
        frame["GR"],
        frame["TVT_input"],
        halfwidths=(1,),
        stride=1,
        min_known=3,
    )
    assert "prefix_ncc_tvt_ensemble" in ncc
    assert np.isfinite(ncc["prefix_ncc_tvt_ensemble"]).all()
    assert np.isfinite(ncc["prefix_ncc_score_max"]).all()
    assert ncc["prefix_ncc_score_max"].between(-1.0, 1.0).all()


def test_build_horizontal_features_can_include_prefix_ncc() -> None:
    frame = pd.DataFrame(
        {
            "MD": np.arange(14, dtype=float),
            "GR": [1, 2, 3, 5, 8, 13, 21, 3, 5, 8, 13, 21, 34, 55],
            "TVT_input": [100, 101, 102, 103, 104, 105, 106, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        }
    )
    features = build_horizontal_features(frame, include_prefix_ncc=True)
    assert "prefix_ncc_delta_ensemble" in features
    assert features["prefix_ncc_known_rows"].iloc[-1] == 7.0


def test_typewell_beam_features_are_finite_on_eval_rows() -> None:
    horizontal = pd.DataFrame(
        {
            "GR": [10.0, 12.0, 15.0, 18.0, 20.0, 18.0, 15.0],
            "TVT_input": [100.0, 101.0, 102.0, np.nan, np.nan, np.nan, np.nan],
        }
    )
    typewell = pd.DataFrame(
        {
            "TVT": np.arange(95.0, 110.0),
            "GR": [5, 7, 10, 12, 15, 18, 20, 18, 15, 12, 10, 8, 7, 6, 5],
        }
    )
    beam = typewell_beam_features(
        horizontal["GR"],
        horizontal["TVT_input"],
        typewell["TVT"],
        typewell["GR"],
    )
    eval_rows = horizontal["TVT_input"].isna()
    assert "typewell_beam_delta_mean" in beam
    assert np.isfinite(beam.loc[eval_rows, "typewell_beam_tvt_mean"]).all()


def test_build_horizontal_features_can_include_typewell_beam() -> None:
    horizontal = pd.DataFrame(
        {
            "MD": np.arange(7, dtype=float),
            "GR": [10.0, 12.0, 15.0, 18.0, 20.0, 18.0, 15.0],
            "TVT_input": [100.0, 101.0, 102.0, np.nan, np.nan, np.nan, np.nan],
        }
    )
    typewell = pd.DataFrame(
        {
            "TVT": np.arange(95.0, 110.0),
            "GR": [5, 7, 10, 12, 15, 18, 20, 18, 15, 12, 10, 8, 7, 6, 5],
        }
    )
    features = build_horizontal_features(
        horizontal,
        typewell,
        include_typewell_beam=True,
    )
    assert "typewell_beam_gr_residual" in features
