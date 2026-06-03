import numpy as np
import pytest

from rogii_wellbore.metrics import rmse


def test_rmse_ignores_nan_pairs() -> None:
    assert rmse(np.array([1.0, np.nan, 3.0]), np.array([2.0, 5.0, 5.0])) == pytest.approx(
        np.sqrt(2.5)
    )
