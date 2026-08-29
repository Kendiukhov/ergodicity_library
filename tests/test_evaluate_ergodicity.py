import numpy as np
import pytest

import ergodicity.tools.evaluate as evaluate


@pytest.fixture(autouse=True)
def stub_unrelated_statistical_tests(monkeypatch):
    """Keep these regression tests focused on time-scale conversion."""
    monkeypatch.setattr(
        evaluate,
        "adfuller",
        lambda values: (0.0, 0.01, 0, len(values), {}, 0.0),
    )
    monkeypatch.setattr(evaluate.stats, "normaltest", lambda values: (0.0, 0.5))


def test_ergodicity_increments_uses_timestep_from_time_grid():
    times = 3.0 + np.arange(21) * 0.25
    elapsed = times - times[0]
    process_data = np.vstack((5.0 + 2.0 * elapsed, 1.0 + 4.0 * elapsed))
    data = np.vstack((times, process_data))

    results = evaluate.test_ergodicity_increments(data)

    assert results["time_average_per_unit"] == pytest.approx(3.0)
    assert results["ensemble_average_per_unit"] == pytest.approx(3.0)


def test_relative_ergodicity_increments_uses_original_time_grid():
    times = 2.0 + np.arange(21) * 0.2
    elapsed = times - times[0]
    rates = np.array([0.3, 0.5])
    process_data = np.vstack(
        (
            5.0 * np.exp(rates[0] * elapsed),
            2.0 * np.exp(rates[1] * elapsed),
        )
    )
    data = np.vstack((times, process_data))

    results = evaluate.test_ergodicity_increments(data, relative_increments=True)

    expected_ensemble_rate = np.mean(np.expm1(rates * 0.2)) / 0.2
    assert results["time_average_per_unit"] == pytest.approx(np.mean(rates))
    assert results["ensemble_average_per_unit"] == pytest.approx(
        expected_ensemble_rate
    )


@pytest.mark.parametrize(
    ("times", "message"),
    [
        (np.array([0.0]), "at least two"),
        (np.array([0.0, 0.5, 0.5]), "strictly increasing"),
        (np.array([0.0, 0.5, 1.1]), "evenly spaced"),
        (np.array([0.0, np.nan, 1.0]), "finite"),
    ],
)
def test_ergodicity_increments_rejects_invalid_time_grids(times, message):
    process_data = np.vstack((np.arange(times.size), np.arange(times.size) + 1.0))
    data = np.vstack((times, process_data))

    with pytest.raises(ValueError, match=message):
        evaluate.test_ergodicity_increments(data)
