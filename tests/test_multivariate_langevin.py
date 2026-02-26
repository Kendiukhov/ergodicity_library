import numpy as np

from ergodicity.process.basic import MultivariateLangevinProcess


def test_multivariate_langevin_simulation_shape():
    process = MultivariateLangevinProcess(
        dims=2,
        drift=lambda x, t: -x,
        diffusion=np.array([[0.2, 0.0], [0.0, 0.3]]),
        initial_state=np.array([1.0, -1.0]),
    )

    times, values = process.simulate(t=2.0, timestep=0.02, num_instances=5, plot=False)

    assert times.ndim == 1
    assert values.ndim == 3
    assert values.shape[0] == 5
    assert values.shape[1] == 2
    assert values.shape[2] == times.shape[0]


def test_multivariate_langevin_deterministic_mean_reversion():
    # Zero diffusion reduces the process to deterministic Euler integration.
    process = MultivariateLangevinProcess(
        dims=1,
        drift=lambda x, t: -x,
        diffusion=0.0,
        initial_state=np.array([2.0]),
    )

    times, values = process.simulate(t=1.0, timestep=0.001, num_instances=1, plot=False)

    trajectory = values[0, 0, :]
    assert np.all(np.diff(trajectory) <= 1e-12)
    assert trajectory[-1] < trajectory[0]
    assert trajectory[-1] > 0
