import numpy as np

from ergodicity.process.basic import BrownianMotion, MultivariateBrownianMotion, PoissonProcess
from ergodicity.process.definitions import Process, simulation_decorator


class UnitDriftProcess(Process):
    """Deterministic process used to expose timestamp/update misalignment."""

    def __init__(self):
        super().__init__(
            name="Unit Drift Test Process",
            multiplicative=False,
            independent=True,
            ito=True,
            process_class=None,
        )
        self.step_sizes = []

    def custom_increment(self, _state, timestep):
        self.step_sizes.append(timestep)
        return timestep


class ExplicitTimeUnitDriftBackend:
    """Minimal external backend whose value equals the requested sample time."""

    def __init__(self, t):
        self.t = t

    def sample_at(self, times):
        return np.asarray(times, dtype=float)


class ExternalUnitDriftProcess(Process):
    def __init__(self):
        super().__init__(
            name="External Unit Drift Test Process",
            multiplicative=False,
            independent=True,
            ito=True,
            process_class=ExplicitTimeUnitDriftBackend,
        )


class ExplicitTimePoissonBackend:
    def __init__(self, rate):
        self.rate = rate

    def sample_at(self, times):
        return np.asarray(times, dtype=float)


class DecoratedUnitDriftProcess(Process):
    def __init__(self):
        super().__init__(
            name="Decorated Unit Drift Test Process",
            multiplicative=False,
            independent=True,
            ito=True,
            process_class=None,
        )

    @simulation_decorator
    def simulate_path(self, _state, timestep):
        return timestep


def test_custom_simulation_has_one_update_per_time_interval(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    process = UnitDriftProcess()

    simulation = process.simulate(t=1.0, timestep=0.1, num_instances=1)
    times, paths = process.separate(simulation)

    np.testing.assert_allclose(times, np.arange(11) * 0.1)
    np.testing.assert_allclose(paths[0], times)
    np.testing.assert_allclose(process.step_sizes, np.full(10, 0.1))


def test_non_divisible_horizon_uses_a_shorter_final_step(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    process = UnitDriftProcess()

    simulation = process.simulate(t=0.25, timestep=0.1, num_instances=1)
    times, paths = process.separate(simulation)

    np.testing.assert_allclose(times, np.array([0.0, 0.1, 0.2, 0.25]))
    np.testing.assert_allclose(paths[0], times)
    np.testing.assert_allclose(process.step_sizes, np.array([0.1, 0.1, 0.05]))


def test_external_simulation_samples_at_reported_times(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    process = ExternalUnitDriftProcess()

    simulation = process.simulate(t=0.25, timestep=0.1, num_instances=1)
    times, paths = process.separate(simulation)

    np.testing.assert_allclose(times, np.array([0.0, 0.1, 0.2, 0.25]))
    np.testing.assert_allclose(paths[0], times)


def test_stochastic_backend_accepts_partial_final_interval(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    process = BrownianMotion()

    simulation = process.simulate(t=0.25, timestep=0.1, num_instances=2)
    times, paths = process.separate(simulation)

    np.testing.assert_allclose(times, np.array([0.0, 0.1, 0.2, 0.25]))
    assert paths.shape == (2, 4)


def test_time_average_integrates_over_reported_times(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    process = UnitDriftProcess()
    simulation = np.array(
        [
            [0.0, 0.2, 1.0],
            [0.0, 0.0, 1.0],
        ]
    )
    monkeypatch.setattr(process, "simulate", lambda **_kwargs: simulation)

    # The piecewise-linear trajectory has area 0.4 over a unit interval.
    assert process.time_average(t=1.0, timestep=0.1) == 0.4


def test_poisson_override_samples_at_reported_times(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    process = PoissonProcess(process_class=ExplicitTimePoissonBackend, rate=2.0)

    simulation = process.simulate(t=0.25, timestep=0.1, num_instances=1)
    times, paths = process.separate(simulation)

    np.testing.assert_allclose(times, np.array([0.0, 0.1, 0.2, 0.25]))
    np.testing.assert_allclose(paths[0], times)


def test_multivariate_brownian_override_uses_elapsed_intervals(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    process = MultivariateBrownianMotion(drift=[0.0], scale=[[1.0]])
    process.custom_increment = lambda _state, step_size: np.array([step_size])

    simulation = process.simulate(t=0.25, timestep=0.1)

    np.testing.assert_allclose(simulation[0], np.array([0.0, 0.1, 0.2, 0.25]))
    np.testing.assert_allclose(simulation[1], simulation[0])


def test_simulation_decorator_uses_elapsed_intervals(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    process = DecoratedUnitDriftProcess()

    simulation = process.simulate_path(t=0.25, timestep=0.1, num_instances=1)

    np.testing.assert_allclose(simulation[0], np.array([0.0, 0.1, 0.2, 0.25]))
    np.testing.assert_allclose(simulation[1], 1.0 + simulation[0])
