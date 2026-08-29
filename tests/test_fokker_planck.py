import numpy as np

from ergodicity.tools.compute import solve_fokker_planck_numerically


def test_fokker_planck_constant_coefficients_match_standard_operator():
    """Constant coefficients reduce to -mu P_x + sigma^2 P_xx / 2."""
    drift = 0.3
    diffusion = 0.4
    initial_condition = lambda x: 1.0 - 0.2 * x - 0.1 * x**2

    x, t, probability = solve_fokker_planck_numerically(
        mu_func=lambda x_value, time: drift,
        sigma_func=lambda x_value, time: diffusion,
        P0_func=initial_condition,
        x_range=(-1.0, 1.0),
        t_range=(0.0, 0.01),
        Nx=7,
        Nt=2,
        boundary_conditions=(initial_condition(-1.0), initial_condition(1.0)),
        plot=False,
    )

    dx = x[1] - x[0]
    dt = t[1] - t[0]
    previous = probability[0]
    first_derivative = (previous[2:] - previous[:-2]) / (2.0 * dx)
    second_derivative = (
        previous[2:] - 2.0 * previous[1:-1] + previous[:-2]
    ) / dx**2
    expected = previous[1:-1] + dt * (
        -drift * first_derivative + 0.5 * diffusion**2 * second_derivative
    )

    np.testing.assert_allclose(probability[1, 1:-1], expected)


def test_fokker_planck_differentiates_variable_coefficient_products():
    """The update applies derivatives to mu*P and sigma^2*P, not only P."""
    time_step = 0.01

    _, _, probability = solve_fokker_planck_numerically(
        mu_func=lambda x_value, time: 2.0 * x_value,
        sigma_func=lambda x_value, time: np.sqrt(1.0 + x_value**2),
        P0_func=lambda x: np.ones_like(x),
        x_range=(-1.0, 1.0),
        t_range=(0.0, time_step),
        Nx=7,
        Nt=2,
        boundary_conditions=(1.0, 1.0),
        plot=False,
    )

    # For P=1, mu=2x, and sigma^2=1+x^2, the exact local operator is
    # -d(2x)/dx + (1/2)d^2(1+x^2)/dx^2 = -2 + 1 = -1.
    np.testing.assert_allclose(probability[1, 1:-1], 1.0 - time_step)
