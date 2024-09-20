"""
solve Submodule

The `solve` submodule is designed to tackle stochastic differential equations (SDEs) and related topics in stochastic calculus, particularly through symbolic computation. The submodule provides tools for applying Ito's Lemma, integrating SDEs, performing transformations for ergodicity, and solving Fokker-Planck equations. It also offers methods to analyze convergence rates and ergodic behavior in stochastic systems.

Key Features:

1. **Ito Calculus**:

   - `ito_differential`: Applies Ito's Lemma to a given function `f(x, t)` where `x` follows a stochastic differential equation (SDE).

   - `ito_integrate`: Performs Ito integration of drift and diffusion terms over time and Wiener processes.

2. **Solving SDEs**:

   - `solve`: Solves stochastic differential equations analytically where possible, by leveraging substitution techniques and Ito's Lemma.

   - `integration_check`: Validates the solution of an SDE by verifying consistency between the initial SDE and the solution found through integration.

3. **Ergodicity**:

   - `ergodicity_transform`: Checks the consistency condition for ergodicity and computes the transformation if the condition is met.

   - `calculate_time_average_dynamics`: Computes the time-average dynamics of a stochastic process using the ergodicity transform.

   - `compare_averages`: Compares the time and ensemble averages to determine long-term behavior.

4. **Convergence Analysis**:

   - `functions_convergence_rate`: Computes the convergence rate of a function to its asymptotic behavior.

   - `mean_convergence_rate_alpha_stable`: Calculates the convergence rate for processes in the domain of attraction of stable distributions.

   - `mad_convergence_rate_alpha_stable`: Determines the rate of convergence of the Mean Absolute Deviation (MAD) for alpha-stable processes.

5. **Fokker-Planck Equation**:

   - `solve_fokker_planck`: Symbolically solves the Fokker-Planck equation for a given drift and diffusion term under specified initial and boundary conditions.

6. **Advanced Theorems**:

   - `apply_girsanov`: Applies Girsanov’s theorem to change the drift of an SDE under a new probability measure, ensuring Novikov's condition holds.

   - `check_novikov_condition`: Validates the applicability of Girsanov's theorem by checking if Novikov's condition is satisfied.

7. **Utilities**:

   - `find_inverse`: Attempts to symbolically invert a function.

   - `asymptotic_approximation`: Finds the leading term in an expression as time tends to infinity, providing insights into the long-term behavior of a process.

   - `ensemble_average` and `time_average`: Calculate the ensemble and time averages of a stochastic process.

Applications:

- **Financial Mathematics**: Solve SDEs related to asset prices, option pricing, and portfolio optimization.

- **Physics**: Model diffusion processes, Brownian motion, and ergodic behavior in physical systems.

- **Engineering**: Analyze control systems and noise in dynamic environments.

- **Mathematical Research**: Study convergence rates, stochastic calculus, and the behavior of complex stochastic systems.

This submodule is a powerful toolkit for anyone involved in the analysis of stochastic processes, offering symbolic solutions and insights into the dynamics of complex systems.
"""
import sympy as sp
import re
from sympy.stats import Normal, E

def ito_differential(f, mu, sigma, x, t):
    """
    Apply Ito's Lemma to a function f(x,t) where x follows the SDE:
    dx = mu(x,t)dt + sigma(x,t)dW

    :param f: The function f(x,t) to which Ito's Lemma is applied
    :type f: sympy expression
    :param mu: The drift term mu(x,t) in the SDE
    :type mu: sympy expression
    :param sigma: The diffusion term sigma(x,t) in the SDE
    :type sigma: sympy expression
    :param x: The symbol representing the stochastic variable x
    :type x: sympy symbol
    :param t: The symbol representing time t
    :type t: sympy symbol
    :return: The resulting df(x,t) according to Ito's Lemma
    :rtype: sympy expression
    """
    # Compute partial derivatives
    df_dx = sp.diff(f, x)
    df_dt = sp.diff(f, t)
    d2f_dx2 = sp.diff(f, x, 2)

    # Apply Ito's Lemma
    df = (df_dt + mu * df_dx + 0.5 * sigma ** 2 * d2f_dx2) * sp.symbols('dt') + sigma * df_dx * sp.symbols('dW(t)')

    # Expand the expression
    expanded = sp.expand(df)

    # Separate dt and dW(t) terms
    dt = sp.Symbol('dt')
    dW = sp.Symbol('dW(t)')
    dt_coeff = expanded.coeff(dt)
    dW_coeff = expanded.coeff(dW)

    # Simplify dt coefficient
    dt_coeff = sp.collect(dt_coeff, x)
    dt_coeff = sp.cancel(dt_coeff)

    # Simplify dW coefficient
    dW_coeff = sp.collect(dW_coeff, x)
    dW_coeff = sp.cancel(dW_coeff)

    df = dt_coeff * dt + dW_coeff * dW

    return df

def extract_ito_terms(differential, x, t):
    """
    Extract the mu(x,t) and sigma(x,t) terms from a differential expression
    of the form: mu(x,t)dt + sigma(x,t)dW(t)

    :param differential: The differential expression, typically the result of applying Ito's Lemma
    :type differential: sympy expression
    :param x: The symbol representing the stochastic variable x
    :type x: sympy symbol
    :param t: The symbol representing time t
    :type t: sympy symbol
    :return: A tuple of sympy expressions (mu(x,t), sigma(x,t))
    :rtype: tuple
    """
    print('differential:', differential)
    dt = sp.symbols('dt')
    dW = sp.symbols('dW(t)')

    # Expand the differential
    expanded_diff = sp.expand(differential)

    # Collect terms with respect to dt and dW
    collected = sp.collect(expanded_diff, [dt, dW], evaluate=False)

    # Extract mu(x,t) and sigma(x,t)
    mu = collected.get(dt, 0)
    sigma = collected.get(dW, 0)

    # Simplify mu and sigma
    mu = sp.simplify(mu)
    sigma = sp.simplify(sigma)

    return mu, sigma

def ito_integrate(mu, sigma, x, t, x0=1):
    """
    Integrate terms by dt and dW(t) from 0 to t.
    The dt term is integrated as a Riemann integral.
    The dW(t) term is integrated using Ito isometry.

    :param mu: The coefficient of dt (drift term)
    :type mu: sympy expression
    :param sigma: The coefficient of dW(t) (diffusion term)
    :type sigma: sympy expression
    :param x: The symbol representing the stochastic variable x
    :type x: sympy symbol
    :param t: The symbol representing time t
    :type t: sympy symbol
    :param x0: The initial value of x at t=0, defaults to 1
    :type x0: int, optional
    :return: The Ito integral expression
    :rtype: sympy expression
    """
    # Check if mu and sigma depend only on t
    if mu.has(x) or sigma.has(x):
        raise ValueError("mu and sigma must depend only on t, not on x")

    # Integrate mu dt (Riemann integral)
    integral_mu = sp.integrate(mu, (t, 0, t))

    # Integrate sigma dW(t) using Ito isometry
    # E[(∫sigma dW)^2] = ∫sigma^2 dt
    integral_sigma_squared = sp.integrate(sigma ** 2, (t, 0, t))

    W = sp.symbols('W(t)')

    integral_sigma = sp.sqrt(integral_sigma_squared / t)

    ito_integral = x0 + integral_mu + integral_sigma*W

    return ito_integral

def find_inverse(f, x, y):
    """
    Attempt to find the inverse of f(x,t) with respect to x.

    :param f: The function f(x,t) to invert
    :type f: sympy expression
    :param x: The symbol representing the variable x
    :type x: sympy symbol
    :param y: The symbol to represent f(x,t) in the inverse function
    :type y: sympy symbol
    :return: The inverse function if found, None otherwise
    :rtype: sympy expression or None
    :exception: Returns None if an error occurs during the calculation
    """
    try:
        # Attempt to solve f(x,t) = y for x
        solution = sp.solve(sp.Eq(f, y), x)

        if solution:
            # Return the first solution if multiple exist
            return solution[0]
        else:
            print("No inverse function found.")
            return None
    except Exception as e:
        print(f"Error in finding inverse: {e}")
        return None


def integrate_with_substitution(f, mu, sigma, x, t, x0=1):
    """
    Perform Ito integration with substitution.

    :param f: The function f(x,t) for substitution
    :type f: sympy expression
    :param mu: The drift term mu(x,t)
    :type mu: sympy expression
    :param sigma: The diffusion term sigma(x,t)
    :type sigma: sympy expression
    :param x: The symbol representing the stochastic variable x
    :type x: sympy symbol
    :param t: The symbol representing time t
    :type t: sympy symbol
    :param x0: The initial value of x at t=0, defaults to 1
    :type x0: int, optional
    :return: The result of the integration after substitution
    :rtype: sympy expression
    :exception: Returns None if an error occurs during the calculation
    """
    try:
        # 1. Find Ito differential df(x,t)
        df = ito_differential(f, mu, sigma, x, t)
        print(f"Ito differential df(x,t): {df}")

        # 2. Extract Ito terms
        mu_f, sigma_f = extract_ito_terms(df, x, t)
        print(f"Extracted mu_f: {mu_f}")
        print(f"Extracted sigma_f: {sigma_f}")

        # f0 = f(x0, 0)
        f0 = f.subs({x: x0, t: 0})
        print('f0', f0)

        # 3. Ito integrate
        y = sp.Function('y')(t)  # New function y(t) to represent f(x,t)
        integral_result = ito_integrate(mu_f.subs(x, y), sigma_f.subs(x, y), y, t, x0=f0)
        print(f"Ito integral result: {integral_result}")

        # 4. Find the reverse function of f
        y_symbol = sp.symbols('y')
        inverse_f = find_inverse(f, x, y_symbol)
        if inverse_f is None:
            raise ValueError("Unable to find inverse of f(x,t)")
        print(f"Inverse of f(x,t): x = {inverse_f}")

        # 5. Substitute back to get x(t)
        final_result = inverse_f.subs(y_symbol, integral_result)
        print(f"Final result: {final_result}")

        return final_result

    except Exception as e:
        print(f"Error in integration with substitution: {e}")
        return None

def find_dx(x, t, W):
    """
    Find dX(X,t,W(t)) for a given X(t,W(t)) using Ito calculus rules,
    and attempt to simplify the result in terms of X where possible.

    :param x: The stochastic process X(t,W(t))
    :type x: sympy expression
    :param t: The symbol representing time t
    :type t: sympy symbol
    :param W: The Wiener process W(t)
    :type W: sympy function
    :return: The SDE in the form dX = ... dt + ... dW, with X substituted where possible
    :rtype: sympy expression
    :exception: Returns None if an error occurs during the calculation
    """
    try:
        # Calculate partial derivatives
        dX_dt = sp.diff(x, t)
        dX_dW = sp.diff(x, W)
        d2X_dW2 = sp.diff(x, W, 2)

        print('x expression:', x)

        # Apply Ito's Lemma
        dt = sp.symbols('dt')
        dW = sp.symbols('dW(t)')

        dX = (dX_dt + 0.5 * d2X_dW2) * dt + dX_dW * dW

        # Simplify and collect terms
        dX = sp.collect(sp.expand(dX), [dt, dW])

        dX = sp.simplify(dX)

        # Attempt to substitute X back into the expression
        X_symbol = sp.symbols('x')
        dX = dX.subs(x, X_symbol)

        dX = sp.collect(dX, [dt, dW])

        print('dX:', dX)

        return dX

    except Exception as e:
        print(f"Error in finding SDE: {e}")
        return None

def integration_check(f, mu, sigma, x, t, x0=1):
    """
    Check the integration of a function f(x,t) with drift mu(x,t) and diffusion sigma(x,t).

    :param f: The function f(x,t) to integrate
    :type f: sympy expression
    :param mu: The drift term mu(x,t)
    :type mu: sympy expression
    :param sigma: The diffusion term sigma(x,t)
    :type sigma: sympy expression
    :param x: The symbol representing the stochastic variable x
    :type x: sympy symbol
    :param t: The symbol representing time t
    :type t: sympy symbol
    :param x0: The initial value of x at t=0, defaults to 1
    :type x0: int, optional
    :return: The result of the integration if successful
    :rtype: sympy expression
    """
    integration_result = integrate_with_substitution(f, mu, sigma, x, t, x0=x0)
    differential_result = find_dx(integration_result, t, sp.symbols('W(t)'))
    mu_new, sigma_new = extract_ito_terms(differential_result, x, t)

    print(f'Integration result: {integration_result}')

    # replace integration result inside mu_new and sigma_new with x if it is there
    mu_new = mu_new.subs(integration_result, x)
    sigma_new = sigma_new.subs(integration_result, x)

    if mu_new == mu and sigma_new == sigma:
        print("Integration successful.")

    else:
        print("Integration failed.")
        print(f"Expected mu: {mu}, sigma: {sigma}")
        print(f"Actual mu: {mu_new}, sigma: {sigma_new}")

    return integration_result

def replace_integration_constants(expr):
    """
    Check if a given expression contains integration constants (C1, C2, etc. or numbers)
    and substitute them with 1.

    :param expr: The expression to check and modify
    :type expr: sympy expression
    :return: The modified expression with integration constants replaced by 1
    :rtype: sympy expression
    """
    # Function to check if a term is a standalone constant
    def is_standalone_constant(term):
        return (isinstance(term, sp.Symbol) and re.match(r'C\d+', term.name)) or \
               (isinstance(term, sp.Number) and term != 0)

    # Function to replace constants in a term
    def replace_constant(term):
        if isinstance(term, sp.Mul):
            factors = term.args
            new_factors = [1 if is_standalone_constant(f) else f for f in factors]
            return sp.Mul(*new_factors)
        elif is_standalone_constant(term):
            return sp.Integer(1)
        else:
            return term

    # Apply the replacement recursively
    return expr.replace(lambda x: True, replace_constant)

def find_substitution(mu, sigma, x, t):
    """
    Find a suitable substitution for solving a stochastic differential equation analytically.

    :param mu: The drift term mu(x,t)
    :type mu: sympy expression
    :param sigma: The diffusion term sigma(x,t)
    :type sigma: sympy expression
    :param x: The symbol representing the stochastic variable x
    :type x: sympy symbol
    :param t: The symbol representing time t
    :type t: sympy symbol
    :return: The substitution function f(t,x) if found, None otherwise
    :rtype: sympy expression or None
    :exception: Returns None if an error occurs during the calculation
    """
    try:
        # Step 1: Find sigma_t and sigma_xx
        sigma_t = sp.diff(sigma, t)
        sigma_xx = sp.diff(sigma, x, 2)

        # Step 2: Check the condition
        condition = sigma * ((1/(sigma ** 2)) * sigma_t - sp.diff(mu / sigma, x) + 0.5 * sigma_xx)
        condition_derivative = sp.diff(condition, x)
        print(f"Condition: {condition}")

        if condition_derivative != 0:
            print("Error: Analytical solution does not exist for this differential.")
            return None

        # Step 3: Find sigma_new
        sigma_new = sp.Function('sigma_new')(t)
        diff_eq = sp.Eq(sigma_new.diff(t) / sigma_new, condition)
        sigma_new_sol = sp.dsolve(diff_eq, sigma_new)

        sigma_new_sol = sigma_new_sol.subs(sp.symbols('C1'), 1)

        # sigma_new_sol = replace_integration_constants(sigma_new_sol)

        # Extract the general solution (without constants)
        if isinstance(sigma_new_sol, sp.Eq):
            sigma_new = sigma_new_sol.rhs
        else:
            sigma_new = sigma_new_sol

        print(f"sigma_new: {sigma_new_sol}")

        # Step 4: Find f
        f = sp.integrate(sigma_new / sigma, x)

        print('Substitution found:', f)

        return f

    except Exception as e:
        print(f"Error in finding substitution: {e}")
        return None

def solve(mu, sigma, x, t, x0=1):
    """
    Solve the stochastic differential equation dx = mu dt + sigma dW(t) for x(t) analytically.

    :param mu: The drift term mu(x,t)
    :type mu: sympy expression
    :param sigma: The diffusion term sigma(x,t)
    :type sigma: sympy expression
    :param x: The symbol representing the stochastic variable x
    :type x: sympy symbol
    :param t: The symbol representing time t
    :type t: sympy symbol
    :param x0: The initial value of x at t=0, defaults to 1
    :type x0: int, optional
    :return: The solution x(t) if found
    :rtype: sympy expression
    """
    substitution = find_substitution(mu, sigma, x, t)
    integration_result = integration_check(substitution, mu, sigma, x, t, x0=x0)

    return integration_result

def ergodicity_transform(mu, sigma, x, t):
    """
    Check the consistency condition for ergodicity and find the ergodicity transform if consistent.
    b_u from the ergodicity economics book (page 57) is set to 1. Then, a_u = c.
    The function is implemented only for the case when mu ang sigma are only functions of x.

    :param mu: The drift term mu(x,t)
    :type mu: sympy expression
    :param sigma: The diffusion term sigma(x,t)
    :type sigma: sympy expression
    :param x: The symbol representing the stochastic variable x
    :type x: sympy symbol
    :param t: The symbol representing time t
    :type t: sympy symbol
    :return: A tuple (is_consistent, u(t,x), a_u, b_u) where u(t,x) is the ergodicity transform
    :rtype: tuple
    :exception: Returns (False, None, None) if an error occurs during the calculation
    """
    try:
        # Step 1: Check consistency
        sigma_x = sp.diff(sigma, x)
        c = sp.Symbol('c')
        consistency_condition = c * sigma + 0.5 * sigma * sigma_x

        # Solve for c
        c_equation = sp.Eq(mu, consistency_condition)
        c_solution = sp.solve(c_equation, c)

        is_consistent = len(c_solution) > 0

        print(f"Consistency check: mu = {mu}")
        print(f"Consistency check: c*sigma + 1/2 * sigma * sigma_x = {consistency_condition}")
        print(f"Is consistent: {is_consistent}")

        if not is_consistent:
            print("Error: The given mu and sigma do not satisfy the ergodicity consistency condition for any c.")
            return False, None, None

        # If multiple solutions, take the simplest one (usually the first)
        c_value = c_solution[0] if isinstance(c_solution, list) else c_solution
        print(f"Consistent for c = {c_value}")

        # Step 2: Find u(t,x)
        u = sp.integrate(1 / sigma, x)

        a_u = c_value
        b_u = 1

        print(f"Ergodicity transform u(t,x) = {u}")

        return True, u, a_u, b_u

    except Exception as e:
        print(f"Error in ergodicity transform calculation: {e}")
        return False, None, None

def dynamic_from_utility(u_function, x, mu_u, sigma_u):
    """
    Calculate the dynamics of the ergodicity transform u(t,x) using Ito calculus.
    It finds a stochastic dynamic corresponding to the given ergodicity transform.
    Hence, it allows to estimate what stochastic dynamics corresponds to the given "utility" function.

    :param u_function: The ergodicity transform u(t,x)
    :type u_function: sympy expression
    :param x: The symbol representing the stochastic variable x
    :type x: sympy symbol
    :param mu_u: The drift term mu_u(x,t)
    :type mu_u: sympy expression
    :param sigma_u: The diffusion term sigma_u(x,t)
    :type sigma_u: sympy expression
    :return: The stochastic dynamic corresponding to the ergodicity transform u(t,x)
    :rtype: sympy expression
    """
    t = sp.symbols('t')
    u = sp.symbols('u')
    u_inverse = find_inverse(u_function, x, u)
    dx = ito_differential(f=u_inverse, mu=mu_u, sigma=sigma_u, x=u, t=t)
    dx = dx.subs(u, u_function)
    return dx

def calculate_time_average_dynamics(u, a_u, x, t):
    """
    Calculate the time average dynamics using the ergodicity transform.
    It is done by inverting the ergodicity transform and applying the inverse to a_u * t.

    :param u: The ergodicity transform u(t,x)
    :type u: sympy expression
    :param a_u: The transformed drift term
    :type a_u: sympy expression
    :param x: The symbol representing the stochastic variable x
    :type x: sympy symbol
    :param t: The symbol representing time t
    :type t: sympy symbol
    :return: The time average dynamics expression
    :rtype: sympy expression
    :exception: Returns None if an error occurs during the calculation
    """
    try:
        # Step 1: Invert the ergodicity transform u
        y = sp.Symbol('y')  # Temporary symbol for inversion
        u_inv = sp.solve(sp.Eq(u, y), x)

        if not u_inv:
            raise ValueError("Could not invert the ergodicity transform u")

        u_inv = u_inv[0]  # Take the first solution if multiple exist

        # Step 2: Apply the inverse to a_u to get the time average dynamics
        time_avg_dynamics = u_inv.subs(y, a_u * t)

        print(f"Inverse of ergodicity transform: x = {u_inv}")
        print(f"Time average dynamics: x(t) = {time_avg_dynamics}")

        return time_avg_dynamics

    except Exception as e:
        print(f"Error in calculating time average dynamics: {e}")
        return None

def time_average(mu, sigma, x, t):
    """
    Calculate the time average of a stochastic process x(t, W(t)) using the ergodicity transform.

    :param mu: The drift term mu(x,t)
    :type mu: sympy expression
    :param sigma: The diffusion term sigma(x,t)
    :type sigma: sympy expression
    :param x: The symbol representing the stochastic variable x
    :type x: sympy symbol
    :param t: The symbol representing time t
    :type t: sympy symbol
    :return: The time average of x(t, W(t))
    :rtype: sympy expression
    :exception: Returns None if an error occurs during the calculation
    """
    try:
        # Step 1: Find the ergodicity transform
        is_consistent, u, a_u, b_u = ergodicity_transform(mu, sigma, x, t)

        if not is_consistent:
            raise ValueError("Ergodicity transform not found")

        # Step 2: Calculate the time average dynamics
        time_avg_dynamics = calculate_time_average_dynamics(u, a_u, x, t)

        print(f"Time average function: {time_avg_dynamics}")

        return time_avg_dynamics

    except Exception as e:
        print(f"Error in calculating time average: {e}")
        return None

def ergodicity_transform_differential(u, a_u, b_u, x, t):
    """
    Calculate the differential form of the ergodicity transform.
    The function is implemented only for the case when mu ang sigma are only functions of x.

    :param u: The ergodicity transform u(t,x)
    :type u: sympy expression
    :param a_u: The transformed drift term a_u
    :type a_u: sympy expression
    :param b_u: The transformed diffusion term b_u
    :type b_u: sympy expression
    :param x: The symbol representing the stochastic variable x
    :type x: sympy symbol
    :param t: The symbol representing time t
    :type t: sympy symbol
    :return: The transformed differential df(x,t) = a_u dt + b_u dW(t)
    :rtype: sympy expression
    :exception: Returns None if an error occurs during the calculation
    """
    try:
        dt = sp.Symbol('dt')
        dW = sp.Function('dW')(t)

        df = a_u * dt + b_u * dW

        print(f"Transformed drift term a_u = {a_u}")
        print(f"Transformed diffusion term b_u = {b_u}")
        print(f"Transformed differential df = {df}")

        return df

    except Exception as e:
        print(f"Error in ergodicity transform differential calculation: {e}")
        return None

def ensemble_average(x_expr, t, W):
    """
    Calculate the expected value of a stochastic process x(t, W(t))
    by replacing W(t) with a standard normal distribution N(0,1).

    :param x_expr: The stochastic process x(t, W(t))
    :type x_expr: sympy expression
    :param t: The symbol representing time t
    :type t: sympy symbol
    :param W: The Wiener process W(t)
    :type W: sympy function
    :return: The expected value of x(t, W(t))
    :rtype: sympy expression
    :exception: Returns None if an error occurs during the calculation
    """
    try:
        # Create a standard normal random variable
        z = Normal('z', 0, 1)

        # Replace W(t) with z in the expression
        x_normal = x_expr.subs(W, z*sp.sqrt(t))

        # Calculate the expected value
        expected_value = E(x_normal)

        print(f"Expected value: {expected_value}")

        return expected_value

    except Exception as e:
        print(f"Error in expected value calculation: {e}")
        return None

def time_average_limit(x_expr, t, W):
    """
    Calculate the time average of a stochastic process x(t, W(t))
    by expressing W(t) as sqrt(t) * N(0,1) and finding the limit of x/t as t -> infinity.

    :param x_expr: The stochastic process x(t, W(t))
    :type x_expr: sympy expression
    :param t: The symbol representing time t
    :type t: sympy symbol
    :param W: The Wiener process W(t)
    :type W: sympy function
    :return: The time average of x(t, W(t))
    :rtype: sympy expression
    :exception: Returns None if an error occurs during the calculation
    """
    try:
        # Create a standard normal random variable
        z = Normal('z', 0, 1)

        # Replace W(t) with sqrt(t) * z in the expression
        x_normal = x_expr.subs(W, sp.sqrt(t) * z)

        # Divide by t
        x_over_t = x_normal / t

        # Calculate the limit as t approaches infinity
        time_avg = sp.limit(x_over_t, t, sp.oo)

        print(f"Time average (limit as t -> inf): {time_avg}")

        return time_avg

    except Exception as e:
        print(f"Error in time average calculation: {e}")
        return None

def compare_growth(term1, term2, t):
    """
    Compare the growth rates of two terms as t approaches infinity.
    A helper function for asymptotic approximation.

    :param term1: The first term to compare
    :type term1: sympy expression
    :param term2: The second term to compare
    :type term2: sympy expression
    :param t: The symbol representing the variable approaching infinity
    :type t: sympy symbol
    :return: 1 if term1 grows faster, -1 if term2 grows faster, 0 if equal
    :rtype: int
    """
    if term1 == term2:
        return 0

    # Extract the power of t in each term
    def extract_t_power(term):
        if term.has(t):
            return max([degree for base, degree in term.as_powers_dict().items() if base == t] + [0])
        return 0

    power1 = extract_t_power(term1)
    power2 = extract_t_power(term2)

    if power1 > power2:
        return 1
    elif power1 < power2:
        return -1
    else:
        # If powers are equal, compare the coefficients
        coeff1 = term1.coeff(t ** power1)
        coeff2 = term2.coeff(t ** power2)
        if coeff1 == coeff2:
            return 0
        limit = sp.limit(sp.Abs(coeff1 / coeff2), t, sp.oo)
        if limit == sp.oo:
            return 1
        elif limit == 0:
            return -1
        else:
            return 0

def asymptotic_approximation(expr, t=sp.symbols('t'), W=sp.symbols('W(t)')):
    """
    Find the asymptotic approximation of an expression as t approaches infinity.
    This basiscally allows to calculate time average behavior of the expression.

    :param expr: The expression to approximate
    :type expr: sympy expression
    :param t: The symbol representing the variable approaching infinity
    :type t: sympy symbol
    :param W: The Wiener process W(t)
    :type W: sympy function
    :returns: The leading term in the expression as t -> infinity
    :rtype: sympy expression
    :exception: Returns None if an error occurs during the calculation
    """
    try:

        z = sp.symbols('z')

        expr = expr.subs(W, z * sp.sqrt(t))

        terms = expr.as_ordered_terms() if expr.is_Add else [expr]

        dominant_term = terms[0]
        for term in terms[1:]:
            comparison = compare_growth(term, dominant_term, t)
            if comparison > 0:
                dominant_term = term
            elif comparison == 0:
                dominant_term += term

        # Special handling for exponential functions
        if dominant_term.has(sp.exp):
            exponents = [arg.args[0] for arg in dominant_term.atoms(sp.exp)]
            if exponents:
                max_exponent = max(exponents, key=lambda x: asymptotic_approximation(x, t))
                dominant_term = sp.exp(asymptotic_approximation(max_exponent, t))

        # Simplify the dominant term
        dominant_term = sp.simplify(dominant_term)

        print(f"Original expression: {expr}")
        print(f"Asymptotic approximation: {dominant_term}")

        return dominant_term

    except Exception as e:
        print(f"Error in asymptotic approximation: {e}")
        return None

def compare_averages(expr, t=sp.symbols('t'), W=sp.symbols('W(t)')):
    """
    Compare the time average and ensemble average of a stochastic process.

    :param expr: The stochastic process x(t, W(t))
    :type expr: sympy expression
    :param t: The symbol representing time t
    :type t: sympy symbol
    :param W: The Wiener process W(t)
    :type W: sympy function
    :return: The ratio of time average to ensemble average
    :rtype: float
    """

    # Calculate time average
    time_avg = asymptotic_approximation(expr, t, W)

    # Calculate ensemble average
    ensemble_avg = ensemble_average(expr, t, W)

    # Calculate the ratio
    ratio = time_avg / ensemble_avg

    print(f"Time to ensemble average ratio: {ratio}")

    return ratio

def functions_convergence_rate(f, g, t):
    """
    Calculate the rate of convergence of f(t) to its asymptotic approximation g(t).

    :param f: The original function
    :type f: sympy expression
    :param g: The asymptotic approximation
    :type g: sympy expression
    :param t: The symbol representing the variable approaching infinity
    :type t: sympy symbol
    :return: The order of convergence
    :rtype: sympy expression
    :exception: Returns None if an error occurs during the calculation
    """
    try:
        # Calculate the relative error
        relative_error = sp.Abs((f - g) / g)

        # Try to find the limit of t^n * relative_error for increasing n
        n = 1
        while True:
            limit = sp.limit(t ** n * relative_error, t, sp.oo)
            if limit == 0:
                n += 1
            elif sp.oo in [limit, -limit]:
                n -= 1
                break
            else:
                break

        if n > 0:
            return f"O(1/t^{n})"
        else:
            return "Convergence rate not found"

    except Exception as e:
        print(f"Error in calculating convergence rate: {e}")
        return None

def mean_convergence_rate_alpha_stable(alpha, n):
    """
    Calculate the convergence rate for a distribution with infinite variance (alpha-stable).

    :param alpha: The stability parameter of the alpha-stable distribution (0 < alpha < 2)
    :type alpha: float
    :param n: The symbol representing the sample size
    :type n: sympy symbol
    :return: The rate of convergence
    :rtype: sympy expression
    :raises: ValueError if alpha is not in the range (0, 2)
    """
    if alpha <= 0 or alpha >= 2:
        raise ValueError("Alpha must be between 0 and 2 exclusively")

    rate = sp.O(1 / (n ** (1 / alpha)))

    print(f"For a distribution in the domain of attraction of a stable law with index {alpha}:")
    print(f"Rate of convergence: {rate}")

    return rate

def mad_convergence_rate_alpha_stable(alpha, n):
    """
    Calculate the convergence rate of Mean Absolute Deviation for alpha-stable processes.

    :param alpha: The stability parameter of the alpha-stable distribution (1 < alpha < 2)
    :type alpha: float
    :param n: The symbol representing the sample size
    :type n: sympy symbol
    :return: The rate of convergence of the sample MAD
    :rtype: sympy expression
    :raises: ValueError if alpha is not in the range (1, 2)
    """
    if alpha <= 1 or alpha >= 2:
        raise ValueError("Alpha must be between 1 and 2 exclusively for finite MAD")

    rate = sp.O(1 / (n ** (1 / alpha - 1)))

    print(f"For an alpha-stable process with alpha = {alpha}:")
    print(f"Rate of convergence of sample MAD: {rate}")
    print(f"Compared to normal rate (1/sqrt(n)): O(1/n^{1 / 2:.4f})")

    return rate

# Function to check Novikov's condition for applicability of Girsanov's theorem
def check_novikov_condition(theta_t, t, T):
    """
    Checks Novikov's condition over a finite time horizon T.

    :param theta_t: The adjustment to the drift (Radon-Nikodym derivative)
    :type theta_t: sympy expression
    :param t: The time variable
    :type t: sympy symbol
    :param T: The finite time horizon
    :type T: float or sympy expression
    :return: Whether Novikov's condition holds over [0, T]
    :rtype: boolean
    """
    # Compute the integral from 0 to T
    novikov_integral = sp.integrate(theta_t ** 2 / 2, (t, 0, T))
    # Compute the expectation
    expectation_novikov = sp.exp(novikov_integral)
    # Check if the expectation is finite
    condition = expectation_novikov.is_finite
    # Ensure that condition is a boolean value
    return bool(condition)

# Function to apply Girsanov's theorem and return the transformed drift
def apply_girsanov(initial_drift, new_drift, diffusion, time_horizon):
    """
    Applies Girsanov's theorem to an Itô process to change the drift of the process.

    :param initial_drift: The initial drift function mu_initial(X_t, t)
    :type initial_drift: sympy expression
    :param new_drift: The new drift function mu_new(X_t, t)
    :type new_drift: sympy expression
    :param diffusion: The diffusion term sigma(X_t, t)
    :type diffusion: sympy expression
    :param time_horizon: The time horizon T for the process
    :type time_horizon: float
    :return: The transformed drift and the new process
    :rtype: sympy expression
    :raises ValueError: If Novikov's condition is not satisfied
    """
    W_t = sp.Symbol('W_t')
    # Define the Radon-Nikodym derivative (theta_t is the change in drift)
    theta_t = (new_drift - initial_drift) / diffusion
    t = sp.symbols('t')
    novikov_condition = check_novikov_condition(theta_t, t, T=time_horizon)
    if not novikov_condition:
        raise ValueError("Novikov's condition is not satisfied. Girsanov's theorem cannot be applied.")

    # Initial PDF under P (assuming W_t is N(0, t))
    initial_pdf = (1 / sp.sqrt(2 * sp.pi * time_horizon)) * sp.exp(-W_t ** 2 / (2 * time_horizon))

    # Girsanov transformation: New measure requires reweighting the original PDF
    radon_nikodym_derivative = sp.exp(-theta_t * W_t - (1 / 2) * sp.integrate(theta_t ** 2, (t, 0, time_horizon)))

    # New PDF under the new measure
    new_pdf = initial_pdf * radon_nikodym_derivative

    # Return the new PDF (simplified)
    return sp.simplify(new_pdf)

def calculate_moment(pdf, x, n):
    """
    Calculate the n-th moment of a probability density function (PDF).

    :param pdf: The probability density function (PDF)
    :type pdf: sympy expression
    :param x: The random variable in the PDF
    :type x: sympy symbol
    :param n: The order of the moment to calculate (e.g., 1 for mean, 2 for variance)
    :type n: int
    :return: The n-th moment of the PDF
    :rtype: sympy expression
    """
    # Define the n-th moment: integral of x^n * f(x) dx
    moment_n = sp.integrate(x ** n * pdf, (x, -sp.oo, sp.oo))

    # Simplify the result
    return sp.simplify(moment_n)











