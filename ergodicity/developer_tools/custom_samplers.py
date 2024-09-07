"""
custom_samplers Submodule Overview

The **`custom_samplers`** submodule provides utility functions for creating custom samplers from both cumulative distribution functions (CDFs) and characteristic functions (CFs). These functions allow for generating random samples from distributions that may not have built-in samplers in standard libraries, by using numerical methods such as inverse transform sampling and Fourier inversion.

Main Features:

1. **create_sampler_from_cdf**:

   - **Purpose**: Constructs a random sampler based on a given cumulative distribution function (CDF).

   - **Method**: Utilizes numerical inverse CDF (bisection method) to generate random samples.

   - **Parameters**:

     - `cdf`: The cumulative distribution function of the desired distribution.

     - `lower_bound`: The lower bound for the support of the distribution.

     - `initial_upper_bound`: The initial upper bound for the support of the distribution.

     - `tolerance`: Numerical tolerance for inverse CDF calculation.

     - `max_iterations`: Maximum iterations allowed to compute the inverse CDF.

   - **Returns**: A callable function that generates random samples from the specified distribution.

2. **characteristic_function_to_pdf**:

   - **Purpose**: Converts a characteristic function (CF) into a probability density function (PDF) via numerical inversion (using Fourier transform).

   - **Parameters**:

     - `cf`: The characteristic function of the desired distribution.

     - `t_values`: Array of `t` values to compute the inverse Fourier transform.

   - **Returns**: Arrays of PDF values and corresponding x-values for the distribution.

3. **create_sampler_from_cf**:

   - **Purpose**: Constructs a random sampler based on a given characteristic function (CF).

   - **Method**: Uses numerical inversion of the characteristic function to obtain a PDF, then generates samples by creating the corresponding CDF and applying inverse transform sampling.

   - **Parameters**:

     - `cf`: The characteristic function of the desired distribution.

     - `t_min` and `t_max`: Range of `t` values for the Fourier transform.

     - `n_points`: Number of points for the Fourier transform.

     - `lower_bound` and `upper_bound`: Bounds on the support of the distribution.

   - **Returns**: A callable function that generates random samples from the distribution defined by the characteristic function.

Use Cases:

1. **Custom Distributions**: For distributions where standard sampling methods do not exist or are inefficient, you can provide the CDF or CF and generate samples numerically.

2. **Distribution Simulation**: Create random samples for simulation purposes using custom distributions, such as non-standard or modified distributions in financial modeling, physics, or statistics.

3. **Numerical Inversion**: Utilize numerical methods like inverse transform sampling and Fourier inversion for specialized distributions, especially in cases where direct analytical sampling is difficult.

Example Usage:

### Creating a Sampler from a CDF:

def exponential_cdf(x, lam):

    return 1 - np.exp(-lam * x)

# Create a sampler for the exponential distribution with lambda = 1

exponential_sampler = create_sampler_from_cdf(exponential_cdf, lower_bound=0, initial_upper_bound=10)

# Generate random samples

samples = [exponential_sampler(lam=1) for _ in range(10)]

print(samples)
"""

import numpy as np
import inspect
import warnings
from ergodicity import custom_warnings as cw

def create_sampler_from_cdf(cdf, lower_bound, initial_upper_bound, tolerance=1e-10, max_iterations=1000, warning = True):
    """
    Creates a sampler function for a given distribution using its CDF.

    :param cdf: function: The cumulative distribution function (CDF) of the desired distribution.
    :type cdf: function
    :param lower_bound: float: The lower bound of the support of the distribution.
    :type lower_bound: float
    :param initial_upper_bound: float: The initial upper bound for the support of the distribution.
    :type initial_upper_bound: float
    :param tolerance: float: The tolerance level for the numerical inverse CDF calculation.
    :type tolerance: float
    :param max_iterations: int: The maximum number of iterations to find the upper bound if the initial upper bound is insufficient.
    :type max_iterations: int
    :return: function: A function that generates a random number from the given distribution using the specified parameters.
    :rtype: function
    """

    def inverse_cdf(u, lower_bound, upper_bound, tolerance, max_iterations, cdf, params):
        """
        Numerically computes the inverse CDF using the bisection method.

        :param u: float: The uniform random variable.
        :type u: float
        :param lower_bound: float: The lower bound of the support of the distribution.
        :type lower_bound: float
        :param upper_bound: float: The upper bound of the support of the distribution.
        :type upper_bound: float
        :param tolerance: float: The tolerance level for the numerical inverse CDF calculation.
        :type tolerance: float
        :param max_iterations: int: The maximum number of iterations to find the upper bound if the initial upper bound is insufficient.
        :type max_iterations: int
        :param cdf: function: The cumulative distribution function.
        :type cdf: function
        :param params: tuple: The parameters of the desired distribution.
        :type params: tuple
        :return: float: The value corresponding to the given u from the inverse CDF.
        :rtype: float
        """
        low, high = lower_bound, upper_bound
        iterations = 0
        while cdf(high, *params) < u and iterations < max_iterations:
            high *= 2
            iterations += 1

        if iterations >= max_iterations:
            raise ValueError("Max iterations reached. Consider increasing the initial upper bound.")

        while high - low > tolerance:
            mid = (low + high) / 2
            if cdf(mid, *params) < u:
                low = mid
            else:
                high = mid
        return (low + high) / 2

    # Get the parameter names of the CDF function
    cdf_params = dict(inspect.signature(cdf).parameters)
    cdf_params.pop('x', None)  # Remove 'x' from the parameters if it exists

    def sampler(**kwargs):
        """
        Generates a random number from the given distribution with specified parameters.

        :param kwargs: dict: The parameters of the desired distribution.
        :type kwargs: dict
        :return: float: A random number from the desired distribution.
        :rtype: float
        """
        # Ensure that all necessary parameters are provided
        if set(cdf_params) != set(kwargs.keys()):
            raise ValueError("Incorrect parameters provided. Expected parameters: {}".format(cdf_params.keys()))

        # Generate a uniform random sample
        u = np.random.uniform(0, 1)

        # Generate a sample from the desired distribution using the inverse CDF
        return inverse_cdf(u, lower_bound, initial_upper_bound, tolerance, max_iterations, cdf, tuple(kwargs.values()))

    if warning:
      print('Make sure you input a correct cumulative distribution function (CDF). This function will not work properly if you do not provide a valid CDF but will not raise an error. Set warning=False to suppress this message.')

    return sampler

# Example Usage
if __name__ == "__main__":
    # Define the CDF of the exponential distribution with lambda as a parameter
    def exponential_cdf(x, lam):
        return 1 - np.exp(-lam * x)

    # Create a sampler for the exponential distribution with an initial upper bound of 10
    exponential_sampler = create_sampler_from_cdf(exponential_cdf, lower_bound=0, initial_upper_bound=10)

    # Generate 10 random numbers from the exponential distribution with lambda = 1
    for _ in range(10):
        print(exponential_sampler(lam=1))

def characteristic_function_to_pdf(cf, t_values):
    """
    Numerically invert the characteristic function to obtain the PDF.

    :param cf: function: The characteristic function of the desired distribution.
    :type cf: function
    :param t_values: numpy.ndarray: An array of t values for the inverse Fourier transform.
    :type t_values: numpy.ndarray
    :return: numpy.ndarray: An array of PDF values obtained by inverting the characteristic function.
    :rtype: numpy.ndarray
    """
    n = len(t_values)
    dt = t_values[1] - t_values[0]

    # Compute the characteristic function values
    cf_values = cf(t_values)

    # print("CF values: ", cf_values)
    # safe CF values to file
    np.savetxt('cf_values.txt', cf_values)

    # Perform inverse FFT to obtain the PDF
    pdf_values = np.real(np.fft.ifft(cf_values))

    # Shift the zero-frequency component to the center
    pdf_values = np.fft.ifftshift(pdf_values)

    # Correct scaling for the PDF
    pdf_values *= n * dt / (2 * np.pi)

    # Generate corresponding x values
    x_values = np.fft.fftfreq(n, d=dt)
    x_values = np.fft.fftshift(x_values) * 2 * np.pi

    # print("X values: ", x_values)
    # safe x values to file
    # np.savetxt('x_values.txt', x_values)


    return pdf_values, x_values

def create_sampler_from_cf(cf, t_min=-0.5, t_max=0.5, n_points=10000, lower_bound=1000, upper_bound=1000, warning = True):
    """
    Creates a sampler function for a given distribution using its characteristic function.

    :param cf: function: The characteristic function of the desired distribution.
    :type cf: function
    :param t_min: float: The minimum t value for the Fourier transform.
    :type t_min: float
    :param t_max: float: The maximum t value for the Fourier transform.
    :type t_max: float
    :param n_points: int: The number of points for the Fourier transform.
    :type n_points: int
    :param lower_bound: float: The lower bound of the support of the distribution.
    :type lower_bound: float
    :param upper_bound: float: The upper bound of the support of the distribution.
    :type upper_bound: float
    :return: function: A function that generates a random number from the given distribution.
    :rtype: function
    """
    # Create an array of t values for the inverse Fourier transform
    t_values = np.linspace(t_min, t_max, n_points)

    # Get the PDF by inverting the characteristic function
    pdf_values, x_values = characteristic_function_to_pdf(cf, t_values)

    # Restrict x_values to the given bounds
    mask = (x_values >= lower_bound) & (x_values <= upper_bound)
    pdf_values = pdf_values[mask]
    x_values = x_values[mask]

    # Normalize PDF values again to ensure total probability is 1 within bounds
    pdf_values /= np.sum(pdf_values * (x_values[1] - x_values[0]))
    # print("PDF values: ", pdf_values)
    # np.savetxt('pdf_values.txt', pdf_values)

    # Create the CDF from the PDF
    cdf_values = np.cumsum(pdf_values)
    cdf_values /= cdf_values[-1]  # Normalize CDF
    # print("CDF values: ", cdf_values)
    # save cdf values to file
    # np.savetxt('cdf_values.txt', cdf_values)

    def sampler():
        """
        Generates a random number from the given distribution using the characteristic function.

        :return: float: A random number from the desired distribution.
        :rtype: float
        """
        # Generate a uniform random sample
        u = np.random.uniform(0, 1)

        # Find the corresponding value in the CDF using linear interpolation
        sample_value = np.interp(u, cdf_values, x_values)

        return sample_value

    if warning:
        print('Make sure you input a correct characteristic function (CF). This function will not work properly if you do not provide a valid CF but will not raise an error. Set warning=False to suppress this message.')

    warnings.warn('Attention!', cw.InDevelopmentWarning)
    warnings.warn('Attention!', cw.KnowWhatYouDoWarning)

    return sampler


# Example Usage
if __name__ == "__main__":
    # Define the characteristic function of the standard normal distribution
    def normal_cf(t):
        return np.exp(-0.5 * t ** 2)


    # Create a sampler for the normal distribution
    normal_sampler = create_sampler_from_cf(normal_cf, t_min=-10, t_max=10, n_points=2048, lower_bound=-5,
                                            upper_bound=5)

    # Generate 10 random numbers from the normal distribution
    samples = [normal_sampler() for _ in range(10)]
    print(samples)




