"""
constructor Submodule

The `Constructor` submodule provides a flexible, interactive framework for users to dynamically create custom stochastic process classes. By gathering user input such as process name, type, and parameters, this module generates Python class definitions for either **Ito** or **Non-Ito** processes, enabling simulation and modeling of a wide variety of stochastic systems.

Key Features:

1. **Custom Process Generation**:

    - Users can define a custom stochastic process interactively by providing basic details, such as:

        - Name of the process

        - Whether the process is of **Ito** or **Non-Ito** type

        - Parameters of the process (e.g., drift and stochastic terms)

        - The mathematical increment of the process (e.g., for Brownian motion or Lévy processes)

2. **Support for Both Ito and Non-Ito Processes**:

    - This module supports both Ito processes, which rely on stochastic calculus involving drift and stochastic terms, and Non-Ito processes, which may involve deterministic or fractional stochastic models.

    - Depending on whether the user selects the process to be **Ito** or **Non-Ito**, the generated class structure is modified accordingly.

3. **Interactive Code Generation**:

    - Once the user specifies the process details, the submodule dynamically generates and prints the Python class code for that process. This class includes:

        - Parameter initialization.

        - Custom increment calculation based on user input.

    The generated code follows the proper inheritance structure from either the `ItoProcess` or `NonItoProcess` base classes, ensuring compatibility with the larger framework.

4. **Simulation-Ready**:

    - The generated class can be used directly within the broader stochastic process framework for simulations. The custom increment function defines the process dynamics in a manner that integrates seamlessly with pre-existing tools.

Example Use Case:

A user might wish to simulate a custom geometric Brownian motion (GBM) with specific drift and volatility parameters. By following the interactive prompts, the user could input:

- Process name: "CustomGBM"

- Process type: Ito

- Parameters: "alpha, beta"

- Drift term: 0.05

- Stochastic term: 0.2

- Increment: `dW = dt**0.5 * np.random.normal(0, 1)`

- Increment equation: `dX = alpha * X * dt + beta * X * dW`

The submodule would then generate a fully functional Python class that the user can modify or simulate directly.

## Use Cases and Applications:

1. **Research and Experimentation**:

    - This module enables researchers to quickly define and simulate new stochastic processes for testing hypotheses or exploring new dynamics.

2. **Rapid Prototyping**:

    - For developers and scientists who need to build custom processes for specific simulations, this tool reduces the time required to write boilerplate code and allows for easy customization.

3. **Educational Purposes**:

    - This submodule is a helpful learning tool for students and practitioners to understand the structure of stochastic processes by generating and analyzing different custom processes.

## Example Workflow:

1. The user runs the `create_custom_process` function.

2. The system prompts for various details about the process (name, parameters, etc.).

3. The user inputs the increment function (drift and diffusion terms for Ito processes).

4. The module dynamically generates the Python code for the process, including initialization and simulation-ready custom increment logic.

5. The user receives a ready-to-use Python class that can be integrated into their broader simulations.

The `Constructor` submodule empowers users with full control over their process definitions, while ensuring consistency with Ito and Non-Ito frameworks in the overall stochastic process toolkit.

"""

from typing import List, Any, Type, Callable
from .definitions import ItoProcess
from .definitions import NonItoProcess
from .definitions import Process
from .default_values import *
import numpy as np
from .definitions import simulation_decorator
from ergodicity.configurations import *

# Function to get user input and create a custom process class
def create_custom_process():
    """
    This function allows the user to create a custom stochastic process class interactively by providing the necessary details such as process name, type, parameters, and increment function. The function then generates the Python class definition for the custom process based on the user input.

    :return: The generated Python class definition for the custom process.
    :rtype: str
    """
    name = input("What will be the name of the process? ")
    is_ito = input("Is this an Ito process? (yes/no) ").strip().lower() == 'yes'
    params = input("What are the parameters of the process? (comma separated, e.g., 'alpha, beta') ").split(',')
    if is_ito:
        drift_term = float(input("What is the drift term of the process? "))
        stochastic_term = float(input("What is the stochastic term of the process? "))
    dW_code = input("What is the increment of the process? (e.g., 'dW = dt**0.5 * np.random.normal(0,1)') ")
    increment_code = input("What is the increment of the process? (e.g., 'dX = mu * X * dt + sigma * X * dW') ")

    param_assignments = '\n        '.join([f'self._{param.strip()} = {param.strip()}' for param in params])

    if is_ito:
        class_code = f"""
    class {name}(ItoProcess):
        def __init__(self, {', '.join(params)}):
            super().__init__(name='{name}', process_class=None, drift_term={drift_term}, stochastic_term={stochastic_term})
            {param_assignments}

        def custom_increment(self, X: float, timestep: float = timestep_default) -> Any:
            {dW_code}
            dX = {increment_code}
            return dX
    """
    else:
        class_code = f"""
    class {name}(NonItoProcess):
        def __init__(self, {', '.join(params)}):
            super().__init__(name='{name}', process_class=None)
            {param_assignments}

        def custom_increment(self, X: float, timestep: float = timestep_default) -> Any:
            {dW_code}
            dX = {increment_code}
            return dX
    """

    print("Generated class definition:")
    print(class_code)

    return class_code
