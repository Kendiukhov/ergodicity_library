# Contributing to Ergodicity Library

Thank you for considering contributing to the Ergodicity Library! The current version of the library is a **beta version** and is in active development. This document will guide you on how you can contribute to its development and provide useful insights into the current state of the library.

## Current Version and Development Process

The **current version of the library is in beta**, meaning most of its functionality is fully implemented and works as expected based on our assessments. However, certain modules and methods may still have issues, which will be indicated with **UserWarnings** when used, along with recommendations on how to avoid any potential problems.

While the library is mostly complete, it hasn't been thoroughly tested by external users yet. We highly value **feedback** from external testers and users, as it will help finalize the beta version and ensure the completeness and correctness of the implementation.

### What We Are Working On

Here are the key areas of development where you can help:

1. **Agents Module**:
    - Ensure full functionality of neural networks, evolutionary algorithms, and machine learning methods.
    - Optimize training processes and manage noisy training data, which arises due to the stochastic nature of simulations.
    - Expand environments, agent behaviors, and integrate additional neural network architectures.
    - Consider future integration with libraries such as **Evotorch** for evolutionary algorithms.

2. **Speed Optimization**:
    - Integrate the library with **C** and **Wolfram Mathematica** to optimize simulation and symbolic computation speed.
    - Improve the efficiency of simulations, especially those that involve long-running stochastic processes.

3. **Domain-Specific Integrations**:
    - Create integrations with libraries used in **economics**, **finance**, and **biology**.
    - Develop custom classes and methods that make applied work in these fields easier for users.

4. **Multi-Core Computations and Parallelism**:
    - Improve and deepen support for multi-core processing. The library already uses parallelism, but there’s more potential for improvement.

5. **New Research in Ergodicity Economics**:
    - Implement tools and methods based on ongoing research in ergodicity economics, especially from papers not covered by **Peters and Adamou (2018)**.

6. **Beta Modules**:
    - Bring certain classes and methods to a fully functional state, particularly those involving **characteristic function-based process generation** and **stochastic PDEs**.
    
7. **Additional Functionality**:
    - Experiment with extending the mathematical tools for ergodic analysis beyond **Ito calculus**.
    - Develop tools for handling **fat-tailed processes** and improve analysis methods for such processes.

8. **Graphical User Interface (GUI)**:
    - Build a user-friendly GUI to facilitate non-programmers' use of the library.

## How You Can Help

Any help in these or other directions is greatly appreciated. Some key areas where you can contribute:

- **Integration with domain-specific libraries** and expanding domain-related functionality.
- **Mathematical functionality**: help expand both symbolic and numerical solutions for stochastic and ergodic analysis.
- **Testing**: your tests will be invaluable in ensuring the library’s robustness.
- **Graphical Interface**: contribute to the development of a GUI to improve usability for a wider audience.

## Advice on Contributing and Usage

The library is a growing project, and here are some key things to keep in mind:

- **UserWarnings**: Pay attention to warnings in the code, as they will indicate whether certain modules are under development or not working as expected.
- **Feedback**: We welcome feedback through GitHub Issues, Google Forms, or direct emails (details available in the repository).
- **Modifications**: Feel free to modify the library without any restrictions. However, please be cautious when changing the **definitions.py** submodule, as it serves as a foundation for many of the library’s functions.
- **Parallelism**: Make use of parallelism when possible. It significantly reduces computation time for many tasks.
- **Development Aids**: Submodules like `constructor`, `custom_classes`, and `lib` are designed to make developing new processes and methods easier. Use them whenever possible.
- **Porting**: If you wish to port the library to another language, it may pose challenges due to Python-specific dependencies. However, critical dependencies rely on large, widely supported projects, so porting is feasible. **Rust** is a good candidate for porting due to its speed and flexibility, and it's a possibility for the future.

## Module Overview

Here’s a brief overview of the current state of the major modules:

- **Process Module**: Ready, except for secondary methods in the definitions submodule. The multiplicative submodule is fully functional.
- **Tools Module**: Fully functional, though additional symbolic computations and parallelism features are planned.
- **Agents Module**: Functional, but neural networks and learning algorithms require optimization. Additional strategies and environments for agents are under development.
- **Integrations and GUI Modules**: Not yet implemented but planned for future releases.
- **Developer Tools Module**: This is mainly for developers contributing to the library, containing technical functions used during development.
- **Speed Optimization**: Multi-core support is present, but further speed optimization is pending based on feedback.

## Future Plans

We plan to focus on the following areas:

- Collecting feedback from users and testers.
- Speed optimizations and enhancing computational performance.
- Developing the GUI and integrating domain-specific libraries.
- Expanding functionality related to fat-tailed processes and stochastic models.

## How to Contribute

1. **Fork the repository** and create a new branch for your feature or bug fix (e.g., `feature-new-functionality`).
2. **Make your changes** and ensure you follow best practices (see below).
3. **Submit a pull request** with a clear description of what you've changed or added.
4. Be open to feedback and further improvements.

## Coding Standards

- Follow **PEP 8** guidelines for Python code.
- Write tests for any new features or modules you implement.
- Document your code where necessary.

## Conclusion

Your contributions will help shape the future of this library. Whether it's optimizing code, implementing new features, or offering feedback, every bit of help is valued. Thank you for taking part in the development of the **Ergodicity Library**!

Feel free to reach out with any questions or feedback. Happy coding!
