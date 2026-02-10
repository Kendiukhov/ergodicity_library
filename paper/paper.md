---
title: 'Ergodicity Library: Integrated stochastic-process simulation, ergodicity analysis, and agent-based experimentation in Python'
tags:
  - Python
  - stochastic processes
  - ergodicity
  - heavy-tailed dynamics
  - agent-based modeling
authors:
  - name: Ihor Kendiukhov
    corresponding: true
    affiliation: 1
affiliations:
  - name: University of Tübingen, Germany
    index: 1
date: 10 February 2026
bibliography: paper.bib
---

# Summary

`ergodicity_library` is an open-source Python package for computational research on stochastic systems, with emphasis on non-ergodicity, multiplicative dynamics, and decision-making under uncertainty [@kendiukhov2025repo; @peters2016evaluating; @peters2019ergodicity]. The package combines three capabilities that are often split across unrelated scripts: (1) simulation of stochastic processes, including heavy-tailed and memory-dependent variants, (2) diagnostics and analytical tooling for time-vs-ensemble behavior, and (3) agent-oriented experimentation where decisions are coupled to stochastic environments.

For non-specialist readers, the core goal is practical: reduce the amount of custom glue code needed to move from a stochastic hypothesis to an interpretable computational result. In many projects, researchers switch repeatedly between process simulation, distributional checks, growth diagnostics, and adaptive decision logic. This software provides these stages in a shared environment instead of forcing users to build each transition manually.

# Statement of need

Many research questions in finance, economics, physics-inspired modeling, and quantitative social simulation require analyzing systems where typical trajectory outcomes differ from ensemble expectations. Standard expected-value pipelines are often insufficient for these settings, especially under multiplicative or heavy-tailed dynamics [@peters2016evaluating; @peters2019ergodicity]. Researchers therefore need workflows that can compare process families, inspect transient behavior, evaluate fitting assumptions, and test decision rules against the same underlying simulations.

The need addressed by `ergodicity_library` is an integrated, research-oriented workflow for these tasks. The intended users are applied researchers, graduate-level users, and research software practitioners who need to:

- simulate heterogeneous stochastic process families with a consistent interface,
- compute diagnostics that explicitly separate time-average and ensemble perspectives,
- fit and evaluate distribution/process assumptions before downstream decisions,
- test agent or portfolio logic under nontrivial stochastic drivers.

This package is not presented as a new stochastic theory. Its contribution is software integration: making these steps operationally coherent in one codebase so that cross-method studies are easier to implement, audit, and reproduce.

# State of the field

The scientific Python ecosystem already provides essential numerical foundations, especially for array programming, optimization/statistics, symbolic math, and plotting [@harris2020numpy; @virtanen2020scipy; @meurer2017sympy; @hunter2007matplotlib]. These tools are robust and widely adopted, but they are intentionally general-purpose.

There are also focused Python packages for stochastic simulation (for example `stochastic`, `sdeint`, and `sdepy`) that support process sampling or numerical SDE integration [@stochastic_pkg; @sdeint_pkg; @sdepy_pkg]. These tools are useful for specific simulation tasks, but typically do not aim to provide one integrated environment spanning process abstraction, ergodicity-oriented diagnostics, fitting workflows, and agent-based decision experiments.

`ergodicity_library` was developed as a separate package, rather than a thin wrapper, because its primary research use-case depends on coupling these stages through a shared object model. The package addresses a gap between (a) low-level scientific primitives and (b) isolated simulation utilities by offering a workflow-centered architecture for stochastic research questions where process choice, diagnostics, and decisions must be evaluated together.

# Software design

The package is structured in three main layers:

- **Processes**: class-based abstractions for stochastic dynamics, including Ito and non-Ito variants, heavy-tailed models, multiplicative constructions, and memory-dependent processes.
- **Tools**: numerical and symbolic helpers for diagnostics, fitting, preasymptotic analysis, multiprocessing/automation, and partial stochastic differential equation experiments.
- **Agents**: utility-oriented agent logic, portfolio/pool interactions, and optimization-oriented components for decision experiments under uncertainty.

A central design choice is a common process abstraction that allows users to switch model families without rewriting surrounding workflow code. This helps maintain comparability when changing assumptions (for example Gaussian to heavy-tailed increments, or memoryless to memory-dependent dynamics).

The implementation emphasizes extensibility through reusable interfaces and experiment scripting. In addition to repository code and examples [@kendiukhov2025repo], the project is accompanied by a long-form book with worked examples and corresponding code contexts [@kendiukhov2024book].

Current limitations are explicit. Some API surfaces remain partial, automated pytest-style coverage is currently sparse relative to package breadth, and optional machine-learning-heavy paths in the agents layer can be environment-sensitive. These constraints are documented so users can adopt the software appropriately and focus on the better-supported workflow paths first.

# Research impact statement

The immediate impact of `ergodicity_library` is methodological and infrastructural: it reduces setup overhead for studies that combine stochastic simulation, diagnostics, and decision layers in one workflow. The package already supports reproducible demonstrations of heavy-tailed spread behavior, multiplicative growth diagnostics, adaptive-memory processes, distribution-fitting checks, and stochastic field visualization [@kendiukhov2024book; @kendiukhov2025repo].

At its current maturity stage, the strongest evidence is implementation breadth plus reproducible artifact support, not large-scale adoption metrics. The software is publicly available under an OSI-approved license, and the package documentation and example corpus provide a practical base for extension in domain-specific projects.

A realistic near-term impact pathway is as a research platform for rapid hypothesis iteration: users can define process assumptions, generate trajectories, test diagnostics, and evaluate decision rules without repeatedly rebuilding scaffolding around each experiment. This can improve reproducibility and shorten iteration time in cross-method stochastic studies.

# AI usage disclosure

Generative AI assistance was used in preparing the JOSS submission materials (manuscript drafting/editing, submission packaging, and figure-selection workflow scripting). The tools used were GPT-5-class coding assistants in an interactive development workflow.

All AI-assisted outputs were reviewed, edited, and validated by the human author. The author made the core scientific/software decisions and is responsible for the accuracy, originality, licensing compliance, and final submitted content.

# Acknowledgements

No external funding is declared for this submission. The author thanks contributors and users of the open scientific Python ecosystem that this project builds upon [@harris2020numpy; @virtanen2020scipy; @meurer2017sympy; @hunter2007matplotlib].

# References
