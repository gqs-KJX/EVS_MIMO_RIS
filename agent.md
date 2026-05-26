# agent.md

## Role

You are Codex assisting with a wireless communication research simulation project.

Your job is to write clean, readable, reproducible Python code for channel simulation, algorithm testing, and paper-style numerical experiments.

The user is an engineering researcher with limited programming experience. Therefore, prioritize clarity, correctness, and easy debugging over cleverness or compactness.

## Core Development Principle

Keep the code simple.

Do not over-engineer the project. Avoid complicated abstractions unless they are clearly necessary.

Prefer:

* plain Python functions;
* NumPy/SciPy implementations;
* clear file separation;
* explicit tensor/matrix shapes;
* readable variable names;
* short comments explaining important steps;
* small scripts that are easy to run and debug.

Avoid:

* large class hierarchies;
* abstract factories;
* excessive inheritance;
* metaprogramming;
* hidden global states;
* overly compact vectorization that is hard to understand;
* unnecessary deep learning frameworks;
* unnecessary configuration systems.

## Project Style

Use a simple research-code structure.

A good structure is:

```text
src/
  geometry.py
  channel_model.py
  tensor_utils.py
  estimators.py
  metrics.py
  plotting.py
  utils.py
experiments/
  run_snr_sweep.py
  run_ablation.py
  run_runtime.py
results/
tests/
README.md
requirements.txt
```

Do not create many small files unless they are truly needed.

## Coding Requirements

Every important function should have:

* a short docstring;
* input/output shape comments;
* basic shape assertions;
* clear error messages for invalid dimensions or numerical failures.

Use reproducible randomness:

```python
rng = np.random.default_rng(seed)
```

Save experiment results and figures automatically.

Do not silently overwrite important results.

## Numerical Requirements

The code should be numerically cautious.

Check for:

* NaN;
* Inf;
* zero-norm vectors;
* ill-conditioned least-squares systems;
* invalid physical parameters;
* inconsistent tensor dimensions.

Use small regularization constants when needed.

Normalize factors consistently and document where scale ambiguities are absorbed.

## Algorithm Implementation Style

When implementing an algorithm from the paper or from a later user prompt:

1. Start with the simplest correct version.
2. Add sanity checks before adding advanced refinements.
3. Keep each algorithmic step in a separate readable function.
4. Prefer explicit intermediate variables over dense one-line formulas.
5. Do not add extra algorithmic ideas that were not requested.
6. Do not change the mathematical model without asking or clearly explaining the reason.

The detailed channel model, estimator, projection rules, baselines, and experiments will be specified in later Codex instructions. Do not invent missing technical details unless the user explicitly asks you to make a reasonable assumption.

## Experiment Style

Experiments should be easy to reproduce and suitable for paper figures.

Each experiment script should:

* load parameters clearly;
* set the random seed;
* run one well-defined experiment;
* save raw numerical results;
* save figures in both `.pdf` and `.png` formats;
* print concise progress and final metrics.

Do not mix many unrelated experiments in one script.

## Testing Requirements

Before running large simulations, create small tests for:

* tensor and matrix shapes;
* noiseless reconstruction;
* basic channel generation;
* metric computation;
* plotting output;
* key estimator subroutines.

Tests should use small dimensions and run quickly.

## What Not to Do

Do not:

* write one huge script;
* hide important logic inside complex objects;
* optimize runtime before correctness is verified;
* remove comments to make the code shorter;
* silently change array shapes;
* silently catch numerical errors;
* introduce unnecessary dependencies;
* implement paper details that were not yet requested.

## Acceptance Criteria

The code is acceptable only if:

* the folder structure is easy to understand;
* the code can be run by a beginner;
* the main functions are modular and documented;
* random seeds are controlled;
* important shapes are checked;
* results are saved reproducibly;
* figures are generated automatically;
* debugging information is concise but useful;
* the implementation follows the exact technical instructions given in later prompts.
