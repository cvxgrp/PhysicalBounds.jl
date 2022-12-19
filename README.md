# PhysicalBounds

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://cvxgrp.github.io/PhysicalBounds.jl/dev/)
[![Build Status](https://github.com/cvxgrp/PhysicalBounds.jl/workflows/CI/badge.svg)](https://github.com/cvxgrp/PhysicalBounds.jl/actions)


## Overview

This package computes performance bounds for physical design problems for the
paper [Bounds on Efficiency Metrics in Photonics](https://arxiv.org/abs/2204.05243).
These design problems are modeled using [WaveOperators.jl](https://github.com/cvxgrp/WaveOperators.jl).

For usage and examples, check out
[the documentation](https://cvxgrp.github.io/PhysicalBounds.jl/dev/).

## Physical constraints

We assume that all fields `z` must satisfy the _Helmhotlz equation_,
which we will write as
```
z + G diag(θ)z = b
```
where `θ` are the design parameters (_e.g.,_ permittivities), `G` is a 
discretization of Green's function, and `b` is a term that comes
from the excitation. [WaveOperators.jl](https://github.com/cvxgrp/WaveOperators.jl)
builds these matrices/vectors automatically and returns an `IntegralEquation`, which
contains all physical information about the system.


## Bounds
We assume that the permittivities `θ` lie in one of two constraint
sets `Θ`, which can be specified by the user:

- `θᵢ ∈ {-1, 1}` gives a discrete choice (e.g., silicon or air) at each point
in the design region

- `-1 ≦ θᵢ ≦ 1` allows us to vary the permittivity between an upper and lower
bound at each point in the design region

Note that we can assume WLOG `θ_max = 1` and `θ_min = -1`, since wave equation
can always be scaled such that this is true.

## Efficiency bounds

The focusing efficiency and related objective functions can be modeled as

```
f(z) = (zᵀPz + 2pᵀz + r) / (zᵀQz + 2qᵀz + s).
```

## Using the package
Several examples can be found in [the documentation](https://cvxgrp.github.io/PhysicalBounds.jl/dev/).

## Citing
If you find this package useful in your work, please cite our paper
```
@article{angeris2022bounds,
  title={Bounds on Efficiency Metrics in Photonics},
  author={Angeris, Guillermo and Diamandis, Theo and Vu{\v{c}}kovi{\'c}, Jelena and Boyd, Stephen},
  journal={arXiv preprint arXiv:2204.05243},
  year={2022}
}
```
