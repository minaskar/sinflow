# sinflow

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://github.com/minaskar/sinflow/blob/master/LICENSE)
[![Documentation Status](https://readthedocs.org/projects/sinflow/badge/?version=latest)](https://sinflow.readthedocs.io/en/latest/?badge=latest)


``sinflow`` is a Python implementation of the sliced iterative normalizing flow (SINF) algorithm
for density estimation and sampling. The package has minimal dependencies, requiring only
``numpy`` and ``scipy``. The code is designed to be easy to use and flexible, with a focus on
performance and scalability. The package is designed to be used in a similar way to ``scikit-learn``,
with a simple and consistent API.

## Documentation

Read the docs at [sinflow.readthedocs.io](https://sinflow.readthedocs.io) for more information, examples and tutorials.

## Installation

To install ``sinflow`` using ``pip`` run:

```bash
pip install sinflow
```

or, to install from source:

```bash
git clone https://github.com/minaskar/sinflow.git
cd pocomc
python setup.py install
```

## Basic example

For instance, if you wanted to draw samples from a 10-dimensional Rosenbrock distribution with a uniform prior, you would do something like:

```python
import sinflow as sf
import numpy as np
from sklearn.datasets import make_moons

# Generate some data
x, _ = make_moons(n_samples=5000, noise=0.15)

# Fit a normalizing flow model
flow = sf.Flow()
flow.fit(x)

# Sample from the model
samples = flow.sample(1000)

# Evaluate the log-likelihood of the samples
log_prob = flow.log_prob(samples)

# Evaluate the forward transformation
z, log_det_forward = flow.forward(x)

# Invert the transformation
x_reconstructed, log_det_inverse = flow.inverse(z)
```


## Attribution & Citation

Please cite the following paper if you found this code useful in your research:

```bash
@article{karamanis2024sinflow,
    title={},
    author={},
    journal={},
    year={2024}
}
```

## Licence

Copyright 2024-Now Minas Karamanis and contributors.

``sinflow`` is free software made available under the GPL-3.0 License. For details see the `LICENSE` file.