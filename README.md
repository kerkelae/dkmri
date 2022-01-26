# `dkmri.py`

`dkmri.py` stands for diffusion kurtosis magnetic resonance imaging in Python.
It is a Python package for estimating diffusion and kurtosis tensor elements from diffusion-weighted magnetic resonance data. The estimation is performed using
regularized non-linear optimization informed by fully-connected feed-forward
neural networks that are trained to learn the mapping from data to kurtosis
measures.

## Installation

`dkmri.py` can be installed with [pip](https://pypi.org/):

```
pip install dkmri
```

## Usage example

This software can be used from the command line or in a Python interpreter. The
command line interface does not require any knowledge about Python, whereas the
Python interface is aimed at people who are comfortable with basic Python
programming.

### Command line interface

`dkmri.py` can be used as follows:

```
dkmri.py <path-of-data-file> <path-of-b-values-file> <path-of-b-vectors-file> [optional arguments]
```

Execute the following to see a description of the arguments:

```
dkmri.py -h
```

### Python interface

See the [example notebook](https://github.com/kerkelae/dkmri/blob/main/docs/example.ipynb).
