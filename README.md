# `dkmri.py`

`dkmri.py` stands for diffusion kurtosis magnetic resonance imaging in Python. It is a Python package for estimating diffusion and kurtosis tensors from diffusion-weighted magnetic resonance data. The estimation is performed using regularized non-linear optimization informed by fully-connected feed-forward neural networks that are trained to learn the mapping from data to kurtosis metrics. Details can be found in the upcoming publication and [source code](https://github.com/kerkelae/dkmri/blob/main/dkmri/dkmri.py).

This software can be used from the command line or in a Python interpreter.

- The command-line interface does not require any knowledge about Python.
- Python interface is for people comfortable with basic Python programming.

## Installation

First, make sure you have installed [Python](https://www.python.org/downloads/).

If you just want to use the command-line interface, the recommended way of installing `dkmri.py` is to use [pipx](https://github.com/pypa/pipx/#install-pipx):

```
pipx install dkmri
```

pipx automatically creates an isolated environment in which the dependencies are installed.

If you want to use the Python interface, you can use [pip](https://pip.pypa.io/en/stable/) (you should install `dkmri.py` in an isolated environment using [venv](https://docs.python.org/3/library/venv.html) or [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) to avoid dependency issues):

```
pip install dkmri
```



## Usage example

### Command-line interface

The command for using `dkmri.py` is

```
dkmri.py data bvals bvecs optional-arguments
```

where `data`, `bvals`, and `bvecs` are the paths of the files containing the
diffusion-weighted data, b-values, and b-vectors, and `optional-arguments` is
where to define things such as which parameter maps to save.

For example, a command for computing a mean kurtosis map from `data.nii.gz` and
saving it in `mk.nii.gz` could be

```
dkmri.py data.nii.gz bvals.txt bvecs.txt -mask mask.nii.gz -mk mk.nii.gz
```

To see a full description of the arguments, execute the following:

```
dkmri.py -h
```

### Python interface

See the [example notebook](https://github.com/kerkelae/dkmri/blob/main/docs/example.ipynb).

## Support

If you have questions, found bugs, or need help, please open an
[issue on Github](https://github.com/kerkelae/dkmri/issues).
