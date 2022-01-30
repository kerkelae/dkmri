# `dkmri.py`

`dkmri.py` stands for diffusion kurtosis magnetic resonance imaging in Python.
It is a Python package for estimating diffusion and kurtosis tensors from 
diffusion-weighted magnetic resonance data. The estimation is performed using
regularized non-linear optimization informed by fully-connected feed-forward
neural networks that are trained to learn the mapping from data to kurtosis
metrics. Details can be found in the upcoming publication and
[source code](https://github.com/kerkelae/dkmri/blob/main/dkmri/dkmri.py).

## Installation

`dkmri.py` can be installed with [pip](https://pypi.org/):

```
pip install dkmri
```

## Usage example

This software can be used from the command line or in a Python interpreter. The
command-line interface does not require any knowledge about Python, whereas the
Python interface is made for people who are comfortable with basic Python
programming.

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
