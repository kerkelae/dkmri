# `dkmri` (WORK IN PROGRESS)

Reproducible and efficient diffusion kurtosis tensor estimation in Python.

## Usage example

`dkmri` can be used from the command line as follows:

```
python dkmri.py \
example-data/data.nii.gz \
example-data/data.bval \
example-data/data.bvec \
-mask example-data/mask.nii.gz \
-md example-data/md.nii.gz \
-ad example-data/ad.nii.gz \
-rd example-data/rd.nii.gz \
-mk example-data/mk.nii.gz \
-ak example-data/ak.nii.gz \
-rk example-data/rk.nii.gz \
-s0 example-data/s0.nii.gz
```

The parameters on lines starting with a dash (`-`) are optional.