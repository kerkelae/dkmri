#!/usr/bin/env python

"""Diffusion kurtosis magnetic resonance imaging."""

import argparse
import warnings

import jax
import jax.numpy as jnp
from jax.scipy.optimize import minimize
import nibabel as nib
import numba
import numpy as np
from sklearn.neural_network import MLPRegressor


jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

# Hard-coded lower and upper limit for kurtosis
MIN_K = -3 / 7
MAX_K = 10


def design_matrix(bvals, bvecs):
    """Return the diffusion kurtosis imaging design matrix.

    Parameters
    ----------
    bvals : numpy.ndarray
        Floating-point array with shape (number of acquisitions,).
    bvecs : numpy.ndarray
        Floating-point array with shape (number of acquisitions, 3).

    Returns
    -------
    numpy.ndarray
    """
    X = np.zeros((len(bvals), 22))
    X[:, 0] = 1
    X[:, 1] = -bvals * bvecs[:, 0] ** 2
    X[:, 2] = -bvals * bvecs[:, 1] ** 2
    X[:, 3] = -bvals * bvecs[:, 2] ** 2
    X[:, 4] = -2 * bvals * bvecs[:, 0] * bvecs[:, 1]
    X[:, 5] = -2 * bvals * bvecs[:, 0] * bvecs[:, 2]
    X[:, 6] = -2 * bvals * bvecs[:, 1] * bvecs[:, 2]
    X[:, 7] = bvals ** 2 * bvecs[:, 0] ** 4 / 6
    X[:, 8] = bvals ** 2 * bvecs[:, 1] ** 4 / 6
    X[:, 9] = bvals ** 2 * bvecs[:, 2] ** 4 / 6
    X[:, 10] = 2 * bvals ** 2 * bvecs[:, 0] ** 3 * bvecs[:, 1] / 3
    X[:, 11] = 2 * bvals ** 2 * bvecs[:, 0] ** 3 * bvecs[:, 2] / 3
    X[:, 12] = 2 * bvals ** 2 * bvecs[:, 1] ** 3 * bvecs[:, 0] / 3
    X[:, 13] = 2 * bvals ** 2 * bvecs[:, 1] ** 3 * bvecs[:, 2] / 3
    X[:, 14] = 2 * bvals ** 2 * bvecs[:, 2] ** 3 * bvecs[:, 0] / 3
    X[:, 15] = 2 * bvals ** 2 * bvecs[:, 2] ** 3 * bvecs[:, 1] / 3
    X[:, 16] = bvals ** 2 * bvecs[:, 0] ** 2 * bvecs[:, 1] ** 2
    X[:, 17] = bvals ** 2 * bvecs[:, 0] ** 2 * bvecs[:, 2] ** 2
    X[:, 18] = bvals ** 2 * bvecs[:, 1] ** 2 * bvecs[:, 2] ** 2
    X[:, 19] = 2 * bvals ** 2 * bvecs[:, 0] ** 2 * bvecs[:, 1] * bvecs[:, 2]
    X[:, 20] = 2 * bvals ** 2 * bvecs[:, 1] ** 2 * bvecs[:, 0] * bvecs[:, 2]
    X[:, 21] = 2 * bvals ** 2 * bvecs[:, 2] ** 2 * bvecs[:, 0] * bvecs[:, 1]
    return X


def params_to_D(params):
    """Return diffusion tensors corresponding to a parameter array.

    Parameters
    ----------
    params : numpy.ndarray
        Floating-point array with shape (..., 22).

    Returns
    -------
    numpy.ndarray
    """
    D = np.zeros(params.shape[0:-1] + (3, 3))
    D[..., 0, 0] = params[..., 1]
    D[..., 1, 1] = params[..., 2]
    D[..., 2, 2] = params[..., 3]
    D[..., 0, 1] = params[..., 4]
    D[..., 1, 0] = params[..., 4]
    D[..., 0, 2] = params[..., 5]
    D[..., 2, 0] = params[..., 5]
    D[..., 1, 2] = params[..., 6]
    D[..., 2, 1] = params[..., 6]
    return D


def params_to_W(params):
    """Return kurtosis tensors corresponding to a parameter array.

    Parameters
    ----------
    params : numpy.ndarray
        Floating-point array with shape (..., 22).

    Returns
    -------
    numpy.ndarray
    """
    W = np.zeros(params.shape[0:-1] + (3, 3, 3, 3))
    W[..., 0, 0, 0, 0] = params[..., 7]
    W[..., 1, 1, 1, 1] = params[..., 8]
    W[..., 2, 2, 2, 2] = params[..., 9]
    W[..., 1, 0, 0, 0] = params[..., 10]
    W[..., 0, 1, 0, 0] = params[..., 10]
    W[..., 0, 0, 1, 0] = params[..., 10]
    W[..., 0, 0, 0, 1] = params[..., 10]
    W[..., 2, 0, 0, 0] = params[..., 11]
    W[..., 0, 2, 0, 0] = params[..., 11]
    W[..., 0, 0, 2, 0] = params[..., 11]
    W[..., 0, 0, 0, 2] = params[..., 11]
    W[..., 0, 1, 1, 1] = params[..., 12]
    W[..., 1, 0, 1, 1] = params[..., 12]
    W[..., 1, 1, 0, 1] = params[..., 12]
    W[..., 1, 1, 1, 0] = params[..., 12]
    W[..., 2, 1, 1, 1] = params[..., 13]
    W[..., 1, 2, 1, 1] = params[..., 13]
    W[..., 1, 1, 2, 1] = params[..., 13]
    W[..., 1, 1, 1, 2] = params[..., 13]
    W[..., 0, 2, 2, 2] = params[..., 14]
    W[..., 2, 0, 2, 2] = params[..., 14]
    W[..., 2, 2, 0, 2] = params[..., 14]
    W[..., 2, 2, 2, 0] = params[..., 14]
    W[..., 1, 2, 2, 2] = params[..., 15]
    W[..., 2, 1, 2, 2] = params[..., 15]
    W[..., 2, 2, 1, 2] = params[..., 15]
    W[..., 2, 2, 2, 1] = params[..., 15]
    W[..., 0, 0, 1, 1] = params[..., 16]
    W[..., 0, 1, 1, 0] = params[..., 16]
    W[..., 1, 1, 0, 0] = params[..., 16]
    W[..., 1, 0, 0, 1] = params[..., 16]
    W[..., 1, 0, 1, 0] = params[..., 16]
    W[..., 0, 1, 0, 1] = params[..., 16]
    W[..., 0, 0, 2, 2] = params[..., 17]
    W[..., 0, 2, 2, 0] = params[..., 17]
    W[..., 2, 2, 0, 0] = params[..., 17]
    W[..., 2, 0, 0, 2] = params[..., 17]
    W[..., 2, 0, 2, 0] = params[..., 17]
    W[..., 0, 2, 0, 2] = params[..., 17]
    W[..., 1, 1, 2, 2] = params[..., 18]
    W[..., 1, 2, 2, 1] = params[..., 18]
    W[..., 2, 2, 1, 1] = params[..., 18]
    W[..., 2, 1, 1, 2] = params[..., 18]
    W[..., 2, 1, 2, 1] = params[..., 18]
    W[..., 1, 2, 1, 2] = params[..., 18]
    W[..., 0, 0, 1, 2] = params[..., 19]
    W[..., 0, 0, 2, 1] = params[..., 19]
    W[..., 0, 1, 0, 2] = params[..., 19]
    W[..., 0, 1, 2, 0] = params[..., 19]
    W[..., 0, 2, 0, 1] = params[..., 19]
    W[..., 0, 2, 1, 0] = params[..., 19]
    W[..., 1, 0, 0, 2] = params[..., 19]
    W[..., 1, 0, 2, 0] = params[..., 19]
    W[..., 1, 2, 0, 0] = params[..., 19]
    W[..., 2, 0, 0, 1] = params[..., 19]
    W[..., 2, 0, 1, 0] = params[..., 19]
    W[..., 2, 1, 0, 0] = params[..., 19]
    W[..., 1, 1, 0, 2] = params[..., 20]
    W[..., 1, 1, 2, 0] = params[..., 20]
    W[..., 1, 0, 1, 2] = params[..., 20]
    W[..., 1, 0, 2, 1] = params[..., 20]
    W[..., 1, 2, 1, 0] = params[..., 20]
    W[..., 1, 2, 0, 1] = params[..., 20]
    W[..., 0, 1, 1, 2] = params[..., 20]
    W[..., 0, 1, 2, 1] = params[..., 20]
    W[..., 0, 2, 1, 1] = params[..., 20]
    W[..., 2, 1, 1, 0] = params[..., 20]
    W[..., 2, 1, 0, 1] = params[..., 20]
    W[..., 2, 0, 1, 1] = params[..., 20]
    W[..., 2, 2, 0, 1] = params[..., 21]
    W[..., 2, 2, 1, 0] = params[..., 21]
    W[..., 2, 0, 2, 1] = params[..., 21]
    W[..., 2, 0, 1, 2] = params[..., 21]
    W[..., 2, 1, 2, 0] = params[..., 21]
    W[..., 2, 1, 0, 2] = params[..., 21]
    W[..., 0, 2, 2, 1] = params[..., 21]
    W[..., 0, 2, 1, 2] = params[..., 21]
    W[..., 0, 1, 2, 2] = params[..., 21]
    W[..., 1, 2, 2, 0] = params[..., 21]
    W[..., 1, 2, 0, 2] = params[..., 21]
    W[..., 1, 0, 2, 2] = params[..., 21]
    evals, _ = np.linalg.eigh(np.nan_to_num(params_to_D(params)))
    MD = np.mean(evals, axis=-1)[..., np.newaxis, np.newaxis, np.newaxis, np.newaxis]
    MD[MD == 0] = np.nan  # To avoid warnings due to division by zero
    W /= MD ** 2
    return W


def tensors_to_params(S0, D, W):
    """Return the parameter array corresponding to tensors.

    Parameters
    ----------
    S0 : numpy.ndarray
        Floating-point array.
    D : numpy.ndarray
        Floating-point array with shape (..., 3, 3).
    W : numpy.ndarray
        Floating-point array with shape (..., 3, 3, 3, 3).

    Returns
    -------
    numpy.ndarray
    """
    params = np.zeros(np.asarray(S0).shape + (22,))
    params[..., 0] = np.log(S0)
    params[..., 1] = D[..., 0, 0]
    params[..., 2] = D[..., 1, 1]
    params[..., 3] = D[..., 2, 2]
    params[..., 4] = D[..., 0, 1]
    params[..., 5] = D[..., 0, 2]
    params[..., 6] = D[..., 1, 2]
    params[..., 7] = W[..., 0, 0, 0, 0]
    params[..., 8] = W[..., 1, 1, 1, 1]
    params[..., 9] = W[..., 2, 2, 2, 2]
    params[..., 10] = W[..., 0, 0, 0, 1]
    params[..., 11] = W[..., 0, 0, 0, 2]
    params[..., 12] = W[..., 1, 1, 1, 0]
    params[..., 13] = W[..., 1, 1, 1, 2]
    params[..., 14] = W[..., 2, 2, 2, 0]
    params[..., 15] = W[..., 2, 2, 2, 1]
    params[..., 16] = W[..., 0, 0, 1, 1]
    params[..., 17] = W[..., 0, 0, 2, 2]
    params[..., 18] = W[..., 1, 1, 2, 2]
    params[..., 19] = W[..., 0, 0, 1, 2]
    params[..., 20] = W[..., 1, 1, 0, 2]
    params[..., 21] = W[..., 2, 2, 0, 1]
    evals, _ = np.linalg.eigh(np.nan_to_num(D))
    MD = np.mean(evals, axis=-1)[..., np.newaxis]
    params[..., 7::] *= MD ** 2
    return params


@jax.jit
def _adc(params, vs):
    """Compute apparent diffusion coefficients along unit vectors `vs`.

    Parameters
    ----------
    params : numpy.ndarray or jaxlib.xla_extension.DeviceArray
        Floating-point array with shape (22,).
    vs : numpy.ndarray or jaxlib.xla_extension.DeviceArray
        Floating-point array with shape (number of directions, 3).

    Returns
    -------
    jaxlib.xla_extension.DeviceArray
    """
    return (
        vs[:, 0] * vs[:, 0] * params[1]
        + vs[:, 1] * vs[:, 1] * params[2]
        + vs[:, 2] * vs[:, 2] * params[3]
        + 2 * vs[:, 0] * vs[:, 1] * params[4]
        + 2 * vs[:, 0] * vs[:, 2] * params[5]
        + 2 * vs[:, 1] * vs[:, 2] * params[6]
    )


def params_to_md(params, mask=None):
    """Compute mean diffusivity.

    Parameters
    ----------
    params : numpy.ndarray
        Floating-point array with shape (..., 22).
    mask : numpy.ndarray, optional
        Boolean array.

    Returns
    -------
    numpy.ndarray
    """
    if mask is None:
        mask = np.ones(params.shape[0:-1]).astype(bool)
    evals, _ = np.linalg.eigh(np.nan_to_num(params_to_D(params[mask])))
    md = np.zeros(mask.shape)
    md[mask] = np.mean(evals, axis=1)
    return md


def params_to_ad(params, mask=None):
    """Compute axial diffusivity.

    Parameters
    ----------
    params : numpy.ndarray
        Floating-point array with shape (..., 22).
    mask : numpy.ndarray, optional
        Boolean array.

    Returns
    -------
    numpy.ndarray
    """
    if mask is None:
        mask = np.ones(params.shape[0:-1]).astype(bool)
    evals, _ = np.linalg.eigh(np.nan_to_num(params_to_D(params[mask])))
    ad = np.zeros(mask.shape)
    ad[mask] = evals[:, 2]
    return ad


def params_to_rd(params, mask=None):
    """Compute radial diffusivity.

    Parameters
    ----------
    params : numpy.ndarray
        Floating-point array with shape (..., 22).
    mask : numpy.ndarray, optional
        Boolean array.

    Returns
    -------
    numpy.ndarray
    """
    if mask is None:
        mask = np.ones(params.shape[0:-1]).astype(bool)
    evals, _ = np.linalg.eigh(np.nan_to_num(params_to_D(params[mask])))
    rd = np.zeros(mask.shape)
    rd[mask] = np.mean(evals[:, 0:2], axis=1)
    return rd


def params_to_fa(params, mask=None):
    """Compute fractional anisotropy.

    Parameters
    ----------
    params : numpy.ndarray
        Floating-point array with shape (..., 22).
    mask : numpy.ndarray, optional
        Boolean array.

    Returns
    -------
    numpy.ndarray
    """
    if mask is None:
        mask = np.ones(params.shape[0:-1]).astype(bool)
    evals, _ = np.linalg.eigh(np.nan_to_num(params_to_D(params[mask])))
    avg_evals = np.mean(evals, axis=-1)
    sum_sq_evals = np.sum(evals ** 2, axis=-1)
    sum_sq_evals[sum_sq_evals == 0] = np.nan  # To avoid warnings for dividing by zero
    fa = np.zeros(mask.shape)
    fa[mask] = np.sqrt(
        1.5
        * (
            (evals[..., 0] - avg_evals) ** 2
            + (evals[..., 1] - avg_evals) ** 2
            + (evals[..., 2] - avg_evals) ** 2
        )
        / sum_sq_evals
    )
    return fa


@jax.jit
def _akc(params, vs):
    """Compute apparent kurtosis coefficients along unit vectors `vs`.

    Parameters
    ----------
    params : numpy.ndarray or jaxlib.xla_extension.DeviceArray
        Floating-point array with shape (22,).
    vs : numpy.ndarray or jaxlib.xla_extension.DeviceArray
        Floating-point array with shape (number of directions, 3).

    Returns
    -------
    jaxlib.xla_extension.DeviceArray
    """
    return (
        vs[:, 0] * vs[:, 0] * vs[:, 0] * vs[:, 0] * params[7]
        + vs[:, 1] * vs[:, 1] * vs[:, 1] * vs[:, 1] * params[8]
        + vs[:, 2] * vs[:, 2] * vs[:, 2] * vs[:, 2] * params[9]
        + 4 * vs[:, 0] * vs[:, 0] * vs[:, 0] * vs[:, 1] * params[10]
        + 4 * vs[:, 0] * vs[:, 0] * vs[:, 0] * vs[:, 2] * params[11]
        + 4 * vs[:, 0] * vs[:, 1] * vs[:, 1] * vs[:, 1] * params[12]
        + 4 * vs[:, 1] * vs[:, 1] * vs[:, 1] * vs[:, 2] * params[13]
        + 4 * vs[:, 0] * vs[:, 2] * vs[:, 2] * vs[:, 2] * params[14]
        + 4 * vs[:, 1] * vs[:, 2] * vs[:, 2] * vs[:, 2] * params[15]
        + 6 * vs[:, 0] * vs[:, 0] * vs[:, 1] * vs[:, 1] * params[16]
        + 6 * vs[:, 0] * vs[:, 0] * vs[:, 2] * vs[:, 2] * params[17]
        + 6 * vs[:, 1] * vs[:, 1] * vs[:, 2] * vs[:, 2] * params[18]
        + 12 * vs[:, 0] * vs[:, 0] * vs[:, 1] * vs[:, 2] * params[19]
        + 12 * vs[:, 0] * vs[:, 1] * vs[:, 1] * vs[:, 2] * params[20]
        + 12 * vs[:, 0] * vs[:, 1] * vs[:, 2] * vs[:, 2] * params[21]
    ) / _adc(params, vs) ** 2


_45_dirs = np.array(
    [
        [1.82043914e-01, -9.27048041e-02, 9.78910534e-01],
        [-3.40163392e-01, 9.66362766e-02, 9.35387779e-01],
        [7.00296424e-02, 4.38017066e-01, 8.96234846e-01],
        [4.06299199e-01, -3.92222958e-01, 8.25276991e-01],
        [-5.02052907e-01, -3.13403177e-01, 8.06052930e-01],
        [-1.66170376e-01, -6.31581895e-01, 7.57292358e-01],
        [5.64556833e-01, 4.31566519e-01, 7.03580786e-01],
        [-2.51657166e-01, 7.00420117e-01, 6.67892454e-01],
        [7.61073783e-01, 3.38394953e-02, 6.47782050e-01],
        [-6.89166960e-01, 4.70041104e-01, 5.51461932e-01],
        [-8.44458835e-01, 8.32112032e-02, 5.29117352e-01],
        [1.66790153e-01, -8.37963970e-01, 5.19612770e-01],
        [-3.38142653e-07, -1.0, 5.11395430e-08],
        [8.25320355e-01, 5.62747972e-01, 4.64868944e-02],
        [4.28784759e-01, 7.88094724e-01, 4.41645035e-01],
        [8.37982166e-01, -3.97841299e-01, 3.73507952e-01],
        [-7.25354257e-01, -5.81937344e-01, 3.67709301e-01],
        [5.99978002e-01, -7.48095908e-01, 2.83511745e-01],
        [1.02962118e-02, 9.60455871e-01, 2.78241815e-01],
        [-4.11173677e-01, -8.83129936e-01, 2.25871034e-01],
        [9.70146317e-01, 1.89897129e-01, 1.50848283e-01],
        [-9.59916320e-01, -2.37455189e-01, 1.48915048e-01],
        [-4.89078081e-01, 8.61410027e-01, 1.37023343e-01],
        [4.89078344e-01, 8.61409859e-01, -1.37023460e-01],
        [9.59916143e-01, -2.37455656e-01, -1.48915448e-01],
        [-9.70146212e-01, 1.89897463e-01, -1.50848537e-01],
        [4.11173208e-01, -8.83130287e-01, -2.25870516e-01],
        [-1.02958307e-02, 9.60455844e-01, -2.78241923e-01],
        [-5.99978279e-01, -7.48095830e-01, -2.83511365e-01],
        [7.25353858e-01, -5.81937942e-01, -3.67709142e-01],
        [-8.37982112e-01, -3.97841173e-01, -3.73508208e-01],
        [-4.28784490e-01, 7.88094865e-01, -4.41645045e-01],
        [-8.25320204e-01, 5.62748203e-01, -4.64867894e-02],
        [-1.66790527e-01, -8.37963935e-01, -5.19612705e-01],
        [8.44458856e-01, 8.32109856e-02, -5.29117353e-01],
        [6.89167026e-01, 4.70041028e-01, -5.51461913e-01],
        [-7.61073594e-01, 3.38395817e-02, -6.47782268e-01],
        [2.51657227e-01, 7.00419875e-01, -6.67892685e-01],
        [-5.64556781e-01, 4.31566701e-01, -7.03580716e-01],
        [1.66170307e-01, -6.31582185e-01, -7.57292132e-01],
        [5.02052846e-01, -3.13403501e-01, -8.06052843e-01],
        [-4.06299100e-01, -3.92222992e-01, -8.25277024e-01],
        [-7.00296293e-02, 4.38016892e-01, -8.96234932e-01],
        [3.40163643e-01, 9.66360119e-02, -9.35387715e-01],
        [-1.82043727e-01, -9.27048310e-02, -9.78910566e-01],
    ]
)  # Directions along which AKC is evaluated for computing MK


def params_to_mk(params, mask=None):
    """Compute mean kurtosis.

    Parameters
    ----------
    params : numpy.ndarray
        Floating-point array with shape (..., 22).
    mask : numpy.ndarray, optional
        Boolean array.

    Returns
    -------
    numpy.ndarray
    """
    if mask is None:
        mask = np.ones(params.shape[0:-1]).astype(bool)
    mk = np.zeros(mask.shape)
    mk_flat = mk[mask]
    params_flat = params[mask]
    size = len(params_flat)
    for i in range(size):
        mk_flat[i] = np.mean(_akc(params_flat[i], _45_dirs))
    mk[mask] = mk_flat
    return mk


def params_to_ak(params, mask=None):
    """Compute axial kurtosis.

    Parameters
    ----------
    params : numpy.ndarray
        Floating-point array with shape (..., 22).
    mask : numpy.ndarray, optional
        Boolean array.

    Returns
    -------
    numpy.ndarray
    """
    if mask is None:
        mask = np.ones(params.shape[0:-1]).astype(bool)
    params_flat = params[mask]
    ak = np.zeros(mask.shape)
    ak_flat = ak[mask]
    size = len(params_flat)
    _, evecs = np.linalg.eigh(np.nan_to_num(params_to_D(params_flat)))
    for i in range(size):
        ak_flat[i] = _akc(params_flat[i], evecs[np.newaxis, i, :, 2])
    ak[mask] = ak_flat
    return ak


_10_dirs = np.array(
    [
        [0.0, 0.0, 1.0],
        [0.0, 5.87785252e-01, 8.09016994e-01],
        [0.0, 9.51056516e-01, 3.09016994e-01],
        [0.0, 9.51056516e-01, -3.09016994e-01],
        [0.0, 5.87785252e-01, -8.09016994e-01],
        [0.0, 1.22464680e-16, -1.0],
        [0.0, -5.87785252e-01, -8.09016994e-01],
        [0.0, -9.51056516e-01, -3.09016994e-01],
        [0.0, -9.51056516e-01, 3.09016994e-01],
        [0.0, -5.87785252e-01, 8.09016994e-01],
    ]
)  # Directions along which AKC is evaluated for computing RK


@numba.njit
def _vec2vec_rotmat(v, k):
    """Compute a rotation matrix for aligning `v` with `k`.

    Parameters
    -----------
    v : numpy.ndarray
        Floating-point array with shape (3,).
    k : numpy.ndarray
        Floating-point array with shape (3,).

    Returns
    ---------
    numpy.ndarray
    """
    v = v / np.linalg.norm(v)
    k = k / np.linalg.norm(k)
    axis = np.cross(v, k)
    if np.linalg.norm(axis) < 1e-15:
        if np.linalg.norm(v - k) > np.linalg.norm(v):
            return -np.eye(3)
        else:
            return np.eye(3)
    axis /= np.linalg.norm(axis)
    angle = np.arccos(np.dot(v, k))
    K = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    R = (
        np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    )  # Rodrigues' rotation formula
    return R


def params_to_rk(params, mask=None):
    """Compute radial kurtosis.

    Parameters
    ----------
    params : numpy.ndarray
        Floating-point array with shape (..., 22).
    mask : numpy.ndarray, optional
        Boolean array.

    Returns
    -------
    numpy.ndarray
    """
    if mask is None:
        mask = np.ones(params.shape[0:-1]).astype(bool)
    params_flat = params[mask]
    rk = np.zeros(mask.shape)
    rk_flat = rk[mask]
    size = len(params_flat)
    _, evecs = np.linalg.eigh(np.nan_to_num(params_to_D(params_flat)))
    for i in range(size):
        R = _vec2vec_rotmat(np.array([1.0, 0, 0]), evecs[i, :, 2])
        vs = (R @ _10_dirs.T).T
        rk_flat[i] = np.mean(_akc(params_flat[i], vs))
    rk[mask] = rk_flat
    return rk


def _mtk(params, mask=None):
    """Compute mean of the kurtosis tensor.

    Parameters
    ----------
    params : numpy.ndarray
        Floating-point array with shape (..., 22).
    mask : numpy.ndarray, optional
        Boolean array.

    Returns
    -------
    numpy.ndarray
    """
    if mask is None:
        mask = np.ones(params.shape[0:-1]).astype(bool)
    W = params_to_W(params)
    W_flat = W[mask]
    mtk = np.zeros(mask.shape)
    mtk_flat = mtk[mask]
    size = len(W_flat)
    for i in range(size):
        mtk_flat[i] = np.trace(np.trace(W_flat[i])) / 5
    mtk[mask] = mtk_flat
    return mtk


def _ols_fit(data, design_matrix, mask=None):
    """Estimate model parameters with ordinary least squares.

    Parameters
    ----------
    data : numpy.ndarray
        Floating-point array with shape (..., number of acquisitions).
    design_matrix : numpy.ndarray
        Floating-point array with shape (number of acquisitions, 22).
    mask : numpy.ndarray, optional
        Boolean array.

    Returns
    -------
    numpy.ndarray
    """
    if mask is None:
        mask = np.ones(data.shape[0:-1]).astype(bool)
    params = np.zeros(mask.shape + (22,))
    params[mask] = (
        np.linalg.pinv(design_matrix.T @ design_matrix)
        @ design_matrix.T
        @ np.log(data[mask])[..., np.newaxis]
    )[..., 0]
    return params


def _nlls_fit(data, design_matrix, mask=None):
    """Estimate model parameters with non-linear least squares.

    Parameters
    ----------
    data : numpy.ndarray
        Floating-point array with shape (..., number of acquisitions).
    design_matrix : numpy.ndarray
        Floating-point array with shape (number of acquisitions, 22).
    mask : numpy.ndarray, optional
        Boolean array.

    Returns
    -------
    numpy.ndarray
    """
    if mask is None:
        mask = np.ones(data.shape[0:-1]).astype(bool)

    data_flat = jnp.asarray(data[mask])
    x0_flat = jnp.asarray(_ols_fit(data_flat, design_matrix))
    design_matrix = jnp.asarray(design_matrix)
    size = len(data_flat)

    def cost(params, design_matrix, y):
        return jnp.mean((jnp.exp(design_matrix @ params) - y) ** 2)

    @jax.jit
    def jit_minimize(i):
        return minimize(
            fun=cost,
            x0=x0_flat[i],
            args=(design_matrix, data_flat[i]),
            method="BFGS",
            options={"maxiter": int(1e4), "line_search_maxiter": int(1e3)},
        )

    params_flat = np.zeros((size, 22))
    for i in range(size):
        results = jit_minimize(i)
        if not results.success:
            warnings.warn(
                f"Fit was not successful in voxel {i} (status = {results.status})"
            )
        params_flat[i] = results.x

    params = np.zeros(mask.shape + (22,))
    params[mask] = params_flat

    return params


@jax.jit
def _akc_mask(W, vs, mask):
    """Compute a mask based on condition AKC >= 0 along all directions in `vs`.

    Parameters
    ----------
    params : numpy.ndarray
        Floating-point array with shape (..., 3, 3, 3, 3).
    vs : numpy.ndarray or jaxlib.xla_extension.DeviceArray
        Floating-point array with shape (number of directions, 3).
    mask : numpy.ndarray, optional
        Boolean array.

    Returns
    -------
    numpy.ndarray
    """
    akc_mask = np.ones(mask.shape)
    for v in vs:
        akc_mask *= (v.T @ (v.T @ W @ v) @ v) >= 0
    akc_mask *= mask
    return akc_mask


def _predict(data, m, akc_mask, seed, mask=None):
    """Train a multilayer perceptron to predict `m` from `data`.

    Parameters
    ----------
    data : numpy.ndarray
        Floating-point array with shape (..., number of acquisitions).
    m : numpy.ndarray
        Floating-point array.
    akc_mask : numpy.ndarray
        Boolean array.
    seed : int
        Seed for pseudo-random number generation to initialize neural network
        weights.
    mask : numpy.ndarray, optional
        Boolean array.

    Returns
    -------
    mk_pred : np.ndarray
    R2 : float
    """
    if mask is None:
        mask = np.ones(data.shape[0:-1]).astype(bool)
    X = data[akc_mask]
    y = m[akc_mask]
    reg = MLPRegressor(
        hidden_layer_sizes=(20, 20), max_iter=int(1e3), random_state=seed,
    ).fit(X, y)
    R2 = reg.score(X, y)
    m_pred = np.zeros(mask.shape)
    m_pred[mask] = reg.predict(data[mask])
    return m_pred, R2


def signal(params, design_matrix, mask=None):
    """Predict signal from model parameters.

    Parameters
    ----------
    params : numpy.ndarray
        Floating-point array with shape (..., 22).
    design_matrix : numpy.ndarray
        Floating-point array with shape (number of acquisitions, 22).
    mask : numpy.ndarray, optional
        Boolean array.

    Returns
    -------
    numpy.ndarray
    """
    if mask is None:
        mask = np.ones(params.shape[0:-1]).astype(bool)
    XB = design_matrix @ params[..., np.newaxis]
    XB[XB > 200] = np.inf  # To avoid overflow warnings
    S_hat = np.exp(XB)[..., 0]
    S_hat[~mask] = 0
    return S_hat


def _calculate_x0(S0, D, mtk_pred, ak_pred, rk_pred, mask=None):
    """Calculate initial positions for the regularized NLLS fit.

    Parameters
    ----------
    S0 : numpy.ndarray
        Floating-point array.
    D : numpy.ndarray
        Floating-point array with shape (..., 3, 3).
    mtk_pred : numpy.ndarray
        Floating-point array.
    ak_pred : numpy.ndarray
        Floating-point array.
    rk_pred : numpy.ndarray
        Floating-point array.
    mask : numpy.ndarray, optional
        Boolean array.

    Returns
    -------
    numpy.ndarray
    """

    if mask is None:
        mask = np.ones(data.shape[0:-1]).astype(bool)

    S0_flat = S0[mask]
    D_flat = D[mask]
    mtk_pred_flat = mtk_pred[mask]
    ak_pred_flat = ak_pred[mask]
    rk_pred_flat = rk_pred[mask]
    size = len(S0_flat)

    evals, evecs = np.linalg.eigh(np.nan_to_num(D_flat))

    @numba.njit
    def p(u):
        P = np.zeros((3, 3, 3, 3))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        P[i, j, k, l] = u[i] * u[j] * u[k] * u[l]
        return P

    @numba.njit
    def d(i, j):
        if i == j:
            return 1
        else:
            return 0

    @numba.njit
    def q(u):
        Q = np.zeros((3, 3, 3, 3))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        Q[i, j, k, l] += (
                            u[i] * u[j] * d(k, l)
                            + u[i] * u[k] * d(j, l)
                            + u[i] * u[l] * d(j, k)
                            + u[j] * u[k] * d(i, l)
                            + u[j] * u[l] * d(i, k)
                            + u[k] * u[l] * d(i, j)
                        ) / 6
        return Q

    I = np.zeros((3, 3, 3, 3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    I[i, j, k, l] += (
                        d(i, j) * d(k, l) + d(i, k) * d(j, l) + d(i, l) * d(j, k)
                    ) / 3

    x0_flat = np.zeros((size, 22))
    for i in range(size):
        u = evecs[i, :, 2]
        md = np.mean(evals[i, :])
        ad = evals[i, 2]
        rd = np.mean(evals[i, 0:2])
        if rd < 0.1:
            rd = 0.1
        mtk = mtk_pred_flat[i]
        atk = ak_pred_flat[i] * ad ** 2 / md ** 2
        rtk = rk_pred_flat[i] * rd ** 2 / md ** 2
        D = rd * np.eye(3) + (ad - rd) * u[:, np.newaxis] @ u[np.newaxis, :]
        W = (
            0.5 * (10 * rtk + 5 * atk - 15 * mtk) * p(u)
            + rtk * I
            + 1.5 * (5 * mtk - atk - 4 * rtk) * q(u)
        )
        x0_flat[i, 0] = np.log(S0_flat[i])
        x0_flat[i, 1] = D[0, 0]
        x0_flat[i, 2] = D[1, 1]
        x0_flat[i, 3] = D[2, 2]
        x0_flat[i, 4] = D[0, 1]
        x0_flat[i, 5] = D[0, 2]
        x0_flat[i, 6] = D[1, 2]
        x0_flat[i, 7] = W[0, 0, 0, 0]
        x0_flat[i, 8] = W[1, 1, 1, 1]
        x0_flat[i, 9] = W[2, 2, 2, 2]
        x0_flat[i, 10] = W[0, 0, 0, 1]
        x0_flat[i, 11] = W[0, 0, 0, 2]
        x0_flat[i, 12] = W[1, 1, 1, 0]
        x0_flat[i, 13] = W[1, 1, 1, 2]
        x0_flat[i, 14] = W[2, 2, 2, 0]
        x0_flat[i, 15] = W[2, 2, 2, 1]
        x0_flat[i, 16] = W[0, 0, 1, 1]
        x0_flat[i, 17] = W[0, 0, 2, 2]
        x0_flat[i, 18] = W[1, 1, 2, 2]
        x0_flat[i, 19] = W[0, 0, 1, 2]
        x0_flat[i, 20] = W[1, 1, 0, 2]
        x0_flat[i, 21] = W[2, 2, 0, 1]
        x0_flat[i, 7::] *= md ** 2

    x0 = np.zeros(mask.shape + (22,))
    x0[mask] = x0_flat

    return x0


def _reg_nlls_fit(
    data,
    design_matrix,
    x0,
    mk_pred,
    ak_pred,
    rk_pred,
    axial_dirs,
    radial_dirs,
    alpha=None,
    mask=None,
    quiet=False,
):
    """Estimate model parameters with regularized non-linear least squares.

    Parameters
    ----------
    data : numpy.ndarray
        Floating-point array with shape (..., number of acquisitions).
    design_matrix : numpy.ndarray
        Floating-point array with shape (number of acquisitions, 22).
    x0 : numpy.ndarray
        Floating-point array with shape (..., 22).
    mk_pred : numpy.ndarray
        Floating-point array.
    ak_pred : numpy.ndarray
        Floating-point array.
    rk_pred : numpy.ndarray
        Floating-point array.
    axial_dirs : numpy.ndarray
        Floating-point array with shape (..., 1, 3).
    radial_dirs : numpy.ndarray
        Floating-point array with shape (..., 10, 3).
    alpha : float, optional
        Constant controlling regularization term magnitude.
    mask : numpy.ndarray, optional
        Boolean array.
    quiet : bool, optional
        Whether not to print messages about computation progress.

    Returns
    -------
    params : numpy.ndarray
    status : numpy.ndarray
    """

    if mask is None:
        mask = np.ones(data.shape[0:-1]).astype(bool)

    if alpha is None:
        mse_dki = np.median(
            np.mean((signal(x0[mask], design_matrix) - data[mask]) ** 2, axis=1)
        )
        mse_mk = np.median((mk_pred[mask] - params_to_mk(x0[mask])) ** 2)
        alpha = 0.1 * mse_dki / mse_mk
    if not quiet:
        print(f"alpha = {np.round(alpha, 5)}")

    data_flat = jnp.asarray(data[mask])
    design_matrix = jnp.asarray(design_matrix)
    x0_flat = jnp.asarray(x0[mask])
    mk_pred_flat = jnp.asarray(mk_pred[mask])
    ak_pred_flat = jnp.asarray(ak_pred[mask])
    rk_pred_flat = jnp.asarray(rk_pred[mask])
    vs = jnp.asarray(_45_dirs)
    axial_dirs_flat = jnp.asarray(axial_dirs[mask])
    radial_dirs_flat = jnp.asarray(radial_dirs[mask])
    size = len(data_flat)

    @jax.jit
    def cost(
        params,
        design_matrix,
        y,
        alpha,
        vs,
        mk_pred,
        ak_pred,
        rk_pred,
        axial_dir,
        radial_dirs,
    ):
        return (
            jnp.mean((jnp.exp(design_matrix @ params) - y) ** 2)
            + alpha * (jnp.mean(_akc(params, vs)) - mk_pred) ** 2
            + alpha * (jnp.mean(_akc(params, axial_dir)) - ak_pred) ** 2
            + alpha * (jnp.mean(_akc(params, radial_dirs)) - rk_pred) ** 2
        )

    @jax.jit
    def jit_minimize(i):
        return minimize(
            fun=cost,
            x0=x0_flat[i],
            args=(
                design_matrix,
                data_flat[i],
                alpha,
                vs,
                mk_pred_flat[i],
                ak_pred_flat[i],
                rk_pred_flat[i],
                axial_dirs_flat[i],
                radial_dirs_flat[i],
            ),
            method="BFGS",
            options={
                "maxiter": int(1e4),
                "line_search_maxiter": int(1e4),
                "gtol": 1e-4,
            },
        )

    params_flat = np.zeros((size, 22))
    status_flat = np.zeros(size)
    for i in range(size):
        if not quiet:
            print(f"{int(i/size*100)}%", end="\r")
        results = jit_minimize(i)
        params_flat[i] = results.x
        status_flat[i] = results.status
    if not quiet:
        print("100%")

    params = np.zeros(mask.shape + (22,))
    params[mask] = params_flat
    status = np.zeros(mask.shape)
    status[mask] = status_flat

    return params, status


class FitResult:
    """Class for storing the regularized NLLS fit results.

    Attributes
    ----------
    params : numpy.ndarray
        Estimated model parameters.
    status : numpy.ndarray
        Codes returned by the optimizer in each voxel where it was run. 0 means
        converged succesfully, 1 means maximum BFGS iterations reached, 3 means
        zoom failed, 4 means saddle point reached, 5 means maximum line search
        iterations reached, and -1 means undefined.
    mk_pred : numpy.ndarray
        Predicted mean kurtosis map.
    ak_pred : numpy.ndarray
        Predicted axial kurtosis map.
    rk_pred : numpy.ndarray
        Predicted radial kurtosis map.
    mask : numpy.ndarray
        Mask defining voxels in which the fit was run.
    params_nlls : numpy.ndarray
        Parameter estimates by the standard NLLS fit.
    akc_mask : nump.ndarray
        Mask defining voxels in which AKC computed from the standard NLLS fit
        results was non-negative along all directions.
    x0 : numpy.ndarray
        Initial positions of the regularized NLLS fit.
    alpha : float
        Value of the constant controlling the magnitude of regularization terms
        in the regularized NLLS fit.

    Notes
    -----
    The model parameters are the following:

            params[..., 0] = log(S0)
            params[..., 1] = D_xx
            params[..., 2] = D_yy
            params[..., 3] = D_zz
            params[..., 4] = D_xy
            params[..., 5] = D_xz
            params[..., 6] = D_yz
            params[..., 7] = W_xxxx * MD ** 2
            params[..., 8] = W_yyyy * MD ** 2
            params[..., 9] = W_zzzz * MD ** 2
            params[..., 10] = W_xxxy * MD ** 2
            params[..., 11] = W_xxxz * MD ** 2
            params[..., 12] = W_yyyx * MD ** 2
            params[..., 13] = W_yyyz * MD ** 2
            params[..., 14] = W_zzzx * MD ** 2
            params[..., 15] = W_zzzy * MD ** 2
            params[..., 16] = W_xxyy * MD ** 2
            params[..., 17] = W_xxzz * MD ** 2
            params[..., 18] = W_yyzz * MD ** 2
            params[..., 19] = W_xxyz * MD ** 2
            params[..., 20] = W_yyxz * MD ** 2
            params[..., 21] = W_zzxy * MD ** 2
    """

    def __init__(
        self,
        params,
        status,
        mk_pred,
        ak_pred,
        rk_pred,
        mask,
        params_nlls,
        akc_mask,
        x0,
        alpha,
    ):
        self.params = params
        self.status = status
        self.mk_pred = mk_pred
        self.ak_pred = ak_pred
        self.rk_pred = rk_pred
        self.mask = mask
        self.params_nlls = params_nlls
        self.akc_mask = akc_mask
        self.x0 = x0
        self.alpha = alpha
        return


def fit(data, bvals, bvecs, mask=None, alpha=None, seed=123, quiet=False):
    """Estimate diffusion and kurtosis tensor elements from data.

    This function does the following:

        1. Remove infinities, nans, and negative values in `data`, and scale
           values in `data` and `bvals`.
        2. Estimate model parameters using standard NLLS.
        3. Train multilayer perceptrons to predict kurtosis measures from data
           in voxels where the apparent kurtosis coefficient computed from the
           NLLS fit results is non-negative along all directions.
        4. Estimate model parameters using regularized NLLS where the
           regularization terms increase the objective function value when
           MK, AK, and RK deviate from their predicted values. Axially
           symmetric tensors with plausible magnitudes are used as initial
           positions for the fit.

    Parameters
    ----------
    data : numpy.ndarray
        Floating-point array with shape (..., number of acquisitions).
    bvals : numpy.ndarray
        Floating-point array with shape (number of acquisitions,).
    bvecs : numpy.ndarray
        Floating-point array with shape (number of acquisitions, 3).
    mask : numpy.ndarray, optional
        Boolean array with the same shape as `data` without the last dimension.
    alpha : float, optional
        Constant controlling regularization term magnitudes in the objective
        function. If not given, `alpha` is equal to the 0.1 times median
        squared error of the standard NLLS fit divided by the median squared
        error of the MK prediction.
    seed : int, optional
        Seed for pseudo-random number generation to initialize neural network
        weights.
    quiet : bool, optional
        Whether not to print messages about computation progress.

    Returns
    -------
    dkmri.FitResult
    """

    if mask is None:
        mask = np.ones(data.shape[0:-1]).astype(bool)

    data = data.astype(float)
    C_data = np.nanmean(data[mask][..., np.where(bvals == np.min(bvals))])
    data = data / C_data
    data[np.isinf(data)] = np.nan
    data[np.isnan(data)] = 0
    min_signal = np.finfo(float).resolution
    data[data < min_signal] = min_signal

    C_bvals = np.mean(bvals)
    bvals = bvals / C_bvals

    if not quiet:
        print("Fitting DKI to data with standard NLLS")
    X = design_matrix(bvals, bvecs)
    params_nlls = _nlls_fit(data, X, mask)

    if not quiet:
        print("Training neural networks to predict kurtosis maps")
    akc_mask = _akc_mask(params_to_W(params_nlls), _45_dirs, mask).astype(bool)
    mk = np.clip(np.nan_to_num(params_to_mk(params_nlls, mask)), MIN_K, MAX_K)
    mk_pred, R2 = _predict(data, mk, akc_mask, seed, mask)
    if not quiet:
        print(f"R^2 = {np.round(R2, 5)} for MK")
    ak = np.clip(np.nan_to_num(params_to_ak(params_nlls, mask)), MIN_K, MAX_K)
    ak_pred, R2 = _predict(data, ak, akc_mask, seed, mask)
    if not quiet:
        print(f"R^2 = {np.round(R2, 5)} for AK")
    rk = np.clip(np.nan_to_num(params_to_rk(params_nlls, mask)), MIN_K, MAX_K)
    rk_pred, R2 = _predict(data, rk, akc_mask, seed, mask)
    if not quiet:
        print(f"R^2 = {np.round(R2, 5)} for RK")

    if not quiet:
        print("Calculating initial positions")
    axial_dirs = np.zeros(mask.shape + (1, 3))
    axial_dirs_flat = axial_dirs[mask]
    radial_dirs = np.zeros(mask.shape + (10, 3))
    radial_dirs_flat = radial_dirs[mask]
    size = len(radial_dirs_flat)
    _, evecs = np.linalg.eigh(np.nan_to_num(params_to_D(params_nlls)))
    evecs_flat = evecs[mask]
    for i in range(size):
        axial_dirs_flat[i] = evecs_flat[i, :, 2].T
        R = _vec2vec_rotmat(np.array([1.0, 0, 0]), evecs_flat[i, :, 2])
        radial_dirs_flat[i] = (R @ _10_dirs.T).T
    axial_dirs[mask] = axial_dirs_flat
    radial_dirs[mask] = radial_dirs_flat
    S0 = np.exp(params_nlls[..., 0])
    D = params_to_D(params_nlls)
    mtk = np.clip(np.nan_to_num(_mtk(params_nlls, mask)), MIN_K, MAX_K)
    mtk_pred, _ = _predict(data, mtk, akc_mask, seed, mask)
    x0 = _calculate_x0(S0, D, mtk_pred, ak_pred, rk_pred, mask)

    if not quiet:
        print("Fitting DKI to data with regularized NLLS")
    params, status = _reg_nlls_fit(
        data,
        X,
        x0,
        mk_pred,
        ak_pred,
        rk_pred,
        axial_dirs,
        radial_dirs,
        alpha,
        mask,
        quiet,
    )

    params[..., 0] += np.log(C_data)
    params[..., 1:7] /= C_bvals
    params[..., 7::] /= C_bvals ** 2

    params_nlls[..., 0] += np.log(C_data)
    params_nlls[..., 1:7] /= C_bvals
    params_nlls[..., 7::] /= C_bvals ** 2

    x0[..., 0] += np.log(C_data)
    x0[..., 1:7] /= C_bvals
    x0[..., 7::] /= C_bvals ** 2

    return FitResult(
        params,
        status,
        mk_pred,
        ak_pred,
        rk_pred,
        mask,
        params_nlls,
        akc_mask,
        x0,
        alpha,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            """Estimate diffusion kurtosis imaging parameters and compute parameter maps.
            The command for using dkmri.py is "dkmri.py data bvals bvecs
            optional-arguments", where data, bvals, and bvecs are the paths of the files
            containing the diffusion-weighted data, b-values, and b-vectors, and
            optional-arguments is where to define things such as which parameter maps to
            save. Visit https://github.com/kerkelae/dkmri for more info."""
        )
    )
    parser.add_argument(
        "data", help="path of a NIfTI file with diffusion-weighted data",
    )
    parser.add_argument(
        "bvals", help="path of a text file with b-values",
    )
    parser.add_argument(
        "bvecs", help="path of a text file with b-vectors",
    )
    parser.add_argument(
        "-mask",
        help=(
            "path of a NIfTI file with a mask definining where to estimate parameters"
        ),
    )
    parser.add_argument(
        "-md", help="path of a NIfTI file in which to save the mean diffusivity map"
    )
    parser.add_argument(
        "-ad", help="path of a NIfTI file in which to save the axial diffusivity map",
    )
    parser.add_argument(
        "-rd", help="path of a NIfTI file in which to save the radial diffusivity map",
    )
    parser.add_argument(
        "-fa",
        help="path of a NIfTI file in which to save the fractional anisotropy map",
    )
    parser.add_argument(
        "-mk", help="path of a NIfTI file in which to save the mean kurtosis map"
    )
    parser.add_argument(
        "-ak", help="path of a NIfTI file in which to save the axial kurtosis map",
    )
    parser.add_argument(
        "-rk", help="path of a NIfTI file in which to save the radial kurtosis map",
    )
    parser.add_argument(
        "-s0", help="path of a NIfTI file in which to save the estimated signal at b=0",
    )
    parser.add_argument(
        "-mk_pred",
        help="path of a NIfTI file in which to save the predicted mean kurtosis map",
    )
    parser.add_argument(
        "-ak_pred",
        help="path of a NIfTI file in which to save the predicted axial kurtosis map",
    )
    parser.add_argument(
        "-rk_pred",
        help="path of a NIfTI file in which to save the predicted radial kurtosis map",
    )
    parser.add_argument(
        "-status",
        help="path of a NIfTI file in which to save the codes returned by the optimizer",
    )
    parser.add_argument(
        "-params",
        help="path of a NIfTI file in which to save the estimated parameters",
    )
    args = parser.parse_args()

    data_img = nib.load(args.data)
    data = data_img.get_fdata()
    affine = data_img.affine
    bvals = np.loadtxt(args.bvals)
    bvecs = np.loadtxt(args.bvecs)
    if bvecs.ndim == 2 and bvecs.shape[0] == 3:
        bvecs = bvecs.T
    if args.mask:
        mask = nib.load(args.mask).get_fdata().astype(bool)
    else:
        mask = None

    fit_result = fit(data, bvals, bvecs, mask)

    if args.md:
        nib.save(
            nib.Nifti1Image(params_to_md(fit_result.params, mask), affine), args.md,
        )
    if args.ad:
        nib.save(
            nib.Nifti1Image(params_to_ad(fit_result.params, mask), affine), args.ad,
        )
    if args.rd:
        nib.save(
            nib.Nifti1Image(params_to_rd(fit_result.params, mask), affine), args.rd,
        )
    if args.fa:
        nib.save(
            nib.Nifti1Image(params_to_fa(fit_result.params, mask), affine), args.fa,
        )
    if args.mk:
        nib.save(
            nib.Nifti1Image(params_to_mk(fit_result.params, mask), affine), args.mk
        )
    if args.ak:
        nib.save(
            nib.Nifti1Image(params_to_ak(fit_result.params, mask), affine), args.ak
        )
    if args.rk:
        nib.save(
            nib.Nifti1Image(params_to_rk(fit_result.params, mask), affine), args.rk
        )
    if args.s0:
        nib.save(nib.Nifti1Image(np.exp(fit_result.params[..., 0]), affine), args.s0)
    if args.mk_pred:
        nib.save(nib.Nifti1Image(fit_result.mk_pred, affine), args.mk_pred)
    if args.ak_pred:
        nib.save(nib.Nifti1Image(fit_result.ak_pred, affine), args.ak_pred)
    if args.rk_pred:
        nib.save(nib.Nifti1Image(fit_result.rk_pred, affine), args.rk_pred)
    if args.status:
        nib.save(nib.Nifti1Image(fit_result.status, affine), args.status)
    if args.params:
        nib.save(nib.Nifti1Image(fit_result.params, affine), args.params)
