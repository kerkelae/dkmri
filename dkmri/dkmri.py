#!/usr/bin/env python

"""Diffusion kurtosis magnetic resonance imaging."""

import argparse

import jax
import jax.numpy as jnp
from jax.scipy.optimize import minimize as jaxminimize
import nibabel as nib
import numba
import numpy as np
from scipy.optimize import minimize as scipyminimize
from sklearn.neural_network import MLPRegressor


jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

SEED = 123

# Parameter array elements are:
#
# params[..., 0] = log(S0)
# params[..., 1] = D_xx
# params[..., 2] = D_yy
# params[..., 3] = D_zz
# params[..., 4] = D_xy
# params[..., 5] = D_xz
# params[..., 6] = D_yz
# params[..., 7] = W_xxxx * MD ** 2
# params[..., 8] = W_yyyy * MD ** 2
# params[..., 9] = W_zzzz * MD ** 2
# params[..., 10] = W_xxxy * MD ** 2
# params[..., 11] = W_xxxz * MD ** 2
# params[..., 12] = W_yyyx * MD ** 2
# params[..., 13] = W_yyyz * MD ** 2
# params[..., 14] = W_zzzx * MD ** 2
# params[..., 15] = W_zzzy * MD ** 2
# params[..., 16] = W_xxyy * MD ** 2
# params[..., 17] = W_xxzz * MD ** 2
# params[..., 18] = W_yyzz * MD ** 2
# params[..., 19] = W_xxyz * MD ** 2
# params[..., 20] = W_yyxz * MD ** 2
# params[..., 21] = W_zzxy * MD ** 2


def _design_matrix(bvals, bvecs):
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


def _params_to_D(params):
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


def _params_to_W(params):
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
    evals, _ = np.linalg.eigh(_params_to_D(params))
    MD = np.mean(evals, axis=-1)[..., np.newaxis, np.newaxis, np.newaxis, np.newaxis]
    MD[MD == 0] = np.nan  # To avoid warnings due to division by zero
    W /= MD ** 2
    return W


def _tensors_to_params(S0, D, W):
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
    evals, _ = np.linalg.eigh(D)
    MD = np.mean(evals, axis=-1)[..., np.newaxis]
    params[..., 7::] *= MD ** 2
    return params


@jax.jit
def _adc(params, vs):
    """Compute apparent diffusion coefficients along unit vectors vs.

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


def _md(params, mask=None):
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
    evals, _ = np.linalg.eigh(_params_to_D(params[mask]))
    md = np.zeros(mask.shape)
    md[mask] = np.mean(evals, axis=1)
    return md


def _ad(params, mask=None):
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
    evals, _ = np.linalg.eigh(_params_to_D(params[mask]))
    ad = np.zeros(mask.shape)
    ad[mask] = evals[:, 2]
    return ad


def _rd(params, mask=None):
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
    evals, _ = np.linalg.eigh(_params_to_D(params[mask]))
    rd = np.zeros(mask.shape)
    rd[mask] = np.mean(evals[:, 0:2], axis=1)
    return rd


@jax.jit
def _akc(params, vs):
    """Compute apparent kurtosis coefficients along unit vectors vs.

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
)


def _mk(params, mask=None):
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


def _ak(params, mask=None):
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
    _, evecs = np.linalg.eigh(_params_to_D(params_flat))
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
)


@numba.njit
def _vec2vec_rotmat(v, k):
    """Compute a rotation matrix defining a rotation that aligns v with k.

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


def _rk(params, mask=None):
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
    _, evecs = np.linalg.eigh(_params_to_D(params_flat))
    for i in range(size):
        R = _vec2vec_rotmat(np.array([1.0, 0, 0]), evecs[i, :, 2])
        vs = (R @ _10_dirs.T).T
        rk_flat[i] = np.mean(_akc(params_flat[i], vs))
    rk[mask] = rk_flat
    return rk


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

    Optimization is done using the BFGS algorithm with the OLS solution as
    initial position.

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
        return jaxminimize(
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
            print(f"Fit was not successful in voxel {i} (status = {results.status})")
        params_flat[i] = results.x

    params = np.zeros(mask.shape + (22,))
    params[mask] = params_flat

    return params


@jax.jit
def _akc_mask(W, vs, mask):
    """Compute a binary mask based on kurtosis tensor positivity.

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
        akc_mask *= (v.T @ (v.T @ W @ v) @ v) > 0
    akc_mask *= mask
    return akc_mask


def _predict(data, m, akc_mask, mask=None, **kwargs):
    """Train a multilayer perceptron to predict `m` from `data` and return the
    predicted map and R-squared on training data.

    Parameters
    ----------
    data : numpy.ndarray
        Floating-point array with shape (..., number of acquisitions).
    m : numpy.ndarray
        Floating-point array.
    akc_mask : numpy.ndarray, optional
        Boolean array.
    mask : numpy.ndarray, optional
        Boolean array.
    **kwargs
        Keyword arguments passed to `sklearn.neural_network.MLPRegressor`.

    Returns
    -------
    np.ndarray
    float
    """
    if mask is None:
        mask = np.ones(data.shape[0:-1]).astype(bool)
    X = data[akc_mask]
    y = np.clip(np.nan_to_num(m[akc_mask]), -3 / 7, 10)
    reg = MLPRegressor(random_state=SEED, **kwargs).fit(X, y)
    R2 = reg.score(X, y)
    m_pred = np.zeros(mask.shape)
    m_pred[mask] = reg.predict(data[mask])
    return m_pred, R2


def _signal(params, design_matrix, mask=None):
    """Predict signal from parameters.

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
    S_hat = np.exp(design_matrix @ params[..., np.newaxis])[..., 0]
    S_hat[~mask] = 0
    return S_hat


def _calculate_x0(S0, D, mk_pred, ak_pred, rk_pred, mask=None):
    """Calculate initial positions for the regularized NLLS fit.
    
    Parameters
    ----------
    S0 : numpy.ndarray
        Floating-point array.
    D : numpy.ndarray
        Floating-point array with shape (..., 3, 3).
    mk_pred : numpy.ndarray
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
    mk_pred_flat = mk_pred[mask]
    ak_pred_flat = ak_pred[mask]
    rk_pred_flat = rk_pred[mask]
    size = len(S0_flat)

    evals, evecs = np.linalg.eigh(D_flat)

    @numba.njit
    def p(u):
        """Return tensor P."""
        P = np.zeros((3, 3, 3, 3))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        P[i, j, k, l] = u[i] * u[j] * u[k] * u[l]
        return P

    @numba.njit
    def d(i, j):
        """Kronecker delta."""
        if i == j:
            return 1
        else:
            return 0

    @numba.njit
    def q(u):
        """Return tensor Q."""
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

        mtk = 0.9 * mk_pred_flat[i]  # Incorrect
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


def _reg_nlls_fit(data, design_matrix, x0, mk_pred, mask=None):
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
    mask : numpy.ndarray, optional
        Boolean array.

    Returns
    -------
    numpy.ndarray
    """

    if mask is None:
        mask = np.ones(data.shape[0:-1]).astype(bool)

    data_flat = jnp.asarray(data[mask])
    x0_flat = jnp.asarray(x0[mask])
    design_matrix = jnp.asarray(design_matrix)
    vs = jnp.asarray(_45_dirs)
    mk_pred_flat = jnp.asarray(mk_pred[mask])
    size = len(data_flat)

    # mse_dki = np.median(
    #    np.mean((_signal(x0_flat, design_matrix) - data_flat) ** 2, axis=1)
    # )
    # mse_mk = np.median((mk_pred[mask] - _mk(x0_flat)) ** 2)
    # alpha = 1.0 * mse_dki / mse_mk
    alpha = 1  # Hard coded for now

    @jax.jit
    def cost(params, design_matrix, y, alpha, vs, mk_pred):
        return (
            jnp.mean((jnp.exp(design_matrix @ params) - y) ** 2)
            + alpha * (jnp.mean(_akc(params, vs)) - mk_pred) ** 2
        )

    jac = jax.jit(jax.jacfwd(cost))
    hess = jax.jit(jax.jacfwd(jax.jacrev(cost)))

    @jax.jit
    def jit_minimize(i):
        return jaxminimize(
            fun=cost,
            x0=x0_flat[i],
            args=(design_matrix, data_flat[i], alpha, vs, mk_pred_flat[i]),
            method="BFGS",
            options={
                "maxiter": int(1e4),
                "line_search_maxiter": int(1e3),
                "gtol": 1e-3,
            },
        )

    fit_status_flat = np.zeros(size)
    params_flat = np.zeros((size, 22))
    for i in range(size):
        print(f"{int(i/size*100)}%", end="\r")
        results = jit_minimize(i)
        if not results.success:
            results = scipyminimize(
                fun=cost,
                x0=x0_flat[i],
                args=(design_matrix, data_flat[i], alpha, vs, mk_pred_flat[i]),
                method="Newton-CG",
                jac=jac,
                hess=hess,
            )
            if not results.success:
                print(
                    f"Fit was not successful in voxel {i} (status = {results.status})"
                )
        params_flat[i] = results.x
        fit_status_flat[i] = results.status

    params = np.zeros(mask.shape + (22,))
    params[mask] = params_flat
    fit_status = np.zeros(mask.shape)
    fit_status[mask] = fit_status_flat

    return params, fit_status


if __name__ == "__main__":

    # Parse arguments

    parser = argparse.ArgumentParser(
        description=(
            "Estimate diffusion and kurtosis tensors, and compute parameter maps."
        )
    )
    parser.add_argument(
        "data", help="path of a NIfTI file with diffusion-weighted data.",
    )
    parser.add_argument(
        "bvals", help="path of a text file with b-values in units of s/mm^2.",
    )
    parser.add_argument(
        "bvecs", help="path of a text file with b-vectors.",
    )
    parser.add_argument(
        "-mask",
        help=(
            "path of a NIfTI file with a binary mask definining the voxels where to "
            "estimate the tensors."
        ),
    )
    parser.add_argument(
        "-md", help="path of a file in which to save the mean diffusivity map."
    )
    parser.add_argument(
        "-ad", help="path of a file in which to save the axial diffusivity map.",
    )
    parser.add_argument(
        "-rd", help="path of a file in which to save the radial diffusivity map.",
    )
    parser.add_argument(
        "-mk", help="path of a file in which to save the mean kurtosis map."
    )
    parser.add_argument(
        "-ak", help="path of a file in which to save the axial kurtosis map.",
    )
    parser.add_argument(
        "-rk", help="path of a file in which to save the radial kurtosis map.",
    )
    parser.add_argument(
        "-s0", help="path of a file in which to save the estimated signal at b=0.",
    )
    parser.add_argument(
        "-mk_pred",
        help="path of a file in which to save the predicted mean kurtosis map.",
    )
    parser.add_argument(
        "-ak_pred",
        help="path of a file in which to save the predicted axial kurtosis map.",
    )
    parser.add_argument(
        "-rk_pred",
        help="path of a file in which to save the predicted radial kurtosis map.",
    )
    parser.add_argument(
        "-fit_status", help="path of a file in which to save the fit status map.",
    )
    parser.add_argument(
        "-params", help="path of a file in which to save the parameters.",
    )
    args = parser.parse_args()

    # Load data

    data_img = nib.load(args.data)
    data = data_img.get_fdata()
    affine = data_img.affine
    bvals = np.loadtxt(args.bvals) * 1e-3
    bvecs = np.loadtxt(args.bvecs)
    if bvecs.ndim == 2 and bvecs.shape[0] == 3:
        bvecs = bvecs.T
    if args.mask:
        mask = nib.load(args.mask).get_fdata().astype(bool)
    else:
        mask = None

    # Clean, scale, and clip data

    data[np.isinf(data)] = np.nan
    data[np.isnan(data)] = 0
    C = np.nanmean(data[..., np.where(bvals == np.min(bvals))])
    data /= C
    min_signal = np.finfo(float).resolution
    data[data < min_signal] = min_signal

    # Fit model to data with NLLS

    print("Fitting DKI to data with standard NLLS")
    X = _design_matrix(bvals, bvecs)
    nlls_params = _nlls_fit(data, X, mask)

    # Predict kurtosis measures

    print("Training a MLP to predict MK")
    mk = np.clip(_mk(nlls_params, mask), -3 / 7, 10)
    akc_mask = _akc_mask(_params_to_W(nlls_params), _45_dirs, mask).astype(bool)
    mk_pred, R2 = _predict(data, mk, akc_mask, mask, hidden_layer_sizes=(50, 50))
    print(f"R^2 on training data for MK = {R2}")

    print("Training a MLP to predict AK")
    ak = np.clip(_ak(nlls_params, mask), -3 / 7, 10)
    akc_mask = _akc_mask(_params_to_W(nlls_params), _45_dirs, mask).astype(bool)
    ak_pred, R2 = _predict(data, ak, akc_mask, mask, hidden_layer_sizes=(50, 50))
    print(f"R^2 on training data for AK = {R2}")

    print("Training a MLP to predict RK")
    rk = np.clip(_rk(nlls_params, mask), -3 / 7, 10)
    akc_mask = _akc_mask(_params_to_W(nlls_params), _45_dirs, mask).astype(bool)
    rk_pred, R2 = _predict(data, rk, akc_mask, mask, hidden_layer_sizes=(50, 50))
    print(f"R^2 on training data for RK = {R2}")

    # Calculate initial positions

    print("Calculating initial positions")
    S0 = np.exp(nlls_params[..., 0])
    D = _params_to_D(nlls_params)
    x0 = _calculate_x0(S0, D, mk_pred, ak_pred, rk_pred, mask)

    # Fit model to data with regularized NLLS

    print("Fitting DKI to data with regularized NLLS")
    params, fit_status = _reg_nlls_fit(data, X, x0, mk_pred, mask)

    # Save results

    if args.params:
        nib.save(nib.Nifti1Image(params, affine), args.params)
    if args.md:
        nib.save(nib.Nifti1Image(_md(params, mask) * 1e-3, affine), args.md)
    if args.ad:
        nib.save(nib.Nifti1Image(_ad(params, mask) * 1e-3, affine), args.ad)
    if args.rd:
        nib.save(nib.Nifti1Image(_rd(params, mask) * 1e-3, affine), args.rd)
    if args.mk:
        nib.save(nib.Nifti1Image(_mk(params, mask), affine), args.mk)
    if args.ak:
        nib.save(nib.Nifti1Image(_ak(params, mask), affine), args.ak)
    if args.rk:
        nib.save(nib.Nifti1Image(_rk(params, mask), affine), args.rk)
    if args.s0:
        nib.save(nib.Nifti1Image(np.exp(params[..., 0] + np.log(C)), affine), args.s0)
    if args.mk_pred:
        nib.save(nib.Nifti1Image(mk_pred, affine), args.mk_pred)
    if args.ak_pred:
        nib.save(nib.Nifti1Image(ak_pred, affine), args.ak_pred)
    if args.rk_pred:
        nib.save(nib.Nifti1Image(rk_pred, affine), args.rk_pred)
    if args.fit_status:
        nib.save(nib.Nifti1Image(fit_status, affine), args.fit_status)

