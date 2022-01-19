"""Diffusion kurtosis magnetic resonance imaging."""

import numpy as np

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
    """Return the diffusion tensors corresponding to a parameter array.

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
    """Return the kurtosis tensors corresponding to a parameter array.

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
