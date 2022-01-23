import numpy as np
import numpy.testing as npt

from .. import dkmri


SEED = 123

params = np.array(
    [
        7.90764792,
        0.88660664,
        0.82186469,
        0.81741033,
        0.25016042,
        0.12341918,
        0.28344717,
        0.97744794,
        0.64809536,
        0.54047796,
        0.09333558,
        -0.06614247,
        0.07547532,
        0.16822022,
        0.12438352,
        0.14840455,
        0.16173709,
        0.17534938,
        0.42078548,
        -0.05851049,
        0.07203667,
        0.12034342,
    ]
)


def test__design_matrix():
    bvals = np.arange(5)
    bvecs = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    desired_X = np.array(
        [
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [0.0, -1.0, -2.0, -0.0, -0.0],
            [0.0, -0.0, -0.0, -3.0, -0.0],
            [0.0, -0.0, -0.0, -0.0, -4.0],
            [0.0, -0.0, -0.0, -0.0, -0.0],
            [0.0, -0.0, -0.0, -0.0, -0.0],
            [0.0, -0.0, -0.0, -0.0, -0.0],
            [0.0, 1 / 6, 2 / 3, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.5, 0.0],
            [0.0, 0.0, 0.0, 0.0, 8 / 3],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    ).T
    X = dkmri._design_matrix(bvals, bvecs)
    npt.assert_almost_equal(X, desired_X)


def test__params_to_D():
    desired_D = np.array(
        [
            [0.88660664, 0.25016042, 0.12341918],
            [0.25016042, 0.82186469, 0.28344717],
            [0.12341918, 0.28344717, 0.81741033],
        ]
    )
    D = dkmri._params_to_D(params)
    npt.assert_almost_equal(D, desired_D)


def test__params_to_W():
    desired_W = np.array(
        [
            [
                [
                    [1.37882815, 0.131663, -0.09330328],
                    [0.131663, 0.22815298, -0.0825373],
                    [-0.09330328, -0.0825373, 0.24735503],
                ],
                [
                    [0.131663, 0.22815298, -0.0825373],
                    [0.22815298, 0.10646858, 0.10161789],
                    [-0.0825373, 0.10161789, 0.16976136],
                ],
                [
                    [-0.09330328, -0.0825373, 0.24735503],
                    [-0.0825373, 0.10161789, 0.16976136],
                    [0.24735503, 0.16976136, 0.17546049],
                ],
            ],
            [
                [
                    [0.131663, 0.22815298, -0.0825373],
                    [0.22815298, 0.10646858, 0.10161789],
                    [-0.0825373, 0.10161789, 0.16976136],
                ],
                [
                    [0.22815298, 0.10646858, 0.10161789],
                    [0.10646858, 0.9142299, 0.23729835],
                    [0.10161789, 0.23729835, 0.59357726],
                ],
                [
                    [-0.0825373, 0.10161789, 0.16976136],
                    [0.10161789, 0.23729835, 0.59357726],
                    [0.16976136, 0.59357726, 0.20934554],
                ],
            ],
            [
                [
                    [-0.09330328, -0.0825373, 0.24735503],
                    [-0.0825373, 0.10161789, 0.16976136],
                    [0.24735503, 0.16976136, 0.17546049],
                ],
                [
                    [-0.0825373, 0.10161789, 0.16976136],
                    [0.10161789, 0.23729835, 0.59357726],
                    [0.16976136, 0.59357726, 0.20934554],
                ],
                [
                    [0.24735503, 0.16976136, 0.17546049],
                    [0.16976136, 0.59357726, 0.20934554],
                    [0.17546049, 0.20934554, 0.76242038],
                ],
            ],
        ]
    )
    W = dkmri._params_to_W(params)
    npt.assert_almost_equal(W, desired_W)


def test__tensors_to_params():
    S0 = np.exp(params[..., 0])
    D = dkmri._params_to_D(params)
    W = dkmri._params_to_W(params)
    npt.assert_almost_equal(dkmri._tensors_to_params(S0, D, W), params)
    return


def test__adc():
    np.random.seed(SEED)
    D = dkmri._params_to_D(params)
    for _ in range(100):
        v = np.random.random((1, 3)) - 0.5
        v /= np.linalg.norm(v)
        desired_adc = (v @ D @ v.T)[0]
        adc = np.asarray(dkmri._adc(params, v))
        npt.assert_almost_equal(adc, desired_adc)
        vs = np.vstack((v, v))
        adcs = np.asarray(dkmri._adc(params, vs))
        npt.assert_almost_equal(adcs[0], adc)
        npt.assert_almost_equal(adcs[1], adc)


def test__md():
    desired_md = 0.8419605533333335
    md = dkmri._md(params)
    npt.assert_almost_equal(md, desired_md)


def test__ad():
    desired_ad = 1.2839527280964818
    ad = dkmri._ad(params)
    npt.assert_almost_equal(ad, desired_ad)


def test__rd():
    desired_rd = 0.6209644659517595
    rd = dkmri._rd(params)
    npt.assert_almost_equal(rd, desired_rd)


def test__akc():
    np.random.seed(SEED)
    D = dkmri._params_to_D(params)
    W = dkmri._params_to_W(params)
    for _ in range(100):
        v = np.random.random((1, 3)) - 0.5
        v /= np.linalg.norm(v)
        md = dkmri._md(params)
        adc = dkmri._adc(params, v)
        desired_akc = (md / adc) ** 2 * v[0] @ (v[0] @ W @ v[0]) @ v[0]
        akc = np.asarray(dkmri._akc(params, v))
        npt.assert_almost_equal(akc, desired_akc)
        vs = np.vstack((v, v))
        akcs = np.asarray(dkmri._akc(params, vs))
        npt.assert_almost_equal(akcs[0], akc)
        npt.assert_almost_equal(akcs[1], akc)


def test__mk():
    desired_mk = 1.1124342668323295
    mk = dkmri._mk(params)
    npt.assert_almost_equal(mk, desired_mk)


def test__ak():
    desired_ak = 0.7109767625600302
    ak = dkmri._ak(params)
    npt.assert_almost_equal(ak, desired_ak)


def test__rk():
    desired_rk = 1.5180490434619633
    rk = dkmri._rk(params)
    npt.assert_almost_equal(rk, desired_rk)

