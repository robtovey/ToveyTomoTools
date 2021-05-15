'''
Created on 10 May 2021

@author: Rob Tovey
'''
import pytest
import numpy as np
import RegTomoRecon as rtr

np.random.seed(1)


def test_compile():
    # test ball projection
    def norm(x): return (x ** 2).sum(-1) ** .5
    for d in range(1, 4):
        x = np.random.rand(10, d)
        y = x.copy()
        rtr.project(x, .5, y)
        x, y = norm(x), norm(y)
        ind = x < .5
        np.testing.assert_allclose(y[ind], x[ind], 1e-10, 1e-10)
        np.testing.assert_allclose(y[np.logical_not(ind)], .5, 1e-10, 1e-10)

        # test symmetric projection
        A = np.random.rand(10, d, d)
        B = A.copy()
        rtr.sym(A, B)
        np.testing.assert_allclose(A + np.transpose(A, (0, 2, 1)), 2 * B, 1e-10, 1e-10)


@pytest.mark.parametrize('dim, stem, spectral', [(dim, stem, spectral) for dim in (1, 2, 3)
                                                 for stem in ('grad', 'grad2') for spectral in (1, 3)])
def test_grad(dim, stem, spectral):
    shape = list(range(7, 7 + dim)) + ([spectral] if spectral > 1 else [])
    x = [x[..., None] for x in np.ogrid[[slice(s) for s in shape[:dim]]]]
    weights = np.random.rand(spectral, 1 + dim)
    if spectral == 1:
        f = weights[0, 0] + sum(weights[0, i + 1] * x[i] for i in range(dim))[..., 0]
        df = np.concatenate([weights[0, i + 1] * np.ones(f.shape)[..., None] for i in range(dim)],
                            axis=-1)
    else:
        f = np.concatenate([weights[j, 0] + sum(weights[j, i + 1] * x[i] for i in range(dim))
                            for j in range(3)], axis=-1)
        df = weights[:, 1:].reshape([1] * dim + [spectral, dim]) + 0 * f[..., None]

    Df = np.random.rand(*df.shape)
    func = rtr.c_diff[stem][str(dim) + '_' + ('0' if spectral == 1 else '1')]
    func(f, Df) if hasattr(func, 'targetdescr') else func[f.shape[:-1], (1,)](f, Df)

    if stem == 'grad':  # Forward differences, Neumann boundary
        for i in range(dim):
            Slice = [slice(None)] * len(shape) + [i]
            Slice[i] = slice(0, -1)
            np.testing.assert_allclose(df[tuple(Slice)], Df[tuple(Slice)])
            Slice[i] = -1
            assert abs(Df[tuple(Slice)]).max() < 1e-10
    else:  # Backward differences, Dirichlet boundary
        for i in range(dim):
            Slice = [slice(None)] * len(shape) + [i]
            Slice[i] = slice(1, None)
            np.testing.assert_allclose(df[tuple(Slice)], Df[tuple(Slice)])
            Slice[i] = 0

            assert abs(Df[tuple(Slice)] - f[tuple(Slice[:-1])]).max() < 1e-10


@pytest.mark.parametrize('dim, stem', [(dim, stem) for dim in (1, 2, 3) for stem in ('grad', 'grad2')])
def test_div(dim, stem):
    shape = list(range(7, 7 + dim))
    # test 1D divergence, 0D spectral component
    f, df = np.random.rand(*shape), np.random.rand(*shape, dim)
    F, Df = np.random.rand(*f.shape), np.random.rand(*df.shape)
    fwrd, bwrd = rtr.c_diff[stem][str(dim) + '_0'], rtr.c_diff[stem + 'T'][str(dim) + '_0']
    fwrd(f, Df) if hasattr(fwrd, 'targetdescr') else fwrd[f.shape[:-1], (1,)](f, Df)
    bwrd(df, F) if hasattr(fwrd, 'targetdescr') else bwrd[f.shape[:-1], (1,)](df, F)
    assert abs((f * F).sum() - (df * Df).sum()) < 1e-10

    # test 1D divergence, 1D spectral component
    f, df = np.random.rand(*shape, 3), np.random.rand(*shape, 3, dim)
    F, Df = np.random.rand(*f.shape), np.random.rand(*df.shape)
    fwrd, bwrd = rtr.c_diff[stem][str(dim) + '_1'], rtr.c_diff[stem + 'T'][str(dim) + '_1']
    fwrd(f, Df) if hasattr(fwrd, 'targetdescr') else fwrd[f.shape[:-1], (1,)](f, Df)
    bwrd(df, F) if hasattr(fwrd, 'targetdescr') else bwrd[f.shape[:-1], (1,)](df, F)
    assert abs((f * F).sum() - (df * Df).sum()) < 1e-10


def get_data(vol, angles):
    if len(vol) == 2:
        data = rtr.tomo_data(np.zeros((len(angles), vol[0])), angles, degrees=False)
    else:  # dim==3
        data = rtr.tomo_data(np.zeros((vol[0], len(angles), vol[1])), angles,
                             tilt_axis=0, stack_dim=1)
    return data


@pytest.mark.parametrize('dim', (2, 3))
def test_null_reconstruction(dim):
    shape = [10] * dim  # volume shape
    angles = np.linspace(0, np.pi, 9)  # detector angles

    data = get_data(shape, angles)
    Xray = data.getOperator(vol_shape=shape, backend='astra')

    for alg in (rtr.FBP(min_val=-1, max_val=1),
                rtr.SART(iterations=10, min_val=-1, max_val=1),
                rtr.SIRT(iterations=10, min_val=-1, max_val=1),):
        recon = alg.run(data=data, op=Xray)
        np.testing.assert_allclose(recon, 0, 1e-6, 1e-6)

    for alg in (rtr.TV(shape, order=1, weight=.9), rtr.TV(shape, order=2, weight=.9),
                rtr.TGV(shape, weight=.9)):
        alg.setParams(data=data, op=Xray)
        recon = alg.run(maxiter=10, x=rtr.getVec(alg.A, rand=True), y=rtr.getVec(alg.A.T, rand=True))
        np.testing.assert_allclose(recon, 0, 1e-6, 1e-6)
