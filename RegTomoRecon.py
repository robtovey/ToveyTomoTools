'''
Created on 25 Oct 2018

@author: Rob Tovey
'''
import numpy as np
from numpy.linalg import norm
from scipy.sparse import dia_matrix
from scipy.sparse.linalg import LinearOperator
# flag for how to compile optimised functions
# I think _GPU=True compiles but is un-tested.
_GPU = False

try:
    import numba
    _params = dict(target='parallel', fastmath=True, cache=True)

    @numba.guvectorize(['T[:],T,T[:]'.replace('T', T) for T in ('f4', 'f8')], '(n),()->(n)', **_params)
    def project(y, scale, out):
        n = 0
        for j in range(y.size):
            n += y[j] ** 2
        if n <= scale ** 2:
            for j in range(y.size):
                out[j] = y[j]
        else:
            n = min(scale * (n ** -.5), 1e8)
            for j in range(y.size):
                out[j] = n * y[j]

    @numba.vectorize(['T(T,T)'.replace('T', T) for T in ('f4', 'f8')], **_params)
    def project1D(y, scale):
        '''
        out = min(1, scale/|y|) * y
        '''
        if y > scale:
            return scale
        elif y < -scale:
            return -scale
        else:
            return y

    @numba.vectorize(['T(T,T)'.replace('T', T) for T in ('f4', 'f8')], **_params)
    def shrink1D(y, scale):
        '''
        out = max(0, |y|-scale) * sign(y)
        '''
        if y > scale:
            return y - scale
        elif y < -scale:
            return y + scale
        else:
            return 0

    @numba.jit(fastmath=True, parallel=True, cache=True)
    def L1_gap(A, X, scale):
        '''
        return |A+scale*sign(X)|
        '''
        out = 0
        for i in numba.prange(A.size):
            a, x = A[i], X[i]
            if x > 0:
                v = abs(a + scale)
            elif x < 0:
                v = abs(a - scale)
            else:
                v = abs(a) - scale
            out = max(out, v)
        return out

    @numba.guvectorize(['T[:,:],T[:,:]'.replace('T', T) for T in ('f4', 'f8')], '(n,n)->(n,n)', nopython=True, target='parallel', fastmath=True, cache=True)
    def sym(A, B):
        '''
        B[i,j,J] = .5(A[i,j,J]+A[i,J,j])
        '''
        for j0 in range(A.shape[1]):
            for j1 in range(j0):
                B[j0, j1] = .5 * (A[j0, j1] + A[j1, j0])
                B[j1, j0] = B[j0, j1]
            B[j0, j0] = A[j0, j0]

    def genbody(lines, var, end, tab, dim, options):
        from itertools import product

        Dim = len(var)
        func = ''
        old = (None,) * dim
        for block in product(*((options,) * dim)):
            # if options = (0,1,2), dim=2, then block = (0,0), (0,1), (0,2), (1,0),...
            stepdown = 0  # number of tabs which need to be subtracted at the end of the block
            first = True  # only the first loop is parallelised
            for i in range(dim):  # Naive loop over spectral dimension
                if block[i] == old[i]:  # action can be commented out
                    func += '#'

                if block[i] == 0:  # var[i] = 0 is fixed for this block
                    func += '\t' * tab + var[i] + ' = 0\n'
                elif block[i] == 2:  # var[i] = -1 is fixed for this block
                    func += '\t' * tab + \
                        var[i] + ' = ' + end[i] + '\n'
                elif block[i] == 1:  # var[i] loops in this block
                    if block[i] == old[i]:
                        func += '\t' * tab + var[i] + ' is looping\n'
                    else:
                        func += ('\t' * tab + 'for ' + var[i] + ' in ' +
                                 ('numba.prange(' if first else 'range(') +
                                 ('1' if options[0] == 0 else '0') + ', ' + end[i] + '):\n')
                        tab += 1
                    stepdown += all(j == options[-1] for j in block[i + 1:])
                    first = False
                else:
                    raise ValueError

            for i in range(dim, Dim):  # Naive loop over spectral dimensions
                func += ('\t' * tab + 'for ' +
                         var[i] + ' in range(' + end[i] + '):\n')
                tab += 1

            # Print actual computations (loop body)
            for i in range(dim):
                func += '\t' * tab + lines[block[i]][i] + '\n'

            tab -= (Dim - dim)
            if stepdown:
                tab -= stepdown
            old = tuple(b for b in block)
            func += '\n'
        return func

    # The following functions write a module with highly optimised code for
    # computing gradients etc.

    def genbody_GPU(lines, var, end, tab, dim, options):
        isPad = len(var) - dim  # either 1 or 0
        func = ''

        func += '\t' * tab + \
            ','.join(var[:dim]) + ' = cuda.grid(' + str(dim) + ')\n'

        # Don't permit out of bound indices
        for i in range(dim):
            func += ('\t' * tab + 'if ' +
                     var[i] + ' >= f.shape[' + str(i) + ']:\n')
            func += ('\t' * (tab + 1) + 'return\n')

        old_options = [i for i in options]
        options = tuple(i for i in (0, 2, 1) if i in old_options)

        for i in range(dim):
            for j in options:
                if j == 0:
                    func += '\t' * tab + 'if ' + var[i] + '== 0:\n'
                elif j == 2:
                    func += ('\t' * tab + ('el' if 0 in options else '')
                             +'if ' + var[i] + ' == ' + end[i] + ':\n')
                else:
                    func += ('\t' * tab + 'else:\n')
                tab += 1

                if isPad:
                    func += ('\t' * tab + 'for ' +
                             var[-1] + ' in range(' + end[-1] + '):\n')
                    tab += 1

                func += '\t' * tab + lines[j][i] + '\n'
                tab -= 1 + isPad

        return func

    # Writes the gradient function
    def gen_g(dim, pad=0, GPU=True):
        start = 'def _g' + str(dim) + '_' + str(pad) + '(f,Df):'

        lines = [[], [], []]
        var = ['i' + str(i) for i in range(dim + pad)]
        end = tuple('f.shape[' + str(i) + ']-1' for i in range(dim))
        # lines[0] = first index lines
        # lines[1] = middle index lines
        for i in range(dim):
            lines[1].append(
                'Df[' + ','.join(var) + ', ' + str(i) + '] = ' +
                'f[' + ','.join(var[:i] + [var[i] + '+1'] + var[(i + 1):]) + ']' +
                ' - f[' + ','.join(var) + ']')
        # lines[2] = end index lines
        for i in range(dim):
            lines[2].append(
                'Df[' + ','.join(var) + ', ' + str(i) + '] = 0')

        for j in range(pad):
            for i in range(3):
                lines[i].append('')
            end += ('f.shape[' + str(dim + j) + ']',)

        tab = 0
        func = '\t' * tab + start + '\n'
        tab += 1
        if _GPU:
            return func + genbody_GPU(lines, var, end, tab, dim, (1, 2))
        else:
            return func + genbody(lines, var, end, tab, dim, (1, 2))

    # Writes the adjoint gradient (divergence) function
    def gen_gt(dim, pad=0, GPU=True):
        start = 'def _gt' + str(dim) + '_' + str(pad) + '(Df,f):'

        lines = [[], [], []]
        var = ['i' + str(i) for i in range(dim + pad)]
        end = tuple('Df.shape[' + str(i) + ']-1' for i in range(dim))
        # lines[0] = first index lines
        lines[0].append(
            'f[' + ','.join(var) + '] = -Df[' + ','.join(var) + ', 0]')
        for i in range(1, dim):
            lines[0].append(
                'f[' + ','.join(var) + '] -= Df[' + ','.join(var) + ', ' + str(i) + ']')
        # lines[1] = middle index lines
        lines[1].append(
            'f[' + ','.join(var) + '] = Df[' + ','.join([var[0] + '-1'] + var[1:]) + ', 0] - Df[' + ','.join(var) + ', 0]')
        for i in range(1, dim):
            lines[1].append(
                'f[' + ','.join(var) + '] += Df[' + ','.join(var[:i] + [var[i] + '-1'] + var[(i + 1):]) + ', ' + str(i) + '] - Df[' + ','.join(var) + ', ' + str(i) + ']')
        # lines[2] = end index lines
        lines[2].append(
            'f[' + ','.join(var) + '] = Df[' + ','.join([var[0] + '-1'] + var[1:]) + ', 0]')
        for i in range(1, dim):
            lines[2].append(
                'f[' + ','.join(var) + '] += Df[' + ','.join(var[:i] + [var[i] + '-1'] + var[(i + 1):]) + ', ' + str(i) + ' ]')

        for j in range(pad):
            for i in range(3):
                lines[i].append('')
            end += ('Df.shape[' + str(dim + j) + ']',)

        tab = 0
        func = '\t' * tab + start + '\n'
        tab += 1
        if _GPU:
            return func + genbody_GPU(lines, var, end, tab, dim, (0, 1, 2))
        else:
            return func + genbody(lines, var, end, tab, dim, (0, 1, 2))

    # Writes the second gradient function, g2(g(f)) = d^2f/dx^2
    def gen_g2(dim, pad=0, GPU=True):
        start = 'def _g2' + str(dim) + '_' + str(pad) + '(f,Df):'

        lines = [[], [], []]
        var = ['i' + str(i) for i in range(dim + pad)]
        end = tuple('f.shape[' + str(i) + ']' for i in range(dim))
        # lines[0] = first index lines
        for i in range(dim):
            lines[0].append(
                'Df[' + ','.join(var) + ', ' + str(i) + '] = f[' + ','.join(var) + ']')
        # lines[1] = middle index lines
        for i in range(dim):
            lines[1].append(
                'Df[' + ','.join(var) + ', ' + str(i) + '] = ' +
                'f[' + ','.join(var) + ']' +
                ' - f[' + ','.join(var[:i] + [var[i] + '-1'] + var[(i + 1):]) + ']')
        # lines[2] = end index lines

        for j in range(pad):
            for i in range(3):
                lines[i].append('')
            end += ('f.shape[' + str(dim + j) + ']',)

        tab = 0
        func = '\t' * tab + start + '\n'
        tab += 1
        if _GPU:
            return func + genbody_GPU(lines, var, end, tab, dim, (0, 1))
        else:
            return func + genbody(lines, var, end, tab, dim, (0, 1))

    # Writes the adjoint to second gradient function
    def gen_g2t(dim, pad=0, GPU=True):
        start = 'def _g2t' + str(dim) + '_' + str(pad) + '(Df,f):'

        lines = [[], [], []]
        var = ['i' + str(i) for i in range(dim + pad)]
        end = tuple('Df.shape[' + str(i) + ']-1' for i in range(dim))
        # lines[0] = first index lines
        # lines[1] = middle index lines
        lines[1].append(
            'f[' + ','.join(var) + '] = Df[' + ','.join(var) + ', 0] - Df[' + ','.join([var[0] + '+1'] + var[1:]) + ', 0]')
        for i in range(1, dim):
            lines[1].append(
                'f[' + ','.join(var) + '] += Df[' + ','.join(var) + ', ' + str(i) + '] - ' +
                'Df[' + ','.join(var[:i] + [var[i] + '+1'] + var[(i + 1):]) + ', ' + str(i) + ']')
        # lines[2] = end index lines
        lines[2].append(
            'f[' + ','.join(var) + '] = Df[' + ','.join(var) + ', 0]')
        for i in range(1, dim):
            lines[2].append(
                'f[' + ','.join(var) + '] += Df[' + ','.join(var) + ', ' + str(i) + ' ]')

        for j in range(pad):
            for i in range(3):
                lines[i].append('')
            end += ('Df.shape[' + str(dim + j) + ']',)

        tab = 0
        func = '\t' * tab + start + '\n'
        tab += 1
        if _GPU:
            return func + genbody_GPU(lines, var, end, tab, dim, (1, 2))
        else:
            return func + genbody(lines, var, end, tab, dim, (1, 2))

    if _GPU:
        numbastr = ['@cuda.jit(', ')\n']
    else:
        numbastr = [
            '@numba.jit(', 'nopython=True, parallel=False, fastmath=True, cache=True)\n']

    def tosig(i, j):
        return ''
        return ('["void( T[' + ','.join(':' * i) + '], T[' + ','.join(':' * j) + '])".replace("T",t) for t in ["f4","f8"]], ')

    try:
        import _bin
    except Exception:
        with open('_bin.py', 'w') as f:
            if _GPU:
                print('import numba\nfrom numba import cuda\n', file=f)
            else:
                print('import numba\n', file=f)

            for i in range(3):  # i+1 is dimension of volume
                for j in range(2):  # j is spectral dimension
                    print(numbastr[0] + tosig(i + 1 + j, i + 2 + j) +
                          numbastr[1] + gen_g(i + 1, j, _GPU), file=f)
#                     print('print("' + str(8 * i + 4 * j + 1) + '/24", end="\\r")', file=f)
                    print(numbastr[0] + tosig(i + 2 + j, i + 1 + j) +
                          numbastr[1] + gen_gt(i + 1, j, _GPU), file=f)
#                     print('print("' + str(8 * i + 4 * j + 2) + '/24", end="\\r")', file=f)

#                     if i != 2 or j != 1:
                    print(numbastr[0] + tosig(i + 1 + j, i + 2 + j) +
                          numbastr[1] + gen_g2(i + 1, j, _GPU), file=f)
#                     print('print("' + str(8 * i + 4 * j + 3) + '/24", end="\\r")', file=f)

                    print(numbastr[0] + tosig(i + 2 + j, i + 1 + j) +
                          numbastr[1] + gen_g2t(i + 1, j, _GPU), file=f)
#                     print('print("' + str(8 * i + 4 * j + 4) + '/24", end="\\r")', file=f)

        import _bin
        print('Compilation successful')

except Exception:
    raise
    # numpy fall-back
    if 'project' not in globals():
        def project(y, scale):
            n = norm(y, 2, -1, True) + 1e-8
            n = np.minimum(1, scale / n)
            return y * n
        def project1D(y, scale): return project(y[:, None], scale)[:, 0]
        def shrink1D(y, scale): return np.sign(y) * np.maximum(0, y.abs() - scale)

        def sym(A): return .5 * (A + np.swapaxes(A, -2, -1))

c_diff = {'grad': {}, 'gradT': {}, 'grad2': {}, 'grad2T': {}, }
for i in range(3):
    for j in range(2):
        end = str(i + 1) + '_' + str(j)
        c_diff['grad'][end] = getattr(_bin, '_g' + end)
        c_diff['gradT'][end] = getattr(_bin, '_gt' + end)
        c_diff['grad2'][end] = getattr(_bin, '_g2' + end)
        c_diff['grad2T'][end] = getattr(_bin, '_g2t' + end)


# TODO: reimplement addition/multiplication etc.
class tomo_data(np.ndarray):
    '''
    Object that behaves like a numpy array but wraps additional functionality 
    for tomography, such as reordering axes for efficient projection and 
    constructing tomography projectors.
    '''
    def __new__(cls, arr, angles, stack_dim=None, tilt_axis=0,
                spect_dim=0, centres=None, real_size=None,
                degrees=False, geom='parallel'):
        dim = arr.ndim - spect_dim

        if stack_dim is None:
            stack_dim = 0 if dim == 2 else 1

        if dim == 3:
            if type(tilt_axis) is int:
                # For dim=3 want axes of arr to be [constant, angle, ...]
                axes = [i for i in range(arr.ndim)
                        if i not in (tilt_axis, stack_dim)]
                axes.insert(0, tilt_axis)
                axes.insert(1, stack_dim)
                arr = np.transpose(arr, axes)
            else:
                # Not sure what orders to use for general tilt
                # TODO: Full specified angles
                raise NotImplementedError

        elif stack_dim > 0:
            # For dim=2 want axes of arr to be [angle, ...]
            axes = [i for i in range(arr.ndim) if i != stack_dim]
            axes.insert(0, stack_dim)
            arr = np.transpose(arr, axes)

        arr = np.ascontiguousarray(arr)
        self = super(tomo_data, cls).__new__(cls, arr.shape, dtype=arr.dtype,
                                             buffer=arr, strides=arr.strides, order='C')

        self.space_dim = dim
        self.spect_dim = spect_dim
        self.centres = centres

        if dim == 2:
            real_size = (self.shape[1],) if real_size is None else real_size
            self.det = (real_size[0] / self.shape[1], self.shape[1])
        else:
            real_size = (self.shape[0], self.shape[2]
                         ) if real_size is None else real_size
            self.det = (real_size[0] / self.shape[0], real_size[1] / self.shape[2],
                        self.shape[0], self.shape[2])

        self.geom = geom
        angles = np.array(angles)
        if degrees:
            if angles.ndim == 1:
                angles *= np.pi / 180
            else:
                # TODO: Full specified angles
                raise NotImplementedError
        self.angles = angles

        return self

    def getOperator(self, vol_shape=None, backend='astra', GPU=True, **kwargs):
        if self.angles.ndim != 1:
            # TODO: Full specified angles
            raise NotImplementedError

        dim = self.space_dim
        if vol_shape is None:
            if dim == 2:
                vol_shape = (self.shape[1],) * 2
            else:
                vol_shape = self.shape[:1] + (self.shape[2],) * 2

        if backend == 'astra':
            if dim == 3:
                vol_shape = list(vol_shape[1:]) + [vol_shape[0]]

            # Only ever use 'cuda' or linear/linearcone projections
            import astra
            GPU = (GPU and astra.astra.use_cuda())
            dim_str = ('3d' if dim == 3 else '')

            vol_geom = astra.creators.create_vol_geom(vol_shape)

            if self.geom == 'parallel':
                if self.angles.ndim == 1:
                    if dim == 2:
                        proj_geom = astra.creators.create_proj_geom(
                            'parallel' + dim_str, *self.det, self.angles)
                    else:
                        proj_geom = astra.creators.create_proj_geom(
                            'parallel' + dim_str, *self.det, -self.angles)

                else:
                    raise NotImplementedError

                if GPU:
                    ID = astra.create_projector('cuda' + dim_str,
                                                proj_geom, vol_geom)
                else:
#                     ID = astra.create_projector('linear' + dim_str,
#                                                 proj_geom, vol_geom)
                    ID = astra.create_projector('line' + dim_str,
                                                proj_geom, vol_geom)
            else:
                raise NotImplementedError

            op = astra.OpTomo(ID)
            op.is_cuda = GPU

        elif backend == 'skimage':
            from skimage.transform import radon, iradon
            theta = self.angles * 180 / np.pi

            def fwrd(x): return radon(x.reshape(vol_shape[-2:]), theta, True).T

            interpolation = kwargs.get('interpolation', 'linear')

            def bwrd(x):
                return iradon(
                    x.reshape(self.shape[-2:]).T, theta, vol_shape[-1],
                    None, interpolation, True)

            if dim == 2:
                op = LinearOperator(
                    shape=(np.prod(self.shape), np.prod(vol_shape)),
                    matvec=fwrd, rmatvec=bwrd)
            else:
                op = LinearOperator(
                    shape=(np.prod(self.shape), np.prod(vol_shape)),
                    matvec=lambda x: _doSliceWise(
                        fwrd, x.reshape(vol_shape), self.shape, False),
                    rmatvec=lambda x: _doSliceWise(bwrd, x.reshape(self.shape), vol_shape, False))
            op.interpolation = interpolation
            op.vshape, op.sshape = vol_shape, self.shape
        else:
            raise NotImplementedError

        try:
            op.T
        except AttributeError:
            op._transpose = op._adjoint
        op.backend = backend
        return op

    def asarray(self):
        return np.ndarray(self.shape, dtype=self.dtype,
                       buffer=self.data, strides=self.strides, order='C')


def _doSliceWise(func, data, shape, inplace=False):
    out = np.empty(shape, dtype=data.dtype)
    if inplace:
        for i in range(shape[0]):
            func(data[i], out[i])
    else:
        for i in range(shape[0]):
            out[i] = func(data[i])
    return out


def _timelen(t):
    '''
    Converts length of time to a reasonable string depending on scale of time
    '''
    if t > 3600:
        H = t // 3600
        M = (t - 3600 * H) // 60
        return '%dh%2d' % (H, M)
    elif t > 60:
        M = t // 60
        T = int(t - 60 * M)
        return '%2dm%2d' % (M, T)
    else:
        return '%2ds' % int(t)


def cleanup_astra():
    '''
    Astra is bad at clearing up after itself on exit. This function performs
    the jobs manually.
    '''
    try:
        import astra
    except ImportError:
        return
    import atexit
    import signal

    def del_astra(*_, **__):
        try:
            astra.data2d.clear()
            astra.data3d.clear()
            astra.projector.clear()
            astra.algorithm.clear()
            astra.matrix.clear()
            astra.functions.clear()
        except Exception:
            pass

    atexit.register(del_astra)
    signal.signal(signal.SIGTERM, del_astra)
    signal.signal(signal.SIGINT, del_astra)


class tomo_alg():
    '''
    Base class for performing reconstructions from tomography data.
    '''
    def __init__(self, op=None):
        self.op = op

    def setParams(self, data, op=None, **kwargs):
        raise NotImplementedError

    def run(self, **kwargs):
        raise NotImplementedError

    def getRecon(self):
        return self.recon


class tomo_iter_alg(tomo_alg):
    '''
    Base class for tomography algorithms which are iterable
    '''
    def __init__(self, op=None):
        tomo_alg.__init__(self, op)

    def run(self, maxiter=100, callback=None, callback_freq=10, **kwargs):
        from time import time
        tic = time()

        self.start(**kwargs)

        if callback is None:
            print('Started reconstruction... ', flush=True, end='')
            self.step(0, maxiter)
            print('Finished after ' + str(int(time() - tic)), 's')
        else:
            callback = ('Iter', 'Time',) + Tuple(callback)
            prints = []

            def padstr(x, L=13):
                x = str(x)
                l = max(0, L - len(x))
                return ' ' * int(l / 2) + x + ' ' * (l - int(l / 2))

            def frmt(x):
                if type(x) == int:
                    x = '% 3d' % x
                elif np.isscalar(x):
                    x = '% 1.3e' % float(x)
                else:
                    x = str(x)
                return padstr(x)

            i = 0
            print(padstr(callback[0], 6), padstr(callback[1], 7),
                  *(padstr(c) for c in callback[2:]))
            prints.append((i, time() - tic,
                           ) + Tuple(self.callback(callback[2:])))
            print(padstr('%3d%%' % (i / maxiter * 100), 6), padstr(_timelen(prints[-1][1]), 7),
                  *(frmt(c) for c in prints[-1][2:]), flush=True)
            while i < maxiter:
                leap = min(callback_freq, maxiter - i)
                self.step(i, leap)
                i += leap
                prints.append((i, time() - tic,) +
                              Tuple(self.callback(callback[2:])))
                print(padstr('%3d%%' % (i / maxiter * 100), 6), padstr(_timelen(prints[-1][1]), 7),
                      *(frmt(c) for c in prints[-1][2:]), flush=True)
            print()

            dtype = [(callback[i], ('S20' if type(prints[0][i]) is str else 'f4'))
                     for i in range(len(callback))]
            prints = np.array(prints, dtype)
            Q = {callback[j]: prints[callback[j]]
                 for j in range(len(callback))}

        if callback is None:
            return self.getRecon()
        else:
            return self.getRecon(), Q

    def start(self, **_init):
        raise NotImplemented

    def step(self, i, niter):
        raise NotImplemented

    def callback(self, names):
        raise NotImplementedError


def Tuple(x):
    ''' If x is iterable then x is cast to tuple, otherwise it is converted to 
    a length 1 tuple'''
    return tuple(x) if (hasattr(x, '__iter__') and (type(x) is not str)) else (x,)


def Vector(*x):
    '''
    Converts an iterable into an array of objects
    '''
    # Test for generators:
    if hasattr(x[0], 'gi_yieldfrom'):
        x = tuple(x[0])

    X = np.array(x, dtype=object)
    if X.ndim != 1:
        X = np.empty(len(x), dtype=object)
        X[:] = x
    return X


class scalar_mat(LinearOperator):
    ''' Wrapper for matrices which are scalar multiples of the identity '''
    def __init__(self, shape, scale=0, dtype='f4'):
        LinearOperator.__init__(self, dtype, shape)
        self.scale = scale
        self._transpose = self._adjoint

    def _matvec(self, x):
        s = self.scale
        if s == 0:
            return np.zeros(self.shape[0], dtype=x.dtype)
        elif s == 1:
            return x.copy()
        elif s == -1:
            return -x
        else:
            return s * x

    def _rmatvec(self, x):
        s = self.scale
        if s == 0:
            return np.zeros(self.shape[1], dtype=x.dtype)
        elif s == 1:
            return x.copy()
        elif s == -1:
            return -x
        else:
            return s * x


class Matrix(LinearOperator):
    ''' Wrapper scipy linear operators to allow for block-matrices '''
    def __init__(self, m, shape=None, _adjoint=None):
        # Create block matrix
        if shape is None:
            buf = np.array(m, dtype=object)
            if buf.ndim < 2:
                buf.shape = [-1, 1]
            elif buf.ndim > 2:
                if type(m) not in (list, tuple):
                    buf = np.empty((1, 1), dtype=object)
                    buf[0, 0] = m
                elif type(m[0]) not in (list, tuple):
                    buf = np.empty((len(m), 1), dtype=object)
                    buf[:, 0] = m
                else:
                    buf = np.empty((len(m), len(m[0])), dtype=object)
                    for i in range(len(m)):
                        buf[i] = m[i]
        else:
            buf = np.empty(shape, dtype=object)
            for i in range(len(m)):
                buf[i] = m[i]

        # Check shapes of blocks
        h, w = 0 * np.empty(buf.shape[0], dtype=int), 0 * \
            np.empty(buf.shape[1], dtype=int)
        for i in range(buf.shape[0]):
            for j in range(buf.shape[1]):
                b = buf[i, j]
                if hasattr(b, 'shape'):
                    h[i], w[j] = b.shape[:2]
        if (h.min() == 0) or (w.min() == 0):
            raise ValueError('Every row and column must have a known shape')
        for i in range(buf.shape[0]):
            for j in range(buf.shape[1]):
                if buf[i, j] is None:
                    buf[i, j] = scalar_mat((h[i], w[j]), 0)
                elif np.isscalar(buf[i, j]):
                    buf[i, j] = scalar_mat((h[i], w[j]), buf[i, j])
                elif not hasattr(buf[i, j], 'shape'):
                    buf[i, j].shape = (h[i], w[j])

        LinearOperator.__init__(self, object, buf.shape)

        self.m = buf
        self.shapes = (h, w)

        if _adjoint is None:
            # Adjoint operator
            buf = np.empty((buf.shape[1], buf.shape[0]), dtype=object)
            for i in range(buf.shape[0]):
                for j in range(buf.shape[1]):
                    # At some point the Hermitian adjoint broke, either a
                    # problem with astra or scipy.
                    buf[i, j] = self.m[j, i].T  # H
            self.mH = Matrix(buf, _adjoint=self)
        else:
            self.mH = _adjoint

    def _matvec(self, x): return self.m.dot(x)

    def _rmatvec(self, y): return self.mH.dot(y)

    def _transpose(self): return self.H

    def _adjoint(self): return self.mH

    def __getitem__(self, *args, **kwargs):
        return self.m.__getitem__(*args, **kwargs)

    def __setitem__(self, *args, **kwargs):
        self.m.__setitem__(*args, **kwargs)


class diff(LinearOperator):
    '''
    Implementation of gradient matrix. Note that to construct second 
    derivatives one has something like:
        diff(order=2) = diff(order=1,bumpup=True) * diff(order=1)
    '''
    def __init__(self, shape, order=1, dtype='f4', bumpup=False):
        if order not in [0, 1, 2]:
            raise ValueError('order = %s not supported' % str(order))

        if order == 0:
            scalar_mat.__init__(self, (np.prod(shape),) * 2, 1, dtype)
        else:
            if order == 1:
                shape2 = (len(shape) * np.prod(shape), np.prod(shape))
                if bumpup:
                    shape2 = tuple(len(shape) * s for s in shape2)
            elif order == 2:
                shape2 = (len(shape) ** 2 * np.prod(shape), np.prod(shape))
            LinearOperator.__init__(self, dtype, shape2)

        self.vol_shape = np.array(shape, dtype='i4')

        self.order = order
        if order == 0:
            self._matvec = self._rmatvec = lambda x: x.copy()
            self._transpose = (lambda: self)
            return
        elif order == 1:
            if bumpup:
                self._matvec = self.__grad2
                self._rmatvec = self.__grad2T
            else:
                self._matvec = self.__grad
                self._rmatvec = self.__gradT
        elif order == 2:
            self._matvec = self.__hess
            self._rmatvec = self.__hessT

        T = LinearOperator(dtype=dtype, shape=(shape2[1], shape2[0]),
                           matvec=self._rmatvec, rmatvec=self._matvec)
        self._transpose = (lambda: T)

    def __grad(self, f):
        # Forward difference with Neumann boundary:
        # df/dx[i] = f[i+1]-f[i]
        ravel = (f.ndim == 1)
        f = f.reshape(self.vol_shape)
        dim = f.ndim
        Df = np.empty(f.shape + (dim,), dtype=f.dtype, order='C')

        mystr = str(dim) + '_' + '0'
        if c_diff['grad'].get(mystr, None) is not None:
            func = c_diff['grad'][mystr]
            if hasattr(func, 'targetdescr'):
                func(f, Df)
            else:
                if dim == 1:
                    tpb = (256,)
                elif dim == 2:
                    tpb = (16, 16)
                else:
                    tpb = (8,) * 3
                blocks = tuple(-(-f.shape[i] // tpb[i]) for i in range(dim))
                func[blocks, tpb](f, Df)
        else:
            for i in range(dim):
                null = [slice(None) for _ in range(dim)]
                x = [slice(None) for _ in range(dim)]
                xp1 = [slice(None) for _ in range(dim)]
                null[i] = -1
                x[i], xp1[i] = slice(-1), slice(1, None)

                Df[tuple(x) + (i,)] = f[tuple(xp1)] - f[tuple(x)]
                Df[tuple(null) + (i,)] = 0

        if ravel:
            return Df.ravel()
        else:
            return Df

    def __gradT(self, Df):
        # Adjoint of forward difference with Neumann boundary
        # is backward difference divergence with Dirichlet boundary.
        # The numerical adjoint assumes Df[...,-1,...] = 0 too.
        ravel = (Df.ndim == 1)
        dim = len(self.vol_shape)
        Df = Df.reshape(*self.vol_shape, dim)

        f = np.empty(Df.shape[:-1], dtype=Df.dtype, order='C')

        mystr = str(dim) + '_' + '0'
        if c_diff['gradT'].get(mystr, None) is not None:
            func = c_diff['gradT'][mystr]
            if hasattr(func, 'targetdescr'):
                func(Df, f)
            else:
                if dim == 1:
                    tpb = (256,)
                elif dim == 2:
                    tpb = (16, 16)
                else:
                    tpb = (8,) * 3
                blocks = tuple(-(-f.shape[i] // tpb[i]) for i in range(dim))
                func[blocks, tpb](Df, f)
        else:
            f *= 0
            # We implement the numerical adjoint
            for i in range(dim):
                x = [slice(None) for _ in range(dim + 1)]
                xm1 = [slice(None) for _ in range(dim + 1)]
                x[i], xm1[i] = slice(1, -1), slice(-2)
                x[-1], xm1[-1] = i, i
                f[tuple(x[:-1])] += Df[tuple(xm1)] - Df[tuple(x)]

                first = [slice(None) for _ in range(dim + 1)]
                first[i], first[-1] = 0, i
                f[tuple(first[:-1])] -= Df[tuple(first)]

                x = [slice(None) for _ in range(dim + 1)]
                xm1 = [slice(None) for _ in range(dim + 1)]
                x[i], xm1[i] = slice(-1, None), slice(-2, -1)
                x[-1], xm1[-1] = i, i
                f[tuple(x[:-1])] += Df[tuple(xm1)]

        if ravel:
            return f.ravel()
        else:
            return f

    def __grad2(self, f):
        # Backward difference with Dirichlet boundary:
        # df/dx[i] = f[i+1]-f[i]
        ravel = (f.ndim == 1)
        dim = len(self.vol_shape)

        if f.size == np.prod(self.vol_shape):
            # compute f -> Df
            f = f.reshape(self.vol_shape)
        else:
            # compute Df -> D^2f
            f = f.reshape(*self.vol_shape, dim)

        Dim = f.ndim
        Df = np.empty(f.shape + (dim,), dtype=f.dtype, order='C')

        mystr = str(dim) + '_' + ('0' if dim == Dim else '1')
        if c_diff['grad2'].get(mystr, None) is not None:
            func = c_diff['grad2'][mystr]
            if hasattr(func, 'targetdescr'):
                func(f, Df)
            else:
                if dim == 1:
                    tpb = (256,)
                elif dim == 2:
                    tpb = (16, 16)
                else:
                    tpb = (8,) * 3
                blocks = tuple(-(-f.shape[i] // tpb[i]) for i in range(dim))
                func[blocks, tpb](f, Df)
        else:
            for i in range(dim):
                x = [slice(None) for _ in range(Dim)]
                xm1 = [slice(None) for _ in range(Dim)]
                x[i], xm1[i] = slice(1, None), slice(-1)
                Df[tuple(x) + (i,)] = f[tuple(x)] - f[tuple(xm1)]

                null = [slice(None) for _ in range(Dim)]
                null[i] = 0
                Df[tuple(null) + (i,)] = f[tuple(null)]

        if ravel:
            return Df.ravel()
        else:
            return Df

    def __grad2T(self, Df):
        # Adjoint of backward difference with Dirichlet boundary
        # is forward difference divergence with Neumann boundary.
        # The numerical adjoint is also Dirichlet boundary.
        ravel = (Df.ndim == 1)
        dim = len(self.vol_shape)

        if Df.size == np.prod(self.vol_shape) * dim:
            # compute Df -> div(Df)
            Df = Df.reshape(*self.vol_shape, dim)
        else:
            # compute D^2f -> div(D^2f)
            Df = Df.reshape(*self.vol_shape, dim, dim)

        Dim = Df.ndim - 1
        f = np.empty(Df.shape[:-1], dtype=Df.dtype, order='C')

        mystr = str(dim) + '_' + ('0' if dim == Dim else '1')
        if c_diff['grad2T'].get(mystr, None) is not None:
            func = c_diff['grad2T'][mystr]
            if hasattr(func, 'targetdescr'):
                func(Df, f)
            else:
                if dim == 1:
                    tpb = (256,)
                elif dim == 2:
                    tpb = (16, 16)
                else:
                    tpb = (8,) * 3
                blocks = tuple(-(-f.shape[i] // tpb[i]) for i in range(dim))
                func[blocks, tpb](Df, f)
        else:
            f *= 0
            # We implement the numeric adjoint
            for i in range(dim):
                x = [slice(None) for _ in range(Dim + 1)]
                xp1 = [slice(None) for _ in range(Dim + 1)]
                x[i], xp1[i] = slice(0, -1), slice(1, None)
                x[-1], xp1[-1] = i, i
                f[tuple(x[:-1])] += Df[tuple(x)] - Df[tuple(xp1)]

                last = [slice(None) for _ in range(Dim + 1)]
                last[i], last[-1] = -1, i
                f[tuple(last[:-1])] += Df[tuple(last)]

        if ravel:
            return f.ravel()
        else:
            return f

    def __hess(self, f):
        # Symmetrised Forward-Backward differences:
        # d^2fdxdy = 1/2(f[i+1,j]+f[i,j+1]+f[i-1,j]+f[i,j-1]
        #                -f[i-1,j+1]-f[i+1,j-1]-2f[i,j])
        # which equates to computing forward differences
        # then backwards then symmetrising.
        ravel = (f.ndim == 1)
        f = f.reshape(self.vol_shape)

        D2f = self.__grad2(self.__grad(f))
        sym(D2f, out=D2f)

        if ravel:
            return D2f.ravel()
        else:
            return D2f

    def __hessT(self, D2f):
        # Adjoint of symmetrisation is symmetrisation
        ravel = (D2f.ndim == 1)
        dim = len(self.vol_shape)
        D2f = D2f.reshape(*self.vol_shape, dim, dim)
        sym(D2f, out=D2f)
        D2f = self.__gradT(self.__grad2T(D2f))

        if ravel:
            return D2f.ravel()
        else:
            return D2f

    def _matvec(self, _):
        # Dummy function for LinearOperator
        return None

    def norm(self, order=2):
        if self.order == 0:
            return 1
        dim = len(self.vol_shape)
        if self.order == 1:
            if order == 1:
                return 2 * dim
            elif order == 2:
                return 2 * dim ** .5
            elif order in ['inf', np.inf]:
                return 2
        elif self.order == 2:
            if order == 1:
                return 4 * dim ** 2
            elif order == 2:
                return 4 * dim
            elif order in ['inf', np.inf]:
                return 4
        else:
            if order == 1:
                return (2 * dim) ** order
            elif order == 2:
                return (4 * dim) ** (order / 2)
            elif order in ['inf', np.inf]:
                return 2 ** order


def getVec(mat, rand=False):
    '''
    Generates a vector in the domain of a Matrix, either random or zeros.
    '''
    if len(mat.shape) in [1, 2]:
        if mat.dtype == object:
            Z = np.empty(mat.shape[-1], dtype=mat.dtype)
            if len(mat.shape) == 1:
                Z = Vector(getVec(m, rand) for m in mat)
            else:
                Z = Vector(getVec(m, rand) for m in mat[0,:])
        else:
            if rand:
                Z = np.random.rand(mat.shape[-1]).astype(mat.dtype)
            else:
                Z = np.zeros(mat.shape[-1], dtype=mat.dtype)
    else:
        raise ValueError('input must either be or dimension 1 or 2')
    return Z


def vecNorm(x):
    ''' Computes the Euclidean norm of a vector'''
    x = x.reshape(-1)
    if x.dtype == object:
        x = np.array([vecNorm(xi) for xi in x])
    return norm(x.reshape(-1))


def vecIP(x, y):
    ''' Computes the inner/dot product of two vectors '''
    x, y = x.reshape(-1), y.reshape(-1)
    if x.dtype == object:
        return sum(vecIP(x[i], y[i]) for i in range(x.size))
    else:
        return x.T.dot(y)


def poweriter(mat, maxiter=300):
    ''' Estimates the norm of a matrix using power iterations '''
    x = getVec(mat, rand=True)
    for _ in range(maxiter):
        x /= vecNorm(x)
        x = mat.T.dot(mat.dot(x))
    return vecNorm(x) ** .5


class FBP(tomo_alg):
    def __init__(self, op=None, filter=None, min_val=None, max_val=None):
        '''
        Performs a filtered back projection on given data. 
        The available filters depends on the choice of backend.

        Parameters:
            op=None: Forward operator to use, dictates which filters are available
            filter=None: choice of filter to use. Defaults and available 
                options given below
            min_val=None: If given, returned array will be thresholded below
            max_val=None: If given, returned array will be thresholded above

        astra supports: default=ram-lak, shepp-logan, cosine, hamming, hann, 
                        tukey, lanczos, triangular, gaussian, barlett-hann, 
                        blackman, nuttall, blackman-harris, blackman-nuttall, 
                        flat-top, kaiser, parzen
        skimage supports: default=ramp, shepp-logan, cosine, hamming, hann 
        '''
        tomo_alg.__init__(self, op)
        self.filter = filter
        self.constraint = [min_val, max_val]
        self.data = None

    def setParams(self, data, op=None, **kwargs):
        self.data = data
        if op is not None:
            self.op = op
        if 'filter' in kwargs:
            self.filter = kwargs['filter']
        if 'min_val' in kwargs:
            self.constraint[0] = kwargs['min_val']
        if 'max_val' in kwargs:
            self.constraint[1] = kwargs['max_val']

    def run(self, data=None, op=None, **kwargs):
        if (data is None) and (self.data is None):
            raise ValueError(
                'Algorithm has not recieved data, please call setParams.')

        self.setParams(data, op, **kwargs)

        dim = data.space_dim
        if self.op is None:
            if hasattr(data, 'getOperator'):
                GPU = kwargs.pop('GPU', True)
                op = data.getOperator(GPU=GPU, **kwargs)
            else:
                raise ValueError(
                    'operator must be provided or data must be an instance of tomo_data')
        else:
            op = self.op

        if op.backend == 'astra':
            if dim == 3:
                small = tomo_data(data[0], data.angles, geom=data.geom)
                kwargs['backend'] = 'astra'
                kwargs['GPU'] = kwargs.get('GPU', op.is_cuda)
                op = small.getOperator(vol_shape=op.vshape[1:], **kwargs)

            from astra import data2d, astra_dict, algorithm

            cfg = astra_dict('FBP' + ('_CUDA' if op.is_cuda else ''))
            cfg['ProjectorId'] = op.proj_id
            cfg['FilterType'] = 'ram-lak' if self.filter is None else self.filter

            if dim == 2:
                rec_id = data2d.create('-vol', op.vg)
                d_id = data2d.create('-sino', op.pg)
                data2d.store(d_id, data.asarray())
                cfg['ProjectionDataId'] = d_id
                cfg['ReconstructionDataId'] = rec_id
                alg_id = algorithm.create(cfg)
                algorithm.run(alg_id, 1)
                recon = data2d.get(rec_id).reshape(op.vshape)
                algorithm.delete(alg_id)
                data2d.delete([rec_id, d_id])
            else:
                def fbp_2d(d, out):
                    rec_id = data2d.create('-vol', op.vg)
                    d_id = data2d.create('-sino', op.pg)
                    data2d.store(d_id, d)
                    cfg['ProjectionDataId'] = d_id
                    cfg['ReconstructionDataId'] = rec_id
                    alg_id = algorithm.create(cfg)
                    algorithm.run(alg_id, 1)
                    out[:,:] = data2d.get(rec_id)
                    algorithm.delete(alg_id)
                    data2d.delete([rec_id, d_id])
                    return out
                recon = _doSliceWise(
                    fbp_2d, data.asarray(), data.shape[:1] + op.vshape, inplace=True)

        elif op.backend == 'skimage':
            from skimage.transform import iradon
            filter = 'ramp' if self.filter is None else self.filter
            theta = data.angles * 180 / np.pi

            def slice_fbp(x):
                return iradon(
                    x.reshape(op.sshape[-2:]).T, theta, op.vshape[-1],
                    filter, op.interpolation, True)

            if dim == 2:
                recon = slice_fbp(data).reshape(op.vshape)
            else:
                recon = _doSliceWise(slice_fbp, data.asarray(), op.vshape)

        else:
            raise ValueError(
                'backend must be astra or skimage for this algorithm')

        if self.constraint[0] is not None:
            recon = np.maximum(self.constraint[0], recon)
        if self.constraint[1] is not None:
            recon = np.minimum(self.constraint[1], recon)
        self.recon = recon
        return recon


class SIRT(tomo_alg):
    def __init__(self, op=None, iterations=100, min_val=None, max_val=None):
        '''
        Performs the SIRT algorithm on given data.

        Parameters:
            op=None: Forward operator to use, dictates geometry and backend
            iterations=100: number of iterations to perform
            min_val=None: If given, returned array will be thresholded below
            max_val=None: If given, returned array will be thresholded above

        '''
        tomo_alg.__init__(self, op)
        self.iterations = iterations
        self.constraint = [min_val, max_val]
        self.data = None

    def setParams(self, data, op=None, **kwargs):
        self.data = data
        if op is not None:
            self.op = op
        if 'iterations' in kwargs:
            self.iterations = kwargs['iterations']
        if 'min_val' in kwargs:
            self.constraint[0] = kwargs['min_val']
        if 'max_val' in kwargs:
            self.constraint[1] = kwargs['max_val']

    def run(self, data=None, op=None, **kwargs):
        if (data is None) and (self.data is None):
            raise ValueError(
                'Algorithm has not recieved data, please call setParams.')

        self.setParams(data, op, **kwargs)

        dim = data.space_dim
        if self.op is None:
            if hasattr(data, 'getOperator'):
                GPU = kwargs.pop('GPU', True)
                op = data.getOperator(GPU=GPU, **kwargs)
            else:
                raise ValueError(
                    'operator must be provided or data must be an instance of tomo_data')
        else:
            op = self.op

        if op.backend == 'astra':
            from astra import data2d, data3d, astra_dict, algorithm
            params = {}
            if self.constraint[0] is not None:
                params['UseMinConstraint'] = True
                params['MinConstraintValue'] = self.constraint[0]
            if self.constraint[1] is not None:
                params['UseMaxConstraint'] = True
                params['MaxConstraintValue'] = self.constraint[1]

            dmod = data2d if dim == 2 else data3d
            rec_id = dmod.create('-vol', op.vg)
            d_id = dmod.create('-sino', op.pg)
            dmod.store(d_id, data.asarray())
            cfg = astra_dict('SIRT' + ('3D' if dim == 3 else '') +
                             ('_CUDA' if op.is_cuda else ''))
            cfg['ProjectionDataId'] = d_id
            cfg['ReconstructionDataId'] = rec_id
            cfg['ProjectorId'] = op.proj_id
            cfg['options'] = params
            alg_id = algorithm.create(cfg)
            algorithm.run(
                alg_id, 100 if self.iterations is None else self.iterations)
            recon = dmod.get(rec_id).reshape(op.vshape)
            algorithm.delete(alg_id)
            dmod.delete([rec_id, d_id])
        else:
            raise ValueError(
                'backend must be astra for this algorithm')

        if self.constraint[0] is not None:
            recon = np.maximum(self.constraint[0], recon)
        if self.constraint[1] is not None:
            recon = np.minimum(self.constraint[1], recon)
        self.recon = recon
        return recon


class SART(tomo_alg):
    def __init__(self, op=None, iterations=100, min_val=None, max_val=None):
        '''
        Performs the SART algorithm on given data.

        Parameters:
            op=None: Forward operator to use, dictates geometry and backend
            iterations=100: number of iterations to perform
            min_val=None: If given, returned array will be thresholded below
            max_val=None: If given, returned array will be thresholded above

        '''
        tomo_alg.__init__(self, op)
        self.iterations = iterations
        self.constraint = [min_val, max_val]
        self.data = None

    def setParams(self, data, op=None, **kwargs):
        self.data = data
        if op is not None:
            self.op = op
        if 'iterations' in kwargs:
            self.iterations = kwargs['iterations']
        if 'min_val' in kwargs:
            self.constraint[0] = kwargs['min_val']
        if 'max_val' in kwargs:
            self.constraint[1] = kwargs['max_val']

    def run(self, data=None, op=None, **kwargs):
        if (data is None) and (self.data is None):
            raise ValueError(
                'Algorithm has not recieved data, please call setParams.')

        self.setParams(data, op, **kwargs)

        dim = data.space_dim
        if self.op is None:
            if hasattr(data, 'getOperator'):
                GPU = kwargs.pop('GPU', True)
                op = data.getOperator(GPU=GPU, **kwargs)
            else:
                raise ValueError(
                    'operator must be provided or data must be an instance of tomo_data')
        else:
            op = self.op

        if op.backend == 'astra':
            if dim == 3:
                small = tomo_data(data[0], data.angles, geom=data.geom)
                kwargs['backend'] = 'astra'
                kwargs['GPU'] = kwargs.get('GPU', op.is_cuda)
                op = small.getOperator(vol_shape=op.vshape[1:], **kwargs)

            from astra import data2d, astra_dict, algorithm

            params = {'ProjectionOrder': kwargs.get(
                'ProjectionOrder', 'random')}
            if self.constraint[0] is not None:
                params['UseMinConstraint'] = True
                params['MinConstraintValue'] = self.constraint[0]
            if self.constraint[1] is not None:
                params['UseMaxConstraint'] = True
                params['MaxConstraintValue'] = self.constraint[1]

            cfg = astra_dict('SART' + ('_CUDA' if op.is_cuda else ''))
            cfg['ProjectorId'] = op.proj_id
            cfg['options'] = params

            if dim == 2:
                rec_id = data2d.create('-vol', op.vg)
                d_id = data2d.create('-sino', op.pg)
                data2d.store(d_id, data.asarray())
                cfg['ProjectionDataId'] = d_id
                cfg['ReconstructionDataId'] = rec_id
                alg_id = algorithm.create(cfg)
                algorithm.run(
                    alg_id, 100 if self.iterations is None else self.iterations)
                recon = data2d.get(rec_id).reshape(op.vshape)
                algorithm.delete(alg_id)
                data2d.delete([rec_id, d_id])
            else:
                def sart_2d(d, out):
                    rec_id = data2d.create('-vol', op.vg)
                    d_id = data2d.create('-sino', op.pg)
                    data2d.store(d_id, d)
                    cfg['ProjectionDataId'] = d_id
                    cfg['ReconstructionDataId'] = rec_id
                    alg_id = algorithm.create(cfg)
                    algorithm.run(
                        alg_id, 100 if self.iterations is None else self.iterations)
                    out[:,:] = data2d.get(rec_id)
                    algorithm.delete(alg_id)
                    data2d.delete([rec_id, d_id])
                    return out
                recon = _doSliceWise(
                    sart_2d, data.asarray(), data.shape[:1] + op.vshape, inplace=True)

        elif op.backend == 'skimage':
            from skimage.transform import iradon_sart
            theta = data.angles * 180 / np.pi
            if all(m is not None for m in self.constraint):
                clip = self.constraint
            else:
                clip = None

            def sart_2d(x):
                return iradon_sart(
                    x.reshape(op.sshape[-2:]).T, theta,
                    clip=clip, relaxation=kwargs.get('relaxation', 0.15))

            if dim == 2:
                recon = sart_2d(data).reshape(op.vshape)
            else:
                recon = _doSliceWise(
                    sart_2d, data.asarray(), op.vshape)
        else:
            raise ValueError(
                'backend must be astra or skimage for this algorithm')

        if self.constraint[0] is not None:
            recon = np.maximum(self.constraint[0], recon)
        if self.constraint[1] is not None:
            recon = np.minimum(self.constraint[1], recon)
        self.recon = recon
        return recon


class smoothFunc():
    '''
    Base class for a smooth function f (bounded derivative), 
    i.e. for all x and y:
        f(y) <= f(x) + f'(x)(y-x) + L/2|y-x|^2
    where L is the Lipshitz constant.
    '''
    def __init__(self, f, grad=None, Lip=None, violation=None):
        self._f, self._grad = f, grad
        if Lip is not None:
            self.Lip = Lip
        else:  # Lip=None means implemented by child class
            assert 'Lip' in dir(self)

        if violation is None:
            self._violation = lambda _: 0
        else:
            self._violation = violation

    def __call__(self, x):
        F = self._f(x.ravel())
        return F[0] if self._grad is None else F

    def grad(self, x):
        if self._grad is None:
            return self._f(x.ravel())[1].ravel()
        else:
            return self._grad(x.ravel()).ravel()

    def violation(self, x): return self._violation(x.ravel())


class L2fidelity(smoothFunc):
    def __init__(self, scale=1, A=None, translation=None, normA=None):
        assert A is not None

        # This implementation is ugly but appears to be more accurate
        # when A and A.T are not exact conjugates.
        if translation is None:
            def f(x):
                x = x.ravel()
                return (self.scale * .5) * x.dot(A.T.dot(A.dot(x)))
            def df(x):
                tmp = self.A.T.dot(self.A.dot(x))
                return tmp if self.scale == 1 else self.scale * tmp

        else:
            translation = translation.reshape(A.shape[0])
            Ad, d2 = A.T.dot(translation), translation.dot(translation)
            def f(x):
                x = x.ravel()
                return (self.scale * .5) * (x.dot(self.A.T.dot(self.A.dot(x)) - 2 * Ad) + d2)
            def df(x):
                tmp = self.A.T.dot(self.A.dot(x)) - Ad
                return tmp if self.scale == 1 else self.scale * tmp

            def f(x): return .5 * ((A.dot(x) - translation) ** 2).sum()
            def df(x):return A.T.dot(A.dot(x) - translation)

        normA = 1.01 * poweriter(A) if normA is None else normA
        smoothFunc.__init__(self, f, df)
        self.scale, self.A, self._normA = scale, A, normA

    @property
    def Lip(self): return self.scale * self._normA ** 2


class proxFunc():
    '''
    Base class for proximal projection which, for a function f and scale t, 
    is defined as:
        prox(f,t,x) = argmin_{y} 1/2|x-y|^2 + t*f(y)
    '''
    def __init__(self, f, prox=None, setprox=None, violation=None, grad=None):
        self._f = f
        self._prox = prox
        if setprox is None:
            def tmp(_): raise NotImplementedError
            self._setprox = tmp
        else:
            self._setprox = setprox

        if violation is None:
            self._violation = lambda _: 0
        else:
            self._violation = violation

        if grad is None:
            def tmp(_): raise NotImplementedError
            self._grad = tmp
        else:
            self._grad = grad

    def __call__(self, x):
        return self._f(x.ravel())

    def setprox(self, t):
        return self._setprox(t)

    def prox(self, x, t=None):
        if t is not None:
            try:
                self.setprox(t)
            except NotImplementedError:
                return self._prox(x.ravel(), t).ravel()
        return self._prox(x.ravel()).ravel()

    def violation(self, x):
        return self._violation(x.ravel())

    def grad(self, x):
        return self._grad(x.ravel()).ravel()


def dualableFunc(f, fstar):
    f.dual = fstar
    fstar.dual = f
    return f


class stackProxFunc(proxFunc):
    '''
    Implementation of the proximal map of a function F which can be expressed 
        F(x,y) = g(x) + h(y)
    '''
    def __init__(self, *args, ignoreDual=False):
        proxFunc.__init__(self, self.__call__)
        self.__funcs = tuple(args)

        if (not ignoreDual) and all(hasattr(f, 'dual') for f in args):
            dualableFunc(self, stackProxFunc(
                *(f.dual for f in args), ignoreDual=True))

    def __call__(self, x):
        return sum(self.__funcs[i](x[i]) for i in range(len(x)))

    def setprox(self, t):
        if np.isscalar(t):
            for f in self.__funcs:
                f.setprox(t)
        else:
            for i in range(len(t)):
                self.__funcs[i].setprox(t[i])

    def prox(self, x):
        return Vector(*tuple(self.__funcs[i].prox(x[i]) for i in range(len(x))))

    def violation(self, x):
        return sum(self.__funcs[i].violation(x[i]) for i in range(len(x)))

    def grad(self, x):
        return Vector(*tuple(self.__funcs[i].grad(x[i]) for i in range(len(x))))


class ZERO(proxFunc):
    ''' f(x) = 0'''
    def __init__(self):
        proxFunc.__init__(self,
                          lambda _: 0,
                          prox=lambda x: x,
                          setprox=lambda _: None,
                          violation=lambda _: 0,
                          grad=lambda x: 0 * x)

        dual = proxFunc(lambda _: 0,
                        prox=lambda x: 0 * x,
                        setprox=lambda _: None,
                        violation=lambda x: abs(x).max())
        dualableFunc(self, dual)


class NONNEG(proxFunc):
    ''' f(x) = 0 if x>=0 else infinity'''
    def __init__(self):
        proxFunc.__init__(self,
                          lambda _: 0,
                          prox=lambda x: np.maximum(0, x),
                          setprox=lambda _: None,
                          violation=lambda x: max(0, -x.min()))

        dual = proxFunc(lambda _: 0,
                        prox=lambda x: np.minimum(0, x),
                        setprox=lambda _: None,
                        violation=lambda x: max(0, x.max()))
        dualableFunc(self, dual)


class L2(proxFunc):
    def __init__(self, scale=1, translation=None):
        '''
        f(x) = scale/2|x-translation|^2
             = s/2|x-d|^2
        df = s(x-d)
        lip = hess = s
        proxf = argmin 1/2|x-X|^2 + st/2|x-d|^2
              = (X+std)/(1+st)

        g(y) = sup <x,y> - f(x)
             = 1/(2s)|y|^2 + <d,y>
        dg = y/s + d
        lip = hess = 1/s
        proxg = argmin 1/2|y-Y|^2 + t/(2s)|y|^2 + t<d,y>
              = (Y-td)/(1+t/s)

        f(X,x) = |X+ix - T-it|^2 = f(X) + f(x)
        g(Y,y) = sup <X,Y>+<x,y> - (X-T)^2 - (x-t)^2
               = g(Y) + g(y)
        proxg(Y,y) = (proxg(Y), proxg(y))
        '''
        if translation is None:
            def f(x): return (scale * .5) * norm(x, 2) ** 2

            def g(y): return (.5 / scale) * norm(y, 2) ** 2

            if scale == 1:
                def df(x): return x

                def dg(y): return y
            else:
                def df(x): return scale * x

                def dg(y): return (1 / scale) * y

            def setproxf(t): self.__proxfparam = 1 / (1 + scale * t)

            def setproxg(t): self.__proxgparam = 1 / (1 + t / scale)

            def proxf(x): return x * self.__proxfparam

            def proxg(y): return y * self.__proxgparam
        else:
            translation = translation.reshape(-1)

            def f(x):
                x = x - translation
                return (scale * .5) * norm(x, 2) ** 2

            def g(y):
                return (.5 / scale) * norm(y, 2) ** 2 + (y.conj() * translation).real.sum()

            if scale == 1:
                def df(x): return x - translation

                def dg(y): return y + translation
            else:
                def df(x): return scale * (x - translation)

                def dg(y): return (1 / scale) * y + translation

            def setproxf(t):
                self.__proxfparam = (
                    1 / (1 + scale * t), translation * (scale * t / (1 + scale * t)))

            def setproxg(t):
                self.__proxgparam = (
                    1 / (1 + t / scale), translation * (t / (1 + t / scale)))

            def proxf(x):
                return x * self.__proxfparam[0] + self.__proxfparam[1]

            def proxg(y):
                return y * self.__proxgparam[0] - self.__proxgparam[1]

        proxFunc.__init__(self, f, prox=proxf, setprox=setproxf, grad=df)
        dual = proxFunc(g, prox=proxg, setprox=setproxg, grad=dg)
        dualableFunc(self, dual)


class L1(proxFunc):
    def __init__(self, scale=1, translation=None):
        '''
        f(x) = scale|x-translation|_1 
             = s|x-d|_1
        df(x) = s*sign(x)
        lip = hess = nan
        prox(x,t) = argmin_X 1/2|X-x|^2 + st|X-d|
                  = d + argmin_X 1/2|X-(x-d)|^2 + st|X|
            X + (st)sign(X) = x-d
            X = x-d -st, 0, x-d +st

        g(y) = sup_x <y,x> - s|x-d|
             = <y,d> if |y|_\infty <= s
        dg(y) = d
        lip = hess = nan
        prox(y,t) = argmin 1/2|Y-y|^2 + t<Y,d> s.t. |Y|< s
                  = proj_{|Y|<s}(y-td)


        '''
        if translation is None:
            def f(x): return scale * norm(x, 1)

            def g(_): return 0

            def setproxf(t): self.__proxfparam = scale * t

            def setproxg(_): self.__proxgparam = 0

            def proxf(x): return shrink1D(x, self.__proxfparam)

            def proxg(y): return project1D(y, scale)
        else:
            translation = translation.reshape(-1)

            def f(x): return scale * norm(x - translation, 1)

            def g(y): return (y.conj() * translation).sum()

            def setproxf(t): self.__proxfparam = scale * t

            def setproxg(t): self.__proxgparam = translation * t

            def proxf(x):
                X = x - translation
                shrink1D(X, self.__proxfparam, out=X)
                X += translation
                return X

            def proxg(y): return project1D(y - self.__proxgparam, scale)

        def violation(x): return max(0, norm(x, np.inf) / scale - 1)

        proxFunc.__init__(self, f, prox=proxf, setprox=setproxf)
        dual = proxFunc(g, prox=proxg,
                        setprox=setproxg, violation=violation)
        dualableFunc(self, dual)


class L1_2(proxFunc):
    def __init__(self, size, scale=1, translation=None):
        '''
        f(x) = scale||x-translation|_2|_1 
             = s||x-d|_2|_1
        df(x) = s*x/|x|_2
        lip = hess = nan
        prox(x,t) = argmin_X 1/2|X-x|^2 + st|X-d|_2
                  = d + argmin_X 1/2|X-(x-d)|^2 + st|X|_2
            X-d + (st)sign(X-d) = x-d
            |X-d| = max(0, |x-d| -st)

        g(y) = sup_x <y,x> - s|x-d|_2
             = <y,d> if |y|_2 <= s
        dg(y) = d
        lip = hess = nan
        prox(y,t) = argmin 1/2|Y-y|^2 + t<Y,d> s.t. |Y|_2< s
                  = proj_{|Y|_2<s}(y-td)

        Vector norm is computed along axis 0
        '''

        self.shape = (size, -1)

        if translation is None:
            def f(x):
                x = norm(x.reshape(self.shape), 2, -1, True)
                return scale * x.sum()

            def g(_): return 0

            def setproxf(t): self.__proxfparam = scale * t

            def setproxg(_): self.__proxgparam = 0

            def proxf(x):
                t = self.__proxfparam
                n = self.__vecnorm(x.reshape(self.shape))
                n = np.maximum(0, 1 - t / n)
                return x * n

            def proxg(y): return project(y.reshape(self.shape), scale)
        else:
            translation = translation.reshape(self.shape)

            def f(x):
                x = x.reshape(self.shape) - translation
                x = norm(x, 2, -1, True)
                return scale * x.sum()

            def g(y): return (y.conj().reshape(self.shape) * translation).sum()

            def setproxf(t): self.__proxfparam = scale * t

            def setproxg(t): self.__proxgparam = translation * t

            def proxf(x):
                t = self.__proxfparam
                x = x.reshape(self.shape) - translation
                n = self.__vecnorm(x)
                n = np.maximum(0, 1 - t / n)
                return x * n + translation

            def proxg(y):
                y = y.reshape(self.shape) - self.__proxgparam
                project(y, scale, out=y)
                return y

        def violation(y):
            y = norm(y.reshape(self.shape), 2, axis=-1)
            return max(0, y.max() / scale - 1)

        proxFunc.__init__(self, f, prox=proxf, setprox=setproxf)
        dual = proxFunc(g, prox=proxg, setprox=setproxg, violation=violation)
        dualableFunc(self, dual)

    def __vecnorm(self, x):
        x = norm(x, 2, -1, True)
        return x + 1e-8


class WaveletL1(L1_2):
    def __init__(self, scale=1, translation=None, wavelet='haar', mode='constant', vol_shape=None, spect_shape=1):
        '''
        f(x) = scale|W*(x-translation)|_1 
             = s|W*(x-d)|_1
        df(x) = s*W^T*sign(W*(x-d))
        lip = hess = nan
        prox(x,t) = argmin_X 1/2|X-x|^2 + st|W*(X-d)|
                  = d + W^T*argmin_X 1/2|X-(x-W*d)|^2 + st|X|
            X + (st)sign(X) = x-W*d
            X = x-W*d -st, 0, x-W*d +st

        g(y) = sup_x <y,x> - s|W*(x-d)|
             = <y,d> if |W*y|_\infty <= s
        dg(y) = d
        lip = hess = nan
        prox(y,t) = argmin 1/2|Y-y|^2 + t<Y,d> s.t. |W*Y|< s
                  = W^T*proj_{|Y|<s}(W*(y-td))


        '''
        self.scale = scale
        vol_shape = (-1,) if vol_shape is None else vol_shape
        self.vol_shape, self.spect_shape = vol_shape, spect_shape
        self._scalar = (spect_shape == 1 or np.prod(spect_shape.shape) == 1)
        self.shape = tuple(vol_shape) + (() if self._scalar else np.prod(spect_shape))
        assert np.prod(self.vol_shape) != -1 or np.prod(self.spect_shape) != -1
        self.size = np.prod(self.shape)

        if wavelet in (None, 'None', 'none'):  # No wavelet, fwrd/bwrd are identity
            self.wavelet = None, mode

            def _fwrd(x, listify=False): return [x.reshape(self.shape)]
            def _bwrd(x): return x[0]

        else:
            if not np.isclose(2 ** np.log2(vol_shape).round(), vol_shape).all():
                raise ValueError(
                    'Volume dimensions must be exact powers of 2 \nfor wavelet implementation to behave as expected.')
            import pywt
            self.wavelet = pywt.Wavelet(wavelet), mode
            assert self.wavelet[0].orthogonal

            def _fwrd(x, listify=False):
                if self._scalar:  # scalar data
                    out = pywt.wavedecn(x.reshape(self.vol_shape), *self.wavelet)
                else:  # spectral data
                    out = pywt.wavedecn(x.reshape(*self.vol_shape, -1), *self.wavelet, axes=range(len(vol_shape)))
                return self.listify(out) if listify else out

            def _bwrd(y):
                if self._scalar:  # non-spectral
                    return pywt.waverecn(y, *self.wavelet)
                else:  # spectral
                    return pywt.waverecn(y, *self.wavelet, axes=range(len(vol_shape)))

        self.fwrd, self.bwrd = _fwrd, _bwrd

        if translation is None:
            def f(x):
                w = self.fwrd(x, listify=True)
                if self._scalar:  # non-spectral data
                    return self.scale * sum(norm(ww.ravel(), 1) for ww in w)
                else:
                    return self.scale * sum(self.__vecnorm(ww.reshape(-1, self.shape[-1])) for ww in w)

            def g(_): return 0

            def setproxf(t): self.__proxfparam = self.scale * t

            def setproxg(_): self.__proxgparam = 0

            def proxf(x):
                w = self.fwrd(x)
                if self._scalar:
                    for ww in self.listify(w):
#                         shrink1D(ww, self.__proxfparam, out=ww)  # inplace edit of w
                        ww[abs(ww) <= self.__proxfparam] = 0
                        ww[ww > self.__proxfparam] -= self.__proxfparam
                        ww[ww < -self.__proxfparam] += self.__proxfparam
                else:
                    t = self.__proxfparam
                    for ww in self.listify(w):
                        n = self.__vecnorm(ww.reshape(-1, self.shape[-1]))
                        ww *= np.maximum(0, 1 - t / n)  # inplace edit of w
                return self.bwrd(w)

            def proxg(y): raise NotImplementedError
        else:
            translation = translation.reshape(self.shape)

            def f(x):
                w = self.fwrd(x.reshape(self.shape) - translation, listify=True)
                if self._scalar:  # non-spectral data
                    return self.scale * sum(norm(ww.ravel(), 1) for ww in w)
                else:
                    return self.scale * sum(self.__vecnorm(ww.reshape(-1, self.shape[-1])) for ww in w)

            def g(y): return (y.conj().reshape(self.shape) * translation).sum()

            def setproxf(t): self.__proxfparam = self.scale * t

            def setproxg(t): self.__proxgparam = translation * t

            def proxf(x):
                w = self.fwrd(x.reshape(self.shape) - translation)
                if self._scalar:
                    for ww in self.listify(w):
                        shrink1D(ww, self.__proxfparam, out=ww)  # inplace edit of w
                else:
                    t = self.__proxfparam
                    for ww in self.listify(w):
                        n = self.__vecnorm(ww.reshape(-1, self.shape[-1]))
                        ww *= np.maximum(0, 1 - t / n)  # inplace edit of w

                return self.bwrd(w) + translation

            def proxg(y): raise NotImplementedError

        def violation(y):
            w = self.listify(y)
            if self._scalar:
                n, inf = abs(w[0]).max(), float('inf')
                for ww in w[1:]:
                    n = max(n, norm(ww.ravel(), inf, axis=1))
            else:
                n = norm(w[0].reshape(-1, self.shape[-1]), 2, axis=1).max()
                for ww in w[1:]:
                    n = max(n, norm(ww.reshape(-1, self.shape[-1]), 2, axis=1).max())
            return max(0, n / self.scale - 1)

        proxFunc.__init__(self, f, prox=proxf, setprox=setproxf)
        dual = proxFunc(g, prox=proxg, setprox=setproxg, violation=violation)
        dualableFunc(self, dual)

    # utility mapping wavelet coefficients to a list of arrays
    def listify(self, w): return sum((list(ww.values()) for ww in w[1:]), [w[0]])


class PDHG(tomo_iter_alg):
    '''
    Implementation of the primal-dual hybrid gradient method as proposed in
        @article{chambolle2011first,
          title={A first-order primal-dual algorithm for convex problems with applications to imaging},
          author={Chambolle, Antonin and Pock, Thomas},
          journal={Journal of Mathematical Imaging and Vision},
          volume={40},
          number={1},
          pages={120--145},
          year={2011},
          publisher={Springer}
        }
    with the adaptivity suggested in:
        @inproceedings{goldstein2015adaptive,
          title={Adaptive primal-dual splitting methods for statistical learning and image processing},
          author={Goldstein, Thomas and Li, Min and Yuan, Xiaoming},
          booktitle={Proceedings of the 28th International Conference on Neural Information Processing Systems-Volume 2},
          pages={2089--2097},
          year={2015}
        }
    '''
    def __init__(self, A, f, g, op=None):
        '''
        Compute saddle points of:
         f(x) + <Ax,y> - g(y)
        '''
        tomo_iter_alg.__init__(self, op)
        self.A = A
        self.f = f
        self.g = g
        self.x, self.y = None, None
        self.s, self.t = None, None
        self.tol = None

    def setParams(self, A=None, f=None, g=None, x=None, y=None, sigma=None,
                  tau=None, balance=None, normA=None, tol=None,
                  steps=None, stepParams=None, **_):
        # Reset problem
        if f is not None:
            self.f = f
        if g is not None:
            self.g = g
        if A is not None:
            self.A = A

        # Set starting point
        if x is not None:
            self.x = x
            self.xm1 = x
        if y is not None:
            self.y = y
            self.ym1 = y

        self.Ax, self.Ay = self.A * self.x, self.A.T * self.y
        self.Axm1, self.Aym1 = self.Ax, self.Ay

        # Set initial step sizes
        # tau/sigma is primal/dual step size respectively
        if (sigma is not None) and (tau is not None):
            sigma, tau = sigma, tau
        elif (balance is None) and (normA is None) and (self.s is not None):
            sigma, tau = self.s, self.t
        else:
            balance = 1 if balance is None else balance
            normA = 1.01 * poweriter(self.A) if normA is None else normA
            sigma, tau = balance / normA, 1 / (balance * normA)
        self.s, self.t = sigma, tau
        self.f.setprox(self.s)
        self.g.setprox(self.t)

        # Set adaptive step criterion
        stepParams = {} if stepParams is None else stepParams
        if steps in [None, 'None', 'none']:
            self._stepsize = lambda _: False
        elif steps[0].lower() == 'a':
            if not (np.isscalar(self.s) and np.isscalar(self.t)):
                raise ValueError(
                    'Step sizes must be scalar for adaptive choice')
            self._stepsize = self._adaptive(**stepParams)
        elif steps[0].lower() == 'b':
            if not (np.isscalar(self.s) and np.isscalar(self.t)):
                raise ValueError(
                    'Step sizes must be scalar for backtracking choice')
            self._stepsize = self._backtrack(**stepParams)
        else:
            raise ValueError('steps must be None, adaptive or backtrack.')

        # Set tolerance
        if tol is not None:
            self.tol = tol
        elif self.tol is None:
            self.tol = 1e-4

    def start(self, A=None, x=None, y=None, **kwargs):
        # Set matrix
        if A is None:
            if self.A is None:
                raise ValueError('Matrix A must be provided')
        else:
            self.A = A

        # Choose starting point:
        if (x is None) and (self.x is None):
            x = getVec(self.A, rand=False)
        if (y is None) and (self.y is None):
            y = getVec(self.A.T, rand=False)

        self.setParams(x=x, y=y, **kwargs)

    def step(self, i, niter):
        for _ in range(niter):
            # Primal step:
            tmp = self.x
            self.x = self.f.prox(self.x - self.t * self.Ay)
            self.xm1, self.Axm1 = tmp, self.Ax
            self.Ax = self.A * self.x

            # Dual step:
            tmp = self.y
            self.y = self.g.prox(self.y + self.s * (2 * self.Ax - self.Axm1))
            self.ym1, self.Aym1 = tmp, self.Ay
            self.Ay = self.A.T * self.y

            # Check step size:
            if self._stepsize(self):
                self.f.setprox(self.t)
                self.g.setprox(self.s)

    def _backtrack(self, beta=0.95, gamma=0.8):
        self.stepParams = {'beta': beta, 'gamma': gamma}

        def stepsize(alg):
            dx = (alg.x - alg.xm1)
            dy = (alg.y - alg.ym1)

            b = (2 * alg.t * alg.s * vecIP(dx, alg.Ay - alg.Aym1).real) / (
                alg.s * vecNorm(dx) ** 2 + alg.t * vecNorm(dy) ** 2)

            if b > gamma:
                b *= beta / gamma
                alg.s /= b
                alg.t /= b
                return True
            else:
                return False

        return stepsize

    def _adaptive(self, alpha=0.5, eta=0.95, delta=1.5, s=1):
        self.stepParams = {'alpha': alpha, 'eta': eta, 'delta': delta, 's': s}
        params = (alpha, eta, s * delta, s / delta)
        a = [alpha / eta]

        def stepsize(alg):
            p = vecNorm((alg.x - alg.xm1) / alg.t - alg.Ay + alg.Aym1)
            d = vecNorm((alg.y - alg.ym1) / alg.s + alg.Ax - alg.Axm1)
            r = p / d

            if r > params[2]:
                a[0] *= params[1]
                self.s *= 1 - a[0]
                self.t /= 1 - a[0]
                return True
            elif r < params[3]:
                a[0] *= params[1]
                self.s /= 1 - a[0]
                self.t *= 1 - a[0]
                return True
            else:
                return False

        return stepsize

    def callback(self, names):
        for n in names:
            if n == 'grad':
                yield self._grad
            elif n == 'gap':
                yield self._gap
            elif n == 'primal':
                yield self._prim
            elif n == 'dual':
                yield self._dual
            elif n == 'violation':
                yield self._violation
            elif n == 'step':
                yield self._step

    @property
    def _grad(self):
        p = vecNorm((self.x - self.xm1) / self.t - self.Ay + self.Aym1)
        d = vecNorm((self.y - self.ym1) / self.s + self.Ax - self.Axm1)
        return p / (1e-8 + vecNorm(self.x)) + d / (1e-8 + vecNorm(self.y))

    @property
    def _prim(self):
        return self.f(self.x) + self.g.dual(self.Ax)

    @property
    def _dual(self):
        return self.f.dual(-self.Ay) + self.g(self.y)

    @property
    def _gap(self):
        '''
        f(x) + f^*(-A^Ty) >= -<x,A^Ty>
        g^*(Ax) + g(y) >= <Ax,y>
        '''
        p = self._prim
        d = self._dual
        z = vecIP(self.Ax, self.y) - vecIP(self.x, self.Ay)
        return (p + d - z) / (abs(p) + abs(d))

    @property
    def _step(self):
        x = vecNorm(self.x - self.xm1) / (1e-8 + vecNorm(self.x))
        y = vecNorm(self.y - self.ym1) / (1e-8 + vecNorm(self.y))
        return x + y

    @property
    def _violation(self):
        return (self.f.violation(self.x) + self.f.dual.violation(-self.Ay)
                +self.g.dual.violation(self.Ax) + self.g.violation(self.y))


class TV(PDHG):
    def __init__(self, shape, order=1,
                 op=None, weight=0.1, pos=True):
        '''
        Minimise the energy:
        1/2|op(u)-data|^2 + weight*|\nabla^{order}u|_1
        where:
            shape: the shape of the output, u
            order=1: integer 0, 1 or 2
            op=None: if provided, the X-ray projector to use
            weight=0.1: positive scalar value, almost definitely less than 1
            pos=True: If true then a non-negativity constraint is applied
        '''
        PDHG.__init__(self, None,
                      stackProxFunc(NONNEG() if pos else ZERO()),
                      None, op=op)
        self.pos = pos
        self.d = diff(shape, order=order)
        self.order = order
        self.weight = weight
        self.data = None

    def setParams(self, data, op=None, x=None, y=None, **kwargs):
        vol_shape = self.d.vol_shape
        if op is not None:
            self.op = op
        elif self.op is None:
            self.op = data.getOperator(vol_shape=vol_shape, **kwargs)
        R = self.op

        normR = [R * (getVec(R) + 1), R.T * (getVec(R.T) + 1)]
        normR += [R.T * normR[0], R * normR[1]]
        normR = [n.max() ** .5 for n in normR]
        normR = min(normR[0] * normR[1], *normR[2:])
        normD = self.d.norm()

        A = Matrix([[(normD / normR) * R], [self.d]])
        normA = normD * 2 ** .5

        dataScale = max(abs(R.T * data.ravel()).max(), 1e-6)
        self.data = data  # pre-scaled
        data = data.__array__() / (dataScale * normD / normR)
        self.dataScale = dataScale * (normD / normR) ** 2

        self.weight = kwargs.get('weight', self.weight)
        gstar = stackProxFunc(
            L2(scale=1, translation=data),
            L1_2(size=np.prod(vol_shape), scale=self.weight)
        )

        # Choose starting point:
        if x is None:
            x = getVec(A, rand=False)
        if y is None:
            y = getVec(A.T, rand=False)

        if 'steps' not in kwargs:
            kwargs['steps'] = 'backtrack'
        if str(kwargs['steps'])[0].lower() == 'b':
            normA /= 3

        self.pos = kwargs.get('pos', self.pos)

        PDHG.setParams(self, A=A,
                       f=stackProxFunc(NONNEG() if self.pos else ZERO()),
                       g=gstar.dual, x=x,
                       y=y, normA=normA, **kwargs)

    def start(self, data=None, x=None, y=None, **kwargs):
        if data is None:
            if self.data is None:
                raise ValueError('data must be provided')
            else:
                data = self.data

        self.setParams(data=data, x=x, y=y, **kwargs)

    def getRecon(self):
        return self.x[0].reshape(self.d.vol_shape) * self.dataScale


class TGV(PDHG):
    def __init__(self, shape,
                 op=None, weight=0.1, pos=True):
        '''
        Minimise the energy:
        1/2|op(u)-data|^2 + weight[0]*|\nabla u - v|_1 + weight[1]*|\nabla v|_1  
        where:
            shape: the shape of the output, u
            order=1: integer 0, 1 or 2
            op=None: if provided, the X-ray projector to use
            weight=0.1: single or pair of positive scalar values, almost definitely 
                less than 1. Note that weight=0.1 is equivalent to weight=(0.1,0.1) 
            pos=True: If true then a non-negativity constraint is applied
        '''
        PDHG.__init__(self, None,
                      stackProxFunc(NONNEG() if pos else ZERO(), ZERO()),
                      None, op=op)
        self.pos = pos
        self.d1 = diff(shape, order=1)
        self.d2 = diff(shape, order=1, bumpup=True)
        self.weight = weight
        self.data = None

    def setParams(self, data, op=None, x=None, y=None, **kwargs):
        vol_shape = self.d1.vol_shape
        if op is not None:
            self.op = op
        elif self.op is None:
            self.op = data.getOperator(vol_shape=vol_shape, **kwargs)
        R = self.op

        normR = [R * (getVec(R) + 1), R.T * (getVec(R.T) + 1)]
        normR += [R.T * normR[0], R * normR[1]]
        normR = [n.max() ** .5 for n in normR]
        normR = min(normR[0] * normR[1], *normR[2:])
        normD = self.d1.norm()
        A = Matrix([[(normD / normR) * R, 0],
                    [self.d1, -1],
                    [0, self.d2]])
        normA = (3 * normD ** 2 + 1) ** .5

        dataScale = max(abs(R.T * data.ravel()).max(), 1e-6)
        self.data = data  # pre-scaled
        data = data.__array__() / (dataScale * normD / normR)
        self.dataScale = dataScale * (normD / normR) ** 2

        weight = kwargs.get('weight', self.weight)
        if np.isscalar(weight):
            weight = (weight, weight)
        self.weight = weight
        gstar = stackProxFunc(
            L2(scale=1, translation=data),
            L1_2(size=np.prod(vol_shape), scale=weight[0]),
            L1_2(size=np.prod(vol_shape), scale=weight[1])
        )

        # Choose starting point:
        if (x is None):
            x = getVec(A, rand=False)
        if (y is None):
            y = getVec(A.T, rand=False)

        if 'steps' not in kwargs:
            kwargs['steps'] = 'backtrack'
        if kwargs['steps'][0].lower() == 'b':
            normA /= 3

        self.pos = kwargs.get('pos', self.pos)

        PDHG.setParams(self, A=A,
                       f=stackProxFunc(
                           NONNEG() if self.pos else ZERO(), ZERO()),
                       g=gstar.dual, x=x,
                       y=y, normA=normA, **kwargs)

    def start(self, data=None, x=None, y=None, **kwargs):
        if data is None:
            if self.data is None:
                raise ValueError('data must be provided')
            else:
                data = self.data

        self.setParams(data=data, x=x, y=y, **kwargs)

    def getRecon(self):
        return self.x[0].reshape(self.d1.vol_shape) * self.dataScale


class FISTA(tomo_iter_alg):
    '''
    The classical implementation of the FISTA method as proposed in
        @article{beck2009fast,
          title={A fast iterative shrinkage-thresholding algorithm for linear inverse problems},
          author={Beck, Amir and Teboulle, Marc},
          journal={SIAM Journal on Imaging Sciences},
          volume={2},
          number={1},
          pages={183--202},
          year={2009},
          publisher={SIAM}
        }
    with more modern variants:
        @article{liang2018improving,
          title={Improving "Fast Iterative Shrinkage-Thresholding Algorithm": Faster, Smarter and Greedier},
          author={Liang, Jingwei and Luo, Tao and Sch{\"o}nlieb, Carola-Bibiane},
          journal={arXiv preprint arXiv:1811.01430},
          year={2018}
        }
    '''
    def __init__(self, f, g, op=None):
        '''
        Compute minimisers of:
         f(x) + g(x)
        '''
        tomo_iter_alg.__init__(self, op)
        self.f, self.g = f, g
        self.x = self.t = self.tol = self.restarting = None

    def setParams(self, f=None, g=None, x=None, tol=None,
                  steps=None, restarting=None, stepParams={}, **_):
        # Reset problem
        if f is not None:
            self.f = f
        if g is not None:
            self.g = g

        # Set starting point
        if x is not None:
            self.x, self.xm1 = x, x

        # tau is initial step size, maybe in the future implement backtracking
        self.t = 1 / self.f.Lip
        self.g.setprox(self.t)

        # Set adaptive step criterion
        if steps[0].lower() == 'c':  # classic Chambolle-Dossal choice
            self._inertia, self._stepsize = self._classic(**stepParams)
        elif steps[0].lower() == 's':  # strong convexity
            self._inertia, self._stepsize = self._strong(**stepParams)
        elif steps[0].lower() == 'g':  # greedy
            self._inertia, self._stepsize = self._greedy(**stepParams)
        else:
            raise ValueError('steps must be classic, strong, or greedy.')

        # Set tolerance
        if tol is not None:
            self.tol = tol
        elif self.tol is None:
            self.tol = 1e-8

        if restarting is not None:
            self.restarting = restarting
        elif self.restarting is None:
            self.restarting = True

    def start(self, x=None, **kwargs):
        # Choose starting point:
        if (x is None) and (self.x is None):
            x = getVec(self.op, rand=False)

        self.setParams(x=x, **kwargs)
        self.counter = 0

    def step(self, i, niter):
        for _ in range(niter):
            # Inertial step:
            a = self._inertia(self)
            if a == 0:
                y = self.x
            elif a == 1:
                y = 2 * self.x - self.xm1
            else:
                y = (1 + a) * self.x - a * self.xm1
            self.x, self.xm1 = self.g.prox(y - self.t * self.f.grad(y)), self.x
#             self.x, self.xm1 = y - self.t * self.f.grad(y), self.x

#             Check for restarting:
            if self.restarting and vecIP(y - self.x, self.x - self.xm1) >= 0:
                self.x = self.g.prox(self.xm1 - self.t * self.f.grad(self.xm1))
                self.counter = 0
                continue

            # Check step size:
            if self._stepsize(self):
                self.g.setprox(self.t)

            self.counter += 1  # increment counter

    def _classic(self, d=20):
        self.stepParams = {'d':d}

        def inertia(alg): return alg.counter / (alg.counter + alg.stepParams['d'])
        def stepsize(alg): False

        return inertia, stepsize

    def _strong(self, convexity=1):  # coefficient of strong convexity
        self.stepParams = {'convexity':convexity}

        def inertia(alg):
            return max(0, (1 - (alg.t * alg.stepParams['convexity']) ** .5) /
                                (1 + (alg.t * alg.stepParams['convexity']) ** .5))
        def stepsize(alg): False

        return inertia, stepsize

    def _greedy(self, xi=0.96, eta=1.3, s=1):
        self.stepParams = {'xi': xi, 'eta':eta, 's': s, 't0': self.t}
        self.t = eta * self.t  # start with a larger t value
        self.g.setprox(self.t)

        self.__greedynorm = 0

        def inertia(alg): return 1
        def stepsize(alg):
            norm = vecNorm(alg.x - alg.xm1)
            if self.__greedynorm == 0:
                self.__greedynorm = self.stepParams['s'] * norm
                return False
            elif norm <= self.__greedynorm:
                return False
            else:
                self.t = max(self.t * self.stepParams['xi'], self.stepParams['t0'])
                return True

        return inertia, stepsize

    def callback(self, names):
        for n in names:
            if n == 'primal':
                yield self._prim
            elif n == 'violation':
                yield self._violation
            elif n == 'step':
                yield self._step
            elif n == 'gap':
                yield self._gap

    @property
    def _prim(self): return self.f(self.x) + self.g(self.x)

    @property
    def _step(self): return vecNorm(self.x - self.xm1) / (1e-8 + vecNorm(self.x))

    @property
    def _violation(self): return self.f.violation(self.x) + self.g.violation(self.x)

    def getRecon(self): return self.x


class Wavelet(FISTA):
    def __init__(self, shape, wavelet='haar', mode='constant',
                 op=None, weight=0.1):
        '''
        Minimise the energy:
        1/2|op(u)-data|^2 + weight*|W*u|_1
        where:
            shape: the shape of the output, u
            wavelet='haar': name of wavelet type (e.g. haar, db1, db2,...), or None for W=identity
            op=None: if provided, the X-ray projector to use
            weight=0.1: float between 0 and 1
        '''
        FISTA.__init__(self, None, None, op=op)

        self.weight, self.vol_shape = weight, shape
        self.wavelet, self.mode = wavelet, mode
        self.data = None

    def setParams(self, data, op=None, x=None, **kwargs):
        if op is not None:
            self.op = op
        elif self.op is None:
            self.op = data.getOperator(vol_shape=self.vol_shape, **kwargs)

        R = self.op
        normR = [R.dot((getVec(R) + 1)), R.T.dot(getVec(R.T) + 1)]
        normR += [R.T.dot(normR[0]), R.dot(normR[1])]
        normR = [n.max() ** .5 for n in normR]
        normR = min(normR[0] * normR[1], *normR[2:])

        self.weight = kwargs.get('weight', self.weight)
        g = WaveletL1(scale=self.weight, wavelet=self.wavelet, mode=self.mode, vol_shape=self.vol_shape)

        self.dataScale = max(max(abs(w).max() for w in g.fwrd(R.T.dot(data.ravel()), listify=True)), 1e-6)
#         print(self.dataScale / R.T.dot(data.ravel()).max())
#         exit()
#         self.dataScale = max((R.T * data.ravel()).max(), 1e-6)

        self.data = data  # pre-scaled
        f = L2fidelity(scale=1, A=R, normA=normR, translation=data.__array__() / self.dataScale)

        # Choose starting point:
        if x is None:
            x = getVec(R, rand=False)

        if 'steps' not in kwargs:
            kwargs['steps'] = 'greedy'

        FISTA.setParams(self, f=f, g=g, x=x, **kwargs)

    def start(self, data=None, x=None, **kwargs):
        if data is None:
            if self.data is None:
                raise ValueError('data must be provided')
            else:
                data = self.data

        self.setParams(data=data, x=x, **kwargs)
        self.counter = 0

    @property
    def _gap(self):
        dF = self.g.fwrd(self.f.grad(self.x), listify=True)
        X = self.g.fwrd(self.x, listify=True)
        return max(L1_gap(df.ravel(), x.ravel(), self.g.scale) for df, x in zip(dF, X))

    def getRecon(self): return self.x.reshape(self.vol_shape) * self.dataScale


cleanup_astra()
if __name__ == '__main__':
    from skimage.data import moon, binary_blobs
    from matplotlib import pyplot as plt

    data2d = moon().astype('f4')
    data3d = binary_blobs(length=128, n_dim=3,
                          volume_fraction=0.01, seed=1).astype('f4')
    angles = np.linspace(0, np.pi, 180)
    # dim=0 is tilt axis
    # dim=1 is y/z
    # At angle=0 we see [x,z]
    # Sinogram = [x,theta,y/z]

    # Read in 2d data with angles
    tomo_data2d = tomo_data(np.zeros((len(angles),) + data2d.shape[:-1]),
                            angles, degrees=False, geom='parallel')
    # Read in 3d data with angles
    tomo_data3d = tomo_data(np.zeros((data3d.shape[0], len(angles), data3d.shape[1])),
                            angles, degrees=False, geom='parallel',
                            tilt_axis=0, stack_dim=1)

#     Xray2 = tomo_data2d.getOperator(
#         vol_shape=data2d.shape, backend='skimage', interpolation='linear')
    Xray2 = tomo_data2d.getOperator(
        vol_shape=data2d.shape, backend='astra', GPU=True)
    Xray3 = tomo_data3d.getOperator(
        vol_shape=data3d.shape, backend='astra', GPU=True)

    vol2d, det2d = Xray2.vshape, Xray2.sshape
    vol3d, det3d = Xray3.vshape, Xray3.sshape

    # Forward projection
    sino2d = (Xray2 * data2d.ravel()).reshape(det2d)
    sino3d = (Xray3 * data3d.ravel()).reshape(det3d)

    # Now pretend this is raw data
    sino2d = tomo_data(sino2d, angles)
    sino3d = tomo_data(sino3d, angles)

    # Backward projection
    bp2d = (Xray2.T * sino2d.ravel()).reshape(vol2d)
    bp3d = (Xray3.T * sino3d.ravel()).reshape(vol3d)

#     # Filtered Back-projection
#     fbp = FBP(filter='ramp')
#     rec2d = fbp.run(data=sino2d, op=Xray2, min_val=0)
#     rec3d = fbp.run(data=sino3d, op=Xray3,
#                     filter='Ram-Lak', min_val=0, max_val=1)

#     # SIRT
#     sirt = SIRT(iterations=100)
#     rec2d = sirt.run(data=sino2d, backend='astra', min_val=0)
#     rec3d = sirt.run(data=sino3d, op=Xray3, min_val=0, max_val=1)

#     # SART
#     sart = SART(iterations=100)
#     rec2d = sart.run(data=sino2d, op=Xray2, min_val=0, max_val=data2d.max())
#     rec3d = sart.run(data=sino3d, op=Xray3, min_val=0)

    # TV reconstruction
    rec2d = TV(vol2d).run(data=sino2d, balance=1, maxiter=100, weight=0.01,
                          callback=('gap', 'primal', 'dual', 'violation', 'step'))[0]
    rec3d = TV(vol3d).run(data=sino3d, balance=1, maxiter=100, weight=0.01,
                          callback=('gap', 'primal', 'dual', 'violation', 'step'))[0]

#     # TV2 reconstruction
#     rec2d = TV(vol2d, order=2).run(data=sino2d, op=None, backend='astra', balance=1, weight=0.01,
#                                    callback=('gap', 'primal', 'step'))[0]
#     rec3d = TV(vol3d, order=2).run(data=sino3d, op=Xray3, balance=1, weight=0.01,
#                                    callback=('gap', 'primal', 'step'))[0]

#     # TGV reconstruction
#     rec2d = TGV(vol2d).run(data=sino2d, balance=1, maxiter=100, weight=0.01,
#                            callback=('gap', 'primal', 'dual', 'violation', 'step'))[0]
#     rec3d = TGV(vol3d).run(data=sino3d, balance=1, maxiter=100, weight=0.01,
# callback=('gap', 'primal', 'dual', 'violation', 'step'))[0]

    # Haar wavelet reconstruction
#     rec2d = Wavelet(vol2d, wavelet='db4').run(data=sino2d, maxiter=100, weight=1e-4,
#                           callback=('primal', 'gap', 'step'))[0]
#     rec3d = Wavelet(vol3d).run(data=sino3d, maxiter=100, weight=1e-4,
#                           callback=('primal', 'gap', 'step'))[0]

    plt.figure('2D Data')
    plt.subplot(221); plt.imshow(data2d); plt.title('data')
    plt.subplot(222); plt.imshow(sino2d, aspect='auto'); plt.title('Sinogram')
    plt.subplot(223); plt.imshow(bp2d); plt.title('Back-projection')
    plt.subplot(224); plt.imshow(rec2d, vmin=0, vmax=data2d.max()); plt.title('Reconstruction')

    half = int(vol3d[0] / 2)
    plt.figure('3D Data')
    plt.subplot(221); plt.imshow(data3d[half]); plt.title('Slice of data')
    plt.subplot(222); plt.imshow(sino3d[:, 90,:]); plt.title('Single projection')
    plt.subplot(223); plt.imshow(bp3d[half]); plt.title('2D slice of back projection')
    plt.subplot(224); plt.imshow(rec3d[half], vmin=0, vmax=data3d[half].max()); plt.title('2D slice of Reconstruction')

    plt.show()
