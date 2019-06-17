import numpy as np
import nengo
import nengo_spa as spa
from PIL import Image
import base64
import matplotlib.cm as cm
import sys

# Python version specific imports
if sys.version_info[0] == 3:
    from io import BytesIO
else:
    import cStringIO


def image_svg(arr):
    """
    Given an array, return an svg image
    """
    if sys.version_info[0] == 3:
        # Python 3

        png = Image.fromarray(arr)
        buffer = BytesIO()
        png.save(buffer, format="PNG")
        buffer.seek(0)
        img_str = base64.b64encode(buffer.read()).decode()
        return '''
            <svg width="100%%" height="100%%" viewbox="0 0 100 100">
            <image width="100%%" height="100%%"
                   xlink:href="data:image/png;base64,%s"
                   style="image-rendering: pixelated;"/>
            </svg>''' % (''.join(img_str))

    else:
        # Python 2

        png = Image.fromarray(arr)
        buffer = cStringIO.StringIO()
        png.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue())
        return '''
            <svg width="100%%" height="100%%" viewbox="0 0 100 100">
            <image width="100%%" height="100%%"
                   xlink:href="data:image/png;base64,%s"
                   style="image-rendering: pixelated;">
            </svg>''' % (''.join(img_str))


class SpatialHeatmap(object):

    def __init__(self, heatmap_vectors, xs, ys, cmap='plasma', vmin=None, vmax=None):

        self.heatmap_vectors = heatmap_vectors
        self.xs = xs
        self.ys = ys

        self.cmap = cmap
        self.vmin = vmin
        self.vmax = vmax

        self.cm = cm.get_cmap(cmap)

        self._nengo_html_ = ""

    def __call__(self, t, x):

        if len(self.heatmap_vectors.shape) == 4:
            # Index for selecting which heatmap vectors to use
            # Used for deconvolving with multiple items in the same plot
            index = int(x[0])
            index = np.clip(index, 0, self.heatmap_vectors.shape[0] - 1)

            hmv = self.heatmap_vectors[index, ...]
            vector = x[1:]
        else:
            hmv = self.heatmap_vectors
            vector = x

        # Generate heatmap values
        vs = np.tensordot(vector, hmv, axes=([0], [2]))

        if self.vmin is None:
            min_val = np.min(vs)
        else:
            min_val = self.vmin

        if self.vmax is None:
            max_val = np.max(vs)
        else:
            max_val = self.vmax

        vs = np.clip(vs, a_min=min_val, a_max=max_val).T

        values = (self.cm(vs)*255).astype(np.uint8)

        self._nengo_html_ = image_svg(values)


def power(s, e):
    x = np.fft.ifft(np.fft.fft(s.v) ** e).real
    return spa.SemanticPointer(data=x)


def encode_point(x, y, x_axis_sp, y_axis_sp):

    return power(x_axis_sp, x) * power(y_axis_sp, y)


def make_good_unitary(dim, eps=1e-3, rng=np.random):
    # created by arvoelke
    a = rng.rand((dim - 1) // 2)
    sign = rng.choice((-1, +1), len(a))
    phi = sign * np.pi * (eps + a * (1 - 2 * eps))
    assert np.all(np.abs(phi) >= np.pi * eps)
    assert np.all(np.abs(phi) <= np.pi * (1 - eps))

    fv = np.zeros(dim, dtype='complex64')
    fv[0] = 1
    fv[1:(dim + 1) // 2] = np.cos(phi) + 1j * np.sin(phi)
    fv[-1:dim // 2:-1] = np.conj(fv[1:(dim + 1) // 2])
    if dim % 2 == 0:
        fv[dim // 2] = 1

    assert np.allclose(np.abs(fv), 1)
    v = np.fft.ifft(fv)
    # assert np.allclose(v.imag, 0, atol=1e-5)
    v = v.real
    assert np.allclose(np.fft.fft(v), fv)
    assert np.allclose(np.linalg.norm(v), 1)
    return spa.SemanticPointer(v)


def get_heatmap_vectors(xs, ys, x_axis_sp, y_axis_sp):
    """
    Precompute spatial semantic pointers for every location in the linspace
    Used to quickly compute heat maps by a simple vectorized dot product (matrix multiplication)
    """
    if x_axis_sp.__class__.__name__ == 'SemanticPointer':
        dim = len(x_axis_sp.v)
    else:
        dim = len(x_axis_sp)
        x_axis_sp = spa.SemanticPointer(data=x_axis_sp)
        y_axis_sp = spa.SemanticPointer(data=y_axis_sp)

    vectors = np.zeros((len(xs), len(ys), dim))

    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            p = encode_point(
                x=x, y=y, x_axis_sp=x_axis_sp, y_axis_sp=y_axis_sp,
            )
            vectors[i, j, :] = p.v

    return vectors
