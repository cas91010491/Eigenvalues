"""
Mesh transformations. Provides various transformation classes.

All transformations are derived from GenericTransform, and should support
2D and 3D without specifying the dimension a priori. They can also support
skipped dimension in 3D.

Method  update(params) sets internal transformation parameters.
See GenericDomain class docstring for further details about params handling.

Method apply(coords) takes mesh coordinate array and returns transformed array.
Calling an object is also supported.

Docstrings of the transformations are used in GUI as name and help messages.
The first line is the name. The rest constitutes the help, possibly with
long lines as paragraphs of text. These normally have backslash as last
character, but this interferes with indentation removal. Instead, docstrings
are raw strings and trailing backslash is removed with the newline character
in postprocessing (build_domain_dict function).
"""
from __future__ import division

import numpy as np
import numpy.linalg as nl
from paramchecker import evaluate, ParamChecker, clip, flatten
from dolfin.cpp import Mesh
from cPickle import dumps


class GenericTransform(ParamChecker):

    """
    Abstract superclass of all mesh coordinate transformations.

    Use update(params, dim) to set parameters and dimension.
    Use apply(coord) to apply transformation to coords.

    Parameters depend on the specific transformation implementation.
    """

    def __init__(self):
        """Call superclass init and set default parameters."""
        super(GenericTransform, self).__init__()
        # initialize to default values
        self.update("")

    def __repr__(self):
        """Return string representation suitable for object reconstruction."""
        return self.name+' with parameters: '+self.params

    def update(self, params):
        """
        Process input parameters.

        self.eval(params) should be used to parse params,
            set self.params and self.values

        self.params will be
            the same as the argument if the argument was useful,
            equal self.default if the argument was '',
            otherwise the previous value of self.params
        self.values contain evaluated params

        """
        raise NotImplementedError("Implement this!")

    def apply(self, coords):
        """
        Apply transformation to coords (using numpy vectorized functions!).

        coords should be a numpy array:
            coords[:,0] - x coordinates
            coords[:,1] - y coordinates
            coords[:,2] - z coordinates
        """
        raise NotImplementedError("Implement this!")


class Linear(GenericTransform):

    """
    Linear tranformation.

    Linear transformation (matrix) to be applied to the mesh coordinates.

    Parameters:
        all entries (as matrix or flat list) OR
        the diagonal OR
        scalar.
    Default: 2 (scale the domain)
    """

    default = "2"
    matrix = None

    @staticmethod
    def format(x):
        """Up to 4, or 9 floats."""
        x = flatten(x)
        if len(x) < 9:
            x = x[:4]
        else:
            x = x[:9]
        return clip(x, float)

    def update(self, params):
        """Form a diagonal or full matrix."""
        entries = self.eval(params)
        if len(entries) == 1:
            self.matrix = np.identity(3) * entries[0]
        elif len(entries) <= 3:
            diag = [1, 1, 1]
            diag[:len(entries)] = entries
            self.matrix = np.diag(diag)
        else:
            # 4 or 9 entries
            dim = int(np.sqrt(len(entries)))
            self.matrix = np.identity(3)
            self.matrix[:dim, :dim] = np.reshape(entries, (dim, dim))

        # we do not want a degenerate matrix
        cond = nl.cond(self.matrix)
        if cond < 1E-10 or cond > 1E+10:
            self.matrix = np.identity(3)
        self.matrix = np.transpose(self.matrix)

    def apply(self, coords):
        """Multiply transformation and coordinate matrices."""
        dim = len(coords[0])
        return np.dot(coords, self.matrix[:dim, :dim])


class Shift(GenericTransform):

    """
    Shift.

    Shift the domain by a vector.

    Parameters: list of coordinates, or just one number to shift in x-direction.
    """

    default = "1"
    full = [1, 0, 0]

    @staticmethod
    def format(x):
        """Three floats."""
        return clip(x[:3], float)

    def update(self, params):
        """Fill first few coordinates."""
        self.vector = np.array(self.pad(self.eval(params)))

    def apply(self, coords):
        """Shift using numpy broadcasting."""
        dim = len(coords[0])
        return coords + self.vector[:dim]


class Norm(GenericTransform):

    """
    Lp to Lq.

    Transform unit Lp ball to unit Lq ball.

    Default values: p=1, q=2 (both optional).

    Gives circle / sphere:
        if (L1 to L2) is applied to a square(vertices on axes) / octahedron OR
        if (Linf to L2) is applied to a square(sides parallel to axes)
    Apply (Linf to L2) to a right/isosceles triangle to get a sector.

    Optional third parameter is an index of a dimension to skip in the norm.
        (1,2,3 for x,y,z)
    """

    # TODO: other examples with skipped dimension
    default = "1 2"
    full = (1, 2, 0)

    @staticmethod
    def format(x):
        """Two positive floats. Then integer from 1,2,3."""
        return clip(np.fabs(x[:2]), float, 0.01) + clip(x[2:3], int, 1, 3)

    def update(self, params):
        """Copy parameter values."""
        self.p, self.q, self.ind = self.pad(self.eval(params))

    def apply(self, coords):
        """Apply norm transformation to all or just two dimensions."""
        dim = len(coords[0])
        skip = range(dim)
        if self.ind > 0 and dim == 3:
            # skipping a dimension
            del skip[self.ind-1]
        temp = (np.transpose(coords[:, skip]) *
                (self.pnorms(coords[:, skip], self.p) /
                 self.pnorms(coords[:, skip], self.q)))
        coords[:, skip] = np.nan_to_num(np.transpose(temp))
        return coords

    @staticmethod
    def pnorms(arr, p):
        """Compute Lp norm."""
        if p == np.inf:
            return np.max(np.absolute(arr), axis=1)
        else:
            return np.power(np.sum(np.power(np.absolute(arr), p), axis=1), 1./p)


class Power(GenericTransform):

    """
    Power of distance.

    Apply Nth power to the distances to the chosen origin.

    Transformation: p -> p|p|^(N-1).

    Parameters:
        power N != 0 (can be negative, but origin should be outside)
        the origin (X, Y, Z)

    Default values: N = 0.5, (X, Y, 0) = (0, 0, 0).

    Example: N=-1 gives a circle/ball inversion (origin must be ouside!)
    """

    default = "0.5"
    full = (0.5, 0, 0, 0)

    @staticmethod
    def format(x):
        """Four floats."""
        return clip(x[:4], float)

    def update(self, params):
        """Set power and origin."""
        params = self.pad(self.eval(params))
        self.N = params[0]
        self.origin = params[1:4]

    def apply(self, coords):
        """Shift to origin, then scale distances and shift back."""
        dim = len(coords[0])
        if self.N in [0, 1]:
            return coords
        temp = coords-self.origin[:dim]
        temp = np.transpose(temp) * self.ppowers(temp, self.N-1)
        return np.nan_to_num(np.transpose(temp)+self.origin[:dim])

    @staticmethod
    def ppowers(arr, p):
        """Compute L2 norm to power p."""
        return np.power(np.sum(arr**2, axis=1), p/2.0)


class Starlike(GenericTransform):

    r"""
    Starlike transformation.

    Apply a starlike transformation
        (r, theta, eta) ->
            (r F(r,theta,eta), theta, eta)

    Note: eta = 0 in 2D!
    Parameters (F, T = 3):
        F - function depending on r, theta and eta
        T - optional transformation type in 3D
            T = -1, -2, -3 (x, y, z respectively):
                direction for the north pole in spherical coordinates
            T = 1, 2, 3:
                cylindrical coordinates with axis along a chosen direction
                (eta is the height in this case).

    Input supports any number of constant substitutions with the syntax
        c=?, d=?: function using c, d

    The default function turns a disk into a hippopede, where eps is \
    the eccentricity of the ellipse which inverts to the hippopede.

    Implementation: lambda is prepended to parameters so that it can evaluate \
    to a python function.
    """

    default = "eps=sqrt(2)/2: sqrt(1-eps^2*sin(theta)^2)"
    def_pref = 'lambda r, theta, eta, '
    noparam = 'lambda r, theta, eta: '

    @staticmethod
    def format(x):
        """Free function input. Then one integer."""
        return [x[0]] + clip(x[1:2], int)

    def update(self, params):
        """Create a function with or without parameters."""
        self.full = [evaluate(self.def_pref + self.default)[0], 3]
        formula, ind = self.pad(self.eval(params, self.noparam))
        if self.params != params:
            formula, ind = self.pad(self.eval(params, self.def_pref))
        self.full = None
        self.values = None
        ind -= 1
        return formula, ind

    def apply(self, coords):
        """Compute polar/spherical coordinate arrays and apply function."""
        formula, ind = self.update(self.params)
        dim = len(coords[0])
        cylindrical = ind < 0
        skip = range(dim)
        if dim == 3:
            # north pole direction for spherical/cylindrical coordinates
            north = clip(abs(ind), int, 1, 3)-1
            del skip[north]
        # r and theta dimensions as complex
        rtheta = coords[:, skip[0]] + coords[:, skip[1]] * 1j
        theta = np.angle(rtheta)
        r = np.absolute(rtheta)
        eta = 0
        if dim == 3:
            h = coords[:, north]
            eta = np.arctan2(r, h)
            if cylindrical:
                eta = h
            else:
                r = np.sqrt(r**2+h**2)
        if cylindrical:
            coords[:, skip] = np.nan_to_num(np.transpose(
                np.transpose(coords[:, skip]) * formula(r, theta, eta)))
        else:
            coords = np.nan_to_num(np.transpose(np.transpose(coords) *
                                                formula(r, theta, eta)))
        return coords


class Complex(GenericTransform):

    r"""
    Complex transformation.

    Apply a complex transformation (z,h) -> (F(z,h), h) (h = 0 in 2D).

    Parameters (F, D = 3):
        function F(z, h) with complex z and real h,
        optional h - dimension parameter (D = 1, 2, 3 for x, y, z).

    Function can use: conj(z), re(z), im(z), abs(z), arg(z).

    Input supports any number of constant substitutions with the syntax
        c=?, d=?: function using c, d

    The default function turns a disk into an epicycloid with a given \
    number of cusps (changing epsilon leads to generalized limacons).

    Implementation: lambda is prepended to parameters so that it can evaluate \
    to a python function.
    """

    default = "n=5, eps=1: z+eps/(n+1)*z^(n+1)"
    def_pref = 'lambda z, h, '
    noparam = 'lambda z, h: '

    @staticmethod
    def format(x):
        """Free function input. Then an integer."""
        return [x[0]] + clip(x[1:2], int, 1, 3)

    def update(self, params):
        """Create a function with or without parameters."""
        self.full = [evaluate(self.def_pref + self.default)[0], 3]
        formula, ind = self.pad(self.eval(params, self.noparam))
        if self.params != params:
            formula, ind = self.pad(self.eval(params, self.def_pref))
        ind -= 1
        self.full = None
        self.values = None
        return formula, ind

    def apply(self, coords):
        """Write two dimensions as complex numbers and apply function."""
        formula, ind = self.update(self.params)
        ind -= 1
        dim = len(coords[0])
        skip = range(dim)
        if dim == 3:
            # skip h coordinate index
            del skip[ind]

        # two dimensions as complex
        z = coords[:, skip[0]] + coords[:, skip[1]] * 1j
        h = 0
        if dim == 3:
            h = coords[:, ind]
        temp = formula(z, h)
        coords[:, skip[0]] = np.nan_to_num(np.real(temp))
        coords[:, skip[1]] = np.nan_to_num(np.imag(temp))
        return coords


def build_transform_dict():
    """
    Return dictionary of all transformations.

    Keys:   transformation names
    Values: transformation classes
    """
    from inspect import getdoc
    import re
    transform_dict = {}
    for subcls in GenericTransform.__subclasses__():
        lines = re.sub(r'\\\n', '', getdoc(subcls)).split("\n")
        subcls.name = lines[0][:-1]
        subcls.help = "\n".join(lines[2:])
        try:
            dumps(subcls())
            transform_dict[subcls.name] = subcls
        except:
            print subcls.name + ' not picklable'
    return transform_dict


def transform_mesh(mesh, lst):
    """
    Apply list of transforms to a mesh.

    Copy the mesh first to save the original mesh object.
    """
    mesh = Mesh(mesh)
    for trans in lst:
        mesh.coordinates()[:] = trans.apply(mesh.coordinates())
    return mesh
