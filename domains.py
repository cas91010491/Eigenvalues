"""
Domain definitions. Implements various domain classes.

Meshes are generated using get(params) method of subclasses of GenericDomain.

Calling an object with params returns the mesh.
See GenericDomain.__call__ docstring for details.

Argument params is a string containing lists/numbers. The format does not need
to be Python's syntax. E.g. space separated numbers will be turned into a list.
Mathematical functions can also be used on numbers or elementwise on lists.

The string is evaluated and formatted to suit each subclass using
format method, full and default attributes of each class.

This rather strange setup was dictated by params coming from user
interaction with text inputs in GUI.

Calling an object makes it easy to use the object in multiprocessing as a
target function, with all parameters already stored in the object.
Note: Methods are not picklable, but object() is OK.

Docstrings of the domains are used in GUI as help messages and domain names.
The first line is the name. The rest constitutes the help, possibly with
long lines as paragraphs of text. These normally have backslash as last
character, but this interferes with indentation removal. Instead, docstrings
are raw strings and trailing backslash is removed with the newline character
in postprocessing (build_domain_dict function).
"""
from __future__ import division

# pylama:ignore=E731
from math import pi, sin, cos, sqrt, tan
import numpy as np
from dolfin import Mesh, MeshEditor, DynamicMeshEditor, RectangleMesh, \
    UnitSquareMesh, Point, UnitCubeMesh, BoxMesh, compile_extension_module
from mshr import Circle, UnitSphereMesh, generate_mesh, Polygon
from paramchecker import evaluate, ParamChecker, clip, flatten

np.seterr(divide='ignore', invalid='ignore')
np.set_printoptions(precision=5, suppress=True)

# compile c++ code
with open("cpp/buildmesh.cpp", "r") as f:
    code = f.read()

builder = compile_extension_module(code=code)  # , source_directory="cpp"
# include_dirs=["."])


def CircleMesh(p, r, s):
    """ mshr has no CircleMesh. """
    return generate_mesh(Circle(p, r, max(int(4 * pi / s), 4)), int(2 / s))


class GenericDomain(ParamChecker):

    """
    Abstract superclass of all domains used for eigenvalue calculations.

    Parameters are set using eval(params) method.
    Mesh can be obtained from get() method, or by calling an object.

    Parameters depend on the specific domain implementation.
    """

    diag = ("left", "right", "crossed")
    dim = "2D"

    def __call__(self, monitor=None):
        """Return the mesh. Monitor could be used to pass progress. """
        self.monitor = monitor
        return self.get()

    def get(self):
        """
        Should return mesh.

        self.eval(params) should be used to parse params,
            set self.params and self.values

        self.params will be
            the same as the argument if the argument was useful,
            equal self.default if the argument was '',
            otherwise the previous value of self.params
        self.values contains evaluated params

        """
        raise NotImplementedError("Implement this!")

    def with_params(self):
        """Check if domain has any parameters."""
        return self.full is not None
#
#  2D domains
#


class RegularDomain(GenericDomain):

    """
    Regular polygon.

    Parameter: number of sides.
    """

    default = "3"
    full = [3]

    @staticmethod
    def format(x):
        """One integer with a lower bound."""
        return clip(x[:1], int, 3)

    def get(self):
        """One triangle per side with common vertex at (0,0)."""
        sides = self.values[0]
        mesh = Mesh()
        editor = MeshEditor()
        editor.open(mesh, 2, 2)
        editor.init_vertices(sides + 1)
        editor.init_cells(sides)
        editor.add_vertex(0, 0, 0)
        for i in range(1, sides + 1):
            editor.add_vertex(i,
                              cos(2 * pi * i / sides),
                              sin(2 * pi * i / sides))
        for i in range(sides - 1):
            editor.add_cell(i, 0, i + 1, i + 2)
        editor.add_cell(sides - 1, 0, sides, 1)
        editor.close()
        return mesh


class PolygonalSectorDomain(GenericDomain):

    """
    Polygonal sector.

    Approximation of a circular sector using K sides of a regular N-gon.

    Parameters: N, K.
    """

    default = "12, 2"
    full = (12, 2)

    @staticmethod
    def format(x):
        """Two integers with lower bounds."""
        return clip(x[:2], int, [3, 2])

    def get(self):
        """Part of the regular polygon construction."""
        sides, angle = self.pad(self.values)
        mesh = Mesh()
        editor = MeshEditor()
        editor.open(mesh, 2, 2)  # dimension
        editor.init_vertices(angle + 2)
        editor.init_cells(angle)
        editor.add_vertex(0, 0, 0)
        for i in range(0, angle + 1):
            editor.add_vertex(i + 1,
                              cos(2 * pi * i / sides),
                              sin(2 * pi * i / sides))
        editor.add_cell(0, 2, 1, 0)
        for i in range(1, angle):
            editor.add_cell(i, 0, i + 1, i + 2)
        editor.close()
        return mesh


class SquareDomain(GenericDomain):

    r"""
    Unit square.

    Unit square [0,1]x[0,1] subdivided into NxK rectangles, with diagonal \
    edge types given by L=0,1,2.

    Parameters: N,K,L (all optional).
    Default: 1,1,0.
    """

    default = ""
    full = (1, 1, 0)

    @staticmethod
    def format(x):
        """Two nonnegative integers. Then integer between 0 and 2."""
        return clip(x[:2], int, 1) + clip(x[2:3], int, 0, 2)

    def get(self):
        """Built in mesh."""
        x = self.pad(self.values)
        return UnitSquareMesh(x[0], x[1], self.diag[x[2]])


class RectangleDomain(GenericDomain):

    r"""
    Rectangle.

    Rectangle centered at the origin with sides of length A and B, \
    subdivided into NxK rectangles, with diagonal edge types given by L=0,1,2.

    Parameters: A,B,N,K,L (all optional).
    Default: 2,1,1,1,0.
    """

    default = "2, 1"
    full = (2, 1, 1, 1, 0)

    @staticmethod
    def format(x):
        """Two positive floats, then three ints."""
        return clip(np.fabs(x[:2]), float, 0.0001) + clip(x[2:4], int, 1) + \
            clip(x[4:5], int, 0, 2)

    def get(self):
        """Built in mesh."""
        x = self.pad(self.values)
        return RectangleMesh(Point(-x[0] / 2.0, -x[1] / 2.0),
                             Point(x[0] / 2.0, x[1] / 2.0),
                             x[2], x[3], self.diag[x[4]])


class CircleDomain(GenericDomain):

    """
    Disk.

    Unit disk centered at the origin with triangles of size L.

    Parameters L, (optional).
    Default: 0.1.
    """

    default = "0.1"
    full = [0.1]

    @staticmethod
    def format(x):
        """One positive float."""
        return clip(np.fabs(x[:1]), float, 0.003)

    def get(self):
        """Built-in mesh."""
        return CircleMesh(Point(0, 0), 1, self.pad(self.values)[0])


class TriangleDomain(GenericDomain):

    """
    Triangle.

    Triangle with vertices (0, 0), (1, 0) and (A, B).

    Parameters: A, B (optional).
    Default: (1/2, sqrt(3)/2) (equilateral triangle).
    """

    default = "1/2, sqrt(3)/2"
    full = (0.5, sqrt(3) / 2)

    @staticmethod
    def format(x):
        """One float. Then one positive float."""
        return clip(x[:1], float) + clip(np.fabs(x[1:2]), float, 0.0001)

    def get(self):
        """Just one cell."""
        topx, topy = self.pad(self.values)
        mesh = Mesh()
        editor = MeshEditor()
        editor.open(mesh, 2, 2)
        editor.init_vertices(3)
        editor.init_cells(1)
        editor.add_vertex(0, 0, 0)
        editor.add_vertex(1, 1, 0)
        editor.add_vertex(2, topx, topy)
        editor.add_cell(0, 0, 1, 2)
        editor.close()
        return mesh


class RightTriangleDomain(TriangleDomain):

    """
    Right triangle.

    Right triangle with vertices (0,0), (1,0) and smallest angle A near (0,0).

    Parameters: A (optional).
    Default: pi/4. (right isosceles triangle)
    """

    default = "pi/4"
    full = (1.0, tan(pi/4))

    @staticmethod
    def format(x):
        """Based on the general triangle."""
        return [1] + clip(abs(tan(x[0])), float, 0.001, 1)


class ParallelogramDomain(GenericDomain):

    """
    Parallelogram.

    Parallelogram with vertices (0,0), (1,0), (A,B) and (1-A,-B).

    Parameters: A, B (optional).
    Default: 0,1.
    """

    default = "0, 1"
    full = (0.0, 1.0)

    @staticmethod
    def format(x):
        """One float. Then positive float."""
        return clip(x[:1], float) + clip(np.fabs(x[1:2]), float, 0.0001)

    def get(self):
        """Two cells."""
        topx, topy = self.pad(self.values)
        mesh = Mesh()
        editor = MeshEditor()
        editor.open(mesh, 2, 2)
        editor.init_vertices(4)
        editor.init_cells(2)
        editor.add_vertex(0, 0, 0)
        editor.add_vertex(1, 1, 0)
        editor.add_vertex(2, topx, topy)
        editor.add_vertex(3, 1 - topx, -topy)
        editor.add_cell(0, 0, 1, 2)
        editor.add_cell(1, 0, 1, 3)
        editor.close()
        return mesh


class KiteDomain(GenericDomain):

    """
    Kite.

    Kite with vertices (0,0), (1,0), (A,B) and (A,-B).

    Parameters: A, B (optional).
    Default: 0.25,0.5.
    """

    default = "0.25, 0.5"
    full = (0.25, 0.5)

    @staticmethod
    def format(x):
        """One float. Then positive float."""
        return clip(x[:1], float) + clip(np.fabs(x[1:2]), float, 0.0001)

    def get(self):
        """Two cells."""
        a, b = self.pad(self.values)
        mesh = Mesh()
        editor = MeshEditor()
        editor.open(mesh, 2, 2)
        editor.init_vertices(4)
        editor.init_cells(2)
        editor.add_vertex(0, 0, 0)
        editor.add_vertex(1, 1, 0)
        editor.add_vertex(2, a, b)
        editor.add_vertex(3, a, -b)
        editor.add_cell(0, 0, 1, 2)
        editor.add_cell(1, 0, 1, 3)
        editor.close()
        return mesh


class IsoscelesDomain(KiteDomain):

    """
    Isosceles triangle.

    Isosceles triangle with vertices (0,0), (1,-+?) and angle A near (0,0).

    Parameters: A (optional).
    Default: pi/3.
    """

    default = "pi/3"
    full = (1.0, tan(pi / 6))

    @staticmethod
    def format(x):
        """Based on kites."""
        return [1] + clip(abs(tan(x[0] / 2)), float, 0.001)


class StarDomain(GenericDomain):

    r"""
    Star-shaped polygon.

    Star-shaped polygon in polar coordinates.

    The first parameter should be a list of polar angles in degrees.
        (randA(n) gives a sorted list of n random angles)

    The second parameter is a list of distances from the origin.
        (randD(n,min) gives random distances from (min,min+1) with default \
    value min=0.5)

    If the list of angles contains at most 2 elements, the first is used to \
    generate a new list of equally spaced angles.

    If the second list is too short, it will be reused cyclically to make \
    rotationally symmetric pattern.
    """

    default = "24, randD(6)"
    full = True

    @staticmethod
    def format(x):
        """Two lists of floats."""
        return [clip(x[0], float), clip(x[1], float)]

    def get(self):
        """Build vertices from polar coordinates."""
        angle, dist = self.values
        if len(angle) < 3:
            angle = np.array(range(int(angle[0]))) * 360.0 / angle[0]
        while len(dist) < len(angle):
            dist = dist * 2
        dist = np.array(dist)
        sides = len(angle)
        mesh = Mesh()
        editor = MeshEditor()
        editor.open(mesh, 2, 2)
        editor.init_vertices(sides + 1)
        editor.init_cells(sides)
        editor.add_vertex(0, 0, 0)
        for i in range(1, sides + 1):
            editor.add_vertex(i, dist[i - 1] * cos(angle[i - 1] / 180.0 * pi),
                              dist[i - 1] * sin(angle[i - 1] / 180.0 * pi))
        for i in range(sides - 1):
            editor.add_cell(i, 0, i + 1, i + 2)
        editor.add_cell(sides - 1, 0, sides, 1)
        editor.close()
        return mesh


class PolygonDomain(GenericDomain):

    r"""
    Polygon.

    Polygon with vertices given as a list ((x1, y1), (x2, y2), ...).

    Parentheses can be omitted inside of the list, and rand/randU can \
    be used to get N random points, see examples).

    Parameters: L - list of vertices, C - convex hull (0 or 1), \
    S - triangle sizes (default=0 (automatic)).

    Examples:
     - rand(8),0 gives arbitrary quadrilaterals,
     - rand(8),1 finds convex hull giving a convex quadrilateral or a triangle,
     - randU(4),1 gives four vertices on a unit circle.
    """

    default = "((0, 0), (2, 0), (2, 2), (1, 2), (1, 1), (0, 1)), 0"
    full = [((0, 0), (2, 0), (2, 2), (1, 2), (1, 1), (0, 1)), 0, 0]

    @staticmethod
    def format(x):
        """List of vertices. Then convex or not. Then integer for sizes."""
        # flatten vertices, then reshape into points
        temp = [np.reshape(clip(flatten(x[0]), float), (-1, 2))] + \
            clip(x[1:3], int, 0, [1, 500])
        if len(temp[0]) < 3:
            return []
        return temp

    def get(self):
        """Built-in polygonal mesh."""
        params = self.pad(self.values)
        if params[1]:
            vertices = convex_hull(params[0])
        else:
            vertices = params[0]
        vertices = [Point(*p) for p in vertices]
        size = params[2]
        try:
            # may fail if not counterclockwise vertices
            poly = Polygon(vertices)
        except:
            poly = Polygon(vertices[::-1])
        return generate_mesh(poly, size)


class IsospectralDomain(GenericDomain):

    r"""
    Isospectral drums.

    Pairs of isospectral of domains.

    Starting with a triangle, build a domain using a sequence of reflections.

    Parameters: N,K and optional (A,B), S.
    The first parameter chooses a pair of domains (N=-1 to 4), while K=0,1 \
    chooses a specific domain. Generating triangle has vertices (0,0), (1,0) \
    and angles 2pi/A, 2pi/B at these vertices.

    The default values of A and B depends on N. Case (N=2, A=B=8) gives the \
    original example due to Gordon, Webb and Wolpert.

    Equilateral generator gives isometric domains (possibly with slits), \
    while other triangles should lead to nonisometric examples \
    (sometimes selfintersecting).

    The examples are based on Figures 4 from Buser, Conway, Doyle and Semmler \
    (order as on the figure). A special, two piece example can be obtained by \
    taking N=-1 (due to Chapman).

    Parameters 3,0,6,12 and 4,0,6,12 give the same shape, but with different \
    slits, while their pairing 3,1 and 4,1 are very different.
    """

    default = "2, 0"
    full = [2, 0, 8, 8]

    @staticmethod
    def format(x):
        """Which pair. Then which one of the two. The two floats for angles."""
        return clip(x[:2], int, [-1, 0], [4, 1]) + clip(x[2:4], float, 2.001)

    # TODO: add more domains
    # TODO: Change angle parametrization
    trees = {
        # generating reflections
        # tree contains side numbers defining reflecion lines
        # after the node reflection is performed, a list of children
        # is used to perform further reflections
        (2, 0): [None, [2, [0], [1, [2, [0, [1]]]]]],
        (2, 1): [None, [0], [1], [2, [1, [0, [2]]]]],
        (1, 0): [None, [1, [2], [0, [1]]], [2, [0]]],
        (1, 1): [None, [1, [0, [2]]], [0], [2, [1]]],
        (0, 0): [None, [0, [1]], [1, [2]], [2, [0]]],
        (0, 1): [None, [0, [2]], [2, [1]], [1, [0]]],
        (3, 0): [None, [0, [1, [2, [1]]]], [1, [2, [0, [2]]]],
                 [2, [0, [1, [0]]]]],
        (3, 1): [None, [0, [2, [1, [2]]]], [1, [0, [2, [0]]]],
                 [2, [1, [0, [1]]]]],
        (4, 0): [None, [2, [0, [1, [0]]]],
                 [1, [2, [0, [2]]], [0, [1], [2, [1]]]]],
        (4, 1): [None, [2, [1], [0, [1]]],
                 [0, [2, [0]]], [1, [0, [2, [1, [2]]]]]],
    }
    # default triangle for each pair (two angles touching side of length 1)
    shape = [[8, 6], [8, 6], [8, 8], [6, 11], [7, 9]]
    # N=-1 vertices (for k=0 and k=1) and triangles (shared)
    bothvertices = [
        [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0),
         (0.0, 2.0), (1.0, 2.0), (1.1, 0.0), (2.1, 1.0),
         (1.1, 1.0), (1.1, 2.0)],
        [(1.0, 0.0), (2.0, 0.0), (1.0, 1.0), (0.0, 0.0),
         (0.0, 1.0), (0.0, 2.0), (1.1, 1.1), (2.1, 1.1),
         (1.1, 2.1), (2.1, 2.1)]]
    triangles = [
        (0, 1, 2), (0, 2, 3), (2, 3, 4), (2, 4, 5), (6, 7, 8), (7, 8, 9)]

    def generate(self, tree, indices):
        """Build triangles and vertices lists using trees of reflections."""
        newindices = indices[:]
        if tree[0] is None:
            self.triangles.append(newindices)
        else:
            newvertex = reflect([self.vertices[i] for i in indices], tree[0])
            newindices[tree[0]] = len(self.vertices)
            self.triangles.append(newindices)
            self.vertices.append(newvertex)
        for node in tree[1:]:
            self.generate(node, newindices)

    def get(self):
        """Build mesh from the tree of triangles created by reflections."""
        # get n
        n, k, a, b = self.pad(self.values)
        # adjust defaults based on n
        self.full[2:4] = self.shape[n]
        # get all parameters
        n, k, a, b = self.pad(self.values)
        if n == -1:
            self.vertices = self.bothvertices[k]
        else:
            tree = self.trees[(n, k)]
            self.triangles = []
            # angles are 2pi/a, 2pi/b
            side = np.sin(2 * pi / b) / np.sin(pi - 2 * pi / a - 2 * pi / b)
            self.vertices = [
                np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array(
                    [side * np.cos(2 * pi / a), side * np.sin(2 * pi / a)])]
            self.generate(tree, [0, 1, 2])

        mesh = Mesh()
        editor = MeshEditor()
        editor.open(mesh, 2, 2)
        editor.init_vertices(len(self.vertices))
        editor.init_cells(len(self.triangles))
        for i, v in enumerate(self.vertices):
            editor.add_vertex(i, *v)
        for i, t in enumerate(self.triangles):
            editor.add_cell(i, *t)
        editor.close()
        return mesh


class AnnulusDomain(GenericDomain):

    r"""
    Polygonal annulus.

    Polygonal annulus formed by two nested regular polygons with the same \
    number of sides N. The larger polygon is inscribed in unit circle, while
    smaller in circle with radius R.

    Use N=4 and Lp to Lq transform with parameters 1 2 to get circular annulus.

    Parameters: N - number of sides, R - length ratio smaller to larger polygon.
    """

    default = "4 0.5"
    full = (5, 0.5)

    @staticmethod
    def format(x):
        """One integer with a lower bound. Then a positive float up to 0.99. """
        return clip(np.fabs(x[:1]), int, 3) + \
            clip(np.fabs(x[1:2]), float, 0.001, 0.999)

    def get(self):
        """One triangle per side in smaller, two triangles in larger."""
        sides, R = self.pad(self.values)
        mesh = Mesh()
        large = [np.array((cos(2 * pi * i / sides), sin(2 * pi * i / sides)))
                 for i in range(1, sides+1)]
        small = np.array([v * R for v in large])
        # centers of edges in large polygon
        center = np.array([(v + w) / 2
                           for v, w in zip(large, large[1:] + [large[0]])])
        large = np.array(large)
        editor = MeshEditor()
        editor.open(mesh, 2, 2)
        editor.init_vertices(3 * sides)
        editor.init_cells(3 * sides)
        for i in range(sides):
            editor.add_vertex(3 * i, *large[i])
            editor.add_vertex(3 * i + 1, *small[i])
            editor.add_vertex(3 * i + 2, *center[i])
        for i, j in zip(range(sides), range(1, sides) + [0]):
            editor.add_cell(3*i, 3*i, 3*i+1, 3*i+2)
            editor.add_cell(3*i+1, 3*i+1, 3*i+2, 3*j+1)
            editor.add_cell(3*i+2, 3*i+2, 3*j+1, 3*j)
        editor.close()
        return mesh


class FunctionDomain(GenericDomain):

    r"""
    Graphs of functions.

    Generates top and bottom bounding curves from expressions involving x or y.
    If only one is given, the set will be symmetric.

    Important: Each expressions must be enclosed in " " or ' '!

    Parameters: (A, B) - domain, F - top curve, B - bottom curve, N - number of\
     sample points (default=100).
    """

    default = '(-1, 1), "1-x^2"'

    full = (-1, 1, "lambda x: 1-x**2", "lambda x: -(1-x**2)", 'x', 100)

    @staticmethod
    def format(x):
        """Two floats. Then at least one function. Finally an integer. """
        x = flatten(x)
        if len(x) < 3:
            return []
        domain = sorted(clip(x[:2], float))
        import re
        var = 'x' if re.search(r'\bx\b', ''.join(x[2:4])) else 'y'
        top = "lambda {}: {}".format(var, x[2])
        if len(x) == 3:
            bottom = "lambda {}: -({})".format(var, x[2])
        else:
            bottom = "lambda {}: {}".format(var, x[3])
        try:
            evaluate(top)[0](domain[0])
            evaluate(bottom)[0](domain[0])
        except:
            return []
        return domain + [top, bottom, var] + clip(np.fabs(x[4:5]),
                                                  int, 1, 1000)

    def get(self):
        """ Generate polygon from top and bottom curves. """
        m, M, top, bottom, var, N = self.pad(self.values)
        top = evaluate(top)[0]
        bottom = evaluate(bottom)[0]
        if var == 'x':
            points = [Point(x, top(x)) for x in np.linspace(m, M, N+2)]
        else:
            points = [Point(top(x), x) for x in np.linspace(m, M, N+2)]
        if abs(top(M)-bottom(M)) < 1E-4:
            points.pop()
        if var == 'x':
            points += [Point(x, bottom(x)) for x in np.linspace(M, m, N+2)]
        else:
            points += [Point(bottom(x), x) for x in np.linspace(M, m, N+2)]
        if abs(top(m)-bottom(m)) < 1E-4:
            points.pop()
        try:
            # may fail if not counterclockwise vertices
            poly = Polygon(points)
        except:
            poly = Polygon(points[::-1])
        return generate_mesh(poly, 1)


class StarlikeDomain(GenericDomain):

    r"""
    Starlike domain.

    Generates a domain using a radius function R(theta).

    Parameters: R - expression in theta, N - number of sample points \
    (default=100).
    """

    default = '"1+cos(theta)"'

    full = ("lambda theta: 1+cos(theta)", 100)

    @staticmethod
    def format(x):
        """Function, then an integer. """
        print x
        radius = "lambda theta: {}".format(x[0])
        try:
            evaluate(radius)[0](0)
            evaluate(radius)[0](np.pi)
        except:
            return []
        return [radius] + clip(np.fabs(x[1:2]), int, 1, 1000)

    def get(self):
        """ Generate polygon from top and bottom curves. """
        radius, N = self.pad(self.values)
        radius = evaluate(radius)[0]

        points = [Point(*(radius(theta)*np.array([cos(theta), sin(theta)])))
                  for theta in np.linspace(0, 2*np.pi, N+2)]
        if abs(radius(0)-radius(2*np.pi)) < 1E-4:
            points.pop()
        try:
            # may fail if not counterclockwise vertices
            poly = Polygon(points)
        except:
            poly = Polygon(points[::-1])
        return generate_mesh(poly, 1)

#
#   3D
#


class GenericDomain3D(GenericDomain):

    """Generic superclass for 3D domains."""

    dim = "3D"


class CubeDomain(GenericDomain3D):

    """
    Unit cube.

    Unit cube [0,1]^3 subdivided into NxKxL rectangles.

    Parameters: N,K,L (all optional).
    Default: 1,1,1.
    """

    default = ""
    full = (1, 1, 1)

    @staticmethod
    def format(x):
        """Three positive integers."""
        return clip(x[:3], int, 1)

    def get(self):
        """Built-in mesh."""
        return UnitCubeMesh(*self.pad(self.values))


class BoxDomain(GenericDomain3D):

    r"""
    Box.

    Box centered at the origin with sides of length A, B and C, \
    subdivided into N x K x L boxes.

    Parameters: A,B,C,N,K,L (all optional).
    Default: 3,2,1,1,1,1.
    """

    default = "3, 2, 1"
    full = (3, 2, 1, 1, 1, 1)

    @staticmethod
    def format(x):
        """Three positive floats. Then three positive integers."""
        return clip(x[:3], float, 0.0001) + clip(x[3:6], int, 1)

    def get(self):
        """Built-in mesh."""
        x = self.pad(self.values)
        return BoxMesh(Point(-x[0] / 2.0, -x[1] / 2.0, -x[2] / 2.0),
                       Point(x[0] / 2.0, x[1] / 2.0, x[2] / 2.0),
                       x[3], x[4], x[5])


class SphereDomain(GenericDomain3D):

    """
    Sphere.

    Unit sphere centered at the origin, with L boundary subdivisions.

    Parameters L (optional).
    Default: 10.
    """

    default = "10"
    full = [10]

    @staticmethod
    def format(x):
        """One positive float."""
        return clip(x[:1], int, 1, 30)

    def get(self):
        """Built-in domain."""
        return UnitSphereMesh(self.pad(self.values)[0])


class TetrahedronDomain(GenericDomain3D):

    """
    Tetrahedron.

    Tetrahedron with permutations of (1,1,0) as vertices.
    """

    def get(self):
        """One cell."""
        mesh = Mesh()
        editor = MeshEditor()
        editor.open(mesh, 3, 3)
        editor.init_vertices(4)
        editor.init_cells(1)
        editor.add_vertex(0, 0, 0, 0)
        editor.add_vertex(1, 1, 1, 0)
        editor.add_vertex(2, 0, 1, 1)
        editor.add_vertex(3, 1, 0, 1)
        editor.add_cell(0, 0, 1, 2, 3)
        editor.close()
        return mesh


class OctahedronDomain(GenericDomain3D):

    """
    Octahedron.

    Octahedron with permutations of (+-1,0,0) as vertices.

    Point (0,0,0) is the common vertex of all initial tetrahedra.
    """

    def get(self):
        """Eight cells."""
        mesh = Mesh()
        editor = MeshEditor()
        editor.open(mesh, 3, 3)
        editor.init_vertices(7)
        editor.init_cells(8)
        editor.add_vertex(0, 1, 0, 0)
        editor.add_vertex(1, 0, 1, 0)
        editor.add_vertex(2, 0, 0, 1)
        editor.add_vertex(3, -1, 0, 0)
        editor.add_vertex(4, 0, -1, 0)
        editor.add_vertex(5, 0, 0, -1)
        editor.add_vertex(6, 0, 0, 0)
        editor.add_cell(0, 6, 0, 1, 2)
        editor.add_cell(1, 6, 0, 1, 5)
        editor.add_cell(2, 6, 0, 4, 2)
        editor.add_cell(3, 6, 0, 4, 5)
        editor.add_cell(4, 6, 3, 1, 2)
        editor.add_cell(5, 6, 3, 1, 5)
        editor.add_cell(6, 6, 3, 4, 2)
        editor.add_cell(7, 6, 3, 4, 5)
        editor.close()
        return mesh


class PyramidDomain(GenericDomain3D):

    r"""
    Pyramid.

    Pyramid with regular N-gon inscribed into the unit circle as base, \
    and (X, Y, Z) as top vertex.

    Parameters: N, Z, X, Y (all optional). Default: 4, 1, 0, 0.
    """

    default = "4, 1"
    full = [4, 1, 0, 0]

    @staticmethod
    def format(x):
        """Positive integer. The positive float. Then two floats."""
        return clip(x[:1], int, 1) + clip(np.fabs(x[1:2]), float, 0.0001) + \
            clip(x[2:4], float)

    def get(self):
        """Build cells based on triangulated regular polygon."""
        sides, h, x, y = self.pad(self.values)
        mesh = Mesh()
        editor = MeshEditor()
        editor.open(mesh, 3, 3)  # dimension
        editor.init_vertices(sides + 2)
        editor.init_cells(sides)
        editor.add_vertex(0, x, y, h)
        for i in range(1, sides + 1):
            editor.add_vertex(i,
                              cos(2 * pi * i / sides),
                              sin(2 * pi * i / sides),
                              0)
        editor.add_vertex(sides + 1, 0, 0, 0)
        for i in range(sides - 1):
            editor.add_cell(i, 0, i + 1, i + 2, sides + 1)
        editor.add_cell(sides - 1, 0, sides, 1, sides + 1)
        editor.close()
        return mesh


class SimplexDomain(GenericDomain3D):

    """
    Simplex.

    Simplex with vertices (0, 0, 0), (1, 0, 0), (A, B, 0) and (C, D, E).

    Parameters: A, B, C, D, E (all optional).
    Default: 0, 1, 0, 0, 1.
    """

    default = "0, 1, 0, 0, 1"
    full = (0, 1, 0, 0, 1)

    @staticmethod
    def format(x):
        """Five floats. Second and fifth positive."""
        return clip(x[:1], float) + clip(np.fabs(x[1:2]), float, 0.0001) + \
            clip(x[2:4], float) + clip(np.fabs(x[4:5]), float, 0.0001)

    def get(self):
        """Single cell."""
        a, b, c, d, e = self.pad(self.values)
        mesh = Mesh()
        editor = MeshEditor()
        editor.open(mesh, 3, 3)  # dimension
        editor.init_vertices(4)
        editor.init_cells(1)
        editor.add_vertex(0, 0, 0, 0)
        editor.add_vertex(1, 1, 0, 0)
        editor.add_vertex(2, a, b, 0)
        editor.add_vertex(3, c, d, e)
        editor.add_cell(0, 0, 1, 2, 3)
        editor.close()
        return mesh


class IcosahedronDomain(GenericDomain3D):

    """
    Icosahedron.

    Icosahedron with center at the origin.
    """

    def get(self):
        """Data in a file."""
        data = from_file("res/icosahedron.dat")
        return polyhedron_mesh(data)


class TruncatedIcosahedronDomain(GenericDomain3D):

    """
    Truncated icosahedron.

    Truncated icosahedron with center at the origin.
    """

    def get(self):
        """Data in a file."""
        data = from_file("res/truncatedicosahedron.dat")
        return polyhedron_mesh(data)


class DodecahedronDomain(GenericDomain3D):

    """
    Dodecahedron.

    Dodecahedron with center at the origin.
    """

    def get(self):
        """Data in a file."""
        data = from_file("res/dodecahedron.dat")
        return polyhedron_mesh(data)


class CylinderDomain(GenericDomain3D):

    r"""
    Cylinder.

    Parameters H, B, T, N.
    Cylinder with height H, top and bottom radii B and T. The second \
    radius is optional (radii will be equal).

    Quality parameters N, K control roundness and number of cells. \
    Default values 100, 20.
    """

    default = "1, 1"
    full = (1, 1, 1, 100, 20)

    @staticmethod
    def format(x):
        """Three positive floats. Then one integer. """
        return clip(x[:3], float, 0.0001) + clip(x[3:5], int, 1)

    def get(self):
        """Use mshr Cylinder, though centers seem to not work."""
        h, b, t, n, k = self.pad(self.values)
        from mshr import Cylinder
        c = Cylinder(Point(0, 0, 0), Point(0, 0, h), b, t, n)
        return generate_mesh(c, k)


#
# File
#
class GenericDomainFile(GenericDomain):

    """
    Superclass for domains loading something from files.

    default should contain file extensions formatted for QFileDialog.
    """

    dim = "FILE"
    dialog = "Open file"

    def eval(self, params, _=''):
        """ Check if file exists, instead of evaluating to list of numbers. """
        import os.path
        if os.path.isfile(params):
            self.params = self.values = params


class MeshFileDomain(GenericDomainFile):

    r"""
    Mesh (XML or other).

    Import mesh from a file.

    Supports many mesh formats, see dolfin_utils/meshconvert.py for details. \
    Files can be gzipped (.gz).
    """

    dialog = "Open mesh file"
    # file types from dolfin_utils/meshconvert
    default = ("dolfin (*.xml *.xml.gz);;gmsh (*.msh *.msh.gz *.gmsh);;"
               "metis *.gra;;scotch *.grf;;diffpack *.grid;;abaqus *.inp;;"
               "NetCDF *.ncdf;;ExodusII (*.exo *.e);;StarCD (*.vrt *.cel);;"
               "Triangle (*.ele *.node);;mesh *.mesh")

    def get(self):
        """ Unzip and convert to xml if possible."""
        mesh = output = None
        import tempfile
        import os
        if not os.path.isfile(self.params):
            return None
        params = self.params
        try:
            split = os.path.splitext(params)
            if split[-1] == '.gz':
                # unzip the mesh
                import gzip
                f = gzip.open(params, 'rb')
                content = f.read()
                f.close()
                input = tempfile.NamedTemporaryFile(
                    suffix=os.path.splitext(split[0])[-1],
                    delete=False)
                input.write(content)
                input.close()
                params = input.name
                split = os.path.splitext(params)
            if split[-1] != '.xml':
                # convert using dolfin's meshconvert
                from dolfin_utils.meshconvert import meshconvert
                output = tempfile.NamedTemporaryFile(
                    suffix='.xml',
                    delete=False)
                output.close()
                meshconvert.convert2xml(params, output.name)
                params = output.name
            # use Mesh constructor with filename
            mesh = Mesh(params)
            try:
                os.unlink(output.name)
            except OSError:
                pass
            os.unlink(input.name)
        # many things could go wrong here
        # just return None as mesh if mesh importing failed
        finally:
            return mesh


#
# helper functions
#
def build_domain_dict():
    """
    Return dictionary of all domains.

    Recursively looks for all subclasses of GenericDomain.
    Keys  : domain names with dimension prepended
    Values: domain objects

    Set help and name attributes for subclasses based on docstrings.
    """
    from inspect import getdoc
    import re
    domain_set = set()
    domain_dict = {}
    # pylint: disable=no-member
    new = set(GenericDomain.__subclasses__())
    while new:
        domain_set.update(new)
        old = new
        new = set()
        for subcls in old:
            if not re.search("Generic", subcls.__name__):
                lines = re.sub(r'\\\n', '', getdoc(subcls)).split("\n")
                subcls.name = lines[0][:-1]
                subcls.help = "\n".join(lines[2:])
                domain_dict["{}. {}".format(subcls.dim, subcls.name)] = subcls()
            new.update([d for d in subcls.__subclasses__()
                        if d not in domain_set])
    return domain_dict


def polyhedron_mesh(data):
    """
    Build polyhedral mesh. Must be strlike with respect to the origin.

    Input:
        data[0] - list of vertices
        data[1] - list of faces
        data[2] - optional other starlike point, instead of the origin
    """
    # TODO: Center of mass of the vertices as origin
    vertex_data = np.array(data[0], dtype='double')
    lowest = np.min(flatten(data[1]))
    face_data = [list(np.array(d) - lowest) for d in data[1]]
    numv = len(vertex_data)  # will be the index of the origin
    if len(data) > 2:
        origin = np.array(data[2], dtype='double')
    else:
        origin = [0.0, 0.0, 0.0]
    mesh = Mesh()
    editor = DynamicMeshEditor()
    editor.open(mesh, "tetrahedron", 3, 3, numv + 1, len(face_data))
    for i, vert in enumerate(vertex_data):
        editor.add_vertex(i, *vert)
    editor.add_vertex(numv, *origin)
    newv = numv + 1  # next vertex index
    newf = 0  # next face index
    for face in face_data:
        if len(face) == 3:
            # triangular face, no splitting
            editor.add_cell(newf, numv, *face)  # face + origin
            newf += 1
        else:
            # split face into triangles using center of mass
            # average face vertices to get the center
            vert = list(np.mean(vertex_data[np.array(face)], axis=0))
            editor.add_vertex(newv, *vert)  # new vertex: face center
            face.append(face[0])
            for i in zip(face[:-1], face[1:]):
                # pairs of vertices
                editor.add_cell(newf, numv, newv, *i)  # + face center + origin
                newf += 1
            newv += 1
    editor.close()
    mesh.order()
    return mesh


def from_file(name):
    """
    Read polyhedron data from file.

    Format:
    first line: numer of vertices, numer of faces
    followed by: one vertex per line
    followed by: one face per line
    """
    import os
    data_path = os.path.dirname(__file__)
    name = os.path.join(data_path, name)
    f = open(name)
    v, c = evaluate(f.readline())[:2]
    vertices = []
    for _ in range(v):
        vertices.append(evaluate(f.readline()))
    faces = []
    for _ in range(c):
        faces.append(evaluate(f.readline()))
    return (vertices, faces)


def reflect(vs, ind):
    """Reflection with respet to the side opposite to vertex with index ind."""
    v1, v2 = vs[:ind] + vs[ind + 1:]
    return vs[ind] + 2 * (v1 - vs[ind] - np.dot(v1 - vs[ind], v1 - v2) /
                          np.dot(v1 - v2, v1 - v2) * (v1 - v2))


def convex_hull(points):
    """
    Find convex hull of a list of Points.

    Using what is available in an installed version of scipy.
    """
    try:
        from scipy.spatial import ConvexHull
        return points[ConvexHull(points).vertices]
    except ImportError:
        from scipy.spatial import Delaunay
        indices = np.unique(Delaunay(points).convex_hull.flatten())
        verts = points[indices]
        angles = np.arctan2(*((verts - np.mean(verts, axis=0)).transpose()))
        return verts[angles.argsort()]


def build_mesh(cells, vertices):
    """
    Assemble a mesh using cells and vertices.

    Using compiled C++ code.
    """
    dim = len(vertices[0])
    return builder.build_mesh(cells.flatten(), vertices.flatten(), dim)[0]


def build_mesh_old(cells, vertices):
    """Assemble a mesh object from cells and vertices."""
    mesh = Mesh()
    editor = MeshEditor()
    dim = len(vertices[0])
    if dim == 2:
        editor.open(mesh, 'triangle', 2, 2)
    else:
        editor.open(mesh, 'tetrahedron', 3, 3)
    editor.init_vertices(len(vertices))
    editor.init_cells(len(cells))
    for i, v in enumerate(vertices):
        editor.add_vertex(i, *v)
    for i, c in enumerate(cells):
        editor.add_cell(i, *c)
    editor.close()
    return mesh
