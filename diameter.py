#!/Users/siudeja/anaconda/bin/python
"""
Compute the diameter of a Mesh.

Accepts numpy array of vertices.

Convex hull is computed first. Then:
2D: rotating calipers,
3D: brute force.
"""

# FIXME use Instant to get c++ speeds

from scipy.spatial import ConvexHull
import numpy as np

# try using numba just-in-time compiler if available
# else use numexpr
try:
    from numba import jit, autojit
    __NUMBA = True
except:
    import numexpr as ne
    __NUMBA = False

    def jit(*args, **kwargs):
        """ Useless decorator. """
        def donothing(f):
            return f
        return donothing

    def autojit(fun):
        """ Another useless decorator. """
        return fun


def diameter(points):
    """ Find diameter for a set of points. """
    if len(points[0]) == 2:
        return bounds2D(points)[0]
    else:
        return diameter3D(points)


def width(points):
    """
    Find the width of the domain.

    Not implemented in 3D!
    """
    if len(points[0]) == 2:
        return bounds2D(points)[1]
    else:
        return None


def diameter3D(points):
    """ Diameter of a 3D set using brute force method. """
    hull = ConvexHull(points)
    hull = points[hull.vertices]
    print hull.shape
    return compute3D(hull)


if __NUMBA:
    @jit("f8(f8[:,:])", nopython=True)
    def compute3D(points):
        """ Another brute force approach. """
        largest = 0
        for i in xrange(1, len(points)):
            for j in xrange(i):
                dist = (points[i, 0]-points[j, 0])**2 + \
                    (points[i, 1]-points[j, 1])**2 + \
                    (points[i, 2]-points[j, 2])**2
                if dist > largest:
                    largest = dist
        return np.sqrt(largest)
else:
    def compute3D(points):
        """ Brute force algorithm for finding diameter. """
        hull = ConvexHull(points)
        hull = points[hull.vertices]
        expr = "sum((a-p)**2, axis=1)"
        largest = [np.max(ne.evaluate(expr, local_dict={'a': hull[:i, :],
                                                        'p': hull[i]}))
                   for i in xrange(1, len(hull))]
        return np.sqrt(np.max(largest))


@autojit
def rotatingCalipers(L, U):
    """
    David Eppstein's implementation of rotating calipers.

    Given lists of lower L and upper U vertices of the convex hall finds all
    ways of sandwiching the points between two parallel lines that touch one
    point each, and yields the sequence of pairs of points touched by each
    pair of lines.
    """
    i = 0
    j = len(L) - 1
    while i < len(U) - 1 or j > 0:
        yield U[i], L[j]

        # if all the way through one side of hull, advance the other side
        if i == len(U) - 1:
            j -= 1
        elif j == 0:
            i += 1

        # still points left on both lists, compare slopes of next hull edges
        # being careful to avoid divide-by-zero in slope calculation
        elif (U[i+1][1]-U[i][1])*(L[j][0]-L[j-1][0]) > \
                (L[j][1]-L[j-1][1])*(U[i+1][0]-U[i][0]):
            i += 1
        else:
            j -= 1
    yield U[i], L[j]


@jit("f8(f8[:,:], f8[:,:])")
def runCalipers(L, U):
    """ Run calipers. """
    calipers = rotatingCalipers(L, U)
    oldp, oldq = calipers.next()
    width = diam = (oldp[0]-oldq[0])**2+(oldp[1]-oldq[1])**2
    for p, q in calipers:
        # update diameter
        dist = (p[0]-q[0])**2+(p[1]-q[1])**2
        if dist > diam:
                diam = dist
        # update width based on distance to new side
        dist = (oldp[0]-p[0])**2+(oldp[1]-p[1])**2
        if dist > 1E-15:
            # q stayed the same, p is new
            # find distance from q to side (oldp, p)
            dist = ((p[0]-oldp[0])*(q[1]-p[1])-(p[1]-oldp[1])*(q[0]-p[0]))**2 \
                / dist
            if dist < width:
                width = dist
        else:
            # q is new
            # find distance from p to side (oldq, q)
            dist = ((q[0]-oldq[0])*(p[1]-q[1])-(q[1]-oldq[1])*(p[0]-q[0]))**2 \
                / ((oldq[0]-q[0])**2+(oldq[1]-q[1])**2)
            if dist < width:
                width = dist
        oldp = p
        oldq = q
    return np.sqrt(diam), np.sqrt(width)


def bounds2D(points):
    """ Diameter and width via rotating calipers. """
    hull = ConvexHull(points)
    hull = points[hull.vertices]
    # extract upper and lower boundary
    leftmost = np.argmin(hull[:, 0])
    rightmost = np.argmax(hull[:, 0])
    if rightmost < leftmost:
        rightmost += len(hull)
    L = np.take(hull, range(leftmost, rightmost+1),
                mode='wrap', axis=0)
    U = np.take(hull, range(rightmost, leftmost+len(hull)+1),
                mode='wrap', axis=0)[::-1]
    return runCalipers(L, U)


from datetime import datetime
A = np.random.random((10, 2))
bounds2D(A)
A = np.random.random(100000)
A = np.exp(A*2j*np.pi)
AA = np.asarray([A.imag, A.real]).T
# A = np.random.random((1000,2))
start = datetime.now()
print bounds2D(AA)
print datetime.now()-start
start = datetime.now()
AA = np.asarray([A.imag, A.real, np.random.random(len(A.real))]).T
# AA = np.random.random((100000,3))
print AA.shape
print diameter3D(AA)
print datetime.now()-start
