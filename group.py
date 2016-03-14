""" Generate vertices from a matrix group. """
from __future__ import division
import gzip
import numpy as np
import scipy.spatial as ss

f = gzip.open('ico.group.gz', 'r')
vs = []
for i, l in enumerate(f):
    v = np.array(eval(l)).dot([1, 1, 1])
    vs.append(v)
f.close()
print i+1

# eliminate very close vertices
if hasattr(ss.cKDTree, 'query_pairs'):
    kd = ss.cKDTree(vs)
else:
    kd = ss.KDTree(vs)
repeated = np.array(list(kd.query_pairs(1E-10)))[:, 1]
vertices = set(range(120)) - set(repeated)
print len(vertices)
print np.array(vs)[list(vertices)]
