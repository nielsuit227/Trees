import numpy as np
from numpy.random import random_integers as randdim
import matplotlib.pyplot as plt
import time


def rkd_tree(data, kdim=5, leafsize=10):
    # Note size
    ndim, ndata = np.shape(data)

    # Initialize hyperrectangles
    hrect = np.zeros((2, ndim))
    hrect[0, :] = data.min(axis=1)
    hrect[1, :] = data.max(axis=1)

    # Project data on highest variance axis'
    u, s, v = np.linalg.svd(data.T, full_matrices=False, compute_uv=True)  # Singular Value Decomposition (for PCs)
    data = np.dot(v, data)

    # Initial data split
    splitdim = np.arange(kdim)
    ind = splitdim[randdim(0, kdim - 1)]
    idx = np.argsort(data[ind, :], kind='quicksort')
    data = data[:, idx]
    splitvalue = data[ind, int(ndata / 2)]

    # Update hrect
    l_hrect = hrect.copy()
    r_hrect = hrect.copy()
    l_hrect[1, ind] = splitvalue
    r_hrect[0, ind] = splitvalue

    # Update Tree
    # leaf_data, leaf_idx, node l_hrect, node r_hrect, prev left, prev right) ]
    tree = [(None, None, l_hrect, r_hrect, None, None)]
    # Stack = list [ (data, idx, depth, parent, is_left) ]
    stack = [(data[:, :int(ndata / 2)], idx[:int(ndata / 2)], 1, 0, True),
             (data[:, int(ndata / 2):], idx[int(ndata / 2):], 1, 0, False)]
    # Now loop until stack (to do) is empty
    while stack:
        # Load bottom of stack
        data, didx, depth, parent, leftbranch = stack.pop()
        ndata = data.shape[1]
        treesize = len(tree)
        # Edit children number parent
        _didx, _data, _left_hrect, _right_hrect, left, right = tree[parent]
        if leftbranch:
            tree[parent] = (_didx, _data, _left_hrect, _right_hrect, treesize, right)
        else:
            tree[parent] = (_didx, _data, _left_hrect, _right_hrect, left, treesize)
        if ndata <= leafsize:  # If leaf, save
            _didx = didx.copy()
            _data = data.copy()
            leaf = (_didx, _data, None, None, 0, 0)
            tree.append(leaf)
        else:  # Else split
            # Split data
            ind = splitdim[randdim(0, 4)]
            idx = np.argsort(data[ind, :], kind='quicksort')
            data = data[:, idx]
            didx = didx[idx]
            nodeprnt = len(tree)
            splitvalue = data[ind, int(ndata / 2)]
            stack.append((data[:, :int(ndata / 2)], didx[:int(ndata / 2)], depth + 1, nodeprnt, True))
            stack.append((data[:, int(ndata / 2):], didx[int(ndata / 2):], depth + 1, nodeprnt, False))
            # Update hrect
            if leftbranch:
                l_hrect = _left_hrect.copy()
                r_hrect = _left_hrect.copy()
            else:
                l_hrect = _right_hrect.copy()
                r_hrect = _right_hrect.copy()
            l_hrect[1, splitdim] = splitvalue
            r_hrect[0, splitdim] = splitvalue
            # Update tree
            # (leaf_data, leaf_idx, node l_hrect, node r_hrect, prev left, prev right)
            tree.append((None, None, l_hrect, r_hrect, None, None))
    return tree


def rad_nearest_neighbor(tree, point, rad):
    stack = [tree[0]]
    inside = []
    while stack:
        leaf_idx, leaf_data, l_hrect, r_hrect, left, right = stack.pop()
        if leaf_idx is not None:
            param = leaf_data.shape[0]
            distance = np.sqrt(((leaf_data-point.reshape((param, 1)))**2).sum(axis=0))
            near = np.where(distance <= rad)
            if len(near[0]):
                idx = leaf_idx[near]
                distance = distance[near]
                inside += (zip(distance, idx))
        else:
            if intersect(l_hrect, rad, point):
                stack.append(tree[left])
            if intersect(r_hrect, rad, point):
                stack.append(tree[right])
    return inside


def intersect(hrect, r2, centroid):
    maxval = hrect[1, :]
    minval = hrect[0, :]
    p = centroid.copy()
    idx = p < minval
    p[idx] = minval[idx]
    idx = p > maxval
    p[idx] = maxval[idx]
    return ((p-centroid)**2).sum() < r2

def sortSecond(val):
    return val[1]

# Load data
X = np.load('A18.npy')
X = X[:10000, 3:]
n, m = np.shape(X)
X = (X-np.outer(np.ones(n), np.mean(X, axis=0)))/np.sqrt(np.var(X, axis=0))

t = time.time()
Tree = rkd_tree(X.T, kdim=5, leafsize=50)
buildtime = time.time() - t
t = time.time()
close = rad_nearest_neighbor(Tree, X[1, :], 5)

# print('Time to find all samples in radius: %.4f' % (time.time()-t))

print('Tree size: %.0f, found in %.3f s' % (len(Tree), buildtime))
print('All done')
