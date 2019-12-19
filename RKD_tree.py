import numpy as np
from numpy.random import random_integers as rdmint
import time
import matplotlib.pyplot as plt


def rkd_tree(data, kdim=5, leafsize=10):
    ndata, ndim = np.shape(data)
    sdim = np.arange(kdim)
    ind = sdim[rdmint(0, kdim-1)]
    idx = np.argsort(data[:, ind], kind='quicksort')
    data = data[idx, :]
    # Add split
    stack = []
    tree = [(None, 0, ind, data[int(ndata/2), ind])]
    stack.append((data[:int(ndata/2), :], 1, int(ndata/2) < leafsize))
    stack.append((data[int(ndata / 2):, :], 2, int(ndata / 2) < leafsize))
    # Iteratively loop through stack (to do list)
    while stack:
        data, node, leaf = stack.pop()
        if leaf:
            tree.append((data, node, None, None))
        else:
            ind = sdim[rdmint(0, kdim-1)]
            idx = np.argsort(data[:, ind], kind='quicksort')
            data = data[idx, :]
            ndata, ndim = np.shape(data)
            stack.append((data[:int(ndata/2), :], node*2+1, int(ndata/2) < leafsize))
            stack.append((data[int(ndata/2):, :], node*2+2, int(ndata/2) < leafsize))
            tree.append((None, node, ind, data[int(ndata/2), ind]))
    return tree


def sortsecond(val):
    return val[1]


def radius_nn(tree, point, radius):
    stack = [tree[0]]
    while stack:
        data, node, sdim, sval = stack.pop()
        if data is not None:
            distance = np.sqrt(np.sum((data-point)**2, 1))
            return data[distance < radius, :]
        else:
            if point[sdim] > sval:
                stack.append(tree[node*2+2])
            else:
                stack.append(tree[node*2+1])


def visualize(data, tree):
    xmi = min(1.5*np.min(data[:, 0]), 0.5*np.min(data[:, 0]))
    xma = 1.5*np.max(data[:, 0])
    ymi = min(1.5*np.min(data[:, 1]), 0.5*np.min(data[:, 1]))
    yma = 1.5*np.max(data[:, 1])
    hrect = [(0, xmi, xma, ymi, yma)]
    plt.scatter(data[:, 0], data[:, 1], c='k', s=1)
    while hrect:
        parent, pxmi, pxma, pymi, pyma = hrect.pop()
        data, node, sdim, sval = tree[parent]
        if data is not None:
            continue
        if sdim == 1:
            hrect.append((2*parent+2, pxmi, pxma, sval, pyma))
            hrect.append((2*parent+1, pxmi, pxma, pymi, sval))
            plt.plot([pxmi, pxma], [sval, sval])
        else:
            hrect.append((2*parent+1, pxmi, sval, pymi, pyma))
            hrect.append((2*parent+2, sval, pxma, pymi, pyma))
            plt.plot([sval, sval], [pymi, pyma])


X = np.load('A18.npy')
X = X[:510000, 3:]
n, m = np.shape(X)
X = (X-np.outer(np.ones(n), np.mean(X, axis=0)))/np.sqrt(np.var(X, axis=0))
u, s, v = np.linalg.svd(X, full_matrices=False, compute_uv=True)  # Singular Value Decomposition (for PCs)
X = np.dot(X, v[:, :2])  # Matrix transformation with dim. limit
t = time.time()
Tree = rkd_tree(X, kdim=2, leafsize=250)
Tree.sort(key=sortsecond)
buildtime = time.time() - t
t = time.time()
close = radius_nn(Tree, X[1, :], 5)
findtime = time.time() - t
visualize(X, Tree)
plt.figure()
plt.scatter(X[1, 0], X[1, 1], c='k', s=5)
plt.scatter(close[:, 0], close[:, 1], c='r', s=4)
plt.scatter(X[:, 0], X[:, 1], s=1)
plt.show()
print('Points included: %.0f' % n)
print('Tree size: %.0f, built in %.3f s' % (len(Tree), buildtime))
print('Cluster found, %.0f points in %.3f s' % (np.size(close, 0), findtime))
print('All done')
