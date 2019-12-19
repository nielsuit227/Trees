import numpy as np
import matplotlib.pyplot as plt
import time


def kdtree(data, kdim=5, leafsize=10):
    ndata, ndim = np.shape(data)
    ind = 0
    idx = np.argsort(data[:, ind], kind='quicksort')
    data = data[idx, :]
    # Add split
    stack = []
    tree = [(None, 0, ind, data[int(ndata/2), ind])]
    stack.append((data[:int(ndata/2), :], 1, int(ndata/2) < leafsize, 0))
    stack.append((data[int(ndata / 2):, :], 2, int(ndata / 2) < leafsize, 0))
    # Iteratively loop through stack (to do list)
    while stack:
        data, node, leaf, depth = stack.pop()
        if leaf:
            tree.append((data, node, None, None))
        else:
            ind = np.remainder(depth+1, 2)
            idx = np.argsort(data[:, ind], kind='quicksort')
            data = data[idx, :]
            ndata, ndim = np.shape(data)
            stack.append((data[:int(ndata/2), :], node*2+1, int(ndata/2) < leafsize, depth+1))
            stack.append((data[int(ndata/2):, :], node*2+2, int(ndata/2) < leafsize, depth+1))
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
    xmi = min(1.1*np.min(data[:, 0]), 0.9*np.min(data[:, 0]))
    xma = 1.1*np.max(data[:, 0])
    ymi = min(1.1*np.min(data[:, 1]), 0.9*np.min(data[:, 1]))
    yma = 1.1*np.max(data[:, 1])
    hrect = [(0, xmi, xma, ymi, yma)]
    store = []
    copy = data.copy()
    plt.scatter(data[:, 0], data[:, 1], c='k', s=1)
    while hrect:
        plt.scatter(copy[:, 0], copy[:, 1], c='k', s=1)
        parent, pxmi, pxma, pymi, pyma = hrect.pop()
        data, node, sdim, sval = tree[parent]
        if data is not None:
            continue
        if sdim == 1:
            hrect.append((2*parent+2, pxmi, pxma, sval, pyma))
            hrect.append((2*parent+1, pxmi, pxma, pymi, sval))
            plt.plot([pxmi, pxma], [sval, sval])
            store.append((pxmi, pxma, sval, sval))
        else:
            hrect.append((2*parent+1, pxmi, sval, pymi, pyma))
            hrect.append((2*parent+2, sval, pxma, pymi, pyma))
            plt.plot([sval, sval], [pymi, pyma])
            store.append((sval, sval, pymi, pyma))
        for i in range(len(store)):
            xmin, xmax, ymin, ymax = store[i]
            plt.plot([xmin, xmax], [ymin, ymax])
        plt.show()


# Load data
X = np.load('A18.npy')
X = X[:1000, 3:]
n, m = X.shape
X = (X - np.outer(np.ones(n), np.mean(X, axis=0))) / np.sqrt(np.var(X, axis=0))  # Normalize data
u, s, v = np.linalg.svd(X, full_matrices=False, compute_uv=True)  # Singular Value Decomposition (for PCs)
X = np.dot(X, v[:, :2])  # Matrix transformation with dim. limit

# Generate tree
Tree = kdtree(X, leafsize=50)
Tree.sort(key=sortsecond)

# Visualize R-NN
plt.show()
close = radius_nn(Tree, point=X[0, :], radius=2)
plt.scatter(X[0, 0], X[0, 1], c='k', s=5)
plt.scatter(close[:, 0], close[:, 1], c='r', s=3)
plt.scatter(X[:, 0], X[:, 1], c='b', s=1)

# Visualize Tree
visualize(X, Tree)
plt.show()
print('Tree size: %.0f' % len(Tree))
print('All done')





