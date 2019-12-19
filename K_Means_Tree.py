import numpy as np
from sklearn.cluster import KMeans as kmeans
import time


def k_means_tree(data, branch_factor=5, leaf_size=50, max_iter=7):
    ndata, ndim = np.shape(data)
    tree = []
    if ndata < branch_factor:
        tree.append((data, 0, data.mean(axis=0)))
    else:
        km = kmeans(n_clusters=branch_factor, n_init=1, max_iter=max_iter).fit(data)
        # Stack is to-do list, (data, parentnumber, node number, leaf)
        tree.append((None, 0, km.cluster_centers_))
        stack = []
        for i in range(branch_factor):
            stack.append((data[km.labels_ == i, :], i+1, np.sum(km.labels_ == i) < leaf_size))
        while stack:
            data, node, leaf = stack.pop()
            if leaf:
                tree.append((data, node, None))
            else:
                km = km.fit(data)
                tree.append((None, node, km.cluster_centers_))
                for i in range(branch_factor):
                    stack.append((data[km.labels_ == i, :], node*branch_factor+i+1,
                                  np.sum(km.labels_ == i) < leaf_size))

    return tree


X = np.load('A18.npy')
X = X[:10000, 3:]
n, m = np.shape(X)
X = (X-np.outer(np.ones(n), np.mean(X, axis=0)))/np.sqrt(np.var(X, axis=0))
u, s, v = np.linalg.svd(X, full_matrices=False, compute_uv=True)  # Singular Value Decomposition (for PCs)
X = np.dot(X, v[:, :2])  # Matrix transformation with dim. limit
t = time.time()
Tree = k_means_tree(X, branch_factor=3, leaf_size=50)
buildtime = time.time() - t
print('Tree size: %.0f, found in %.3f s' % (len(Tree), buildtime))
print('All done')
