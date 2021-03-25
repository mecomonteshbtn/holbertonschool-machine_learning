#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 08:33:12 2021

@author: Robinson Montes
"""
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib import cm


X = np.loadtxt("mnist2500_X.txt")
labels = np.loadtxt("mnist2500_labels.txt")
Y = TSNE(n_components=2).fit_transform(X)
fig = plt.figure()
plt.scatter(Y[:, 0], Y[:, 1], 20, labels, cmap=cm.coolwarm)
plt.colorbar()
plt.title('PCA')
plt.show()
fig.savefig('tsne_sklearn.png')
