#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 08:33:12 2021

@author: Robinson Montes
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


pca = __import__('1-pca').pca

X = np.loadtxt("mnist2500_X.txt")
labels = np.loadtxt("mnist2500_labels.txt")
Y = pca(X, 2)
fig = plt.figure()
plt.scatter(Y[:, 0], Y[:, 1], 20, labels, cmap=cm.coolwarm)
plt.colorbar()
plt.title('PCA')
plt.show()
fig.savefig('pca.png')
