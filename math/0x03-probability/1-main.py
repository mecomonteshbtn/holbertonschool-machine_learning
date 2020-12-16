#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  16 11:34:46 2020

@author: Robinson Montes
"""

import numpy as np
Poisson = __import__('poisson').Poisson

np.random.seed(0)
data = np.random.poisson(5., 100).tolist()
p1 = Poisson(data)
print('P(9):', p1.pmf(9))

p2 = Poisson(lambtha=5)
print('P(9):', p2.pmf(9))
