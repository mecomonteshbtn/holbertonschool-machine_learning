#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  16 11:34:46 2020

@author: Robinson Montes
"""

import numpy as np
Exponential = __import__('exponential').Exponential

np.random.seed(0)
data = np.random.exponential(0.5, 100).tolist()
e1 = Exponential(data)
print('F(1):', e1.cdf(1))

e2 = Exponential(lambtha=2)
print('F(1):', e2.cdf(1))
