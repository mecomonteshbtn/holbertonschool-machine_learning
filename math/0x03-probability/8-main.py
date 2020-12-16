#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  16 11:34:46 2020

@author: Robinson Montes
"""

import numpy as np
Normal = __import__('normal').Normal

np.random.seed(0)
data = np.random.normal(70, 10, 100).tolist()
n1 = Normal(data)
print('PSI(90):', n1.pdf(90))

n2 = Normal(mean=70, stddev=10)
print('PSI(90):', n2.pdf(90))
