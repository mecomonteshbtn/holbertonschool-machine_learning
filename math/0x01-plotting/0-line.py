#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

y = np.arange(0, 11) ** 3

plt.xlim(0, len(y)-1)
plt.plot(range(len(y)), y, color='red')
plt.show()
