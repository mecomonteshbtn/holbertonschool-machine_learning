#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

a = plt.bar(range(3), fruit[0, :], 0.5, color='red')
b = plt.bar(range(3), fruit[1, :], 0.5, color='yellow', bottom=fruit[0, :])
o = plt.bar(range(3), fruit[2, :], 0.5, color='#ff8000',
            bottom=fruit[0, :] + fruit[1, :])
p = plt.bar(range(3), fruit[3, :], 0.5, color='#ffe5b4',
            bottom=fruit[0, :] + fruit[1, :] + fruit[2, :])

plt.xticks(range(3), ('Farrah', 'Fred', 'Felicia'))
plt.yticks(np.arange(0, 81, 10))
plt.ylabel('Quantity of Fruit')
plt.title('Number of Fruit per Person')
plt.legend((a[0], b[0], o[0], p[0]),
           ('apples', 'bananas', 'oranges', 'peaches'))
plt.show()
