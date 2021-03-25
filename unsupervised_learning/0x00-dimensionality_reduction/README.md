# 0x00. Dimensionality Reduction

---
## Description 
- What is eigendecomposition?
- What is singular value decomposition?
- What is the difference between eig and svd?
- What is dimensionality reduction and what are - its purposes?
- What is principal components analysis (PCA)?
- What is t-distributed stochastic neighbor - embedding (t-SNE)?
- What is a manifold?
- What is the difference between linear and - non-linear dimensionality reduction?
- Which techniques are linear/non-linear?

---
## Resources

**Read or watch**:

*   [Bayes’ Theorem - The Simplest Case](https://www.youtube.com/watch?v=XQoLVl31ZfQ "Bayes' Theorem - The Simplest Case")
*   [A visual guide to Bayesian thinking](https://www.youtube.com/watch?v=BrK7X_XlGB8 "A visual guide to Bayesian thinking")
*   [Base Rates](http://onlinestatbook.com/2/probability/base_rates.html "Base Rates")
*   [Bayesian statistics: a comprehensive course](https://www.youtube.com/playlist?list=PLFDbGp5YzjqXQ4oE4w9GVWdiokWB9gEpm "Bayesian statistics: a comprehensive course")

---
## Files
| File | Description |
| ------ | ------ |
| [0-pca.py](0-pca.py) | Function pca that performs PCA on a dataset. |
| [1-pca.py](1-pca.py) | Function pca that performs PCA on a dataset. |
| [2-P_init.py](2-P_init.py) | Function P_init that initializes all variables required to calculate the P affinities in t-SNE. |
| [3-entropy.py](3-entropy.py) | Function HP that calculates the Shannon entropy and P affinities relative to a data point. |
| [4-P_affinities.py](4-P_affinities.py) | Function P_affinities that calculates the symmetric P affinities of a data set. |
| [5-Q_affinities.py](5-Q_affinities.py) | Function Q_affinities that calculates the Q affinities. |
| [6-grads.py](6-grads.py) | Function grads that calculates the gradients of Y. |
| [7-cost.py](7-cost.py) | Function cost that calculates the cost of the t-SNE transformation. |
| [8-tsne.py](8-tsne.py) | Function tsne that performs a t-SNE transformation. |

---
### Build with
- Python (python 3.6)
- Numpy (numpy 1.15)
- Ubuntu 20.04 LTS 

---

### [0. Likelihood](./0-likelihood.py)
You are conducting a study on a revolutionary cancer drug and are looking to find the probability that a patient who takes this drug will develop severe side effects. During your trials, `n` patients take the drug and `x` patients develop severe side effects. You can assume that `x` follows a binomial distribution.

Write a function `def likelihood(x, n, P):` that calculates the likelihood of obtaining this data given various hypothetical probabilities of developing severe side effects:
*   `x` is the number of patients that develop severe side effects
*   `n` is the total number of patients observed
*   `P` is a 1D `numpy.ndarray` containing the various hypothetical probabilities of developing severe side effects
*   If `n` is not a positive integer, raise a `ValueError` with the message `n must be a positive integer`
*   If `x` is not an integer that is greater than or equal to `0`, raise a `ValueError` with the message `x must be an integer that is greater than or equal to 0`
*   If `x` is greater than `n`, raise a `ValueError` with the message `x cannot be greater than n`
*   If `P` is not a 1D `numpy.ndarray`, raise a `TypeError` with the message `P must be a 1D numpy.ndarray`
*   If any value in `P` is not in the range `[0, 1]`, raise a `ValueError` with the message `All values in P must be in the range [0, 1]`
*   Returns: a 1D `numpy.ndarray` containing the likelihood of obtaining the data, `x` and `n`, for each probability in `P`, respectively
```   
    alexa@ubuntu-xenial:0x07-bayesian_prob$ ./0-main.py 
    [0.00000000e+00 2.71330957e-04 8.71800070e-02 3.07345706e-03
     5.93701546e-07 1.14387595e-12 1.09257177e-20 6.10151799e-32
     9.54415702e-49 1.00596671e-78 0.00000000e+00]
    alexa@ubuntu-xenial:0x07-bayesian_prob$
```

### [1. Intersection](./1-intersection.py)
Based on `0-likelihood.py`, write a function `def intersection(x, n, P, Pr):` that calculates the intersection of obtaining this data with the various hypothetical probabilities:
*   `x` is the number of patients that develop severe side effects
*   `n` is the total number of patients observed
*   `P` is a 1D `numpy.ndarray` containing the various hypothetical probabilities of developing severe side effects
*   `Pr` is a 1D `numpy.ndarray` containing the prior beliefs of `P`
*   If `n` is not a positive integer, raise a `ValueError` with the message `n must be a positive integer`
*   If `x` is not an integer that is greater than or equal to `0`, raise a `ValueError` with the message `x must be an integer that is greater than or equal to 0`
*   If `x` is greater than `n`, raise a `ValueError` with the message `x cannot be greater than n`
*   If `P` is not a 1D `numpy.ndarray`, raise a `TypeError` with the message `P must be a 1D numpy.ndarray`
*   If `Pr` is not a `numpy.ndarray` with the same shape as `P`, raise a `TypeError` with the message `Pr must be a numpy.ndarray with the same shape as P`
*   If any value in `P` or `Pr` is not in the range `[0, 1]`, raise a `ValueError` with the message `All values in {P} must be in the range [0, 1]` where `{P}` is the incorrect variable
*   If `Pr` does not sum to `1`, raise a `ValueError` with the message `Pr must sum to 1` **Hint: use [numpy.isclose](/rltoken/7pptg2vy0_-c0qQ9MnZu1w "numpy.isclose")**
*   All exceptions should be raised in the above order
*   Returns: a 1D `numpy.ndarray` containing the intersection of obtaining `x` and `n` with each probability in `P`, respectively

```
    alexa@ubuntu-xenial:0x07-bayesian_prob$ ./1-main.py 
    [0.00000000e+00 2.46664506e-05 7.92545518e-03 2.79405187e-04
     5.39728678e-08 1.03988723e-13 9.93247059e-22 5.54683454e-33
     8.67650639e-50 9.14515194e-80 0.00000000e+00]
    alexa@ubuntu-xenial:0x07-bayesian_prob$
```

### [2. Marginal Probability](./2-marginal.py)
Based on `1-intersection.py`, write a function `def marginal(x, n, P, Pr):` that calculates the marginal probability of obtaining the data:
*   `x` is the number of patients that develop severe side effects
*   `n` is the total number of patients observed
*   `P` is a 1D `numpy.ndarray` containing the various hypothetical probabilities of patients developing severe side effects
*   `Pr` is a 1D `numpy.ndarray` containing the prior beliefs about `P`
*   If `n` is not a positive integer, raise a `ValueError` with the message `n must be a positive integer`
*   If `x` is not an integer that is greater than or equal to `0`, raise a `ValueError` with the message `x must be an integer that is greater than or equal to 0`
*   If `x` is greater than `n`, raise a `ValueError` with the message `x cannot be greater than n`
*   If `P` is not a 1D `numpy.ndarray`, raise a `TypeError` with the message `P must be a 1D numpy.ndarray`
*   If `Pr` is not a `numpy.ndarray` with the same shape as `P`, raise a `TypeError` with the message `Pr must be a numpy.ndarray with the same shape as P`
*   If any value in `P` or `Pr` is not in the range `[0, 1]`, raise a `ValueError` with the message `All values in {P} must be in the range [0, 1]` where `{P}` is the incorrect variable
*   If `Pr` does not sum to `1`, raise a `ValueError` with the message `Pr must sum to 1`
*   All exceptions should be raised in the above order
*   Returns: the marginal probability of obtaining `x` and `n`


```
    alexa@ubuntu-xenial:0x07-bayesian_prob$ ./2-main.py 
    0.008229580791426582
    alexa@ubuntu-xenial:0x07-bayesian_prob$
```

### [3. Posterior](./3-posterior.py)
Based on `2-marginal.py`, write a function `def posterior(x, n, P, Pr):` that calculates the posterior probability for the various hypothetical probabilities of developing severe side effects given the data:
*   `x` is the number of patients that develop severe side effects
*   `n` is the total number of patients observed
*   `P` is a 1D `numpy.ndarray` containing the various hypothetical probabilities of developing severe side effects
*   `Pr` is a 1D `numpy.ndarray` containing the prior beliefs of `P`
*   If `n` is not a positive integer, raise a `ValueError` with the message `n must be a positive integer`
*   If `x` is not an integer that is greater than or equal to `0`, raise a `ValueError` with the message `x must be an integer that is greater than or equal to 0`
*   If `x` is greater than `n`, raise a `ValueError` with the message `x cannot be greater than n`
*   If `P` is not a 1D `numpy.ndarray`, raise a `TypeError` with the message `P must be a 1D numpy.ndarray`
*   If `Pr` is not a `numpy.ndarray` with the same shape as `P`, raise a `TypeError` with the message `Pr must be a numpy.ndarray with the same shape as P`
*   If any value in `P` or `Pr` is not in the range `[0, 1]`, raise a `ValueError` with the message `All values in {P} must be in the range [0, 1]` where `{P}` is the incorrect variable
*   If `Pr` does not sum to `1`, raise a `ValueError` with the message `Pr must sum to 1`
*   All exceptions should be raised in the above order
*   Returns: the posterior probability of each probability in `P` given `x` and `n`, respectively
```    
    alexa@ubuntu-xenial:0x07-bayesian_prob$ ./3-main.py 
    [0.00000000e+00 2.99729127e-03 9.63044824e-01 3.39513268e-02
     6.55839819e-06 1.26359684e-11 1.20692303e-19 6.74011797e-31
     1.05430721e-47 1.11125368e-77 0.00000000e+00]
    alexa@ubuntu-xenial:0x07-bayesian_prob$
```

### [4. Continuous Posterior ](./4-P_affinities.py)
Based on `3-posterior.py`, write a function `def posterior(x, n, p1, p2):` that calculates the posterior probability that the probability of developing severe side effects falls within a specific range given the data:
*   `x` is the number of patients that develop severe side effects
*   `n` is the total number of patients observed
*   `p1` is the lower bound on the range
*   `p2` is the upper bound on the range
*   You can assume the prior beliefs of `p` follow a uniform distribution
*   If `n` is not a positive integer, raise a `ValueError` with the message `n must be a positive integer`
*   If `x` is not an integer that is greater than or equal to `0`, raise a `ValueError` with the message `x must be an integer that is greater than or equal to 0`
*   If `x` is greater than `n`, raise a `ValueError` with the message `x cannot be greater than n`
*   If `p1` or `p2` are not floats within the range `[0, 1]`, raise a`ValueError` with the message `{p} must be a float in the range [0, 1]` where `{p}` is the corresponding variable
*   if `p2` <= `p1`, raise a `ValueError` with the message `p2 must be greater than p1`
*   The only import you are allowed to use is `from scipy import math, special`
*   Returns: the posterior probability that `p` is within the range `[p1, p2]` given `x` and `n`
```
    alexa@ubuntu-xenial:0x07-bayesian_prob$ ./100-main.py 
    0.6098093274896035
    alexa@ubuntu-xenial:0x07-bayesian_prob$
```

### [5. Q affinities](./5-Q_affinities.py)
Write a function def Q_affinities(Y): that calculates the Q affinities:
*    Y is a numpy.ndarray of shape (n, ndim) containing the low dimensional transformation of X
*        n is the number of points
*        ndim is the new dimensional representation of X
*    Returns: Q, num
*        Q is a numpy.ndarray of shape (n, n) containing the Q affinities
*        num is a numpy.ndarray of shape (n, n) containing the numerator of the Q affinities
*    Hint: See page 7 of t-SNE
```
alexa@ubuntu-xenial:0x00-dimensionality_reduction$ cat 5-main.py 
#!/usr/bin/env python3

import numpy as np
Q_affinities = __import__('5-Q_affinities').Q_affinities

np.random.seed(0)
Y = np.random.randn(2500, 2)
Q, num = Q_affinities(Y)
print('num:', num.shape)
print(num)
print(np.sum(num))
print('Q:', Q.shape)
print(Q)
print(np.sum(Q))
alexa@ubuntu-xenial:0x00-dimensionality_reduction$ ./5-main.py 
num: (2500, 2500)
[[0.         0.1997991  0.34387413 ... 0.08229525 0.43197616 0.29803545]
 [0.1997991  0.         0.08232739 ... 0.0780192  0.36043254 0.20418429]
 [0.34387413 0.08232739 0.         ... 0.07484357 0.16975081 0.17792688]
 ...
 [0.08229525 0.0780192  0.07484357 ... 0.         0.13737822 0.22790422]
 [0.43197616 0.36043254 0.16975081 ... 0.13737822 0.         0.65251175]
 [0.29803545 0.20418429 0.17792688 ... 0.22790422 0.65251175 0.        ]]
2113140.980877581
Q: (2500, 2500)
[[0.00000000e+00 9.45507652e-08 1.62731275e-07 ... 3.89445137e-08
  2.04423728e-07 1.41039074e-07]
 [9.45507652e-08 0.00000000e+00 3.89597234e-08 ... 3.69209645e-08
  1.70567198e-07 9.66259681e-08]
 [1.62731275e-07 3.89597234e-08 0.00000000e+00 ... 3.54181605e-08
  8.03310395e-08 8.42001935e-08]
 ...
 [3.89445137e-08 3.69209645e-08 3.54181605e-08 ... 0.00000000e+00
  6.50113847e-08 1.07850932e-07]
 [2.04423728e-07 1.70567198e-07 8.03310395e-08 ... 6.50113847e-08
  0.00000000e+00 3.08787608e-07]
 [1.41039074e-07 9.66259681e-08 8.42001935e-08 ... 1.07850932e-07
  3.08787608e-07 0.00000000e+00]]
1.0000000000000004
alexa@ubuntu-xenial:0x00-dimensionality_reduction$
```

## [6. Gradients](./6-grads.py)
Write a function def grads(Y, P): that calculates the gradients of Y:
*    Y is a numpy.ndarray of shape (n, ndim) containing the low dimensional transformation of X
*    P is a numpy.ndarray of shape (n, n) containing the P affinities of X
*    Do not multiply the gradients by the scalar 4 as described in the paper’s equation
*    Returns: (dY, Q)
*        dY is a numpy.ndarray of shape (n, ndim) containing the gradients of Y
*        Q is a numpy.ndarray of shape (n, n) containing the Q affinities of Y
*    You may use Q_affinities = __import__('5-Q_affinities').Q_affinities
*    Hint: See page 8 of t-SNE
```
alexa@ubuntu-xenial:0x00-dimensionality_reduction$ cat 6-main.py 
#!/usr/bin/env python3

import numpy as np
pca = __import__('1-pca').pca
P_affinities = __import__('4-P_affinities').P_affinities
grads = __import__('6-grads').grads

np.random.seed(0)
X = np.loadtxt("mnist2500_X.txt")
X = pca(X, 50)
P = P_affinities(X)
Y = np.random.randn(X.shape[0], 2)
dY, Q = grads(Y, P)
print('dY:', dY.shape)
print(dY)
print('Q:', Q.shape)
print(Q)
print(np.sum(Q))
alexa@ubuntu-xenial:0x00-dimensionality_reduction$ ./6-main.py 
dY: (2500, 2)
[[ 1.28824814e-05  1.55400363e-05]
 [ 3.21435525e-05  4.35358938e-05]
 [-1.02947106e-05  3.53998421e-07]
 ...
 [-2.27447049e-05 -3.05191863e-06]
 [ 9.69379032e-06  1.00659610e-06]
 [ 5.75113416e-05  7.65517123e-09]]
Q: (2500, 2500)
[[0.00000000e+00 9.45507652e-08 1.62731275e-07 ... 3.89445137e-08
  2.04423728e-07 1.41039074e-07]
 [9.45507652e-08 0.00000000e+00 3.89597234e-08 ... 3.69209645e-08
  1.70567198e-07 9.66259681e-08]
 [1.62731275e-07 3.89597234e-08 0.00000000e+00 ... 3.54181605e-08
  8.03310395e-08 8.42001935e-08]
 ...
 [3.89445137e-08 3.69209645e-08 3.54181605e-08 ... 0.00000000e+00
  6.50113847e-08 1.07850932e-07]
 [2.04423728e-07 1.70567198e-07 8.03310395e-08 ... 6.50113847e-08
  0.00000000e+00 3.08787608e-07]
 [1.41039074e-07 9.66259681e-08 8.42001935e-08 ... 1.07850932e-07
  3.08787608e-07 0.00000000e+00]]
1.0000000000000004
alexa@ubuntu-xenial:0x00-dimensionality_reduction$ 
```

### [7. Cost](./7-cost.py)
Write a function def cost(P, Q): that calculates the cost of the t-SNE transformation:
*    P is a numpy.ndarray of shape (n, n) containing the P affinities
*    Q is a numpy.ndarray of shape (n, n) containing the Q affinities
*    Returns: C, the cost of the transformation
*    Hint 1: See page 5 of t-SNE
*    Hint 2: Watch out for division by 0 errors! Take the minimum of all values in p and q with almost 0 (ex. 1e-12)
```
alexa@ubuntu-xenial:0x00-dimensionality_reduction$ cat 7-main.py 
#!/usr/bin/env python3

import numpy as np
pca = __import__('1-pca').pca
P_affinities = __import__('4-P_affinities').P_affinities
grads = __import__('6-grads').grads
cost = __import__('7-cost').cost

np.random.seed(0)
X = np.loadtxt("mnist2500_X.txt")
X = pca(X, 50)
P = P_affinities(X)
Y = np.random.randn(X.shape[0], 2)
_, Q = grads(Y, P)
C = cost(P, Q)
print(C)
alexa@ubuntu-xenial:0x00-dimensionality_reduction$ ./7-main.py 
4.531113944164374
alexa@ubuntu-xenial:0x00-dimensionality_reduction$
```

### [8. t-SNE](./8-tsne.py)
Write a function def tsne(X, ndims=2, idims=50, perplexity=30.0, iterations=1000, lr=500): that performs a t-SNE transformation:
*    X is a numpy.ndarray of shape (n, d) containing the dataset to be transformed by t-SNE
*        n is the number of data points
*        d is the number of dimensions in each point
*    ndims is the new dimensional representation of X
*    idims is the intermediate dimensional representation of X after PCA
*    perplexity is the perplexity
*    iterations is the number of iterations
*    lr is the learning rate
*    Every 100 iterations, not including 0, print Cost at iteration {iteration}: {cost}
*        {iteration} is the number of times Y has been updated and {cost} is the corresponding cost
*    After every iteration, Y should be re-centered by subtracting its mean
*    Returns: Y, a numpy.ndarray of shape (n, ndim) containing the optimized low dimensional transformation of X
*    You should use:
```
        pca = __import__('1-pca').pca
        P_affinities = __import__('4-P_affinities').P_affinities
        grads = __import__('6-grads').grads
        cost = __import__('7-cost').cost
```
*    For the first 100 iterations, perform early exaggeration with an exaggeration of 4
*    a(t) = 0.5 for the first 20 iterations and 0.8 thereafter
*    Hint 1: See Algorithm 1 on page 9 of t-SNE. But WATCH OUT! There is a mistake in the gradient descent step
*    Hint 2: See Section 3.4 starting on page 9 of t-SNE for early exaggeration
```
alexa@ubuntu-xenial:0x00-dimensionality_reduction$ cat 8-main.py 
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
tsne = __import__('8-tsne').tsne

np.random.seed(0)
X = np.loadtxt("mnist2500_X.txt")
labels = np.loadtxt("mnist2500_labels.txt")
Y = tsne(X, perplexity=50.0, iterations=3000, lr=750)
plt.scatter(Y[:, 0], Y[:, 1], 20, labels)
plt.colorbar()
plt.title('t-SNE')
plt.show()
alexa@ubuntu-xenial:0x00-dimensionality_reduction$ ./8-main.py 
Cost at iteration 100: 15.132745380504453
Cost at iteration 200: 1.4499349051185875
Cost at iteration 300: 1.299196107400927
Cost at iteration 400: 1.225553022181153
Cost at iteration 500: 1.1797532644514792
Cost at iteration 600: 1.147630679133037
Cost at iteration 700: 1.1235015025736461
Cost at iteration 800: 1.1044968276172742
Cost at iteration 900: 1.0890468673949136
Cost at iteration 1000: 1.0762018736143135
Cost at iteration 1100: 1.0652921250043608
Cost at iteration 1200: 1.0558751316523143
Cost at iteration 1300: 1.0476533388700722
Cost at iteration 1400: 1.040398188071647
Cost at iteration 1500: 1.0339353593266645
Cost at iteration 1600: 1.028128752446572
Cost at iteration 1700: 1.022888534479414
Cost at iteration 1800: 1.018126557673678
Cost at iteration 1900: 1.0137760713813615
Cost at iteration 2000: 1.0097825451815519
Cost at iteration 2100: 1.006100712557423
Cost at iteration 2200: 1.0026950513450208
Cost at iteration 2300: 0.999533533326889
Cost at iteration 2400: 0.9965894332394137
Cost at iteration 2500: 0.9938399255561283
Cost at iteration 2600: 0.9912653473151111
Cost at iteration 2700: 0.9888485527807178
Cost at iteration 2800: 0.9865746432480411
Cost at iteration 2900: 0.9844307720012043
Cost at iteration 3000: 0.9824051809484148

Awesome! We can see pretty good clusters! For comparison, here’s how PCA performs on the same dataset:

alexa@ubuntu-xenial:0x00-dimensionality_reduction$ cat pca.py 
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
pca = __import__('1-pca').pca

X = np.loadtxt("mnist2500_X.txt")
labels = np.loadtxt("mnist2500_labels.txt")
Y = pca(X, 2)
plt.scatter(Y[:, 0], Y[:, 1], 20, labels)
plt.colorbar()
plt.title('PCA')
plt.show()
alexa@ubuntu-xenial:0x00-dimensionality_reduction$ ./pca.py 
```

---
## Authors

* **Robinson Montes** - [mecomonteshbtn](https://github.com/mecomonteshbtn)
