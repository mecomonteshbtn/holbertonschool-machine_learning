# 0x05. Regularization

---
## Read or watch:
*    [Regularization (mathematics)](https://en.wikipedia.org/wiki/Regularization_(mathematics))
*    [An Overview of Regularization Techniques in Deep Learning](https://www.analyticsvidhya.com/blog/2018/04/fundamentals-deep-learning-regularization-techniques/)
*    [L2 Regularization and Back-Propagation](https://jamesmccaffrey.wordpress.com/2017/02/19/l2-regularization-and-back-propagation/)
*    [Intuitions on L1 and L2 Regularisation](https://towardsdatascience.com/intuitions-on-l1-and-l2-regularisation-235f2db4c261)
*    [Analysis of Dropout](https://pgaleone.eu/deep-learning/regularization/2017/01/10/anaysis-of-dropout/)
*    [Early stopping](https://en.wikipedia.org/wiki/Early_stopping)
*    [How to use early stopping properly for training deep neural network?](https://stats.stackexchange.com/questions/231061/how-to-use-early-stopping-properly-for-training-deep-neural-network)
*    [Data Augmentation | How to use Deep Learning when you have Limited Data ](https://medium.com/nanonets/how-to-use-deep-learning-when-you-have-limited-data-part-2-data-augmentation-c26971dc8ced)
*    [deeplearning.ai videos:](https://www.deeplearning.ai/)
*        [Regularization](https://www.youtube.com/watch?v=6g0t3Phly2M&index=4&list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc)
*        [Why Regularization Reduces Overfitting](https://www.youtube.com/watch?v=NyG-7nRpsW8&index=5&list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc)
*        [Dropout Regularization](https://www.youtube.com/watch?v=D8PJAL-MZv8&index=6&list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc)
*        [Understanding Dropout](https://www.youtube.com/watch?v=ARq74QuavAo&index=7&list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc)
*        [Other Regularization Methods](https://www.youtube.com/watch?v=BOCLq2gpcGU&index=8&list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc)

---
## References:
*    [numpy.linalg.norm](https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.linalg.norm.html)
*    [numpy.random.binomial](https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.binomial.html)
*    [tf.contrib.layers.l2_regularizer](https://github.com/tensorflow/docs/blob/r1.12/site/en/api_docs/python/tf/contrib/layers/l2_regularizer.md)
*    [tf.layers.Dense#kernel_regularizer](https://github.com/tensorflow/docs/blob/r1.12/site/en/api_docs/python/tf/layers/Dense.md#kernel_regularizer)
*    [tf.losses.get_regularization_loss](https://github.com/tensorflow/docs/blob/r1.12/site/en/api_docs/python/tf/losses/get_regularization_losses.md)
*    [tf.layers.Dropout](https://github.com/tensorflow/docs/blob/r1.12/site/en/api_docs/python/tf/layers/Dropout.md)
*    [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](http://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf)
*    [Early Stopping - but when?](https://page.mi.fu-berlin.de/prechelt/Biblio/stop_tricks1997.pdf)
*    [L2 Regularization versus Batch and Weight Normalization](https://arxiv.org/pdf/1706.05350.pdf)

---
## Learning Objectives
At the end of this project, you are expected to be able to explain to anyone, without the help of Google:
General
*    What is regularization? What is its purpose?
*    What is are L1 and L2 regularization? What is the difference between the two methods?
*    What is dropout?
*    What is early stopping?
*    What is data augmentation?
*    How do you implement the above regularization methods in Numpy? Tensorflow?
*    What are the pros and cons of the above regularization methods?

---
## General Requirements
*    Allowed editors: vi, vim, emacs
*    All your files will be interpreted/compiled on Ubuntu 16.04 LTS using python3 (version 3.5)
*    Your files will be executed with numpy (version 1.15)
*    All your files should end with a new line
*    The first line of all your files should be exactly #!/usr/bin/env python3
*    A README.md file, at the root of the folder of the project, is mandatory
*    Your code should use the pycodestyle style (version 2.4)
*    All your modules should have documentation (python3 -c 'print(__import__("my_module").__doc__)')
*    All your classes should have documentation (python3 -c 'print(__import__("my_module").MyClass.__doc__)')
*    All your functions (inside and outside a class) should have documentation (python3 -c 'print(__import__("my_module").my_function.__doc__)' and python3 -c 'print(__import__("my_module").MyClass.my_function.__doc__)')
*    Unless otherwise noted, you are not allowed to import any module except import numpy as np and import tensorflow as tf
*    You are not allowed to use the keras module in tensorflow
*    You should not import any module unless it is being used
*    All your files must be executable
*    The length of your files will be tested using wc
*    When initializing layer weights, use tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG").

---
### [0. L2 Regularization Cost](./0-l2_reg_cost.py)
Write a function def l2_reg_cost(cost, lambtha, weights, L, m): that calculates the cost of a neural network with L2 regularization:
*    cost is the cost of the network without L2 regularization
*    lambtha is the regularization parameter
*    weights is a dictionary of the weights and biases (numpy.ndarrays) of the neural network
*    L is the number of layers in the neural network
*    m is the number of data points used
*    Returns: the cost of the network accounting for L2 regularization

ubuntu@alexa-ml:~/0x05-regularization$ cat 0-main.py 
#!/usr/bin/env python3

import numpy as np
l2_reg_cost = __import__('0-l2_reg_cost').l2_reg_cost

if __name__ == '__main__':
    np.random.seed(0)

    weights = {}
    weights['W1'] = np.random.randn(256, 784)
    weights['W2'] = np.random.randn(128, 256)
    weights['W3'] = np.random.randn(10, 128)

    cost = np.abs(np.random.randn(1))

    print(cost)
    cost = l2_reg_cost(cost, 0.1, weights, 3, 1000)
    print(cost)
ubuntu@alexa-ml:~/0x05-regularization$ ./0-main.py 
[0.41842822]
[0.45158952]
ubuntu@alexa-ml:~/0x05-regularization$ 


### [1. Gradient Descent with L2 Regularization](./1-l2_reg_gradient_descent.py)
Write a function def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L): that updates the weights and biases of a neural network using gradient descent with L2 regularization:
*    Y is a one-hot numpy.ndarray of shape (classes, m) that contains the correct labels for the data
*        classes is the number of classes
*        m is the number of data points
*    weights is a dictionary of the weights and biases of the neural network
*    cache is a dictionary of the outputs of each layer of the neural network
*    alpha is the learning rate
*    lambtha is the L2 regularization parameter
*    L is the number of layers of the network
*    The neural network uses tanh activations on each layer except the last, which uses a softmax activation
*    The weights and biases of the network should be updated in place

ubuntu@alexa-ml:~/0x05-regularization$ cat 1-main.py
#!/usr/bin/env python3

import numpy as np
l2_reg_gradient_descent = __import__('1-l2_reg_gradient_descent').l2_reg_gradient_descent


def one_hot(Y, classes):
    """convert an array to a one-hot matrix"""
    m = Y.shape[0]
    one_hot = np.zeros((classes, m))
    one_hot[Y, np.arange(m)] = 1
    return one_hot

if __name__ == '__main__':
    lib= np.load('../data/MNIST.npz')
    X_train_3D = lib['X_train']
    Y_train = lib['Y_train']
    X_train = X_train_3D.reshape((X_train_3D.shape[0], -1)).T
    Y_train_oh = one_hot(Y_train, 10)

    np.random.seed(0)

    weights = {}
    weights['W1'] = np.random.randn(256, 784)
    weights['b1'] = np.zeros((256, 1))
    weights['W2'] = np.random.randn(128, 256)
    weights['b2'] = np.zeros((128, 1))
    weights['W3'] = np.random.randn(10, 128)
    weights['b3'] = np.zeros((10, 1))

    cache = {}
    cache['A0'] = X_train
    cache['A1'] = np.tanh(np.matmul(weights['W1'], cache['A0']) + weights['b1'])
    cache['A2'] = np.tanh(np.matmul(weights['W2'], cache['A1']) + weights['b2'])
    Z3 = np.matmul(weights['W3'], cache['A2']) + weights['b3']
    cache['A3'] = np.exp(Z3) / np.sum(np.exp(Z3), axis=0)
    print(weights['W1'])
    l2_reg_gradient_descent(Y_train_oh, weights, cache, 0.1, 0.1, 3)
    print(weights['W1'])
ubuntu@alexa-ml:~/0x05-regularization$ ./1-main.py
[[ 1.76405235  0.40015721  0.97873798 ...  0.52130375  0.61192719
  -1.34149673]
 [ 0.47689837  0.14844958  0.52904524 ...  0.0960042  -0.0451133
   0.07912172]
 [ 0.85053068 -0.83912419 -1.01177408 ... -0.07223876  0.31112445
  -1.07836109]
 ...
 [-0.60467085  0.54751161 -1.23317415 ...  0.82895532  1.44161136
   0.18972404]
 [-0.41044606  0.85719512  0.71789835 ... -0.73954771  0.5074628
   1.23022874]
 [ 0.43129249  0.60767018 -0.07749988 ... -0.26611561  2.52287972
   0.73131543]]
[[ 1.76405199  0.40015713  0.97873779 ...  0.52130364  0.61192707
  -1.34149646]
 [ 0.47689827  0.14844955  0.52904513 ...  0.09600419 -0.04511329
   0.07912171]
 [ 0.85053051 -0.83912402 -1.01177388 ... -0.07223874  0.31112438
  -1.07836088]
 ...
 [-0.60467073  0.5475115  -1.2331739  ...  0.82895516  1.44161107
   0.189724  ]
 [-0.41044598  0.85719495  0.71789821 ... -0.73954756  0.5074627
   1.2302285 ]
 [ 0.4312924   0.60767006 -0.07749987 ... -0.26611556  2.52287922
   0.73131529]]
ubuntu@alexa-ml:~/0x05-regularization$


### [2. L2 Regularization Cost](./2-l2_reg_cost.py)
Write the function def l2_reg_cost(cost): that calculates the cost of a neural network with L2 regularization:
*    cost is a tensor containing the cost of the network without L2 regularization
*    Returns: a tensor containing the cost of the network accounting for L2 regularization


### [3. Create a Layer with L2 Regularization](./3-l2_reg_create_layer.py)
Write a function def l2_reg_create_layer(prev, n, activation, lambtha): that creates a tensorflow layer that includes L2 regularization:
*    prev is a tensor containing the output of the previous layer
*    n is the number of nodes the new layer should contain
*    activation is the activation function that should be used on the layer
*    lambtha is the L2 regularization parameter
*    Returns: the output of the new layer

ubuntu@alexa-ml:~/0x05-regularization$ cat 3-main.py 
#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
l2_reg_cost = __import__('2-l2_reg_cost').l2_reg_cost
l2_reg_create_layer = __import__('3-l2_reg_create_layer').l2_reg_create_layer

def one_hot(Y, classes):
    """convert an array to a one-hot matrix"""
    m = Y.shape[0]
    one_hot = np.zeros((m, classes))
    one_hot[np.arange(m), Y] = 1
    return one_hot

if __name__ == '__main__':
    lib= np.load('../data/MNIST.npz')
    X_train_3D = lib['X_train']
    Y_train = lib['Y_train']
    X_train = X_train_3D.reshape((X_train_3D.shape[0], -1))
    Y_train_oh = one_hot(Y_train, 10)

    tf.set_random_seed(0)
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y = tf.placeholder(tf.float32, shape=[None, 10])
    h1 = l2_reg_create_layer(x, 256, tf.nn.tanh, 0.1)
    y_pred = l2_reg_create_layer(x, 10, None, 0.)
    cost = tf.losses.softmax_cross_entropy(y, y_pred)
    l2_cost = l2_reg_cost(cost)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(l2_cost, feed_dict={x: X_train, y: Y_train_oh}))
ubuntu@alexa-ml:~/0x05-regularization$ ./3-main.py 
2018-11-26 19:18:37.910515: I tensorflow/core/common_runtime/process_util.cc:69] Creating new thread pool with default inter op setting: 2. Tune using inter_op_parallelism_threads for best performance.
[41.243492]
ubuntu@alexa-ml:~/0x05-regularization$


### [4. Forward Propagation with Dropout](./4-dropout_forward_prop.py)
Write a function def dropout_forward_prop(X, weights, L, keep_prob): that conducts forward propagation using Dropout:
*    X is a numpy.ndarray of shape (nx, m) containing the input data for the network
*        nx is the number of input features
*        m is the number of data points
*    weights is a dictionary of the weights and biases of the neural network
*    L the number of layers in the network
*    keep_prob is the probability that a node will be kept
*    All layers except the last should use the tanh activation function
*    The last layer should use the softmax activation function
*    Returns: a dictionary containing the outputs of each layer and the dropout mask used on each layer (see example for format)

ubuntu@alexa-ml:~/0x05-regularization$ cat 4-main.py 
#!/usr/bin/env python3

import numpy as np
dropout_forward_prop = __import__('4-dropout_forward_prop').dropout_forward_prop


def one_hot(Y, classes):
    """convert an array to a one-hot matrix"""
    m = Y.shape[0]
    one_hot = np.zeros((classes, m))
    one_hot[Y, np.arange(m)] = 1
    return one_hot

if __name__ == '__main__':
    lib= np.load('../data/MNIST.npz')
    X_train_3D = lib['X_train']
    Y_train = lib['Y_train']
    X_train = X_train_3D.reshape((X_train_3D.shape[0], -1)).T
    Y_train_oh = one_hot(Y_train, 10)

    np.random.seed(0)

    weights = {}
    weights['W1'] = np.random.randn(256, 784)
    weights['b1'] = np.zeros((256, 1))
    weights['W2'] = np.random.randn(128, 256)
    weights['b2'] = np.zeros((128, 1))
    weights['W3'] = np.random.randn(10, 128)
    weights['b3'] = np.zeros((10, 1))

    cache = dropout_forward_prop(X_train, weights, 3, 0.8)
    for k, v in sorted(cache.items()):
        print(k, v)
ubuntu@alexa-ml:~/0x05-regularization$ ./4-main.py
A0 [[0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]
 ...
 [0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]]
A1 [[-1.24999999 -1.25       -1.24999945 ... -1.25       -1.25
  -1.25      ]
 [ 1.25        1.24999777  1.25       ...  0.37738875  1.24999717
  -1.24999889]
 [ 0.19383179 -0.80653094 -1.24950714 ...  1.24253535  1.08653948
  -1.20190135]
 ...
 [-1.25       -1.25        0.         ... -0.         -1.25
  -1.24999852]
 [-1.0858595  -1.25        0.         ...  1.24972487 -0.88878698
  -1.24999933]
 [ 1.25        1.24999648  0.2057473  ...  0.          1.23194191
  -1.24908257]]
A2 [[-1.25        0.          1.24985922 ... -1.25        0.
   1.24996854]
 [-0.         -0.         -0.         ... -1.24996232 -0.70684864
   1.25      ]
 [-1.25        0.          0.18486152 ... -1.24999999 -1.25
  -1.24999989]
 ...
 [ 1.2404131   1.25        1.25       ...  1.1670038   1.25
  -0.        ]
 [ 1.25        1.25       -1.24998041 ...  1.2400913  -1.25
   1.23620006]
 [ 0.93426582  1.25        1.25       ...  1.24999867 -1.25
  -0.        ]]
A3 [[9.13222086e-07 1.53352996e-09 4.02988574e-13 ... 2.93685964e-04
  2.21615443e-11 7.95945899e-04]
 [4.10709405e-16 4.27810333e-11 7.38725096e-07 ... 2.05423847e-17
  2.66482686e-09 1.74341031e-12]
 [9.82953561e-01 9.88655425e-01 9.73580864e-01 ... 1.14493065e-03
  9.28074126e-10 1.92423905e-13]
 ...
 [3.03047424e-04 1.11981605e-02 4.72284535e-05 ... 1.25781567e-20
  9.57462819e-01 3.33328605e-13]
 [3.20689297e-11 7.42324257e-08 5.62529910e-19 ... 2.05682936e-16
  1.07622653e-12 1.41200115e-02]
 [5.06603174e-06 8.50852457e-11 5.51467429e-10 ... 9.98493133e-01
  1.97896353e-14 2.38078250e-05]]
D1 [[1 1 1 ... 1 1 1]
 [1 1 1 ... 1 1 1]
 [1 1 1 ... 1 1 1]
 ...
 [1 1 0 ... 0 1 1]
 [1 1 0 ... 1 1 1]
 [1 1 1 ... 0 1 1]]
D2 [[1 0 1 ... 1 0 1]
 [0 0 0 ... 1 1 1]
 [1 0 1 ... 1 1 1]
 ...
 [1 1 1 ... 1 1 0]
 [1 1 1 ... 1 1 1]
 [1 1 1 ... 1 1 0]]
ubuntu@alexa-ml:~/0x05-regularization$


### [5. Gradient Descent with Dropout](./5-dropout_gradient_descent.py)
Write a function def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L): that updates the weights of a neural network with Dropout regularization using gradient descent:
*    Y is a one-hot numpy.ndarray of shape (classes, m) that contains the correct labels for the data
*        classes is the number of classes
*        m is the number of data points
*    weights is a dictionary of the weights and biases of the neural network
*    cache is a dictionary of the outputs and dropout masks of each layer of the neural network
*    alpha is the learning rate
*    keep_prob is the probability that a node will be kept
*    L is the number of layers of the network
*    All layers use thetanh activation function except the last, which uses the softmax activation function
*    The weights of the network should be updated in place

ubuntu@alexa-ml:~/0x05-regularization$ cat 5-main.py
#!/usr/bin/env python3

import numpy as np
dropout_forward_prop = __import__('4-dropout_forward_prop').dropout_forward_prop
dropout_gradient_descent = __import__('5-dropout_gradient_descent').dropout_gradient_descent


def one_hot(Y, classes):
    """convert an array to a one-hot matrix"""
    m = Y.shape[0]
    one_hot = np.zeros((classes, m))
    one_hot[Y, np.arange(m)] = 1
    return one_hot

if __name__ == '__main__':
    lib= np.load('../data/MNIST.npz')
    X_train_3D = lib['X_train']
    Y_train = lib['Y_train']
    X_train = X_train_3D.reshape((X_train_3D.shape[0], -1)).T
    Y_train_oh = one_hot(Y_train, 10)

    np.random.seed(0)

    weights = {}
    weights['W1'] = np.random.randn(256, 784)
    weights['b1'] = np.zeros((256, 1))
    weights['W2'] = np.random.randn(128, 256)
    weights['b2'] = np.zeros((128, 1))
    weights['W3'] = np.random.randn(10, 128)
    weights['b3'] = np.zeros((10, 1))

    cache = dropout_forward_prop(X_train, weights, 3, 0.8)
    print(weights['W2'])
    dropout_gradient_descent(Y_train_oh, weights, cache, 0.1, 0.8, 3)
    print(weights['W2'])
ubuntu@alexa-ml:~/0x05-regularization$ ./5-main.py
[[-1.9282086  -0.71324613 -1.33191318 ... -2.14202626 -0.07737407
   0.99832167]
 [-0.0237149  -0.18364778  0.08337452 ... -0.06093055 -0.03924408
  -2.17625294]
 [-0.16181888  0.49237435 -0.47196279 ...  0.97504077  0.16272698
   0.56159916]
 ...
 [ 0.39842474 -0.09870005  1.32173992 ... -0.33210834  0.66215988
   0.87211421]
 [ 0.15767221  0.42236212  1.004765   ...  0.69883284  0.70857088
  -0.44427252]
 [ 2.68588811 -0.60351958 -1.0759598  ... -1.2437044   0.69462324
   1.00090403]]
[[-1.92044686 -0.71894673 -1.32811693 ... -2.14071955 -0.07158198
   0.98206832]
 [-0.03706116 -0.17088483  0.07798748 ... -0.07245569 -0.0491215
  -2.16245276]
 [-0.17198668  0.49842244 -0.47369328 ...  0.96880194  0.15497217
   0.5693131 ]
 ...
 [ 0.41997262 -0.11452751  1.32873227 ... -0.31312321  0.67162237
   0.85928296]
 [ 0.13702353  0.44237056  1.00139188 ...  0.68128208  0.69020934
  -0.43055442]
 [ 2.66514017 -0.59204122 -1.08943163 ... -1.26238074  0.69280683
   1.02353101]]
ubuntu@alexa-ml:~/0x05-regularization$


### [6. Create a Layer with Dropout](./6-dropout_create_layer.py)
Write a function def dropout_create_layer(prev, n, activation, keep_prob): that creates a layer of a neural network using dropout:
*    prev is a tensor containing the output of the previous layer
*    n is the number of nodes the new layer should contain
*    activation is the activation function that should be used on the layer
*    keep_prob is the probability that a node will be kept
*    Returns: the output of the new layer

ubuntu@alexa-ml:~/0x05-regularization$ cat 6-main.py
#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
dropout_create_layer = __import__('6-dropout_create_layer').dropout_create_layer

if __name__ == '__main__':
    tf.set_random_seed(0)
    np.random.seed(0)
    x = tf.placeholder(tf.float32, shape=[None, 784])
    X = np.random.randint(0, 256, size=(10, 784))
    a = dropout_create_layer(x, 256, tf.nn.tanh, 0.8)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(a, feed_dict={x: X}))
ubuntu@alexa-ml:~/0x05-regularization$ ./6-main.py
2018-11-26 21:00:33.541659: I tensorflow/core/common_runtime/process_util.cc:69] Creating new thread pool with default inter op setting: 2. Tune using inter_op_parallelism_threads for best performance.
[[-1.         -1.          1.         ... -1.          0.9999985
   1.        ]
 [ 1.         -1.          1.         ... -1.          1.
   1.        ]
 [-1.         -1.          1.         ...  1.          1.
   1.        ]
 ...
 [ 1.         -1.         -1.         ... -0.99999994 -1.
   1.        ]
 [ 1.         -1.          1.         ...  1.         -0.8955073
   1.        ]
 [ 1.         -1.          1.         ...  1.          1.
  -1.        ]]
ubuntu@alexa-ml:~/0x05-regularization$ 


### [7. Early Stopping](./7-early_stopping.py)
Write the function def early_stopping(cost, opt_cost, threshold, patience, count): that determines if you should stop gradient descent early:
*    Early stopping should occur when the validation cost of the network has not decreased relative to the optimal validation cost by more than the threshold over a specific patience count
*    cost is the current validation cost of the neural network
*    opt_cost is the lowest recorded validation cost of the neural network
*    threshold is the threshold used for early stopping
*    patience is the patience count used for early stopping
*    count is the count of how long the threshold has not been met
*    Returns: a boolean of whether the network should be stopped early, followed by the updated count

ubuntu@alexa-ml:~/0x05-regularization$ cat 7-main.py 
#!/usr/bin/env python3

early_stopping = __import__('7-early_stopping').early_stopping

if __name__ == '__main__':
    print(early_stopping(1.0, 1.9, 0.5, 15, 5))
    print(early_stopping(1.1, 1.5, 0.5, 15, 2))
    print(early_stopping(1.0, 1.5, 0.5, 15, 8))
    print(early_stopping(1.0, 1.5, 0.5, 15, 14))
ubuntu@alexa-ml:~/0x05-regularization$ ./7-main.py 
(False, 0)
(False, 3)
(False, 9)
(True, 15)
ubuntu@alexa-ml:~/0x05-regularization$

---
## Authors

* **Robinson Montes** - [mecomonteshbtn](https://github.com/mecomonteshbtn)
