# 0x06. Keras

---
## Read or watch:
*    [Keras Explained](https://www.youtube.com/watch?v=j_pJmXJwMLA&feature=youtu.be&t=228)
*    [Keras](https://www.tensorflow.org/guide/keras/sequential_model)
*    [Hierarchical Data Format](https://en.wikipedia.org/wiki/Hierarchical_Data_Format)
*    [Keras with tensorflow video](https://www.youtube.com/watch?v=qFJeN9V1ZsI)

## References:
*    [tf.keras](https://github.com/tensorflow/docs/tree/r1.12/site/en/api_docs/python/tf/keras)
*        [tf.keras.models](https://github.com/tensorflow/docs/tree/r1.12/site/en/api_docs/python/tf/keras/models)
*        [tf.keras.activations](https://github.com/tensorflow/docs/tree/r1.12/site/en/api_docs/python/tf/keras/activations)
*        [tf.keras.callbacks](https://github.com/tensorflow/docs/tree/r1.12/site/en/api_docs/python/tf/keras/callbacks)
*        [tf.keras.initializers](https://github.com/tensorflow/docs/tree/r1.12/site/en/api_docs/python/tf/keras/initializers)
*        [tf.keras.layers](https://github.com/tensorflow/docs/tree/r1.12/site/en/api_docs/python/tf/keras/layers)
*        [tf.keras.losses](https://github.com/tensorflow/docs/tree/r1.12/site/en/api_docs/python/tf/keras/losses)
*        [tf.keras.metrics](https://github.com/tensorflow/docs/tree/r1.12/site/en/api_docs/python/tf/keras/metrics)
*        [tf.keras.optimizers](https://github.com/tensorflow/docs/tree/r1.12/site/en/api_docs/python/tf/keras/optimizers)
*        [tf.keras.regularizers](https://github.com/tensorflow/docs/tree/r1.12/site/en/api_docs/python/tf/keras/regularizers)
*        [tf.keras.utils](https://github.com/tensorflow/docs/tree/r1.12/site/en/api_docs/python/tf/keras/utils)

---
## General Requirements
*    Allowed editors: vi, vim, emacs
*    All your files will be interpreted/compiled on Ubuntu 16.04 LTS using python3 (version 3.5)
*    Your files will be executed with numpy (version 1.15) and tensorflow (version 1.12)
*    All your files should end with a new line
*    The first line of all your files should be exactly #!/usr/bin/env python3
*    A README.md file, at the root of the folder of the project, is mandatory
*    Your code should use the pycodestyle style (version 2.4)
*    All your modules should have documentation (python3 -c 'print(__import__("my_module").__doc__)')
*    All your classes should have documentation (python3 -c 'print(__import__("my_module").MyClass.__doc__)')
*    All your functions (inside and outside a class) should have documentation (python3 -c 'print(__import__("my_module").my_function.__doc__)' and python3 -c 'print(__import__("my_module").MyClass.my_function.__doc__)')
*    Unless otherwise noted, you are not allowed to import any module except import tensorflow.keras as K
*    All your files must be executable
*    The length of your files will be tested using wc

---
### [0. Sequential](./0-sequential.py)
Write a function def build_model(nx, layers, activations, lambtha, keep_prob): that builds a neural network with the Keras library:
*    nx is the number of input features to the network
*    layers is a list containing the number of nodes in each layer of the network
*    activations is a list containing the activation functions used for each layer of the network
*    lambtha is the L2 regularization parameter
*    keep_prob is the probability that a node will be kept for dropout
*    You are not allowed to use the Input class
*    Returns: the keras model
```
ubuntu@alexa-ml:~/0x06-keras$ cat 0-main.py 
#!/usr/bin/env python3

build_model = __import__('0-sequential').build_model

if __name__ == '__main__':
    network = build_model(784, [256, 256, 10], ['tanh', 'tanh', 'softmax'], 0.001, 0.95)
    network.summary()
    print(network.losses)
ubuntu@alexa-ml:~/0x06-keras$ ./0-main.py 
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                (None, 256)               200960    
_________________________________________________________________
dropout (Dropout)            (None, 256)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 256)               65792     
_________________________________________________________________
dropout_1 (Dropout)          (None, 256)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 10)                2570      
=================================================================
Total params: 269,322
Trainable params: 269,322
Non-trainable params: 0
_________________________________________________________________
[<tf.Tensor 'dense/kernel/Regularizer/add:0' shape=() dtype=float32>, <tf.Tensor 'dense_1/kernel/Regularizer/add:0' shape=() dtype=float32>, <tf.Tensor 'dense_2/kernel/Regularizer/add:0' shape=() dtype=float32>]
ubuntu@alexa-ml:~/0x06-keras$
```

### [1. Input](./1-input.py)
Write a function def build_model(nx, layers, activations, lambtha, keep_prob): that builds a neural network with the Keras library:
*    nx is the number of input features to the network
*    layers is a list containing the number of nodes in each layer of the network
*    activations is a list containing the activation functions used for each layer of the network
*    lambtha is the L2 regularization parameter
*    keep_prob is the probability that a node will be kept for dropout
*    You are not allowed to use the Sequential class
*    Returns: the keras model

ubuntu@alexa-ml:~/0x06-keras$ cat 1-main.py 
#!/usr/bin/env python3

build_model = __import__('1-input').build_model

if __name__ == '__main__':
    network = build_model(784, [256, 256, 10], ['tanh', 'tanh', 'softmax'], 0.001, 0.95)
    network.summary()
    print(network.losses)
ubuntu@alexa-ml:~/0x06-keras$ ./1-main.py 
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 784)               0         
_________________________________________________________________
dense (Dense)                (None, 256)               200960    
_________________________________________________________________
dropout (Dropout)            (None, 256)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 256)               65792     
_________________________________________________________________
dropout_1 (Dropout)          (None, 256)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 10)                2570      
=================================================================
Total params: 269,322
Trainable params: 269,322
Non-trainable params: 0
_________________________________________________________________
[<tf.Tensor 'dense/kernel/Regularizer/add:0' shape=() dtype=float32>, <tf.Tensor 'dense_2/kernel/Regularizer/add:0' shape=() dtype=float32>, <tf.Tensor 'dense_1/kernel/Regularizer/add:0' shape=() dtype=float32>]
ubuntu@alexa-ml:~/0x06-keras$


### [2. Optimize](./2-optimize.py)
Write a function def optimize_model(network, alpha, beta1, beta2): that sets up Adam optimization for a keras model with categorical crossentropy loss and accuracy metrics:
*    network is the model to optimize
*    alpha is the learning rate
*    beta1 is the first Adam optimization parameter
*    beta2 is the second Adam optimization parameter
*    Returns: None


### [3. One Hot](./3-one_hot.py)
Write a function def one_hot(labels, classes=None): that converts a label vector into a one-hot matrix:
*    The last dimension of the one-hot matrix must be the number of classes
*    Returns: the one-hot matrix

ubuntu@alexa-ml:~/0x06-keras$ cat 3-main.py 
#!/usr/bin/env python3

import numpy as np
one_hot = __import__('3-one_hot').one_hot

if __name__ == '__main__':
    labels = np.load('../data/MNIST.npz')['Y_train'][:10]
    print(labels)
    print(one_hot(labels))   
ubuntu@alexa-ml:~/0x06-keras$ ./3-main.py 
[5 0 4 1 9 2 1 3 1 4]
[[0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
 [0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]]
ubuntu@alexa-ml:~/0x06-keras$


### [4. Train](./4-train.py)
Write a function def train_model(network, data, labels, batch_size, epochs, verbose=True, shuffle=False): that trains a model using mini-batch gradient descent:
*    network is the model to train
*    data is a numpy.ndarray of shape (m, nx) containing the input data
*    labels is a one-hot numpy.ndarray of shape (m, classes) containing the labels of data
*    batch_size is the size of the batch used for mini-batch gradient descent
*    epochs is the number of passes through data for mini-batch gradient descent
*    verbose is a boolean that determines if output should be printed during training
*    shuffle is a boolean that determines whether to shuffle the batches every epoch. Normally, it is a good idea to shuffle, but for reproducibility, we have chosen to set the default to False.
*    Returns: the History object generated after training the model

ubuntu@alexa-ml:~/0x06-keras$ cat 4-main.py
#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
build_model = __import__('1-input').build_model
optimize_model = __import__('2-optimize').optimize_model
one_hot = __import__('3-one_hot').one_hot
train_model = __import__('4-train').train_model


if __name__ == '__main__':
    datasets = np.load('../data/MNIST.npz')
    X_train = datasets['X_train']
    X_train = X_train.reshape(X_train.shape[0], -1)
    Y_train = datasets['Y_train']
    Y_train_oh = one_hot(Y_train)

    np.random.seed(0)
    tf.set_random_seed(0)
    lambtha = 0.0001
    keep_prob = 0.95
    network = build_model(784, [256, 256, 10], ['relu', 'relu', 'softmax'], lambtha, keep_prob)
    alpha = 0.001
    beta1 = 0.9
    beta2 = 0.999
    optimize_model(network, alpha, beta1, beta2)
    batch_size = 64
    epochs = 5
    train_model(network, X_train, Y_train_oh, batch_size, epochs)
ubuntu@alexa-ml:~/0x06-keras$ ./4-main.py
Epoch 1/5
50000/50000 [==============================] - 4s 88us/step - loss: 0.3521 - acc: 0.9206
Epoch 2/5
50000/50000 [==============================] - 4s 86us/step - loss: 0.1973 - acc: 0.9659
Epoch 3/5
50000/50000 [==============================] - 4s 77us/step - loss: 0.1596 - acc: 0.9755
Epoch 4/5
50000/50000 [==============================] - 5s 92us/step - loss: 0.1388 - acc: 0.9805
Epoch 5/5
50000/50000 [==============================] - 5s 96us/step - loss: 0.1244 - acc: 0.9838
ubuntu@alexa-ml:~/0x06-keras$


### [5. Validate](./5-train.py)
Based on 4-train.py, update the function def train_model(network, data, labels, batch_size, epochs, validation_data=None, verbose=True, shuffle=False): to also analyze validaiton data:
*    validation_data is the data to validate the model with, if not None

ubuntu@alexa-ml:~/0x06-keras$ cat 5-main.py 
#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
build_model = __import__('1-input').build_model
optimize_model = __import__('2-optimize').optimize_model
one_hot = __import__('3-one_hot').one_hot
train_model = __import__('5-train').train_model


if __name__ == '__main__':
    datasets = np.load('../data/MNIST.npz')
    X_train = datasets['X_train']
    X_train = X_train.reshape(X_train.shape[0], -1)
    Y_train = datasets['Y_train']
    Y_train_oh = one_hot(Y_train)
    X_valid = datasets['X_valid']
    X_valid = X_valid.reshape(X_valid.shape[0], -1)
    Y_valid = datasets['Y_valid']
    Y_valid_oh = one_hot(Y_valid)

    np.random.seed(0)
    tf.set_random_seed(0)
    lambtha = 0.0001
    keep_prob = 0.95
    network = build_model(784, [256, 256, 10], ['relu', 'relu', 'softmax'], lambtha, keep_prob)
    alpha = 0.001
    beta1 = 0.9
    beta2 = 0.999
    optimize_model(network, alpha, beta1, beta2)
    batch_size = 64
    epochs = 5
    train_model(network, X_train, Y_train_oh, batch_size, epochs, validation_data=(X_valid, Y_valid_oh))
ubuntu@alexa-ml:~/0x06-keras$ ./5-main.py 
Train on 50000 samples, validate on 10000 samples
Epoch 1/5
50000/50000 [==============================] - 4s 78us/step - loss: 0.3521 - acc: 0.9206 - val_loss: 0.2151 - val_acc: 0.9613
Epoch 2/5
50000/50000 [==============================] - 4s 79us/step - loss: 0.1975 - acc: 0.9658 - val_loss: 0.1777 - val_acc: 0.9702
Epoch 3/5
50000/50000 [==============================] - 4s 75us/step - loss: 0.1594 - acc: 0.9753 - val_loss: 0.1657 - val_acc: 0.9733
Epoch 4/5
50000/50000 [==============================] - 4s 78us/step - loss: 0.1386 - acc: 0.9803 - val_loss: 0.1768 - val_acc: 0.9690
Epoch 5/5
50000/50000 [==============================] - 4s 76us/step - loss: 0.1259 - acc: 0.9836 - val_loss: 0.1558 - val_acc: 0.9758
ubuntu@alexa-ml:~/0x06-keras$


### [6. Early Stopping](./6-train.py)
Based on 5-train.py, update the function def train_model(network, data, labels, batch_size, epochs, validation_data=None, early_stopping=False, patience=0, verbose=True, shuffle=False): to also train the model using early stopping:
*    early_stopping is a boolean that indicates whether early stopping should be used
*        early stopping should only be performed if validation_data exists
*        early stopping should be based on validation loss
*    patience is the patience used for early stopping

ubuntu@alexa-ml:~/0x06-keras$ cat 6-main.py 
#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
build_model = __import__('1-input').build_model
optimize_model = __import__('2-optimize').optimize_model
one_hot = __import__('3-one_hot').one_hot
train_model = __import__('6-train').train_model


if __name__ == '__main__':
    datasets = np.load('../data/MNIST.npz')
    X_train = datasets['X_train']
    X_train = X_train.reshape(X_train.shape[0], -1)
    Y_train = datasets['Y_train']
    Y_train_oh = one_hot(Y_train)
    X_valid = datasets['X_valid']
    X_valid = X_valid.reshape(X_valid.shape[0], -1)
    Y_valid = datasets['Y_valid']
    Y_valid_oh = one_hot(Y_valid)

    np.random.seed(0)
    tf.set_random_seed(0)
    lambtha = 0.0001
    keep_prob = 0.95
    network = build_model(784, [256, 256, 10], ['relu', 'relu', 'softmax'], lambtha, keep_prob)
    alpha = 0.001
    beta1 = 0.9
    beta2 = 0.999
    optimize_model(network, alpha, beta1, beta2)
    batch_size = 64
    epochs = 30
    train_model(network, X_train, Y_train_oh, batch_size, epochs,
                validation_data=(X_valid, Y_valid_oh), early_stopping=True,
                patience=3)
ubuntu@alexa-ml:~/0x06-keras$ ./6-main.py 
Train on 50000 samples, validate on 10000 samples
Epoch 1/30
50000/50000 [==============================] - 4s 90us/step - loss: 0.3521 - acc: 0.9206 - val_loss: 0.2151 - val_acc: 0.9613
Epoch 2/30
50000/50000 [==============================] - 4s 83us/step - loss: 0.1973 - acc: 0.9659 - val_loss: 0.1761 - val_acc: 0.9706
Epoch 3/30
50000/50000 [==============================] - 5s 95us/step - loss: 0.1596 - acc: 0.9755 - val_loss: 0.1617 - val_acc: 0.9729

...

Epoch 16/30
50000/50000 [==============================] - 5s 102us/step - loss: 0.0884 - acc: 0.9907 - val_loss: 0.1399 - val_acc: 0.9774
ubuntu@alexa-ml:~/0x06-keras$


### [7. Learning Rate Decay](./7-train.py)
Based on 6-train.py, update the function def train_model(network, data, labels, batch_size, epochs, validation_data=None, early_stopping=False, patience=0, learning_rate_decay=False, alpha=0.1, decay_rate=1, verbose=True, shuffle=False): to also train the model with learning rate decay:

    learning_rate_decay is a boolean that indicates whether learning rate decay should be used
        learning rate decay should only be performed if validation_data exists
        the decay should be performed using inverse time decay
        the learning rate should decay in a stepwise fashion after each epoch
        each time the learning rate updates, Keras should print a message
    alpha is the initial learning rate
    decay_rate is the decay rate

ubuntu@alexa-ml:~/0x06-keras$ cat 7-main.py 
#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
build_model = __import__('1-input').build_model
optimize_model = __import__('2-optimize').optimize_model
one_hot = __import__('3-one_hot').one_hot
train_model = __import__('7-train').train_model


if __name__ == '__main__':
    datasets = np.load('../data/MNIST.npz')
    X_train = datasets['X_train']
    X_train = X_train.reshape(X_train.shape[0], -1)
    Y_train = datasets['Y_train']
    Y_train_oh = one_hot(Y_train)
    X_valid = datasets['X_valid']
    X_valid = X_valid.reshape(X_valid.shape[0], -1)
    Y_valid = datasets['Y_valid']
    Y_valid_oh = one_hot(Y_valid)

    np.random.seed(0)
    tf.set_random_seed(0)
    lambtha = 0.0001
    keep_prob = 0.95
    network = build_model(784, [256, 256, 10], ['relu', 'relu', 'softmax'], lambtha, keep_prob)
    alpha = 0.001
    beta1 = 0.9
    beta2 = 0.999
    optimize_model(network, alpha, beta1, beta2)
    batch_size = 64
    epochs = 1000
    train_model(network, X_train, Y_train_oh, batch_size, epochs,
                validation_data=(X_valid, Y_valid_oh), early_stopping=True,
                patience=3, learning_rate_decay=True, alpha=alpha)
ubuntu@alexa-ml:~/0x06-keras$ ./7-main.py
Train on 50000 samples, validate on 10000 samples

Epoch 00001: LearningRateScheduler reducing learning rate to 0.001.
Epoch 1/1000
50000/50000 [==============================] - 4s 78us/step - loss: 0.3521 - acc: 0.9206 - val_loss: 0.2151 - val_acc: 0.9613

Epoch 00002: LearningRateScheduler reducing learning rate to 0.0005.
Epoch 2/1000
50000/50000 [==============================] - 4s 73us/step - loss: 0.1826 - acc: 0.9700 - val_loss: 0.1682 - val_acc: 0.9742

Epoch 00003: LearningRateScheduler reducing learning rate to 0.0003333333333333333.
Epoch 3/1000
50000/50000 [==============================] - 4s 75us/step - loss: 0.1484 - acc: 0.9794 - val_loss: 0.1558 - val_acc: 0.9765


...

Epoch 00084: LearningRateScheduler reducing learning rate to 1.1904761904761905e-05.
Epoch 84/1000
50000/50000 [==============================] - 4s 81us/step - loss: 0.0483 - acc: 0.9994 - val_loss: 0.1005 - val_acc: 0.9828
ubuntu@alexa-ml:~/0x06-keras$


### [8. Save Only the Best](./8-train.py)
Based on 7-train.py, update the function def train_model(network, data, labels, batch_size, epochs, validation_data=None, early_stopping=False, patience=0, learning_rate_decay=False, alpha=0.1, decay_rate=1, save_best=False, filepath=None, verbose=True, shuffle=False): to also save the best iteration of the model:
*    save_best is a boolean indicating whether to save the model after each epoch if it is the best
*        a model is considered the best if its validation loss is the lowest that the model has obtained
*    filepath is the file path where the model should be saved

ubuntu@alexa-ml:~/0x06-keras$ cat 8-main.py 
#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
build_model = __import__('1-input').build_model
optimize_model = __import__('2-optimize').optimize_model
one_hot = __import__('3-one_hot').one_hot
train_model = __import__('8-train').train_model


if __name__ == '__main__':
    datasets = np.load('../data/MNIST.npz')
    X_train = datasets['X_train']
    X_train = X_train.reshape(X_train.shape[0], -1)
    Y_train = datasets['Y_train']
    Y_train_oh = one_hot(Y_train)
    X_valid = datasets['X_valid']
    X_valid = X_valid.reshape(X_valid.shape[0], -1)
    Y_valid = datasets['Y_valid']
    Y_valid_oh = one_hot(Y_valid)

    np.random.seed(0)
    tf.set_random_seed(0)
    lambtha = 0.0001
    keep_prob = 0.95
    network = build_model(784, [256, 256, 10], ['relu', 'relu', 'softmax'], lambtha, keep_prob)
    alpha = 0.001
    beta1 = 0.9
    beta2 = 0.999
    optimize_model(network, alpha, beta1, beta2)
    batch_size = 64
    epochs = 1000
    train_model(network, X_train, Y_train_oh, batch_size, epochs,
                validation_data=(X_valid, Y_valid_oh), early_stopping=True,
                patience=3, learning_rate_decay=True, alpha=alpha,
                save_best=True, filepath='network1.h5')
ubuntu@alexa-ml:~/0x06-keras$ ./8-main.py 
Train on 50000 samples, validate on 10000 samples
Epoch 00001: LearningRateScheduler reducing learning rate to 0.001.
Epoch 1/1000
50000/50000 [==============================] - 4s 83us/step - loss: 0.3521 - acc: 0.9206 - val_loss: 0.2151 - val_acc: 0.9613

Epoch 00002: LearningRateScheduler reducing learning rate to 0.0005.
Epoch 2/1000
50000/50000 [==============================] - 4s 77us/step - loss: 0.1826 - acc: 0.9700 - val_loss: 0.1682 - val_acc: 0.9742

Epoch 00003: LearningRateScheduler reducing learning rate to 0.0003333333333333333.
Epoch 3/1000
50000/50000 [==============================] - 4s 76us/step - loss: 0.1484 - acc: 0.9794 - val_loss: 0.1558 - val_acc: 0.9765

...

Epoch 00084: LearningRateScheduler reducing learning rate to 1.1904761904761905e-05.
Epoch 84/1000
50000/50000 [==============================] - 4s 76us/step - loss: 0.0483 - acc: 0.9995 - val_loss: 0.1005 - val_acc: 0.9824
ubuntu@alexa-ml:~/0x06-keras$ ls network1.h5 
network1.h5
ubuntu@alexa-ml:~/0x06-keras$


### [9. Save and Load Model](./9-model.py)
Write the following functions:

    def save_model(network, filename): saves an entire model:
        network is the model to save
        filename is the path of the file that the model should be saved to
        Returns: None
    def load_model(filename): loads an entire model:
        filename is the path of the file that the model should be loaded from
        Returns: the loaded model

ubuntu@alexa-ml:~/0x06-keras$ cat 9-main.py
#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
optimize_model = __import__('2-optimize').optimize_model
one_hot = __import__('3-one_hot').one_hot
train_model = __import__('8-train').train_model
model = __import__('9-model')

if __name__ == '__main__':
    datasets = np.load('../data/MNIST.npz')
    X_train = datasets['X_train']
    X_train = X_train.reshape(X_train.shape[0], -1)
    Y_train = datasets['Y_train']
    Y_train_oh = one_hot(Y_train)
    X_valid = datasets['X_valid']
    X_valid = X_valid.reshape(X_valid.shape[0], -1)
    Y_valid = datasets['Y_valid']
    Y_valid_oh = one_hot(Y_valid)

    np.random.seed(0)
    tf.set_random_seed(0)
    network = model.load_model('network1.h5')
    batch_size = 32
    epochs = 1000
    train_model(network, X_train, Y_train_oh, batch_size, epochs,
                validation_data=(X_valid, Y_valid_oh), early_stopping=True,
                patience=2, learning_rate_decay=True, alpha=0.001)
    model.save_model(network, 'network2.h5')
    network.summary()
    print(network.get_weights())
    del network

    network2 = model.load_model('network2.h5')
    network2.summary()
    print(network2.get_weights())
ubuntu@alexa-ml:~/0x06-keras$ ./9-main.py
Train on 50000 samples, validate on 10000 samples
Epoch 00001: LearningRateScheduler reducing learning rate to 0.001.
Epoch 1/1000
50000/50000 [==============================] - 7s 138us/step - loss: 0.1857 - acc: 0.9620 - val_loss: 0.1703 - val_acc: 0.9675

Epoch 00002: LearningRateScheduler reducing learning rate to 0.0005.
Epoch 2/1000
50000/50000 [==============================] - 6s 128us/step - loss: 0.1050 - acc: 0.9866 - val_loss: 0.1318 - val_acc: 0.9794

Epoch 00003: LearningRateScheduler reducing learning rate to 0.0003333333333333333.
Epoch 3/1000
50000/50000 [==============================] - 6s 121us/step - loss: 0.0832 - acc: 0.9919 - val_loss: 0.1197 - val_acc: 0.9798

...

Epoch 00018: LearningRateScheduler reducing learning rate to 5.555555555555556e-05.
Epoch 18/1000
50000/50000 [==============================] - 7s 136us/step - loss: 0.0451 - acc: 0.9991 - val_loss: 0.0979 - val_acc: 0.9824
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 784)               0         
_________________________________________________________________
dense (Dense)                (None, 256)               200960    
_________________________________________________________________
dropout (Dropout)            (None, 256)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 256)               65792     
_________________________________________________________________
dropout_1 (Dropout)          (None, 256)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 10)                2570      
=================================================================
Total params: 269,322
Trainable params: 269,322
Non-trainable params: 0
_________________________________________________________________
[array([[-4.5759862e-32, -1.1166585e-32, -4.8618426e-32, ...,
         7.3602036e-33, -5.0511110e-33,  2.7104314e-32],
       [-5.1027422e-32,  4.8105390e-32,  5.7754159e-32, ...,
        -4.2603841e-32, -5.3894100e-32, -9.4006010e-33],
       [-2.6995851e-32, -2.7303489e-32, -1.6941460e-32, ...,
         5.3161938e-32, -1.8428570e-32, -5.6048969e-32],
       ...,
       [-1.6340068e-32,  8.8453599e-34,  4.6327743e-32, ...,
         4.8267561e-33,  1.8723424e-32, -5.2483525e-32],
       [ 4.2798962e-32, -6.1208579e-33, -8.6091796e-33, ...,
         4.6586352e-32, -2.4117143e-32,  8.6849754e-33],
       [-3.9918836e-32, -1.3367825e-32, -5.6599606e-32, ...,
         5.3246614e-32, -4.8722713e-33, -5.0168054e-32]], dtype=float32), array([ 0.00301141,  0.11106261,  0.01185344,  0.01681166,  0.02567326,
       -0.04051443,  0.02836942,  0.0334205 ,  0.12040944, -0.07719209,
       -0.04880118, -0.04885611,  0.15551251, -0.0015661 ,  0.09485239,
       -0.03239933, -0.06574348,  0.07019318, -0.02889021,  0.00120761,
        0.00289638,  0.061031  ,  0.00149272, -0.1057554 ,  0.03843166,
        0.05751475, -0.01821275, -0.00359686,  0.02726471,  0.04887023,
       -0.09000898,  0.06808096, -0.00745843,  0.16093527, -0.02702073,
        0.00330158, -0.01691936, -0.00485621, -0.01686262, -0.02125342,
        0.00108959,  0.00779402,  0.13222499, -0.06067068,  0.04236976,
       -0.00803121, -0.00283712,  0.02479446,  0.02656144,  0.00154453,
        0.13125737,  0.01164243,  0.01800169, -0.04880525, -0.02819832,
       -0.05462935, -0.04556927,  0.0278921 , -0.05708537, -0.0747561 ,
        0.01275936, -0.03510941, -0.01113027,  0.04357697, -0.01531685,
       -0.04704923, -0.00235724, -0.07797859,  0.09687679, -0.0026693 ,
        0.06528118, -0.03962515, -0.00108486, -0.01176789,  0.04870636,
        0.0394779 , -0.04774741,  0.05873551,  0.01720481,  0.10343516,
       -0.03856955,  0.05167955, -0.00957883, -0.02624257,  0.06043729,
       -0.04707929,  0.03259643,  0.08223059, -0.0012835 ,  0.04932084,
       -0.04287738,  0.01486364, -0.04679639,  0.08369008, -0.08601271,
       -0.0182047 ,  0.00126242, -0.02583828,  0.08934206, -0.00715361,
        0.04692224,  0.01849566, -0.00458118,  0.01739533, -0.06382778,
       -0.05426462, -0.07452562, -0.0012134 ,  0.11089198,  0.01632233,
       -0.04663077,  0.01432451, -0.03560297, -0.08001371, -0.10429831,
       -0.03735451, -0.00408611, -0.12141566,  0.01984059,  0.00802785,
        0.01461906, -0.01420708,  0.00686091,  0.06332626,  0.00764306,
        0.04721671, -0.01236838, -0.04268207, -0.01361568, -0.06188548,
       -0.0838668 ,  0.03986252,  0.02611774, -0.1197499 , -0.03030316,
        0.08601489,  0.03008686,  0.03682374,  0.13999882,  0.00157587,
       -0.1026821 , -0.07841676, -0.01978161, -0.01064411,  0.00699465,
       -0.0797313 ,  0.03154324,  0.00848868,  0.04554785, -0.08009008,
        0.04862264,  0.01171482, -0.01754008,  0.01736041, -0.03311204,
        0.02760461, -0.00951267, -0.0117144 , -0.06521405, -0.03734919,
       -0.00146205, -0.01550307,  0.00211244, -0.06127467, -0.01866053,
       -0.05132233,  0.02999958,  0.00214538,  0.06267863, -0.01051223,
       -0.02010772,  0.00676741, -0.04834704,  0.01962986,  0.01890873,
        0.08378886, -0.05787354,  0.09091029,  0.03466658, -0.00298357,
       -0.01442834,  0.02384408,  0.0085625 , -0.05677909, -0.00320599,
       -0.00627259,  0.02726623,  0.00708877,  0.08952815,  0.00049586,
       -0.05715492, -0.05443761, -0.08709867, -0.10241982, -0.05339433,
        0.02938809,  0.00070681,  0.1044661 , -0.02404852, -0.0199181 ,
        0.02580458, -0.01485083, -0.01173928, -0.03401828,  0.03114421,
        0.01035686,  0.02646278, -0.00828437,  0.04220432, -0.0353339 ,
       -0.04094112, -0.08582541,  0.05103881,  0.00706739, -0.01197457,
       -0.05006712,  0.04120311, -0.06406615,  0.0578991 ,  0.00891703,
        0.06913868, -0.01173124,  0.09464837, -0.01412672, -0.03633853,
       -0.07205615, -0.02358213, -0.00065915, -0.067533  ,  0.06598742,
        0.01933156, -0.0228131 , -0.00552226,  0.01547089, -0.03557153,
       -0.04881606, -0.03681552, -0.01457811,  0.03056423,  0.10093366,
       -0.03832528,  0.0053501 ,  0.06375276, -0.00329764,  0.00965369,
       -0.03741526,  0.08444448, -0.07581469,  0.07575754,  0.11538294,
       -0.02900385,  0.06924497, -0.00062138, -0.06835096,  0.07924626,
       -0.07495385], dtype=float32), array([[-0.02356569, -0.04907231, -0.01683468, ...,  0.03239757,
        -0.01283053,  0.05668125],
       [ 0.02751517, -0.01467951,  0.01286861, ..., -0.02291515,
         0.05908852,  0.00577715],
       [ 0.00034507,  0.00042468, -0.00015304, ..., -0.00018566,
         0.00017563, -0.0005187 ],
       ...,
       [-0.03459045, -0.01961152, -0.05777662, ...,  0.01153925,
         0.0385097 ,  0.03362792],
       [-0.00886679,  0.02552748,  0.02426226, ..., -0.01080185,
         0.05787499, -0.02336088],
       [ 0.03688884,  0.00791977,  0.00632477, ..., -0.00356403,
         0.06188476, -0.05196051]], dtype=float32), array([ 0.04276604,  0.0839251 ,  0.12818155,  0.07179586,  0.01363824,
        0.11040296,  0.13408126,  0.00113129, -0.01164962,  0.16563916,
       -0.06423326,  0.04941095,  0.10967087,  0.071988  , -0.07952698,
       -0.00353061,  0.09681981,  0.00588965,  0.06409904,  0.17110778,
        0.09098865,  0.03793814,  0.02660166,  0.01043977,  0.15460846,
        0.08049833,  0.01238438,  0.12799194, -0.00571567, -0.09562721,
       -0.07235042,  0.00720909,  0.02359622,  0.03847999,  0.03463023,
        0.01936127,  0.08349653,  0.03534501,  0.11887035,  0.18138833,
        0.12218701,  0.1396505 , -0.06249057,  0.04689002,  0.06208638,
        0.00691178,  0.16164613,  0.03111066,  0.1428282 ,  0.00946504,
       -0.01745889,  0.0735269 ,  0.1042906 ,  0.16713966,  0.11473349,
       -0.02385673,  0.00854273,  0.00657116,  0.07558189,  0.05253383,
        0.05701437, -0.0922313 , -0.07271408,  0.13432159,  0.02373713,
        0.0760445 ,  0.15186088,  0.0722203 ,  0.03637095,  0.00304131,
        0.01004457, -0.03102851,  0.01230234,  0.01195357,  0.03313484,
       -0.03725894,  0.10755351, -0.0109304 ,  0.09696213,  0.1315541 ,
       -0.06765564,  0.08152518,  0.16889827,  0.17124896,  0.15823211,
        0.10992618, -0.03088786, -0.05439347,  0.02937236, -0.05253729,
        0.08112361,  0.03349514,  0.05405692,  0.1114902 ,  0.16356425,
       -0.06671461,  0.16571335,  0.07301888,  0.00044013,  0.04260354,
        0.09786872,  0.13165115, -0.01215676,  0.09708365, -0.02336201,
        0.05292292,  0.01573434, -0.03491879,  0.05330826,  0.20068868,
        0.09243125,  0.1541582 ,  0.07882609, -0.06355512, -0.0151039 ,
       -0.00028123,  0.08451017,  0.0917349 , -0.05669888,  0.05047854,
        0.01841846, -0.05679559, -0.03347985, -0.08964503,  0.11728358,
        0.05049109,  0.04792545,  0.0340204 , -0.03711318,  0.1833046 ,
        0.03391251,  0.14137387,  0.18467614,  0.05684291,  0.04667677,
        0.04486008,  0.08924367,  0.0847597 ,  0.0667199 ,  0.08874153,
        0.03056497,  0.02150222, -0.03666061,  0.11014843, -0.03888047,
        0.07234956,  0.07279025,  0.04791832,  0.06558265,  0.09905945,
        0.00855816,  0.06454726,  0.1669972 ,  0.20946756, -0.03132073,
        0.04246802,  0.176783  , -0.0695762 , -0.02401024, -0.0836548 ,
        0.04278667,  0.10616542,  0.00394475,  0.03428684, -0.03344549,
       -0.04554497,  0.17681304,  0.06315447, -0.01827188,  0.11631732,
        0.16133408,  0.09948727,  0.17811266,  0.09946909, -0.0331389 ,
        0.02470633,  0.04113306,  0.19572598,  0.04792014, -0.00455976,
        0.04240628,  0.00645264,  0.07237791, -0.01748033, -0.0667926 ,
       -0.04278271,  0.03710879,  0.07079187,  0.03941205, -0.04929861,
        0.07921859,  0.04423168,  0.04986018,  0.1406923 ,  0.19001177,
       -0.06512053, -0.03004923,  0.0697403 , -0.05313126,  0.09124914,
       -0.06638056,  0.19721031, -0.0174092 , -0.0104068 , -0.04129931,
       -0.00642987,  0.0756193 ,  0.08065893, -0.07171095,  0.04835083,
        0.04631289, -0.00369183,  0.2001069 ,  0.02488809,  0.03141034,
        0.09326123,  0.07790846,  0.04581561,  0.02532499,  0.10345417,
       -0.06265671,  0.08512945,  0.05879917,  0.09843485,  0.15000436,
        0.00726045,  0.08259177,  0.09668669,  0.12184898,  0.05166332,
        0.03919809,  0.00591196,  0.01772584,  0.05691071,  0.04831712,
        0.06039805, -0.06845759,  0.11142325, -0.04337049, -0.00530344,
        0.0504744 , -0.07494399, -0.0438296 , -0.013664  ,  0.16828087,
        0.10732908, -0.0182483 ,  0.03078499,  0.03641264,  0.09930968,
       -0.01113053, -0.01751937, -0.02086665,  0.04415414,  0.05891679,
        0.06461747], dtype=float32), array([[-0.03693896,  0.18147089, -0.31610748, ...,  0.04973613,
        -0.07057659, -0.27543774],
       [-0.23216796,  0.12269747, -0.05088847, ..., -0.18649793,
         0.18711695, -0.12467869],
       [ 0.13921684,  0.01261891,  0.03768756, ..., -0.08711316,
         0.15737216, -0.13199079],
       ...,
       [ 0.04805786, -0.1247745 , -0.00850563, ...,  0.05719882,
        -0.1833688 , -0.07768662],
       [ 0.20461436,  0.16149858,  0.24725884, ..., -0.04838214,
        -0.03026397,  0.08451419],
       [-0.1986867 , -0.24403669,  0.05925792, ...,  0.06884798,
         0.01846102,  0.27381638]], dtype=float32), array([ 0.01995393, -0.10105042, -0.00989409, -0.01942602,  0.00440886,
       -0.05625921, -0.05427512, -0.09168304,  0.19373162,  0.03098004],
      dtype=float32)]
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 784)               0         
_________________________________________________________________
dense (Dense)                (None, 256)               200960    
_________________________________________________________________
dropout (Dropout)            (None, 256)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 256)               65792     
_________________________________________________________________
dropout_1 (Dropout)          (None, 256)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 10)                2570      
=================================================================
Total params: 269,322
Trainable params: 269,322
Non-trainable params: 0
_________________________________________________________________
[array([[-4.5759862e-32, -1.1166585e-32, -4.8618426e-32, ...,
         7.3602036e-33, -5.0511110e-33,  2.7104314e-32],
       [-5.1027422e-32,  4.8105390e-32,  5.7754159e-32, ...,
        -4.2603841e-32, -5.3894100e-32, -9.4006010e-33],
       [-2.6995851e-32, -2.7303489e-32, -1.6941460e-32, ...,
         5.3161938e-32, -1.8428570e-32, -5.6048969e-32],
       ...,
       [-1.6340068e-32,  8.8453599e-34,  4.6327743e-32, ...,
         4.8267561e-33,  1.8723424e-32, -5.2483525e-32],
       [ 4.2798962e-32, -6.1208579e-33, -8.6091796e-33, ...,
         4.6586352e-32, -2.4117143e-32,  8.6849754e-33],
       [-3.9918836e-32, -1.3367825e-32, -5.6599606e-32, ...,
         5.3246614e-32, -4.8722713e-33, -5.0168054e-32]], dtype=float32), array([ 0.00301141,  0.11106261,  0.01185344,  0.01681166,  0.02567326,
       -0.04051443,  0.02836942,  0.0334205 ,  0.12040944, -0.07719209,
       -0.04880118, -0.04885611,  0.15551251, -0.0015661 ,  0.09485239,
       -0.03239933, -0.06574348,  0.07019318, -0.02889021,  0.00120761,
        0.00289638,  0.061031  ,  0.00149272, -0.1057554 ,  0.03843166,
        0.05751475, -0.01821275, -0.00359686,  0.02726471,  0.04887023,
       -0.09000898,  0.06808096, -0.00745843,  0.16093527, -0.02702073,
        0.00330158, -0.01691936, -0.00485621, -0.01686262, -0.02125342,
        0.00108959,  0.00779402,  0.13222499, -0.06067068,  0.04236976,
       -0.00803121, -0.00283712,  0.02479446,  0.02656144,  0.00154453,
        0.13125737,  0.01164243,  0.01800169, -0.04880525, -0.02819832,
       -0.05462935, -0.04556927,  0.0278921 , -0.05708537, -0.0747561 ,
        0.01275936, -0.03510941, -0.01113027,  0.04357697, -0.01531685,
       -0.04704923, -0.00235724, -0.07797859,  0.09687679, -0.0026693 ,
        0.06528118, -0.03962515, -0.00108486, -0.01176789,  0.04870636,
        0.0394779 , -0.04774741,  0.05873551,  0.01720481,  0.10343516,
       -0.03856955,  0.05167955, -0.00957883, -0.02624257,  0.06043729,
       -0.04707929,  0.03259643,  0.08223059, -0.0012835 ,  0.04932084,
       -0.04287738,  0.01486364, -0.04679639,  0.08369008, -0.08601271,
       -0.0182047 ,  0.00126242, -0.02583828,  0.08934206, -0.00715361,
        0.04692224,  0.01849566, -0.00458118,  0.01739533, -0.06382778,
       -0.05426462, -0.07452562, -0.0012134 ,  0.11089198,  0.01632233,
       -0.04663077,  0.01432451, -0.03560297, -0.08001371, -0.10429831,
       -0.03735451, -0.00408611, -0.12141566,  0.01984059,  0.00802785,
        0.01461906, -0.01420708,  0.00686091,  0.06332626,  0.00764306,
        0.04721671, -0.01236838, -0.04268207, -0.01361568, -0.06188548,
       -0.0838668 ,  0.03986252,  0.02611774, -0.1197499 , -0.03030316,
        0.08601489,  0.03008686,  0.03682374,  0.13999882,  0.00157587,
       -0.1026821 , -0.07841676, -0.01978161, -0.01064411,  0.00699465,
       -0.0797313 ,  0.03154324,  0.00848868,  0.04554785, -0.08009008,
        0.04862264,  0.01171482, -0.01754008,  0.01736041, -0.03311204,
        0.02760461, -0.00951267, -0.0117144 , -0.06521405, -0.03734919,
       -0.00146205, -0.01550307,  0.00211244, -0.06127467, -0.01866053,
       -0.05132233,  0.02999958,  0.00214538,  0.06267863, -0.01051223,
       -0.02010772,  0.00676741, -0.04834704,  0.01962986,  0.01890873,
        0.08378886, -0.05787354,  0.09091029,  0.03466658, -0.00298357,
       -0.01442834,  0.02384408,  0.0085625 , -0.05677909, -0.00320599,
       -0.00627259,  0.02726623,  0.00708877,  0.08952815,  0.00049586,
       -0.05715492, -0.05443761, -0.08709867, -0.10241982, -0.05339433,
        0.02938809,  0.00070681,  0.1044661 , -0.02404852, -0.0199181 ,
        0.02580458, -0.01485083, -0.01173928, -0.03401828,  0.03114421,
        0.01035686,  0.02646278, -0.00828437,  0.04220432, -0.0353339 ,
       -0.04094112, -0.08582541,  0.05103881,  0.00706739, -0.01197457,
       -0.05006712,  0.04120311, -0.06406615,  0.0578991 ,  0.00891703,
        0.06913868, -0.01173124,  0.09464837, -0.01412672, -0.03633853,
       -0.07205615, -0.02358213, -0.00065915, -0.067533  ,  0.06598742,
        0.01933156, -0.0228131 , -0.00552226,  0.01547089, -0.03557153,
       -0.04881606, -0.03681552, -0.01457811,  0.03056423,  0.10093366,
       -0.03832528,  0.0053501 ,  0.06375276, -0.00329764,  0.00965369,
       -0.03741526,  0.08444448, -0.07581469,  0.07575754,  0.11538294,
       -0.02900385,  0.06924497, -0.00062138, -0.06835096,  0.07924626,
       -0.07495385], dtype=float32), array([[-0.02356569, -0.04907231, -0.01683468, ...,  0.03239757,
        -0.01283053,  0.05668125],
       [ 0.02751517, -0.01467951,  0.01286861, ..., -0.02291515,
         0.05908852,  0.00577715],
       [ 0.00034507,  0.00042468, -0.00015304, ..., -0.00018566,
         0.00017563, -0.0005187 ],
       ...,
       [-0.03459045, -0.01961152, -0.05777662, ...,  0.01153925,
         0.0385097 ,  0.03362792],
       [-0.00886679,  0.02552748,  0.02426226, ..., -0.01080185,
         0.05787499, -0.02336088],
       [ 0.03688884,  0.00791977,  0.00632477, ..., -0.00356403,
         0.06188476, -0.05196051]], dtype=float32), array([ 0.04276604,  0.0839251 ,  0.12818155,  0.07179586,  0.01363824,
        0.11040296,  0.13408126,  0.00113129, -0.01164962,  0.16563916,
       -0.06423326,  0.04941095,  0.10967087,  0.071988  , -0.07952698,
       -0.00353061,  0.09681981,  0.00588965,  0.06409904,  0.17110778,
        0.09098865,  0.03793814,  0.02660166,  0.01043977,  0.15460846,
        0.08049833,  0.01238438,  0.12799194, -0.00571567, -0.09562721,
       -0.07235042,  0.00720909,  0.02359622,  0.03847999,  0.03463023,
        0.01936127,  0.08349653,  0.03534501,  0.11887035,  0.18138833,
        0.12218701,  0.1396505 , -0.06249057,  0.04689002,  0.06208638,
        0.00691178,  0.16164613,  0.03111066,  0.1428282 ,  0.00946504,
       -0.01745889,  0.0735269 ,  0.1042906 ,  0.16713966,  0.11473349,
       -0.02385673,  0.00854273,  0.00657116,  0.07558189,  0.05253383,
        0.05701437, -0.0922313 , -0.07271408,  0.13432159,  0.02373713,
        0.0760445 ,  0.15186088,  0.0722203 ,  0.03637095,  0.00304131,
        0.01004457, -0.03102851,  0.01230234,  0.01195357,  0.03313484,
       -0.03725894,  0.10755351, -0.0109304 ,  0.09696213,  0.1315541 ,
       -0.06765564,  0.08152518,  0.16889827,  0.17124896,  0.15823211,
        0.10992618, -0.03088786, -0.05439347,  0.02937236, -0.05253729,
        0.08112361,  0.03349514,  0.05405692,  0.1114902 ,  0.16356425,
       -0.06671461,  0.16571335,  0.07301888,  0.00044013,  0.04260354,
        0.09786872,  0.13165115, -0.01215676,  0.09708365, -0.02336201,
        0.05292292,  0.01573434, -0.03491879,  0.05330826,  0.20068868,
        0.09243125,  0.1541582 ,  0.07882609, -0.06355512, -0.0151039 ,
       -0.00028123,  0.08451017,  0.0917349 , -0.05669888,  0.05047854,
        0.01841846, -0.05679559, -0.03347985, -0.08964503,  0.11728358,
        0.05049109,  0.04792545,  0.0340204 , -0.03711318,  0.1833046 ,
        0.03391251,  0.14137387,  0.18467614,  0.05684291,  0.04667677,
        0.04486008,  0.08924367,  0.0847597 ,  0.0667199 ,  0.08874153,
        0.03056497,  0.02150222, -0.03666061,  0.11014843, -0.03888047,
        0.07234956,  0.07279025,  0.04791832,  0.06558265,  0.09905945,
        0.00855816,  0.06454726,  0.1669972 ,  0.20946756, -0.03132073,
        0.04246802,  0.176783  , -0.0695762 , -0.02401024, -0.0836548 ,
        0.04278667,  0.10616542,  0.00394475,  0.03428684, -0.03344549,
       -0.04554497,  0.17681304,  0.06315447, -0.01827188,  0.11631732,
        0.16133408,  0.09948727,  0.17811266,  0.09946909, -0.0331389 ,
        0.02470633,  0.04113306,  0.19572598,  0.04792014, -0.00455976,
        0.04240628,  0.00645264,  0.07237791, -0.01748033, -0.0667926 ,
       -0.04278271,  0.03710879,  0.07079187,  0.03941205, -0.04929861,
        0.07921859,  0.04423168,  0.04986018,  0.1406923 ,  0.19001177,
       -0.06512053, -0.03004923,  0.0697403 , -0.05313126,  0.09124914,
       -0.06638056,  0.19721031, -0.0174092 , -0.0104068 , -0.04129931,
       -0.00642987,  0.0756193 ,  0.08065893, -0.07171095,  0.04835083,
        0.04631289, -0.00369183,  0.2001069 ,  0.02488809,  0.03141034,
        0.09326123,  0.07790846,  0.04581561,  0.02532499,  0.10345417,
       -0.06265671,  0.08512945,  0.05879917,  0.09843485,  0.15000436,
        0.00726045,  0.08259177,  0.09668669,  0.12184898,  0.05166332,
        0.03919809,  0.00591196,  0.01772584,  0.05691071,  0.04831712,
        0.06039805, -0.06845759,  0.11142325, -0.04337049, -0.00530344,
        0.0504744 , -0.07494399, -0.0438296 , -0.013664  ,  0.16828087,
        0.10732908, -0.0182483 ,  0.03078499,  0.03641264,  0.09930968,
       -0.01113053, -0.01751937, -0.02086665,  0.04415414,  0.05891679,
        0.06461747], dtype=float32), array([[-0.03693896,  0.18147089, -0.31610748, ...,  0.04973613,
        -0.07057659, -0.27543774],
       [-0.23216796,  0.12269747, -0.05088847, ..., -0.18649793,
         0.18711695, -0.12467869],
       [ 0.13921684,  0.01261891,  0.03768756, ..., -0.08711316,
         0.15737216, -0.13199079],
       ...,
       [ 0.04805786, -0.1247745 , -0.00850563, ...,  0.05719882,
        -0.1833688 , -0.07768662],
       [ 0.20461436,  0.16149858,  0.24725884, ..., -0.04838214,
        -0.03026397,  0.08451419],
       [-0.1986867 , -0.24403669,  0.05925792, ...,  0.06884798,
         0.01846102,  0.27381638]], dtype=float32), array([ 0.01995393, -0.10105042, -0.00989409, -0.01942602,  0.00440886,
       -0.05625921, -0.05427512, -0.09168304,  0.19373162,  0.03098004],
      dtype=float32)]
ubuntu@alexa-ml:~/0x06-keras$


### [10. Save and Load Weights](./10-weights.py)
Write the following functions:
*    def save_weights(network, filename, save_format='h5'): saves a model’s weights:
*        network is the model whose weights should be saved
*        filename is the path of the file that the weights should be saved to
*        save_format is the format in which the weights should be saved
*        Returns: None
*    def load_weights(network, filename): loads a model’s weights:
*        network is the model to which the weights should be loaded
*        filename is the path of the file that the weights should be loaded from
*        Returns: None

ubuntu@alexa-ml:~/0x06-keras$ cat 10-main.py
#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
optimize_model = __import__('2-optimize').optimize_model
one_hot = __import__('3-one_hot').one_hot
train_model = __import__('8-train').train_model
model = __import__('9-model')
weights = __import__('10-weights')

if __name__ == '__main__':

    np.random.seed(0)
    tf.set_random_seed(0)
    network = model.load_model('network2.h5')
    weights.save_weights(network, 'weights2.h5')
    del network

    network2 = model.load_model('network1.h5')
    print(network2.get_weights())
    weights.load_weights(network2, 'weights2.h5')
    print(network2.get_weights())
ubuntu@alexa-ml:~/0x06-keras$ ./10-main.py
[array([[-4.5759862e-32, -1.1166585e-32, -4.8618426e-32, ...,
         7.3602036e-33,  1.0679910e-31,  1.2662074e-31],
       [-8.2763765e-32,  7.8024272e-32,  5.7754159e-32, ...,
        -9.4508500e-32, -8.7413357e-32, -9.4006010e-33],
       [-2.6995851e-32, -1.2755118e-31, -1.6941460e-32, ...,
         6.5502971e-32, -1.8428570e-32, -5.6048969e-32],
       ...,
       [-1.6340068e-32,  8.8453599e-34,  7.1772270e-32, ...,
        -1.0205528e-31,  1.8723424e-32, -5.2483525e-32],
       [ 9.4941317e-32,  1.2825350e-31, -1.1253217e-31, ...,
         4.6586352e-32, -2.4117143e-32,  1.1352287e-31],
       [-9.7321176e-32, -1.3367825e-32, -5.6599606e-32, ...,
         8.6363094e-32,  1.0301788e-31, -8.1369723e-32]], dtype=float32), array([ 0.0177226 ,  0.06462484,  0.04383771,  0.02025572,  0.02124701,
        0.01649328,  0.03585848,  0.02832156,  0.09563339, -0.0320198 ,
       -0.02401314, -0.01727396,  0.09775663,  0.05088813,  0.05385403,
       -0.00634286, -0.02179774,  0.05148961, -0.01412386,  0.02191018,
        0.037662  ,  0.03735024,  0.03957009, -0.0624669 ,  0.02667639,
        0.05000881, -0.00634354, -0.00797467,  0.02604088,  0.03435197,
       -0.01682431,  0.11080757,  0.01270892,  0.07789762, -0.01329599,
        0.01139759, -0.00412893,  0.01966521,  0.00463509,  0.03268059,
        0.01650209,  0.02445128,  0.07404366, -0.005305  ,  0.03757716,
       -0.0051098 ,  0.01824963,  0.01275243,  0.00756213,  0.00327058,
        0.07863108,  0.01877724,  0.07315809, -0.02174345,  0.01156692,
       -0.01366173, -0.00898941,  0.02254858, -0.02089196, -0.01966818,
        0.0668253 , -0.03074929,  0.00088954,  0.0460937 ,  0.01529284,
       -0.0277822 ,  0.0443285 , -0.03543003,  0.06722342,  0.00941583,
        0.06746895, -0.01015567,  0.03700616, -0.01015506,  0.04658443,
        0.0246244 , -0.01086085,  0.03586575,  0.02957562,  0.0795339 ,
        0.01083806,  0.03108063, -0.0042541 , -0.0060223 ,  0.04784374,
       -0.01985301,  0.02277074,  0.0519317 ,  0.04299914,  0.04197961,
        0.00578057,  0.01539827, -0.00962491,  0.05623633, -0.05821526,
       -0.0182047 ,  0.01303304, -0.00524886,  0.04236404,  0.02388879,
        0.04200788,  0.0137086 ,  0.03104705,  0.0350742 , -0.00849691,
       -0.01244089, -0.01251533,  0.03520569,  0.07352988,  0.02679587,
       -0.00384042,  0.01439581, -0.01616368, -0.02628596, -0.01794489,
       -0.0049483 ,  0.0043509 , -0.05845036,  0.034833  ,  0.02032258,
        0.02034523, -0.01289587,  0.01331374,  0.06758583,  0.0092988 ,
        0.03365651,  0.03152261, -0.0215671 ,  0.01884756, -0.04153776,
       -0.03377955,  0.03307581,  0.01457593, -0.04662341,  0.00936631,
        0.05457631,  0.04623857,  0.03763595,  0.10441331,  0.00435298,
       -0.03383221, -0.03733158,  0.02570176, -0.01029879, -0.00076294,
       -0.03513486,  0.01957691,  0.0447083 ,  0.00027619, -0.04478331,
        0.01631684,  0.0038563 , -0.01531905,  0.02119915, -0.02418118,
        0.03542251, -0.00980224, -0.00917668, -0.05598968, -0.01618161,
        0.03264032,  0.00265676,  0.01004918, -0.02667106, -0.01810119,
       -0.02692634,  0.02332897,  0.05421867,  0.04119879, -0.00425939,
       -0.01128155,  0.01259288, -0.01888148,  0.02109184,  0.02451956,
        0.05364591, -0.02414621,  0.08432777,  0.03706892,  0.03144047,
        0.03789293,  0.0376611 ,  0.05489098, -0.02124977,  0.04278056,
        0.03244171,  0.04450765,  0.0494053 ,  0.05821519,  0.02111511,
       -0.02286131,  0.00112884, -0.04057874, -0.05840421, -0.01502772,
        0.02632144,  0.06633449,  0.06921737,  0.0106327 , -0.00618735,
        0.0391633 , -0.00141498, -0.01173928,  0.0103368 ,  0.03712434,
        0.03100942,  0.00394621,  0.01445687,  0.03565611, -0.00825073,
       -0.01699825, -0.04524225,  0.05463513,  0.03570695,  0.03913242,
        0.01571981,  0.04191232,  0.00437594,  0.05253635,  0.00096878,
        0.06827752, -0.02368168,  0.04884164, -0.00148936,  0.00053679,
       -0.01244268, -0.01750242,  0.01580064, -0.03205935,  0.03782367,
        0.00463354, -0.00243089,  0.00545296,  0.02300755, -0.01602074,
        0.01387006, -0.006294  ,  0.01320113,  0.04677416,  0.06170861,
       -0.03699086, -0.01016229,  0.04539239,  0.02359739,  0.0145643 ,
       -0.01645944,  0.04060456, -0.03677353,  0.03378868,  0.06703849,
        0.00832724,  0.05729571,  0.0413073 , -0.0562811 ,  0.05869971,
       -0.01291762], dtype=float32), array([[-0.02763553, -0.04549099, -0.01421679, ...,  0.00026949,
         0.02943806,  0.07522127],
       [ 0.0633354 , -0.01512731,  0.02787657, ..., -0.00856919,
         0.03388717, -0.0014368 ],
       [-0.0299193 ,  0.10278308,  0.01729781, ...,  0.03885916,
         0.04814061, -0.06035944],
       ...,
       [-0.08057359,  0.01194098, -0.00049156, ...,  0.03789978,
         0.08766969,  0.04268201],
       [-0.01834538,  0.03133831, -0.01002202, ..., -0.00806595,
         0.0831728 , -0.01953515],
       [ 0.0448166 , -0.03766302,  0.00889443, ...,  0.0161895 ,
         0.05756581, -0.08010282]], dtype=float32), array([ 5.5307318e-02,  4.0775526e-02,  6.9632463e-02,  4.1506812e-02,
        3.1677265e-02,  8.6648464e-02,  8.2473107e-02,  2.8982372e-03,
        1.5261679e-02,  5.9189815e-02, -5.5269329e-03,  4.3383237e-02,
        5.4784771e-02,  4.3615751e-02, -2.2406084e-02,  1.8750880e-02,
        3.5285313e-02,  2.1491092e-02,  5.4440022e-02,  8.9392602e-02,
        6.1208613e-02,  2.1459768e-02,  2.9832296e-02,  1.5609627e-02,
        5.9434008e-02,  3.4845695e-02,  3.2393962e-02,  6.1814342e-02,
        1.3897495e-02, -2.2510218e-02, -1.6878489e-02,  5.1495846e-02,
        1.4861853e-02,  3.7172899e-02,  2.5111690e-02,  3.6285538e-02,
        3.3623077e-02,  4.0942207e-02,  6.1658897e-02,  7.9718955e-02,
        7.6505780e-02,  4.3993089e-02,  1.0139066e-02,  2.8833061e-02,
        4.4250060e-02,  3.0961584e-02,  6.8841629e-02,  3.5665996e-02,
        6.5915473e-02,  2.5379207e-02,  1.3025421e-02,  4.9711294e-02,
        4.1152507e-02,  5.9514582e-02,  5.1758796e-02, -1.5247701e-03,
        8.2781939e-03,  3.3234626e-02,  3.3428732e-02,  4.3685947e-02,
        2.2547856e-02, -3.1755213e-02, -2.0772126e-03,  6.9056489e-02,
        2.6067186e-02,  3.6236260e-02,  5.9837375e-02,  6.6909231e-02,
        4.1837137e-02,  1.2497462e-02,  1.4223246e-02,  1.5956669e-03,
        2.3557320e-02,  2.3733480e-02,  4.1000772e-02, -3.7258938e-02,
        5.6154344e-02,  1.0620966e-02,  5.5723712e-02,  6.3426025e-02,
        4.5602396e-03,  6.6749461e-02,  8.1990726e-02,  6.7610607e-02,
        8.4063791e-02,  5.4124992e-02,  5.1208504e-04, -1.3421856e-02,
        4.7130946e-02,  2.6670665e-02,  4.6774808e-02,  4.8953645e-02,
        3.4138449e-02,  5.5932049e-02,  5.4851782e-02,  1.1128555e-02,
        8.1390083e-02,  7.9631321e-02,  2.5094852e-02,  3.8221266e-02,
        6.0003459e-02,  5.8516972e-02,  1.1409077e-02,  6.5148070e-02,
        3.0753543e-03,  1.9499557e-02,  1.8256320e-02,  3.3567799e-03,
        4.0481985e-02,  9.5163435e-02,  5.7423785e-02,  7.3187068e-02,
        8.3566688e-02, -1.8573154e-02,  2.7278475e-02,  1.6625520e-02,
        5.6265943e-02,  7.0197746e-02, -1.5031311e-02,  2.0655040e-02,
        5.6676593e-02, -1.0340380e-02, -2.0987576e-02, -2.7081056e-02,
        4.7245417e-02, -4.5332154e-03,  3.1245383e-02,  4.4200245e-02,
       -7.7734790e-03,  7.1721226e-02,  3.9938949e-02,  5.2151024e-02,
        7.9258814e-02,  5.0831709e-02,  6.5722317e-02,  6.1673053e-02,
        5.2433431e-02,  4.7275424e-02,  6.1355818e-02,  5.4950465e-02,
        1.7830636e-02,  1.4279577e-02, -1.0280591e-02,  5.5611655e-02,
        6.7540788e-04,  4.2292260e-02,  3.9729364e-02,  4.3594014e-02,
        6.3487485e-02,  5.5273369e-02,  1.3429743e-02,  3.1067422e-02,
        6.5337121e-02,  7.6377437e-02, -8.8757332e-03,  1.4823711e-02,
        8.5845485e-02, -1.0519940e-02, -2.0235716e-03, -2.6282754e-02,
        4.0837917e-02,  7.4328348e-02,  4.4820365e-05,  2.8655652e-02,
       -5.6666541e-03, -5.7983552e-03,  6.7330860e-02,  3.0146427e-02,
        2.4013918e-02,  5.5991143e-02,  7.2535887e-02,  6.3376404e-02,
        7.0380263e-02,  3.9769225e-02, -1.4912275e-02,  1.0049780e-02,
        2.1283727e-02,  9.0736590e-02,  3.2191835e-02,  9.3562230e-03,
        4.9692221e-02,  1.6099367e-02,  2.3009399e-02,  2.0799084e-02,
       -3.6532918e-03,  4.2757737e-03,  3.9379943e-02,  6.9785737e-02,
        4.0044118e-02,  1.9199701e-03,  6.7429245e-02,  1.0203039e-02,
        4.6101581e-02,  5.5401411e-02,  8.7589227e-02, -8.1283012e-03,
       -7.3671568e-04,  3.7098803e-02, -1.6199560e-03,  6.6842586e-02,
       -1.7458135e-02,  9.9052437e-02,  2.7792415e-02,  1.5263462e-02,
        1.3427766e-03,  2.0454351e-02,  4.9139824e-02,  3.7541296e-02,
       -5.7511381e-03,  1.7297510e-02,  3.9793283e-02,  2.4025520e-02,
        8.4716327e-02,  2.8523466e-02,  5.9406329e-02,  5.9621383e-02,
        4.6683144e-02,  4.0367983e-02,  3.9707076e-02,  5.8731150e-02,
       -2.4590671e-02,  5.9299793e-02,  3.3536449e-02,  5.9727013e-02,
        6.3208684e-02,  1.4358658e-02,  4.3202370e-02,  4.0660433e-02,
        6.3481241e-02,  4.4692829e-02,  2.1778919e-02,  1.9559182e-02,
        2.2026088e-02,  2.3985770e-02,  1.8337095e-02,  3.3931546e-02,
       -9.6331118e-03,  6.8637185e-02,  2.5095488e-03,  1.9238470e-02,
        5.2302588e-02, -2.9246639e-02,  8.7012574e-03,  5.0136014e-03,
        8.5062608e-02,  6.3873649e-02, -4.9092164e-03,  3.1776085e-02,
        3.1228958e-02,  5.6337129e-02, -1.2336310e-02, -6.9134715e-03,
        5.9988117e-03,  5.3026587e-02,  2.9054791e-02,  3.1528916e-02],
      dtype=float32), array([[-0.08405132,  0.20399688, -0.28256863, ...,  0.03432821,
        -0.0726487 , -0.24081036],
       [-0.23760298,  0.10214677, -0.04478079, ..., -0.20114845,
         0.19178632, -0.14628552],
       [ 0.16553086, -0.03510654,  0.0563013 , ..., -0.06046436,
         0.18303622, -0.13291068],
       ...,
       [ 0.08879574, -0.11073965,  0.02149505, ...,  0.04346587,
        -0.27625164, -0.03672246],
       [ 0.19350938,  0.156426  ,  0.24151093, ..., -0.05120337,
        -0.00342973,  0.11809952],
       [-0.18886602, -0.2591552 ,  0.06622873, ...,  0.06190929,
        -0.01110658,  0.28826615]], dtype=float32), array([ 0.00926645, -0.02207932, -0.01024525, -0.01336748,  0.01504286,
       -0.02302895, -0.02060063, -0.02706093,  0.06492013,  0.00593618],
      dtype=float32)]
[array([[-4.5759862e-32, -1.1166585e-32, -4.8618426e-32, ...,
         7.3602036e-33, -5.0511110e-33,  2.7104314e-32],
       [-5.1027422e-32,  4.8105390e-32,  5.7754159e-32, ...,
        -4.2603841e-32, -5.3894100e-32, -9.4006010e-33],
       [-2.6995851e-32, -2.7303489e-32, -1.6941460e-32, ...,
         5.3161938e-32, -1.8428570e-32, -5.6048969e-32],
       ...,
       [-1.6340068e-32,  8.8453599e-34,  4.6327743e-32, ...,
         4.8267561e-33,  1.8723424e-32, -5.2483525e-32],
       [ 4.2798962e-32, -6.1208579e-33, -8.6091796e-33, ...,
         4.6586352e-32, -2.4117143e-32,  8.6849754e-33],
       [-3.9918836e-32, -1.3367825e-32, -5.6599606e-32, ...,
         5.3246614e-32, -4.8722713e-33, -5.0168054e-32]], dtype=float32), array([ 0.00301141,  0.11106261,  0.01185344,  0.01681166,  0.02567326,
       -0.04051443,  0.02836942,  0.0334205 ,  0.12040944, -0.07719209,
       -0.04880118, -0.04885611,  0.15551251, -0.0015661 ,  0.09485239,
       -0.03239933, -0.06574348,  0.07019318, -0.02889021,  0.00120761,
        0.00289638,  0.061031  ,  0.00149272, -0.1057554 ,  0.03843166,
        0.05751475, -0.01821275, -0.00359686,  0.02726471,  0.04887023,
       -0.09000898,  0.06808096, -0.00745843,  0.16093527, -0.02702073,
        0.00330158, -0.01691936, -0.00485621, -0.01686262, -0.02125342,
        0.00108959,  0.00779402,  0.13222499, -0.06067068,  0.04236976,
       -0.00803121, -0.00283712,  0.02479446,  0.02656144,  0.00154453,
        0.13125737,  0.01164243,  0.01800169, -0.04880525, -0.02819832,
       -0.05462935, -0.04556927,  0.0278921 , -0.05708537, -0.0747561 ,
        0.01275936, -0.03510941, -0.01113027,  0.04357697, -0.01531685,
       -0.04704923, -0.00235724, -0.07797859,  0.09687679, -0.0026693 ,
        0.06528118, -0.03962515, -0.00108486, -0.01176789,  0.04870636,
        0.0394779 , -0.04774741,  0.05873551,  0.01720481,  0.10343516,
       -0.03856955,  0.05167955, -0.00957883, -0.02624257,  0.06043729,
       -0.04707929,  0.03259643,  0.08223059, -0.0012835 ,  0.04932084,
       -0.04287738,  0.01486364, -0.04679639,  0.08369008, -0.08601271,
       -0.0182047 ,  0.00126242, -0.02583828,  0.08934206, -0.00715361,
        0.04692224,  0.01849566, -0.00458118,  0.01739533, -0.06382778,
       -0.05426462, -0.07452562, -0.0012134 ,  0.11089198,  0.01632233,
       -0.04663077,  0.01432451, -0.03560297, -0.08001371, -0.10429831,
       -0.03735451, -0.00408611, -0.12141566,  0.01984059,  0.00802785,
        0.01461906, -0.01420708,  0.00686091,  0.06332626,  0.00764306,
        0.04721671, -0.01236838, -0.04268207, -0.01361568, -0.06188548,
       -0.0838668 ,  0.03986252,  0.02611774, -0.1197499 , -0.03030316,
        0.08601489,  0.03008686,  0.03682374,  0.13999882,  0.00157587,
       -0.1026821 , -0.07841676, -0.01978161, -0.01064411,  0.00699465,
       -0.0797313 ,  0.03154324,  0.00848868,  0.04554785, -0.08009008,
        0.04862264,  0.01171482, -0.01754008,  0.01736041, -0.03311204,
        0.02760461, -0.00951267, -0.0117144 , -0.06521405, -0.03734919,
       -0.00146205, -0.01550307,  0.00211244, -0.06127467, -0.01866053,
       -0.05132233,  0.02999958,  0.00214538,  0.06267863, -0.01051223,
       -0.02010772,  0.00676741, -0.04834704,  0.01962986,  0.01890873,
        0.08378886, -0.05787354,  0.09091029,  0.03466658, -0.00298357,
       -0.01442834,  0.02384408,  0.0085625 , -0.05677909, -0.00320599,
       -0.00627259,  0.02726623,  0.00708877,  0.08952815,  0.00049586,
       -0.05715492, -0.05443761, -0.08709867, -0.10241982, -0.05339433,
        0.02938809,  0.00070681,  0.1044661 , -0.02404852, -0.0199181 ,
        0.02580458, -0.01485083, -0.01173928, -0.03401828,  0.03114421,
        0.01035686,  0.02646278, -0.00828437,  0.04220432, -0.0353339 ,
       -0.04094112, -0.08582541,  0.05103881,  0.00706739, -0.01197457,
       -0.05006712,  0.04120311, -0.06406615,  0.0578991 ,  0.00891703,
        0.06913868, -0.01173124,  0.09464837, -0.01412672, -0.03633853,
       -0.07205615, -0.02358213, -0.00065915, -0.067533  ,  0.06598742,
        0.01933156, -0.0228131 , -0.00552226,  0.01547089, -0.03557153,
       -0.04881606, -0.03681552, -0.01457811,  0.03056423,  0.10093366,
       -0.03832528,  0.0053501 ,  0.06375276, -0.00329764,  0.00965369,
       -0.03741526,  0.08444448, -0.07581469,  0.07575754,  0.11538294,
       -0.02900385,  0.06924497, -0.00062138, -0.06835096,  0.07924626,
       -0.07495385], dtype=float32), array([[-0.02356569, -0.04907231, -0.01683468, ...,  0.03239757,
        -0.01283053,  0.05668125],
       [ 0.02751517, -0.01467951,  0.01286861, ..., -0.02291515,
         0.05908852,  0.00577715],
       [ 0.00034507,  0.00042468, -0.00015304, ..., -0.00018566,
         0.00017563, -0.0005187 ],
       ...,
       [-0.03459045, -0.01961152, -0.05777662, ...,  0.01153925,
         0.0385097 ,  0.03362792],
       [-0.00886679,  0.02552748,  0.02426226, ..., -0.01080185,
         0.05787499, -0.02336088],
       [ 0.03688884,  0.00791977,  0.00632477, ..., -0.00356403,
         0.06188476, -0.05196051]], dtype=float32), array([ 0.04276604,  0.0839251 ,  0.12818155,  0.07179586,  0.01363824,
        0.11040296,  0.13408126,  0.00113129, -0.01164962,  0.16563916,
       -0.06423326,  0.04941095,  0.10967087,  0.071988  , -0.07952698,
       -0.00353061,  0.09681981,  0.00588965,  0.06409904,  0.17110778,
        0.09098865,  0.03793814,  0.02660166,  0.01043977,  0.15460846,
        0.08049833,  0.01238438,  0.12799194, -0.00571567, -0.09562721,
       -0.07235042,  0.00720909,  0.02359622,  0.03847999,  0.03463023,
        0.01936127,  0.08349653,  0.03534501,  0.11887035,  0.18138833,
        0.12218701,  0.1396505 , -0.06249057,  0.04689002,  0.06208638,
        0.00691178,  0.16164613,  0.03111066,  0.1428282 ,  0.00946504,
       -0.01745889,  0.0735269 ,  0.1042906 ,  0.16713966,  0.11473349,
       -0.02385673,  0.00854273,  0.00657116,  0.07558189,  0.05253383,
        0.05701437, -0.0922313 , -0.07271408,  0.13432159,  0.02373713,
        0.0760445 ,  0.15186088,  0.0722203 ,  0.03637095,  0.00304131,
        0.01004457, -0.03102851,  0.01230234,  0.01195357,  0.03313484,
       -0.03725894,  0.10755351, -0.0109304 ,  0.09696213,  0.1315541 ,
       -0.06765564,  0.08152518,  0.16889827,  0.17124896,  0.15823211,
        0.10992618, -0.03088786, -0.05439347,  0.02937236, -0.05253729,
        0.08112361,  0.03349514,  0.05405692,  0.1114902 ,  0.16356425,
       -0.06671461,  0.16571335,  0.07301888,  0.00044013,  0.04260354,
        0.09786872,  0.13165115, -0.01215676,  0.09708365, -0.02336201,
        0.05292292,  0.01573434, -0.03491879,  0.05330826,  0.20068868,
        0.09243125,  0.1541582 ,  0.07882609, -0.06355512, -0.0151039 ,
       -0.00028123,  0.08451017,  0.0917349 , -0.05669888,  0.05047854,
        0.01841846, -0.05679559, -0.03347985, -0.08964503,  0.11728358,
        0.05049109,  0.04792545,  0.0340204 , -0.03711318,  0.1833046 ,
        0.03391251,  0.14137387,  0.18467614,  0.05684291,  0.04667677,
        0.04486008,  0.08924367,  0.0847597 ,  0.0667199 ,  0.08874153,
        0.03056497,  0.02150222, -0.03666061,  0.11014843, -0.03888047,
        0.07234956,  0.07279025,  0.04791832,  0.06558265,  0.09905945,
        0.00855816,  0.06454726,  0.1669972 ,  0.20946756, -0.03132073,
        0.04246802,  0.176783  , -0.0695762 , -0.02401024, -0.0836548 ,
        0.04278667,  0.10616542,  0.00394475,  0.03428684, -0.03344549,
       -0.04554497,  0.17681304,  0.06315447, -0.01827188,  0.11631732,
        0.16133408,  0.09948727,  0.17811266,  0.09946909, -0.0331389 ,
        0.02470633,  0.04113306,  0.19572598,  0.04792014, -0.00455976,
        0.04240628,  0.00645264,  0.07237791, -0.01748033, -0.0667926 ,
       -0.04278271,  0.03710879,  0.07079187,  0.03941205, -0.04929861,
        0.07921859,  0.04423168,  0.04986018,  0.1406923 ,  0.19001177,
       -0.06512053, -0.03004923,  0.0697403 , -0.05313126,  0.09124914,
       -0.06638056,  0.19721031, -0.0174092 , -0.0104068 , -0.04129931,
       -0.00642987,  0.0756193 ,  0.08065893, -0.07171095,  0.04835083,
        0.04631289, -0.00369183,  0.2001069 ,  0.02488809,  0.03141034,
        0.09326123,  0.07790846,  0.04581561,  0.02532499,  0.10345417,
       -0.06265671,  0.08512945,  0.05879917,  0.09843485,  0.15000436,
        0.00726045,  0.08259177,  0.09668669,  0.12184898,  0.05166332,
        0.03919809,  0.00591196,  0.01772584,  0.05691071,  0.04831712,
        0.06039805, -0.06845759,  0.11142325, -0.04337049, -0.00530344,
        0.0504744 , -0.07494399, -0.0438296 , -0.013664  ,  0.16828087,
        0.10732908, -0.0182483 ,  0.03078499,  0.03641264,  0.09930968,
       -0.01113053, -0.01751937, -0.02086665,  0.04415414,  0.05891679,
        0.06461747], dtype=float32), array([[-0.03693896,  0.18147089, -0.31610748, ...,  0.04973613,
        -0.07057659, -0.27543774],
       [-0.23216796,  0.12269747, -0.05088847, ..., -0.18649793,
         0.18711695, -0.12467869],
       [ 0.13921684,  0.01261891,  0.03768756, ..., -0.08711316,
         0.15737216, -0.13199079],
       ...,
       [ 0.04805786, -0.1247745 , -0.00850563, ...,  0.05719882,
        -0.1833688 , -0.07768662],
       [ 0.20461436,  0.16149858,  0.24725884, ..., -0.04838214,
        -0.03026397,  0.08451419],
       [-0.1986867 , -0.24403669,  0.05925792, ...,  0.06884798,
         0.01846102,  0.27381638]], dtype=float32), array([ 0.01995393, -0.10105042, -0.00989409, -0.01942602,  0.00440886,
       -0.05625921, -0.05427512, -0.09168304,  0.19373162,  0.03098004],
      dtype=float32)]
ubuntu@alexa-ml:~/0x06-keras$


### [11. Save and Load Configuration](./11-config.py)
Write the following functions:
*    def save_config(network, filename): saves a model’s configuration in JSON format:
*        network is the model whose configuration should be saved
*        filename is the path of the file that the configuration should be saved to
*        Returns: None
*    def load_config(filename): loads a model with a specific configuration:
*        filename is the path of the file containing the model’s configuration in JSON format
*        Returns: the loaded model

ubuntu@alexa-ml:~/0x06-keras$ cat 11-main.py
#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
optimize_model = __import__('2-optimize').optimize_model
one_hot = __import__('3-one_hot').one_hot
train_model = __import__('8-train').train_model
model = __import__('9-model')
config = __import__('11-config')

if __name__ == '__main__':
    np.random.seed(0)
    tf.set_random_seed(0)
    network = model.load_model('network1.h5')
    config.save_config(network, 'config1.json')
    del network

    network2 = config.load_config('config1.json')
    network2.summary()
    print(network2.get_weights())
ubuntu@alexa-ml:~/0x06-keras$ ./11-main.py
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 784)               0         
_________________________________________________________________
dense (Dense)                (None, 256)               200960    
_________________________________________________________________
dropout (Dropout)            (None, 256)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 256)               65792     
_________________________________________________________________
dropout_1 (Dropout)          (None, 256)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 10)                2570      
=================================================================
Total params: 269,322
Trainable params: 269,322
Non-trainable params: 0
_________________________________________________________________
[array([[ 0.07595653,  0.09581027, -0.01175209, ..., -0.05243838,
         0.04540078, -0.09269386],
       [-0.01323028, -0.051954  , -0.01268669, ...,  0.00432736,
         0.03686089, -0.07104349],
       [-0.00924175, -0.04997446,  0.0242543 , ..., -0.06823482,
         0.05516547,  0.03175139],
       ...,
       [ 0.03273007, -0.04632739,  0.03379987, ..., -0.07104938,
        -0.05403581, -0.03537126],
       [ 0.09671515,  0.01242327,  0.08824161, ...,  0.00318845,
        -0.09294248,  0.00738481],
       [ 0.02152885,  0.01395665,  0.0101698 , ..., -0.00165461,
        -0.04027275, -0.00877788]], dtype=float32), array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0.], dtype=float32), array([[-0.06833467, -0.03180679,  0.00837077, ..., -0.1575552 ,
        -0.05222473, -0.13664919],
       [-0.05603951, -0.09797473,  0.00573276, ...,  0.16201185,
         0.10563677,  0.08692238],
       [ 0.0773556 , -0.07601337,  0.04726284, ..., -0.00312303,
         0.07468981, -0.11122718],
       ...,
       [-0.09624373, -0.03031957,  0.05009373, ...,  0.11220471,
        -0.12641405, -0.15056057],
       [ 0.07753017, -0.04575136, -0.06678326, ...,  0.03294286,
        -0.10902938, -0.08459996],
       [ 0.01357522, -0.07630654, -0.08225919, ...,  0.08785751,
        -0.07642032, -0.01332911]], dtype=float32), array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0.], dtype=float32), array([[-0.06897541, -0.16557665, -0.04771167, ...,  0.01455407,
         0.03382928, -0.17569515],
       [-0.05053294,  0.09438621,  0.11519638, ..., -0.01245164,
        -0.0719116 , -0.18455806],
       [ 0.09228224,  0.14074004,  0.06882233, ...,  0.05615992,
        -0.15130006,  0.02174817],
       ...,
       [ 0.00889782, -0.00705951,  0.04887312, ..., -0.08805028,
        -0.14918058, -0.1591385 ],
       [-0.14299504, -0.10059351, -0.10517051, ..., -0.06911735,
        -0.09655877,  0.04620347],
       [-0.16582027, -0.08827206,  0.16611351, ...,  0.01500075,
        -0.19330625, -0.11692349]], dtype=float32), array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=float32)]
ubuntu@alexa-ml:~/0x06-keras$


### [12. Test](./12-test.py)
Write a function def test_model(network, data, labels, verbose=True): that tests a neural network:
*    network is the network model to test
*    data is the input data to test the model with
*    labels are the correct one-hot labels of data
*    verbose is a boolean that determines if output should be printed during the testing process
*    Returns: the loss and accuracy of the model with the testing data, respectively

ubuntu@alexa-ml:~/0x06-keras$ cat 12-main.py 
#!/usr/bin/env python3

import numpy as np
one_hot = __import__('3-one_hot').one_hot
load_model = __import__('9-model').load_model
test_model = __import__('12-test').test_model


if __name__ == '__main__':
    datasets = np.load('../data/MNIST.npz')
    X_test = datasets['X_test']
    X_test = X_test.reshape(X_test.shape[0], -1)
    Y_test = datasets['Y_test']
    Y_test_oh = one_hot(Y_test)

    network = load_model('network2.h5')
    print(test_model(network, X_test, Y_test_oh))
ubuntu@alexa-ml:~/0x06-keras$ ./12-main.py 
10000/10000 [==============================] - 0s 43us/step
[0.09121923210024833, 0.9832]
ubuntu@alexa-ml:~/0x06-keras$


### [13. Predict](./13-predict.py)
Write a function def predict(network, data, verbose=False): that makes a prediction using a neural network:
*    network is the network model to make the prediction with
*    data is the input data to make the prediction with
*    verbose is a boolean that determines if output should be printed during the prediction process
*    Returns: the prediction for the data

ubuntu@alexa-ml:~/0x06-keras$ cat 13-main.py
#!/usr/bin/env python3

import numpy as np
one_hot = __import__('3-one_hot').one_hot
load_model = __import__('9-model').load_model
predict = __import__('13-predict').predict


if __name__ == '__main__':
    datasets = np.load('../data/MNIST.npz')
    X_test = datasets['X_test']
    X_test = X_test.reshape(X_test.shape[0], -1)
    Y_test = datasets['Y_test']

    network = load_model('network2.h5')
    Y_pred = predict(network, X_test)
    print(Y_pred)
    print(np.argmax(Y_pred, axis=1))
    print(Y_test)
ubuntu@alexa-ml:~/0x06-keras$ ./13-main.py
2018-11-30 21:13:04.692277: I tensorflow/core/common_runtime/process_util.cc:69] Creating new thread pool with default inter op setting: 2. Tune using inter_op_parallelism_threads for best performance.
[[1.09882777e-07 1.85020565e-06 7.01209501e-07 ... 9.99942422e-01
  2.60075751e-07 8.19494835e-06]
 [1.37503928e-08 1.84829651e-06 9.99997258e-01 ... 2.15385221e-09
  8.63893135e-09 8.08128995e-14]
 [1.03242555e-05 9.99097943e-01 1.67965060e-04 ... 5.23889903e-04
  7.54134162e-05 1.10524084e-07]
 ...
 [1.88145090e-11 5.88180065e-08 1.43965796e-12 ... 3.95040814e-07
  1.28503856e-08 2.26610467e-07]
 [2.37400890e-08 2.48911092e-09 1.20860308e-10 ... 1.69956849e-08
  5.97703838e-05 3.89016153e-10]
 [2.68221925e-08 1.28844213e-10 5.13091347e-09 ... 1.14895975e-11
  1.83396942e-09 7.46730282e-12]]
[7 2 1 ... 4 5 6]
[7 2 1 ... 4 5 6]
ubuntu@alexa-ml:~/0x06-keras$

---
## Authors

* **Robinson Montes** - [mecomonteshbtn](https://github.com/mecomonteshbtn)
