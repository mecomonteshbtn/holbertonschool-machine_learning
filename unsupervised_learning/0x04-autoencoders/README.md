# 0x04. Autoencoders

---
## Read or watch:
*    [Autoencoder - definition](https://www.youtube.com/watch?v=FzS3tMl4Nsc&t=73s)
*    [Autoencoder - loss function](https://www.youtube.com/watch?v=xTU79Zs4XKY)
*    [Deep learning - deep autoencoder](https://www.youtube.com/watch?v=z5ZYm_wJ37c)
*    [Introduction to autoencoders](https://www.jeremyjordan.me/autoencoders/)
*    [Variational Autoencoders - EXPLAINED! up to 12:55](https://www.youtube.com/watch?v=fcvYpzHmhvA)
*    [Variational Autoencoders](https://www.youtube.com/watch?v=9zKuYvjFFS8)
*    [Intuitively Understanding Variational Autoencoders](https://towardsdatascience.com/intuitively-understanding-variational-autoencoders-1bfe67eb5daf)
*    [Deep Generative Models up to Generative Adversarial Networks](https://towardsdatascience.com/deep-generative-models-25ab2821afd3)

### Definitions to skim:
*    [Kullback–Leibler divergence recall its use in t-SNE](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence)
*    [Autoencoder](https://en.wikipedia.org/wiki/Autoencoder)
*    [Generative model](https://en.wikipedia.org/wiki/Generative_model)

### References:
*    [The Deep Learning textbook - Chapter 14: Autoencoders](https://www.deeplearningbook.org/contents/autoencoders.html)
*    [Reducing the Dimensionality of Data with Neural Networks 2006](https://www.cs.toronto.edu/~hinton/science.pdf)

---
## Learning Objectives
 - What is an autoencoder?
 - What is latent space?
 - What is a bottleneck?
 - What is a sparse autoencoder?
 - What is a convolutional autoencoder?
 - What is a generative model?
 - What is a variational autoencoder?
 - What is the Kullback-Leibler divergence?

---
## Files
| File | Description |
| ------ | ------ |
| [0-vanilla.py](0-vanilla.py) | Function autoencoder that creates an autoencoder. |
| [1-sparse.py](1-sparse.py) | Function sparse that creates a sparse autoencoder. |
| [2-convolutional.py](2-convolutional.py) | Function autoencoder that creates a convolutional autoencoder. |
| [3-variational.py](3-variational.py) | Function autoencoder that creates a variational autoencoder. |

---
## Build with
- Python (python 3.6)
- Numpy (numpy 1.19)
- Ubuntu 20.04 LTS
- tensorflow (version 1.15)

---
### [0. "Vanilla" Autoencoder](./0-vanilla.py)
Write a function def autoencoder(input_dims, hidden_layers, latent_dims): that creates an autoencoder:
*    input_dims is an integer containing the dimensions of the model input
*    hidden_layers is a list containing the number of nodes for each hidden layer in the encoder, respectively
     *   the hidden layers should be reversed for the decoder
*    latent_dims is an integer containing the dimensions of the latent space representation
*    Returns: encoder, decoder, auto
     *   encoder is the encoder model
     *   decoder is the decoder model
     *   auto is the full autoencoder model
*    The autoencoder model should be compiled using adam optimization and binary cross-entropy loss
*    All layers should use a relu activation except for the last layer in the decoder, which should use sigmoid
```
$ cat 0-main.py
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist

autoencoder = __import__('0-vanilla').autoencoder

(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((-1, 784))
x_test = x_test.reshape((-1, 784))
np.random.seed(0)
tf.set_random_seed(0)
encoder, decoder, auto = autoencoder(784, [128, 64], 32)
auto.fit(x_train, x_train, epochs=50,batch_size=256, shuffle=True,
                validation_data=(x_test, x_test))
encoded = encoder.predict(x_test[:10])
print(np.mean(encoded))
reconstructed = decoder.predict(encoded)

for i in range(10):
    ax = plt.subplot(2, 10, i + 1)
    ax.axis('off')
    plt.imshow(x_test[i].reshape((28, 28)))
    ax = plt.subplot(2, 10, i + 11)
    ax.axis('off')
    plt.imshow(reconstructed[i].reshape((28, 28)))
plt.show()
$ ./0-main.py
Epoch 1/50
60000/60000 [==============================] - 5s 85us/step - loss: 0.2504 - val_loss: 0.1667
Epoch 2/50
60000/60000 [==============================] - 5s 84us/step - loss: 0.1498 - val_loss: 0.1361
Epoch 3/50
60000/60000 [==============================] - 5s 83us/step - loss: 0.1312 - val_loss: 0.1242
Epoch 4/50
60000/60000 [==============================] - 5s 79us/step - loss: 0.1220 - val_loss: 0.1173
Epoch 5/50
60000/60000 [==============================] - 5s 79us/step - loss: 0.1170 - val_loss: 0.1132

...

Epoch 46/50
60000/60000 [==============================] - 5s 80us/step - loss: 0.0852 - val_loss: 0.0850
Epoch 47/50
60000/60000 [==============================] - 5s 81us/step - loss: 0.0851 - val_loss: 0.0846
Epoch 48/50
60000/60000 [==============================] - 5s 84us/step - loss: 0.0850 - val_loss: 0.0848
Epoch 49/50
60000/60000 [==============================] - 5s 80us/step - loss: 0.0849 - val_loss: 0.0845
Epoch 50/50
60000/60000 [==============================] - 5s 85us/step - loss: 0.0848 - val_loss: 0.0844
6.5280433
```

### [1. Sparse Autoencoder](./1-sparse.py)
Write a function def autoencoder(input_dims, hidden_layers, latent_dims, lambtha): that creates a sparse autoencoder:
*    input_dims is an integer containing the dimensions of the model input
*    hidden_layers is a list containing the number of nodes for each hidden layer in the encoder, respectively
     *   the hidden layers should be reversed for the decoder
*    latent_dims is an integer containing the dimensions of the latent space representation
*    lambtha is the regularization parameter used for L1 regularization on the encoded output
*    Returns: encoder, decoder, auto
     *   encoder is the encoder model
     *   decoder is the decoder model
     *   auto is the sparse autoencoder model
*    The sparse autoencoder model should be compiled using adam optimization and binary cross-entropy loss
*    All layers should use a relu activation except for the last layer in the decoder, which should use sigmoid
```
$ cat 1-main.py
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist

autoencoder = __import__('1-sparse').autoencoder

(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((-1, 784))
x_test = x_test.reshape((-1, 784))
np.random.seed(0)
tf.set_random_seed(0)
encoder, decoder, auto = autoencoder(784, [128, 64], 32, 10e-6)
auto.fit(x_train, x_train, epochs=100,batch_size=256, shuffle=True,
                validation_data=(x_test, x_test))
encoded = encoder.predict(x_test[:10])
print(np.mean(encoded))
reconstructed = decoder.predict(encoded)

for i in range(10):
    ax = plt.subplot(2, 10, i + 1)
    ax.axis('off')
    plt.imshow(x_test[i].reshape((28, 28)))
    ax = plt.subplot(2, 10, i + 11)
    ax.axis('off')
    plt.imshow(reconstructed[i].reshape((28, 28)))
plt.show()
$ ./1-main.py
Epoch 1/50
60000/60000 [==============================] - 6s 102us/step - loss: 0.3123 - val_loss: 0.2538
Epoch 2/100
60000/60000 [==============================] - 6s 96us/step - loss: 0.2463 - val_loss: 0.2410
Epoch 3/100
60000/60000 [==============================] - 5s 90us/step - loss: 0.2400 - val_loss: 0.2381
Epoch 4/100
60000/60000 [==============================] - 5s 80us/step - loss: 0.2379 - val_loss: 0.2360
Epoch 5/100
60000/60000 [==============================] - 5s 82us/step - loss: 0.2360 - val_loss: 0.2339

...

Epoch 96/100
60000/60000 [==============================] - 5s 80us/step - loss: 0.1602 - val_loss: 0.1609
Epoch 97/100
60000/60000 [==============================] - 5s 84us/step - loss: 0.1601 - val_loss: 0.1608
Epoch 98/100
60000/60000 [==============================] - 5s 87us/step - loss: 0.1601 - val_loss: 0.1601
Epoch 99/100
60000/60000 [==============================] - 5s 89us/step - loss: 0.1601 - val_loss: 0.1604
Epoch 100/100
60000/60000 [==============================] - 5s 82us/step - loss: 0.1597 - val_loss: 0.1601
0.016292876
```

### [2. Convolutional Autoencoder](./2-convolutional.py)
Write a function def autoencoder(input_dims, filters, latent_dims): that creates a convolutional autoencoder:
*    input_dims is a tuple of integers containing the dimensions of the model input
*    filters is a list containing the number of filters for each convolutional layer in the encoder, respectively
     *   the filters should be reversed for the decoder
*    latent_dims is a tuple of integers containing the dimensions of the latent space representation
*    Each convolution in the encoder should use a kernel size of (3, 3) with same padding and relu activation, followed by max pooling of size (2, 2)
*    Each convolution in the decoder, except for the last two, should use a filter size of (3, 3) with same padding and relu activation, followed by upsampling of size (2, 2)
     *   The second to last convolution should instead use valid padding
     *   The last convolution should have the same number of filters as the number of channels in input_dims with sigmoid activation and no upsampling
*    Returns: encoder, decoder, auto
     *   encoder is the encoder model
     *   decoder is the decoder model
     *   auto is the full autoencoder model
*    The autoencoder model should be compiled using adam optimization and binary cross-entropy loss
```
$ cat 2-main.py
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist

autoencoder = __import__('2-convolutional').autoencoder

(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.expand_dims(x_train, axis=3)
x_test = np.expand_dims(x_test, axis=3)
print(x_train.shape)
print(x_test.shape)
np.random.seed(0)
tf.set_random_seed(0)
encoder, decoder, auto = autoencoder((28, 28, 1), [16, 8, 8], (4, 4, 8))
auto.fit(x_train, x_train, epochs=50, batch_size=256, shuffle=True,
                validation_data=(x_test, x_test))
encoded = encoder.predict(x_test[:10])
print(np.mean(encoded))
reconstructed = decoder.predict(encoded)[:,:,:,0]

for i in range(10):
    ax = plt.subplot(2, 10, i + 1)
    ax.axis('off')
    plt.imshow(x_test[i,:,:,0])
    ax = plt.subplot(2, 10, i + 11)
    ax.axis('off')
    plt.imshow(reconstructed[i])
plt.show()
$ ./2-main.py
Epoch 1/50
60000/60000 [==============================] - 49s 810us/step - loss: 63.9743 - val_loss: 43.5109
Epoch 2/50
60000/60000 [==============================] - 48s 804us/step - loss: 39.9287 - val_loss: 37.1333
Epoch 3/50
60000/60000 [==============================] - 48s 803us/step - loss: 35.7883 - val_loss: 34.1952
Epoch 4/50
60000/60000 [==============================] - 48s 792us/step - loss: 33.4408 - val_loss: 32.2462
Epoch 5/50
60000/60000 [==============================] - 47s 791us/step - loss: 31.8871 - val_loss: 30.9729

...

Epoch 46/50
60000/60000 [==============================] - 45s 752us/step - loss: 23.9016 - val_loss: 23.6926
Epoch 47/50
60000/60000 [==============================] - 45s 754us/step - loss: 23.9029 - val_loss: 23.7102
Epoch 48/50
60000/60000 [==============================] - 45s 750us/step - loss: 23.8331 - val_loss: 23.5239
Epoch 49/50
60000/60000 [==============================] - 46s 771us/step - loss: 23.8047 - val_loss: 23.5510
Epoch 50/50
60000/60000 [==============================] - 46s 772us/step - loss: 23.7744 - val_loss: 23.4939
2.4494107
```

### [3. Variational Autoencoder](./3-variational.py)
Write a function def autoencoder(input_dims, hidden_layers, latent_dims): that creates a variational autoencoder:
*    input_dims is an integer containing the dimensions of the model input
*    hidden_layers is a list containing the number of nodes for each hidden layer in the encoder, respectively
     *   the hidden layers should be reversed for the decoder
*    latent_dims is an integer containing the dimensions of the latent space representation
*    Returns: encoder, decoder, auto
     *   encoder is the encoder model, which should output the latent representation, the mean, and the log variance, respectively
     *   decoder is the decoder model
     *   auto is the full autoencoder model
*    The autoencoder model should be compiled using adam optimization and binary cross-entropy loss
*    All layers should use a relu activation except for the mean and log variance layers in the encoder, which should use None, and the last layer in the decoder, which should use sigmoid
```
$ cat 3-main.py
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist

autoencoder = __import__('3-variational').autoencoder

(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((-1, 784))
x_test = x_test.reshape((-1, 784))
np.random.seed(0)
tf.set_random_seed(0)
encoder, decoder, auto = autoencoder(784, [512], 2)
auto.fit(x_train, x_train, epochs=50, batch_size=256, shuffle=True,
                validation_data=(x_test, x_test))
encoded, mu, log_sig = encoder.predict(x_test[:10])
print(mu)
print(np.exp(log_sig / 2))
reconstructed = decoder.predict(encoded).reshape((-1, 28, 28))
x_test = x_test.reshape((-1, 28, 28))

for i in range(10):
    ax = plt.subplot(2, 10, i + 1)
    ax.axis('off')
    plt.imshow(x_test[i])
    ax = plt.subplot(2, 10, i + 11)
    ax.axis('off')
    plt.imshow(reconstructed[i])
plt.show()


l1 = np.linspace(-3, 3, 25)
l2 = np.linspace(-3, 3, 25)
L = np.stack(np.meshgrid(l1, l2, sparse=False, indexing='ij'), axis=2)
G = decoder.predict(L.reshape((-1, 2)), batch_size=125)

for i in range(25*25):
    ax = plt.subplot(25, 25, i + 1)
    ax.axis('off')
    plt.imshow(G[i].reshape((28, 28)))
plt.show()
$ ./3-main.py
Epoch 1/50
60000/60000 [==============================] - 15s 242us/step - loss: 214.4525 - val_loss: 177.2306
Epoch 2/50
60000/60000 [==============================] - 11s 175us/step - loss: 171.7558 - val_loss: 168.7191
Epoch 3/50
60000/60000 [==============================] - 11s 182us/step - loss: 167.4977 - val_loss: 166.5061
Epoch 4/50
60000/60000 [==============================] - 11s 179us/step - loss: 165.6473 - val_loss: 165.1279
Epoch 5/50
60000/60000 [==============================] - 11s 181us/step - loss: 164.0918 - val_loss: 163.7083

...

Epoch 46/50
60000/60000 [==============================] - 15s 249us/step - loss: 148.1491 - val_loss: 151.3205
Epoch 47/50
60000/60000 [==============================] - 12s 204us/step - loss: 148.0358 - val_loss: 151.2141
Epoch 48/50
60000/60000 [==============================] - 11s 179us/step - loss: 147.9396 - val_loss: 151.3823
Epoch 49/50
60000/60000 [==============================] - 13s 223us/step - loss: 147.8144 - val_loss: 151.4026
Epoch 50/50
60000/60000 [==============================] - 11s 189us/step - loss: 147.6572 - val_loss: 151.1969
[[-0.33454233 -3.0770888 ]
 [-0.68772286  0.52945304]
 [ 3.1372023  -1.5037178 ]
 [-0.46997875  2.4711971 ]
 [-2.239822   -0.91364074]
 [ 2.7829633  -1.2185467 ]
 [-0.8319831  -0.97430193]
 [-1.3994675  -0.16924876]
 [-0.2642493  -0.45080736]
 [-0.3476941  -1.5133704 ]]
[[0.07307572 0.18656202]
 [0.04450396 0.03617072]
 [0.15917557 0.09816898]
 [0.07885559 0.056187  ]
 [0.11542598 0.07378525]
 [0.14280568 0.0857826 ]
 [0.0790622  0.07540198]
 [0.08175724 0.05216441]
 [0.05364255 0.05444151]
 [0.04280119 0.07214296]]
```
---
## Authors

* **Robinson Montes** - [mecomonteshbtn](https://github.com/mecomonteshbtn)
