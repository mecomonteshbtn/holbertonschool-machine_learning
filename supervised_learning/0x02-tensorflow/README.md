# 0x02. Tensorflow

## Read or watch:
*    Low Level Intro (Excluding Datasets and Feature columns)
*    Graphs
*    Tensors
*    Variables
*    Placeholders
*    Save and Restore (Up to Save and restore models, excluded)
*    TensorFlow, why there are 3 files after saving the model?
*    Exporting and Importing a MetaGraph
*    TensorFlow - import meta graph and use variables from it

---
## References:
*    tf.Graph
*    tf.Session
*        tf.Session.run
*    tf.Tensor
*    tf.Variable
*    tf.constant
*    tf.placeholder
*    tf.Operation
*    tf.layers
*        tf.layers.Dense
*    tf.contrib.layers.variance_scaling_initializer
*    tf.nn
*        tf.nn.relu
*        tf.nn.sigmoid
*        tf.nn.tanh
*    tf.losses
*        tf.losses.softmax_cross_entropy
*    tf.train
*        tf.train.GradientDescentOptimizer
*            tf.train.GradientDescentOptimizer.minimize
*        tf.train.Saver
*            tf.train.Saver.save
*            tf.train.Saver.restore
*    tf.add_to_collection
*    tf.get_collection
*    tf.global_variables_initializer
*    tf.argmax
*    tf.equal
*    tf.set_random_seed
*    tf.name_scope

---
## Learning Objectives
*    What is tensorflow?
*    What is a session? graph?
*    What are tensors?
*    What are variables? constants? placeholders? How do you use them?
*    What are operations? How do you use them?
*    What are namespaces? How do you use them?
*    How to train a neural network in tensorflow
*    What is a checkpoint?
*    How to save/load a model with tensorflow
*    What is the graph collection?
*    How to add and get variables from the collection

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
*    Unless otherwise noted, you are not allowed to import any module except import tensorflow as tf
*    You are not allowed to use the keras module in tensorflow
*    All your files must be executable
*    The length of your files will be tested using wc

---
## More Info
Installing Tensorflow 1.12
```
$ pip install --user tensorflow==1.12
```
Optimize Tensorflow (Optional)

In order to get full use of your computer’s hardware, you will need to build tensorflow from source.

Here are some extra reading on why/how to do this:
*    How to compile Tensorflow with SSE4.2 and AVX instructions?
*    Installing Bazel on Ubuntu
*    Build from Source
*    Performance
*    Python Configuration Error: ‘PYTHON_BIN_PATH’ environment variable is not set

The following instructions assume you already have tensorflow (version 1.12) installed and that you do not have access to a GPU:

### 0. Install All Dependencies:
```
$ sudo apt-get install pkg-config zip g++ zlib1g-dev unzip python python3-dev
```
### 1. Install Bazel
```
$ wget https://github.com/bazelbuild/bazel/releases/download/0.18.1/bazel-0.18.1-installer-linux-x86_64.sh
$ chmod +x bazel-0.18.1-installer-linux-x86_64.sh
$ sudo ./bazel-0.18.1-installer-linux-x86_64.sh --bin=/bin

Add the line source /usr/local/lib/bazel/bin/bazel-complete.bash to your ~/.bashrc if you want bash to tab complete bazel.
```
### 2. Clone Tensorflow Repository
```
$ git clone https://github.com/tensorflow/tensorflow.git
$ cd tensorflow
$git checkout r1.12
```
### 3. Build and Install Tensorflow
```
$ export PYTHON_BIN_PATH=/usr/bin/python3 # or wherever python3 is located
$ bazel build --config=mkl --config=opt //tensorflow/tools/pip_package:build_pip_package
$ ./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
$ pip install --user /tmp/tensorflow_pkg/tensorflow-1.12.0-cp35-cp35m-linux_x86_64.whl
```
### 4. Remove Tensorflow Repository
```
$ cd ..
$ rm -rf tensorflow
```
Now tensorflow will be able to fully utilize the parallel processing capabilities of your computer’s hardware, which will make your training MUCH faster!

---
### [0. Placeholders](./0-create_placeholders.py)
Write the function def create_placeholders(nx, classes): that returns two placeholders, x and y, for the neural network:
*    nx: the number of feature columns in our data
*    classes: the number of classes in our classifier
*    Returns: placeholders named x and y, respectively
*        x is the placeholder for the input data to the neural network
*        y is the placeholder for the one-hot labels for the input data
```
ubuntu@alexa-ml:~/0x02-tensorflow$ cat 0-main.py 
#!/usr/bin/env python3

import tensorflow as tf

create_placeholders = __import__('0-create_placeholders').create_placeholders

x, y = create_placeholders(784, 10)
print(x)
print(y)
ubuntu@alexa-ml:~/0x02-tensorflow$ ./0-main.py 
Tensor("x:0", shape=(?, 784), dtype=float32)
Tensor("y:0", shape=(?, 10), dtype=float32)
ubuntu@alexa-ml:~/0x02-tensorflow$ 
```

### [1. Layers](./1-create_layer.py)
Write the function def create_layer(prev, n, activation):
*    prev is the tensor output of the previous layer
*    n is the number of nodes in the layer to create
*    activation is the activation function that the layer should use
*    use tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG") to implement He et. al initialization for the layer weights
*    each layer should be given the name layer
*    Returns: the tensor output of the layer
```
ubuntu@alexa-ml:~/0x02-tensorflow$ cat 1-main.py 
#!/usr/bin/env python3

import tensorflow as tf

create_placeholders = __import__('0-create_placeholders').create_placeholders
create_layer = __import__('1-create_layer').create_layer

x, y = create_placeholders(784, 10)
l = create_layer(x, 256, tf.nn.tanh)
print(l)
ubuntu@alexa-ml:~/0x02-tensorflow$ ./1-main.py 
Tensor("layer/Tanh:0", shape=(?, 256), dtype=float32)
ubuntu@alexa-ml:~/0x02-tensorflow$ 
```

### [2. Forward Propagation](./2-forward_prop.py)
Write the function def forward_prop(x, layer_sizes=[], activations=[]): that creates the forward propagation graph for the neural network:
*    x is the placeholder for the input data
*    layer_sizes is a list containing the number of nodes in each layer of the network
*    activations is a list containing the activation functions for each layer of the network
*    Returns: the prediction of the network in tensor form
*    For this function, you should import your create_layer function with create_layer = __import__('1-create_layer').create_layer
```
ubuntu@alexa-ml:~/0x02-tensorflow$ cat 2-main.py 
#!/usr/bin/env python3

import tensorflow as tf

create_placeholders = __import__('0-create_placeholders').create_placeholders
forward_prop = __import__('2-forward_prop').forward_prop

x, y = create_placeholders(784, 10)
y_pred = forward_prop(x, [256, 256, 10], [tf.nn.tanh, tf.nn.tanh, None])
print(y_pred)
ubuntu@alexa-ml:~/0x02-tensorflow$ ./2-main.py 
Tensor("layer_2/BiasAdd:0", shape=(?, 10), dtype=float32)
ubuntu@alexa-ml:~/0x02-tensorflow$ 
```

### [3. Accuracy](./3-calculate_accuracy.py)
Write the function def calculate_accuracy(y, y_pred): that calculates the accuracy of a prediction:
*    y is a placeholder for the labels of the input data
*    y_pred is a tensor containing the network’s predictions
*    Returns: a tensor containing the decimal accuracy of the prediction
```
ubuntu@alexa-ml:~/0x02-tensorflow$ cat 3-main.py 
#!/usr/bin/env python3

import tensorflow as tf

create_placeholders = __import__('0-create_placeholders').create_placeholders
forward_prop = __import__('2-forward_prop').forward_prop
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy

x, y = create_placeholders(784, 10)
y_pred = forward_prop(x, [256, 256, 10], [tf.nn.tanh, tf.nn.tanh, None])
accuracy = calculate_accuracy(y, y_pred)
print(accuracy)
ubuntu@alexa-ml:~/0x02-tensorflow$ ./3-main.py 
Tensor("Mean:0", shape=(), dtype=float32)
ubuntu@alexa-ml:~/0x02-tensorflow$
```

### [4. Loss](./4-calculate_loss.py)
Write the function def calculate_loss(y, y_pred): that calculates the softmax cross-entropy loss of a prediction:
*    y is a placeholder for the labels of the input data
*    y_pred is a tensor containing the network’s predictions
*    Returns: a tensor containing the loss of the prediction
```
ubuntu@alexa-ml:~/0x02-tensorflow$ cat 4-main.py 
#!/usr/bin/env python3

import tensorflow as tf

create_placeholders = __import__('0-create_placeholders').create_placeholders
forward_prop = __import__('2-forward_prop').forward_prop
calculate_loss = __import__('4-calculate_loss').calculate_loss

x, y = create_placeholders(784, 10)
y_pred = forward_prop(x, [256, 256, 10], [tf.nn.tanh, tf.nn.tanh, None])
loss = calculate_loss(y, y_pred)
print(loss)
ubuntu@alexa-ml:~/0x02-tensorflow$ ./4-main.py 
Tensor("softmax_cross_entropy_loss/value:0", shape=(), dtype=float32)
ubuntu@alexa-ml:~/0x02-tensorflow$ 
```

### [5. Train_Op](./5-create_train_op.py)
Write the function def create_train_op(loss, alpha): that creates the training operation for the network:
*    loss is the loss of the network’s prediction
*    alpha is the learning rate
*    Returns: an operation that trains the network using gradient descent
```
ubuntu@alexa-ml:~/0x02-tensorflow$ cat 5-main.py 
#!/usr/bin/env python3

import tensorflow as tf

create_placeholders = __import__('0-create_placeholders').create_placeholders
forward_prop = __import__('2-forward_prop').forward_prop
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_train_op = __import__('5-create_train_op').create_train_op


x, y = create_placeholders(784, 10)
y_pred = forward_prop(x, [256, 256, 10], [tf.nn.tanh, tf.nn.tanh, None])
loss = calculate_loss(y, y_pred)
train_op = create_train_op(loss, 0.01)
print(train_op)
ubuntu@alexa-ml:~/0x02-tensorflow$ ./5-main.py 
name: "GradientDescent"
op: "NoOp"
input: "^GradientDescent/update_layer/kernel/ApplyGradientDescent"
input: "^GradientDescent/update_layer/bias/ApplyGradientDescent"
input: "^GradientDescent/update_layer_1/kernel/ApplyGradientDescent"
input: "^GradientDescent/update_layer_1/bias/ApplyGradientDescent"
input: "^GradientDescent/update_layer_2/kernel/ApplyGradientDescent"
input: "^GradientDescent/update_layer_2/bias/ApplyGradientDescent"

ubuntu@alexa-ml:~/0x02-tensorflow$ 
```

### [6. Train](./6-train.py)
Write the function def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations, alpha, iterations, save_path="/tmp/model.ckpt"): that builds, trains, and saves a neural network classifier:
*    X_train is a numpy.ndarray containing the training input data
*    Y_train is a numpy.ndarray containing the training labels
*    X_valid is a numpy.ndarray containing the validation input data
*    Y_valid is a numpy.ndarray containing the validation labels
*    layer_sizes is a list containing the number of nodes in each layer of the network
*    activations is a list containing the activation functions for each layer of the network
*    alpha is the learning rate
*    iterations is the number of iterations to train over
*    save_path designates where to save the model
*    Add the following to the graph’s collection
*        placeholders x and y
*        tensors y_pred, loss, and accuracy
*        operation train_op
*    After every 100 iterations, the 0th iteration, and iterations iterations, print the following:
*        After {i} iterations: where i is the iteration
*        \tTraining Cost: {cost} where {cost} is the training cost
*        \tTraining Accuracy: {accuracy} where {accuracy} is the training accuracy
*        \tValidation Cost: {cost} where {cost} is the validation cost
*        \tValidation Accuracy: {accuracy} where {accuracy} is the validation accuracy
*    Reminder: the 0th iteration represents the model before any training has occurred
*    After training has completed, save the model to save_path
*    You may use the following imports:
*        calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
*        calculate_loss = __import__('4-calculate_loss').calculate_loss
*        create_placeholders = __import__('0-create_placeholders').create_placeholders
*        create_train_op = __import__('5-create_train_op').create_train_op
*        forward_prop = __import__('2-forward_prop').forward_prop
*    You are not allowed to use tf.saved_model
*    Returns: the path where the model was saved
```
ubuntu@alexa-ml:~/0x02-tensorflow$ cat 6-main.py 
#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
train = __import__('6-train').train

def one_hot(Y, classes):
    """convert an array to a one-hot matrix"""
    one_hot = np.zeros((Y.shape[0], classes))
    one_hot[np.arange(Y.shape[0]), Y] = 1
    return one_hot

if __name__ == '__main__':
    lib= np.load('../data/MNIST.npz')
    X_train_3D = lib['X_train']
    Y_train = lib['Y_train']
    X_train = X_train_3D.reshape((X_train_3D.shape[0], -1))
    Y_train_oh = one_hot(Y_train, 10)
    X_valid_3D = lib['X_valid']
    Y_valid = lib['Y_valid']
    X_valid = X_valid_3D.reshape((X_valid_3D.shape[0], -1))
    Y_valid_oh = one_hot(Y_valid, 10)

    layer_sizes = [256, 256, 10]
    activations = [tf.nn.tanh, tf.nn.tanh, None]
    alpha = 0.01
    iterations = 1000

    tf.set_random_seed(0)
    save_path = train(X_train, Y_train_oh, X_valid, Y_valid_oh, layer_sizes,
                      activations, alpha, iterations, save_path="./model.ckpt")
    print("Model saved in path: {}".format(save_path))
ubuntu@alexa-ml:~/0x02-tensorflow$ ./6-main.py 
2018-11-03 01:04:55.281078: I tensorflow/core/common_runtime/process_util.cc:69] Creating new thread pool with default inter op setting: 2. Tune using inter_op_parallelism_threads for best performance.
After 0 iterations:
    Training Cost: 2.8232274055480957
    Training Accuracy: 0.08726000040769577
    Validation Cost: 2.810533285140991
    Validation Accuracy: 0.08640000224113464
After 100 iterations:
    Training Cost: 0.8393384218215942
    Training Accuracy: 0.7824000120162964
    Validation Cost: 0.7826032042503357
    Validation Accuracy: 0.8061000108718872
After 200 iterations:
    Training Cost: 0.6094841361045837
    Training Accuracy: 0.8396000266075134
    Validation Cost: 0.5562412142753601
    Validation Accuracy: 0.8597999811172485

...

After 1000 iterations:
    Training Cost: 0.352960467338562
    Training Accuracy: 0.9004999995231628
    Validation Cost: 0.32148978114128113
    Validation Accuracy: 0.909600019454956
Model saved in path: ./model.ckpt
ubuntu@alexa-ml:~/0x02-tensorflow$ ls model.ckpt*
model.ckpt.data-00000-of-00001  model.ckpt.index  model.ckpt.meta
ubuntu@alexa-ml:~/0x02-tensorflow$
```

### [7. Evaluate](./7-evaluate.py)
Write the function def evaluate(X, Y, save_path): that evaluates the output of a neural network:
*    X is a numpy.ndarray containing the input data to evaluate
*    Y is a numpy.ndarray containing the one-hot labels for X
*    save_path is the location to load the model from
*    You are not allowed to use tf.saved_model
*    Returns: the network’s prediction, accuracy, and loss, respectively
```
ubuntu@alexa-ml:~/0x02-tensorflow$ cat 7-main.py 
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
evaluate = __import__('7-evaluate').evaluate

def one_hot(Y, classes):
    """convert an array to a one-hot matrix"""
    one_hot = np.zeros((Y.shape[0], classes))
    one_hot[np.arange(Y.shape[0]), Y] = 1
    return one_hot

if __name__ == '__main__':
    lib= np.load('../data/MNIST.npz')
    X_test_3D = lib['X_test']
    Y_test = lib['Y_test']
    X_test = X_test_3D.reshape((X_test_3D.shape[0], -1))
    Y_test_oh = one_hot(Y_test, 10)

    Y_pred_oh, accuracy, cost = evaluate(X_test, Y_test_oh, './model.ckpt')
    print("Test Accuracy:", accuracy)
    print("Test Cost:", cost)

    Y_pred = np.argmax(Y_pred_oh, axis=1)

    fig = plt.figure(figsize=(10, 10))
    for i in range(100):
        fig.add_subplot(10, 10, i + 1)
        plt.imshow(X_test_3D[i])
        plt.title(str(Y_test[i]) + ' : ' + str(Y_pred[i]))
        plt.axis('off')
    plt.tight_layout()
    plt.show()
ubuntu@alexa-ml:~/0x02-tensorflow$ ./7-main.py
2018-11-03 02:08:30.767168: I tensorflow/core/common_runtime/process_util.cc:69] Creating new thread pool with default inter op setting: 2. Tune using inter_op_parallelism_threads for best performance.
Test Accuracy: 0.9096
Test Cost: 0.32148978
```
## Authors

* **Robinson Montes** - [mecomonteshbtn](https://github.com/mecomonteshbtn)
