# 0x0E. Time Series Forecasting

---
## Read or watch:
*    [Time Series Prediction](https://www.youtube.com/watch?v=d4Sn6ny_5LI)
*    [Time Series Forecasting](https://www.tensorflow.org/tutorials/structured_data/time_series)
*    [Time Series Talk : Stationarity](https://www.youtube.com/watch?v=oY-j2Wof51c)
*    [tf.data: Build TensorFlow input pipelines](https://www.tensorflow.org/guide/data)
*    [Tensorflow Datasets](https://github.com/tensorflow/docs/blob/master/site/en/r1/guide/datasets.md)

## Definitions to skim
*    [Time Series](https://en.wikipedia.org/wiki/Time_series)
*    [Stationary Process](https://en.wikipedia.org/wiki/Stationary_process)

## References:
*    [tf.keras.layers.SimpleRNN](https://github.com/tensorflow/docs/blob/r1.12/site/en/api_docs/python/tf/keras/layers/SimpleRNN.md)
*    [tf.keras.layers.GRU](https://github.com/tensorflow/docs/blob/r1.12/site/en/api_docs/python/tf/keras/layers/GRU.md)
*    [tf.keras.layers.LSTM](https://github.com/tensorflow/docs/blob/r1.12/site/en/api_docs/python/tf/keras/layers/LSTM.md)
*    [tf.data.Dataset](https://github.com/tensorflow/docs/blob/r1.12/site/en/api_docs/python/tf/data/Dataset.md)

---
## Learning Objectives
At the end of this project, you are expected to be able to explain to anyone, without the help of Google:
-    What is time series forecasting?
-    What is a stationary process?
-    What is a sliding window?
-    How to preprocess time series data
-    How to create a data pipeline in tensorflow for time series data
-    How to perform time series forecasting with RNNs in tensorflow

## General Requirements
-    Allowed editors: vi, vim, emacs
-    All your files will be interpreted/compiled on Ubuntu 16.04 LTS using python3 (version 3.5)
-    Your files will be executed with numpy (version 1.15) and tensorflow (version 1.12)
-    All your files should end with a new line
-    The first line of all your files should be exactly #!/usr/bin/env python3
-    All of your files must be executable
-    A README.md file, at the root of the folder of the project, is mandatory
-    Your code should follow the pycodestyle style (version 2.4)
-    All your modules should have documentation (python3 -c 'print(__import__("my_module").__doc__)')
-    All your classes should have documentation (python3 -c 'print(__import__("my_module").MyClass.__doc__)')
-    All your functions (inside and outside a class) should have documentation (python3 -c 'print(__import__("my_module").my_function.__doc__)' and python3 -c 'print(__import__("my_module").MyClass.my_function.__doc__)')

---
## Files
| File | Description |
| ------ | ------ |
| [forecast_btc.py](forecast_btc.py) | Script that creates, trains, and validates a keras model for the forecasting of BTC. |
| [preprocess_data.py](preprocess_data.py) | Script to preprocess this data. |

---
## Build with
- Python (python 3.7)
- Numpy (numpy 1.19)
- Ubuntu 20.04 LTS 

---
### [0. When to Invest](./forecast_btc.py)
Bitcoin (BTC) became a trending topic after its [price](https://www.google.com/search?q=bitcoin+price) peaked in 2018. Many have sought to predict its value in order to accrue wealth. Let’s attempt to use our knowledge of RNNs to attempt just that.

Given the [coinbase](https://drive.google.com/file/d/16MgiuBfQKzXPoWFWi2w-LKJuZ7LgivpE/view) and [bitstamp](https://drive.google.com/file/d/15A-rLSrfZ0td7muSrYHy0WX9ZqrMweES/view) datasets, write a script, forecast_btc.py, that creates, trains, and validates a keras model for the forecasting of BTC:

*    Your model should use the past 24 hours of BTC data to predict the value of BTC at the close of the following hour (approximately how long the average transaction takes):
*    The datasets are formatted such that every row represents a 60 second time window containing:
     *    The start time of the time window in Unix time
     *    The open price in USD at the start of the time window
     *    The high price in USD within the time window
     *    The low price in USD within the time window
     *    The close price in USD at end of the time window
     *    The amount of BTC transacted in the time window
     *    The amount of Currency (USD) transacted in the time window
     *    The [volume-weighted](https://en.wikipedia.org/wiki/Volume-weighted_average_price#:~:text=In%20finance%2C%20volume%2Dweighted%20average,traded%20over%20the%20trading%20horizon.) average price in USD for the time window
*    Your model should use an RNN architecture of your choosing
*    Your model should use mean-squared error (MSE) as its cost function
*    You should use a tf.data.Dataset to feed data to your model

Because the dataset is raw, you will need to create a script, preprocess_data.py to preprocess this data. Here are some things to consider:

*    Are all of the data points useful?
*    Are all of the data features useful?
*    Should you rescale the data?
*    Is the current time window relevant?
*    How should you save this preprocessed data?

---
## Authors

* **Robinson Montes** - [mecomonteshbtn](https://github.com/mecomonteshbtn)
