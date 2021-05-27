# 0x11. Attention
## Description 


- How to use Transformers for Machine Translation
- How to write a custom train/test loop in Keras
- How to use Tensorflow Datasets

---
## Files
| File | Description |
| ------ | ------ |
| [0-dataset.py](0-dataset.py) | Class Dataset that loads and preps a dataset for machine translation. |
| [1-dataset.py](1-dataset.py) | Update the class Dataset in the instance method encode that encodes a translation into tokens. |
| [2-dataset.py](2-dataset.py) | Update the class Dataset adding the instance method tf_encode. |
| [3-dataset.py](3-dataset.py) | Update the class Dataset to set up the data pipeline. |
| [4-create_masks.py](4-create_masks.py) | Function def create_masks(inputs, target): that creates all masks for training/validation. |
| [5-transformer.py](5-transformer.py) | Function train_transformer that creates and trains a transformer model for machine translation of Portuguese to English using our previously created dataset. |

---
## Build with
- Python (python 3.6.12)
- Numpy (numpy 1.16)
- Ubuntu 16.04 LTS 
- Tensorflow (tensorflow 1.15)

---
## Authors

* **Robinson Montes** - [mecomonteshbtn](https://github.com/mecomonteshbtn)
