# 0x11. Attention

---
## Description 

- What is the attention mechanism?
- How to apply attention to RNNs
- What is a transformer?
- How to create an encoder-decoder transformer model
- What is GPT?
- What is BERT?
- What is self-supervised learning?
- How to use BERT for specific NLP tasks
- What is SQuAD? GLUE?

---
## Resources:
*    [Attention Model Intuition](https://www.youtube.com/watch?v=SysgYptB198)
*    [Attention Model](https://www.youtube.com/watch?v=quoGRI-1l0A)
*    [How Transformers work in deep learning and NLP: an intuitive introduction](https://theaisummer.com/transformer/)
*    [Transformers](https://towardsdatascience.com/transformers-explained-visually-part-1-overview-of-functionality-95a6dd460452)
*    [Bert, GPT : The Illustrated GPT-2 - Visualizing Transformer Language Models](https://jalammar.github.io/illustrated-gpt2/)
*    [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/)
*    [Glue](https://mccormickml.com/2019/11/05/GLUE/)
*    [Self supervised learning](https://lilianweng.github.io/lil-log/2019/11/10/self-supervised-learning.html)

---
## Files
| File | Description |
| ------ | ------ |
| [0-rnn_encoder.py](0-rnn_encoder.py) | Class RNNEncoder that inherits from tensorflow.keras.layers.Layer to encode for machine translation. |
| [1-self_attention.py](1-self_attention.py) | Class SelfAttention that inherits from tensorflow.keras.layers.Layer to calculate the attention for machine translation based on [this paper](https://arxiv.org/pdf/1409.0473.pdf). |
| [2-rnn_decoder.py](2-rnn_decoder.py) | Class RNNDecoder that inherits from tensorflow.keras.layers.Layer to decode for machine translation. |
| [4-positional_encoding.py](4-positional_encoding.py) | Function positional_encoding that calculates the positional encoding for a transformer. |
| [5-sdp_attention.py](5-sdp_attention.py) | Function sdp_attention that calculates the scaled dot product attention. |
| [6-multihead_attention.py](6-multihead_attention.py) | Class MultiHeadAttention that inherits from tensorflow.keras.layers.Layer to perform multi head attention. |
| [7-transformer_encoder_block.py](7-transformer_encoder_block.py) | Class EncoderBlock that inherits from tensorflow.keras.layers.Layer to create an encoder block for a transformer. |
| [8-transformer_decoder_block.py](8-transformer_decoder_block.py) | Class DecoderBlock that inherits from tensorflow.keras.layers.Layer to create an encoder block for a transformer. |
| [9-transformer_encoder.py](9-transformer_encoder.py) | Class Encoder that inherits from tensorflow.keras.layers.Layer to create the encoder for a transformer. |
| [10-transformer_decoder.py](10-transformer_decoder.py) | Class Decoder that inherits from tensorflow.keras.layers.Layer to create the decoder for a transformer. |
| [11-transformer.py](11-transformer.py) | Class Transformer that inherits from tensorflow.keras.Model to create a transformer network. |
| []() | . |

---
### Build with
- Python (python 3.7)
- Numpy (numpy 1.19)
- Ubuntu 20.04 LTS 

---
### [0. RNN Encoder](./0-rnn_encoder.py)
*    [Encoder-Decoder Architecture](https://compstat-lmu.github.io/seminar_nlp_ss20/recurrent-neural-networks-and-their-applications-in-nlp.html)
Create a class RNNEncoder that inherits from tensorflow.keras.layers.Layer to encode for machine translation:
*    Class constructor def __init__(self, vocab, embedding, units, batch):
     *   vocab is an integer representing the size of the input vocabulary
     *   embedding is an integer representing the dimensionality of the embedding vector
     *   units is an integer representing the number of hidden units in the RNN cell
     *   batch is an integer representing the batch size
     *   Sets the following public instance attributes:
     *   batch - the batch size
     *   units - the number of hidden units in the RNN cell
     *   embedding - a keras Embedding layer that converts words from the vocabulary into an embedding vector
     *   gru - a keras GRU layer with units units
     *   Should return both the full sequence of outputs as well as the last hidden state
     *   Recurrent weights should be initialized with glorot_uniform
*    Public instance method def initialize_hidden_state(self):
     *   Initializes the hidden states for the RNN cell to a tensor of zeros
     *   Returns: a tensor of shape (batch, units)containing the initialized hidden states
*    Public instance method def call(self, x, initial):
     *   x is a tensor of shape (batch, input_seq_len) containing the input to the encoder layer as word indices within the vocabulary
     *   initial is a tensor of shape (batch, units) containing the initial hidden state
     *   Returns: outputs, hidden
     *   outputs is a tensor of shape (batch, input_seq_len, units)containing the outputs of the encoder
     *   hidden is a tensor of shape (batch, units) containing the last hidden state of the encoder
```
$ cat 0-main.py
#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
RNNEncoder = __import__('0-rnn_encoder').RNNEncoder

encoder = RNNEncoder(1024, 128, 256, 32)
print(encoder.batch)
print(encoder.units)
print(type(encoder.embedding))
print(type(encoder.gru))

initial = encoder.initialize_hidden_state()
print(initial)
x = tf.convert_to_tensor(np.random.choice(1024, 320).reshape((32, 10)))
outputs, hidden = encoder(x, initial)
print(outputs)
print(hidden)
$ ./0-main.py
32
256
<class 'tensorflow.python.keras.layers.embeddings.Embedding'>
<class 'tensorflow.python.keras.layers.recurrent.GRU'>
Tensor("zeros:0", shape=(32, 256), dtype=float32)
Tensor("rnn_encoder/gru/transpose_1:0", shape=(32, 10, 256), dtype=float32)
Tensor("rnn_encoder/gru/while/Exit_2:0", shape=(32, 256), dtype=float32)
$

Ignore the Warning messages in the output
```

### [1. Self Attention](./1-self_attention.py)
Create a class SelfAttention that inherits from tensorflow.keras.layers.Layer to calculate the attention for machine translation based on [this paper](https://arxiv.org/pdf/1409.0473.pdf):
*    Class constructor def __init__(self, units):
     *   units is an integer representing the number of hidden units in the alignment model
     *   Sets the following public instance attributes:
     *   W - a Dense layer with units units, to be applied to the previous decoder hidden state
     *   U - a Dense layer with units units, to be applied to the encoder hidden states
     *   V - a Dense layer with 1 units, to be applied to the tanh of the sum of the outputs of W and U
*    Public instance method def call(self, s_prev, hidden_states):
     *   s_prev is a tensor of shape (batch, units) containing the previous decoder hidden state
     *   hidden_states is a tensor of shape (batch, input_seq_len, units)containing the outputs of the encoder
     *   Returns: context, weights
     *   context is a tensor of shape (batch, units) that contains the context vector for the decoder
     *   weights is a tensor of shape (batch, input_seq_len, 1) that contains the attention weights
```
$ cat 1-main.py
#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention

attention = SelfAttention(256)
print(attention.W)
print(attention.U)
print(attention.V)
s_prev = tf.convert_to_tensor(np.random.uniform(size=(32, 256)), preferred_dtype='float32')
hidden_states = tf.convert_to_tensor(np.random.uniform(size=(32, 10, 256)), preferred_dtype='float32')
context, weights = attention(s_prev, hidden_states)
print(context)
print(weights)
$ ./1-main.py
<tensorflow.python.keras.layers.core.Dense object at 0x12309d3c8>
<tensorflow.python.keras.layers.core.Dense object at 0xb28536b38>
<tensorflow.python.keras.layers.core.Dense object at 0xb28536e48>
Tensor("self_attention/Sum:0", shape=(32, 256), dtype=float64)
Tensor("self_attention/transpose_1:0", shape=(32, 10, 1), dtype=float64)
$

Ignore the Warning messages in the output
```

### [2. RNN Decoder](./2-rnn_decoder.py)
Create a class RNNDecoder that inherits from tensorflow.keras.layers.Layer to decode for machine translation:
*    Class constructor def __init__(self, vocab, embedding, units, batch):
     *   vocab is an integer representing the size of the output vocabulary
     *   embedding is an integer representing the dimensionality of the embedding vector
     *   units is an integer representing the number of hidden units in the RNN cell
     *   batch is an integer representing the batch size
     *   Sets the following public instance attributes:
     *   embedding - a keras Embedding layer that converts words from the vocabulary into an embedding vector
     *   gru - a keras GRU layer with units units
     *   Should return both the full sequence of outputs as well as the last hidden state
     *   Recurrent weights should be initialized with glorot_uniform
     *   F - a Dense layer with vocab units
*    Public instance method def call(self, x, s_prev, hidden_states):
     *   x is a tensor of shape (batch, 1) containing the previous word in the target sequence as an index of the target vocabulary
     *   s_prev is a tensor of shape (batch, units) containing the previous decoder hidden state
     *   hidden_states is a tensor of shape (batch, input_seq_len, units)containing the outputs of the encoder
     *   You should use SelfAttention = __import__('1-self_attention').SelfAttention
     *   You should concatenate the context vector with x in that order
     *   Returns: y, s
     *   y is a tensor of shape (batch, vocab) containing the output word as a one hot vector in the target vocabulary
     *   s is a tensor of shape (batch, units) containing the new decoder hidden state
```
$ cat 2-main.py
#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
RNNDecoder = __import__('2-rnn_decoder').RNNDecoder

decoder = RNNDecoder(2048, 128, 256, 32)
print(decoder.embedding)
print(decoder.gru)
print(decoder.F)
x = tf.convert_to_tensor(np.random.choice(2048, 32).reshape((32, 1)))
s_prev = tf.convert_to_tensor(np.random.uniform(size=(32, 256)).astype('float32'))
hidden_states = tf.convert_to_tensor(np.random.uniform(size=(32, 10, 256)).astype('float32'))
y, s = decoder(x, s_prev, hidden_states)
print(y)
print(s)
$ ./2-main.py
<tensorflow.python.keras.layers.embeddings.Embedding object at 0x1321113c8>
<tensorflow.python.keras.layers.recurrent.GRU object at 0xb375aab00>
<tensorflow.python.keras.layers.core.Dense object at 0xb375d5128>
Tensor("rnn_decoder/dense/BiasAdd:0", shape=(32, 2048), dtype=float32)
Tensor("rnn_decoder/gru/while/Exit_2:0", shape=(32, 256), dtype=float32)
$

Ignore the Warning messages in the output
```

### [3. Positional Encoding](./4-positional_encoding.py)
Write the function def positional_encoding(max_seq_len, dm): that calculates the positional encoding for a transformer:
*    max_seq_len is an integer representing the maximum sequence length
*    dm is the model depth
*    Returns: a numpy.ndarray of shape (max_seq_len, dm) containing the positional encoding vectors
*    You should use import numpy as np
```
$ cat 4-main.py
#!/usr/bin/env python3

import numpy as np
positional_encoding = __import__('4-positional_encoding').positional_encoding

PE = positional_encoding(30, 512)
print(PE.shape)
print(PE)
$ ./4-main.py
(30, 512)
[[ 0.00000000e+00  1.00000000e+00  0.00000000e+00 ...  1.00000000e+00
   0.00000000e+00  1.00000000e+00]
 [ 8.41470985e-01  5.40302306e-01  8.21856190e-01 ...  9.99999994e-01
   1.03663293e-04  9.99999995e-01]
 [ 9.09297427e-01 -4.16146837e-01  9.36414739e-01 ...  9.99999977e-01
   2.07326584e-04  9.99999979e-01]
 ...
 [ 9.56375928e-01 -2.92138809e-01  7.91416314e-01 ...  9.99995791e-01
   2.79890525e-03  9.99996083e-01]
 [ 2.70905788e-01 -9.62605866e-01  9.53248145e-01 ...  9.99995473e-01
   2.90256812e-03  9.99995788e-01]
 [-6.63633884e-01 -7.48057530e-01  2.94705106e-01 ...  9.99995144e-01
   3.00623096e-03  9.99995481e-01]]
$
```

### [4. Scaled Dot Product Attention](./5-sdp_attention.py)
Write the function def sdp_attention(Q, K, V, mask=None) that calculates the scaled dot product attention:
*    Q is a tensor with its last two dimensions as (..., seq_len_q, dk) containing the query matrix
*    K is a tensor with its last two dimensions as (..., seq_len_v, dk) containing the key matrix
*    V is a tensor with its last two dimensions as (..., seq_len_v, dv) containing the value matrix
*    mask is a tensor that can be broadcast into (..., seq_len_q, seq_len_v) containing the optional mask, or defaulted to None
     *   if mask is not None, multiply -1e9 to the mask and add it to the scaled matrix multiplication
*    The preceding dimensions of Q, K, and V are the same
*    Returns: output, weights
     *   outputa tensor with its last two dimensions as (..., seq_len_q, dv) containing the scaled dot product attention
     *   weights a tensor with its last two dimensions as (..., seq_len_q, seq_len_v) containing the attention weights
```
$ cat 5-main.py
#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention

np.random.seed(0)
Q = tf.convert_to_tensor(np.random.uniform(size=(50, 10, 256)).astype('float32'))
K = tf.convert_to_tensor(np.random.uniform(size=(50, 15, 256)).astype('float32'))
V = tf.convert_to_tensor(np.random.uniform(size=(50, 15, 512)).astype('float32'))
output, weights = sdp_attention(Q, K, V)
print(output)
print(weights)
$ ./5-main.py
Tensor("MatMul_1:0", shape=(50, 10, 512), dtype=float32)
Tensor("Softmax:0", shape=(50, 10, 15), dtype=float32)
$
```

### [5. Multi Head Attention](./6-multihead_attention.py)
*    [Why multi-head self attention works: math, intuitions and 10+1 hidden insights](https://theaisummer.com/self-attention/)

Create a class MultiHeadAttention that inherits from tensorflow.keras.layers.Layer to perform multi head attention:
*    Class constructor def __init__(self, dm, h):
     *   dm is an integer representing the dimensionality of the model
     *   h is an integer representing the number of heads
     *   dm is divisible by h
     *   Sets the following public instance attributes:
     *   h - the number of heads
     *   dm - the dimensionality of the model
     *   depth - the depth of each attention head
     *   Wq - a Dense layer with dm units, used to generate the query matrix
     *   Wk - a Dense layer with dm units, used to generate the key matrix
     *   Wv - a Dense layer with dm units, used to generate the value matrix
     *   linear - a Dense layer with dm units, used to generate the attention output
*    Public instance method def call(self, Q, K, V, mask):
     *   Q is a tensor of shape (batch, seq_len_q, dk) containing the input to generate the query matrix
     *   K is a tensor of shape (batch, seq_len_v, dk) containing the input to generate the key matrix
     *   V is a tensor of shape (batch, seq_len_v, dv) containing the input to generate the value matrix
     *   mask is always None
     *   Returns: output, weights
     *   outputa tensor with its last two dimensions as (..., seq_len_q, dm) containing the scaled dot product attention
     *   weights a tensor with its last three dimensions as (..., h, seq_len_q, seq_len_v) containing the attention weights
*    You should use sdp_attention = __import__('5-sdp_attention').sdp_attention
```
$ cat 6-main.py
#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention

mha = MultiHeadAttention(512, 8)
print(mha.dm)
print(mha.h)
print(mha.depth)
print(mha.Wq)
print(mha.Wk)
print(mha.Wv)
print(mha.linear)
Q = tf.convert_to_tensor(np.random.uniform(size=(50, 15, 256)).astype('float32'))
K = tf.convert_to_tensor(np.random.uniform(size=(50, 15, 256)).astype('float32'))
V = tf.convert_to_tensor(np.random.uniform(size=(50, 15, 256)).astype('float32'))
output, weights = mha(Q, K, V, None)
print(output)
print(weights)
$ ./6-main.py
512
8
64
<tensorflow.python.keras.layers.core.Dense object at 0xb2c585b38>
<tensorflow.python.keras.layers.core.Dense object at 0xb2c585e48>
<tensorflow.python.keras.layers.core.Dense object at 0xb2c5b1198>
<tensorflow.python.keras.layers.core.Dense object at 0xb2c5b14a8>
Tensor("multi_head_attention/dense_3/BiasAdd:0", shape=(50, 15, 512), dtype=float32)
Tensor("multi_head_attention/Softmax:0", shape=(50, 8, 15, 15), dtype=float32)
$

Ignore the Warning messages in the output
```

### [6. Transformer Encoder Block](./7-transformer_encoder_block.py)
Create a class EncoderBlock that inherits from tensorflow.keras.layers.Layer to create an encoder block for a transformer:
*    Class constructor def __init__(self, dm, h, hidden, drop_rate=0.1):
     *   dm - the dimensionality of the model
     *   h - the number of heads
     *   hidden - the number of hidden units in the fully connected layer
     *   drop_rate - the dropout rate
     *   Sets the following public instance attributes:
     *   mha - a MultiHeadAttention layer
     *   dense_hidden - the hidden dense layer with hidden units and relu activation
     *   dense_output - the output dense layer with dm units
     *   layernorm1 - the first layer norm layer, with epsilon=1e-6
     *   layernorm2 - the second layer norm layer, with epsilon=1e-6
     *   dropout1 - the first dropout layer
     *   dropout2 - the second dropout layer
*    Public instance method call(self, x, training, mask=None):
     *   x - a tensor of shape (batch, input_seq_len, dm)containing the input to the encoder block
     *   training - a boolean to determine if the model is training
     *   mask - the mask to be applied for multi head attention
     *   Returns: a tensor of shape (batch, input_seq_len, dm) containing the block’s output
*    You should use MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention
```
$ cat 7-main
#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
EncoderBlock = __import__('7-transformer_encoder_block').EncoderBlock

eblock = EncoderBlock(512, 8, 2048)
print(eblock.mha)
print(eblock.dense_hidden)
print(eblock.dense_output)
print(eblock.layernorm1)
print(eblock.layernorm2)
print(eblock.dropout1)
print(eblock.dropout2)
x = tf.random.uniform((32, 10, 512))
output = eblock(x, True, None)
print(output)
$ ./7-main.py
<6-multihead_attention.MultiHeadAttention object at 0x12c61b390>
<tensorflow.python.keras.layers.core.Dense object at 0xb31ae1860>
<tensorflow.python.keras.layers.core.Dense object at 0xb31ae1b70>
<tensorflow.python.keras.layers.normalization.LayerNormalization object at 0xb31ae1e80>
<tensorflow.python.keras.layers.normalization.LayerNormalization object at 0xb31aea128>
<tensorflow.python.keras.layers.core.Dropout object at 0xb31aea390>
<tensorflow.python.keras.layers.core.Dropout object at 0xb31aea518>
Tensor("encoder_block/layer_normalization_1/batchnorm/add_1:0", shape=(32, 10, 512), dtype=float32)
$

Ignore the Warning messages in the output
```

### [7. Transformer Decoder Block](./8-transformer_decoder_block.py)
Create a class DecoderBlock that inherits from tensorflow.keras.layers.Layer to create an encoder block for a transformer:
*    Class constructor def __init__(self, dm, h, hidden, drop_rate=0.1):
     *   dm - the dimensionality of the model
     *   h - the number of heads
     *   hidden - the number of hidden units in the fully connected layer
     *   drop_rate - the dropout rate
     *   Sets the following public instance attributes:
     *   mha1 - the first MultiHeadAttention layer
     *   mha2 - the second MultiHeadAttention layer
     *   dense_hidden - the hidden dense layer with hidden units and relu activation
     *   dense_output - the output dense layer with dm units
     *   layernorm1 - the first layer norm layer, with epsilon=1e-6
     *   layernorm2 - the second layer norm layer, with epsilon=1e-6
     *   layernorm3 - the third layer norm layer, with epsilon=1e-6
     *   dropout1 - the first dropout layer
     *   dropout2 - the second dropout layer
     *   dropout3 - the third dropout layer
*    Public instance method def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
     *   x - a tensor of shape (batch, target_seq_len, dm)containing the input to the decoder block
     *   encoder_output - a tensor of shape (batch, input_seq_len, dm)containing the output of the encoder
     *   training - a boolean to determine if the model is training
     *   look_ahead_mask - the mask to be applied to the first multi head attention layer
     *   padding_mask - the mask to be applied to the second multi head attention layer
     *   Returns: a tensor of shape (batch, target_seq_len, dm) containing the block’s output
*    You should use MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention
```
$ cat 8-main.py
#!/usr/bin/env python3

import tensorflow as tf
DecoderBlock = __import__('8-transformer_decoder_block').DecoderBlock

dblock = DecoderBlock(512, 8, 2048)
print(dblock.mha1)
print(dblock.mha2)
print(dblock.dense_hidden)
print(dblock.dense_output)
print(dblock.layernorm1)
print(dblock.layernorm2)
print(dblock.layernorm3)
print(dblock.dropout1)
print(dblock.dropout2)
print(dblock.dropout3)
x = tf.random.uniform((32, 15, 512))
hidden_states = tf.random.uniform((32, 10, 512))
output = dblock(x, hidden_states, False, None, None)
print(output)
$ ./8-main.py
<6-multihead_attention.MultiHeadAttention object at 0x1313f4400>
<6-multihead_attention.MultiHeadAttention object at 0xb368bc9b0>
<tensorflow.python.keras.layers.core.Dense object at 0xb368c37b8>
<tensorflow.python.keras.layers.core.Dense object at 0xb368c3ac8>
<tensorflow.python.keras.layers.normalization.LayerNormalization object at 0xb368c3dd8>
<tensorflow.python.keras.layers.normalization.LayerNormalization object at 0xb368cb080>
<tensorflow.python.keras.layers.normalization.LayerNormalization object at 0xb368cb2e8>
<tensorflow.python.keras.layers.core.Dropout object at 0xb368cb550>
<tensorflow.python.keras.layers.core.Dropout object at 0xb368cb6d8>
<tensorflow.python.keras.layers.core.Dropout object at 0xb368cb828>
Tensor("decoder_block/layer_normalization_2/batchnorm/add_1:0", shape=(32, 15, 512), dtype=float32)
$

Ignore the Warning messages in the output
```

### [8. Transformer Encoder](./9-transformer_encoder.py)
Create a class Encoder that inherits from tensorflow.keras.layers.Layer to create the encoder for a transformer:
*    Class constructor def __init__(self, N, dm, h, hidden, input_vocab, max_seq_len, drop_rate=0.1):
     *   N - the number of blocks in the encoder
     *   dm - the dimensionality of the model
     *   h - the number of heads
     *   hidden - the number of hidden units in the fully connected layer
     *   input_vocab - the size of the input vocabulary
     *   max_seq_len - the maximum sequence length possible
     *   drop_rate - the dropout rate
     *   Sets the following public instance attributes:
     *   N - the number of blocks in the encoder
     *   dm - the dimensionality of the model
     *   embedding - the embedding layer for the inputs
     *   positional_encoding - a numpy.ndarray of shape (max_seq_len, dm) containing the positional encodings
     *   blocks - a list of length N containing all of the EncoderBlock‘s
     *   dropout - the dropout layer, to be applied to the positional encodings
*    Public instance method call(self, x, training, mask):
     *   x - a tensor of shape (batch, input_seq_len, dm)containing the input to the encoder
     *   training - a boolean to determine if the model is training
     *   mask - the mask to be applied for multi head attention
     *   Returns: a tensor of shape (batch, input_seq_len, dm) containing the encoder output
*    You should use positional_encoding = __import__('4-positional_encoding').positional_encoding and EncoderBlock = __import__('7-transformer_encoder_block').EncoderBlock
```
$ cat 9-main.py
#!/usr/bin/env python3

import tensorflow as tf
Encoder = __import__('9-transformer_encoder').Encoder

encoder = Encoder(6, 512, 8, 2048, 10000, 1000)
print(encoder.dm)
print(encoder.N)
print(encoder.embedding)
print(encoder.positional_encoding)
print(encoder.blocks)
print(encoder.dropout)
x = tf.random.uniform((32, 10))
output = encoder(x, True, None)
print(output)
$ ./9-main.py
512
6
<tensorflow.python.keras.layers.embeddings.Embedding object at 0xb2981acc0>
[[ 0.00000000e+00  1.00000000e+00  0.00000000e+00 ...  1.00000000e+00
   0.00000000e+00  1.00000000e+00]
 [ 8.41470985e-01  5.40302306e-01  8.21856190e-01 ...  9.99999994e-01
   1.03663293e-04  9.99999995e-01]
 [ 9.09297427e-01 -4.16146837e-01  9.36414739e-01 ...  9.99999977e-01
   2.07326584e-04  9.99999979e-01]
 ...
 [-8.97967480e-01 -4.40061818e-01  4.26195541e-01 ...  9.94266169e-01
   1.03168405e-01  9.94663903e-01]
 [-8.55473152e-01  5.17847165e-01  9.86278111e-01 ...  9.94254673e-01
   1.03271514e-01  9.94653203e-01]
 [-2.64607527e-02  9.99649853e-01  6.97559894e-01 ...  9.94243164e-01
   1.03374623e-01  9.94642492e-01]]
ListWrapper([<7-transformer_encoder_block.EncoderBlock object at 0xb2981aef0>, <7-transformer_encoder_block.EncoderBlock object at 0xb29850ba8>, <7-transformer_encoder_block.EncoderBlock object at 0xb298647b8>, <7-transformer_encoder_block.EncoderBlock object at 0xb29e502e8>, <7-transformer_encoder_block.EncoderBlock object at 0xb29e5add8>, <7-transformer_encoder_block.EncoderBlock object at 0xb29e6c908>])
<tensorflow.python.keras.layers.core.Dropout object at 0xb29e7c470>
Tensor("encoder/encoder_block_5/layer_normalization_11/batchnorm/add_1:0", shape=(32, 10, 512), dtype=float32)
$

Ignore the Warning messages in the output
```

### [9. Transformer Decoder](./10-transformer_decoder.py)
Create a class Decoder that inherits from tensorflow.keras.layers.Layer to create the decoder for a transformer:
*    Class constructor def __init__(self, N, dm, h, hidden, target_vocab, max_seq_len, drop_rate=0.1):
*    N - the number of blocks in the encoder
     *   dm - the dimensionality of the model
     *   h - the number of heads
     *   hidden - the number of hidden units in the fully connected layer
     *   target_vocab - the size of the target vocabulary
     *   max_seq_len - the maximum sequence length possible
     *   drop_rate - the dropout rate
     *   Sets the following public instance attributes:
     *   N - the number of blocks in the encoder
     *   dm - the dimensionality of the model
     *   embedding - the embedding layer for the targets
     *   positional_encoding - a numpy.ndarray of shape (max_seq_len, dm) containing the positional encodings
     *   blocks - a list of length N containing all of the DecoderBlock‘s
     *   dropout - the dropout layer, to be applied to the positional encodings
*    Public instance method def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
     *   x - a tensor of shape (batch, target_seq_len, dm)containing the input to the decoder
     *   encoder_output - a tensor of shape (batch, input_seq_len, dm)containing the output of the encoder
     *   training - a boolean to determine if the model is training
     *   look_ahead_mask - the mask to be applied to the first multi head attention layer
     *   padding_mask - the mask to be applied to the second multi head attention layer
     *   Returns: a tensor of shape (batch, target_seq_len, dm) containing the decoder output
*    You should use positional_encoding = __import__('4-positional_encoding').positional_encoding and DecoderBlock = __import__('8-transformer_decoder_block').DecoderBlock
```
$ cat 10-main.py
#!/usr/bin/env python3

import tensorflow as tf
Decoder = __import__('10-transformer_decoder').Decoder

decoder = Decoder(6, 512, 8, 2048, 12000, 1500)
print(decoder.dm)
print(decoder.N)
print(decoder.embedding)
print(decoder.positional_encoding)
print(decoder.blocks)
print(decoder.dropout)
x = tf.random.uniform((32, 15))
hidden_states = tf.random.uniform((32, 10, 512))
output = decoder(x, hidden_states, True, None, None)
print(output)
$ ./10-main.py
512
6
<tensorflow.python.keras.layers.embeddings.Embedding object at 0xb2cdede48>
[[ 0.00000000e+00  1.00000000e+00  0.00000000e+00 ...  1.00000000e+00
   0.00000000e+00  1.00000000e+00]
 [ 8.41470985e-01  5.40302306e-01  8.21856190e-01 ...  9.99999994e-01
   1.03663293e-04  9.99999995e-01]
 [ 9.09297427e-01 -4.16146837e-01  9.36414739e-01 ...  9.99999977e-01
   2.07326584e-04  9.99999979e-01]
 ...
 [ 9.99516416e-01 -3.10955511e-02 -8.59441209e-01 ...  9.87088496e-01
   1.54561841e-01  9.87983116e-01]
 [ 5.13875021e-01 -8.57865061e-01 -6.94580536e-02 ...  9.87071278e-01
   1.54664258e-01  9.87967088e-01]
 [-4.44220699e-01 -8.95917390e-01  7.80301396e-01 ...  9.87054048e-01
   1.54766673e-01  9.87951050e-01]]
ListWrapper([<8-transformer_decoder_block.DecoderBlock object at 0xb2ce0f0b8>, <8-transformer_decoder_block.DecoderBlock object at 0xb2ce29ef0>, <8-transformer_decoder_block.DecoderBlock object at 0xb2d711b00>, <8-transformer_decoder_block.DecoderBlock object at 0xb2d72c710>, <8-transformer_decoder_block.DecoderBlock object at 0xb2d744320>, <8-transformer_decoder_block.DecoderBlock object at 0xb2d755ef0>])
<tensorflow.python.keras.layers.core.Dropout object at 0xb2d76db38>
Tensor("decoder/decoder_block_5/layer_normalization_17/batchnorm/add_1:0", shape=(32, 15, 512), dtype=float32)
$

Ignore the Warning messages in the output
```

### [10. Transformer Network](./
Create a class Transformer that inherits from tensorflow.keras.Model to create a transformer network:
*    Class constructor def __init__(self, N, dm, h, hidden, input_vocab, target_vocab, max_seq_input, max_seq_target, drop_rate=0.1):
     *   N - the number of blocks in the encoder and decoder
     *   dm - the dimensionality of the model
     *   h - the number of heads
     *   hidden - the number of hidden units in the fully connected layers
     *   input_vocab - the size of the input vocabulary
     *   target_vocab - the size of the target vocabulary
     *   max_seq_input - the maximum sequence length possible for the input
     *   max_seq_target - the maximum sequence length possible for the target
     *   drop_rate - the dropout rate
     *   Sets the following public instance attributes:
     *   encoder - the encoder layer
     *   decoder - the decoder layer
     *   linear - a final Dense layer with target_vocab units
*    Public instance method def call(self, inputs, target, training, encoder_mask, look_ahead_mask, decoder_mask):
     *   inputs - a tensor of shape (batch, input_seq_len)containing the inputs
     *   target - a tensor of shape (batch, target_seq_len)containing the target
     *   training - a boolean to determine if the model is training
     *   encoder_mask - the padding mask to be applied to the encoder
     *   look_ahead_mask - the look ahead mask to be applied to the decoder
     *   decoder_mask - the padding mask to be applied to the decoder
     *   Returns: a tensor of shape (batch, target_seq_len, target_vocab) containing the transformer output
*    You should use Encoder = __import__('9-transformer_encoder').Encoder and Decoder = __import__('10-transformer_decoder').Decoder
```
$ cat 11-main.py
#!/usr/bin/env python3

import tensorflow as tf
Transformer = __import__('11-transformer').Transformer

transformer = Transformer(6, 512, 8, 2048, 10000, 12000, 1000, 1500)
print(transformer.encoder)
print(transformer.decoder)
print(transformer.linear)
x = tf.random.uniform((32, 10))
y = tf.random.uniform((32, 15))
output = transformer(x, y, True, None, None, None)
print(output)
$ ./11-main.py
<9-transformer_encoder.Encoder object at 0xb2edc5128>
<10-transformer_decoder.Decoder object at 0xb2f412b38>
<tensorflow.python.keras.layers.core.Dense object at 0xb2fd68898>
Tensor("transformer/dense_96/BiasAdd:0", shape=(32, 15, 12000), dtype=float32)
$

Ignore the Warning messages in the output
```

---
## Authors

* **Robinson Montes** - [mecomonteshbtn](https://github.com/mecomonteshbtn)
