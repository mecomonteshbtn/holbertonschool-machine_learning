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
## Authors

* **Robinson Montes** - [mecomonteshbtn](https://github.com/mecomonteshbtn)
