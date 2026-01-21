# Supported platforms

This project was tested on the following machines:
- Windows 10 + 96GB RAM + Geforce RTX 4060 Ti 16GB (Ada)
- Windows 10 + 128GB RAM + Quadro RTX 5000 16GB (Turing)

Therefore for Linux flows some steps needs to be changed and scripts slightly changed for GPUs with less VRAM or installation of pytorch changed for different GPU generations.

# Installation
Including the versions which worked for me quick justification/explanations why:

- Updating NVIDIA GPU driver R580: https://www.nvidia.com/en-us/drivers/results/
- CUDA Toolkit SDK 13.0 update 2: https://developer.nvidia.com/cuda-downloads
  - Check CUDA GPU compute capabilities for your GPU: https://developer.nvidia.com/cuda-gpus
  - My GPUS: Turing has 7.5 and Ada 8.9 compute capabilities
  - https://en.wikipedia.org/wiki/CUDA#GPUs_supported
    - 7.5 compute capability is still supported by the latest CUDA Toolkit SDK 13.0 (therefore getting this version)
- Python 3.10.11 (not latest) - https://www.python.org/downloads/release/python-31011/
  - Because ` PyTorch requires Python 3.10 or later` statement is slightly misleading. It needs to be within 3.10 release, not 3.15 etc... And ignoring this will cause cryptic problems/errors.
- Pytorch https://pytorch.org/get-started/locally/
  - Getting pytorch for CUDA SDK 13.0 (Windows and Linux step should be identical)
  - `pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130`
  - If getting `ERROR: No matching distribution found for typing-extensions>=4.10.0` then `python -m pip install --upgrade pip` should fix it
- Pandas `pip install pandas`
- Numpy `pip install numpy`
- SentencePiece `pip install sentencepiece` https://github.com/google/sentencepiece 
  - If we want to build our own protobuf (protobufer protoc) to read/modify the vocabulary `winget install protobuf` https://protobuf.dev/installation/
  - And to use the protoc `protoc --python_out=. sentencepiece_model.proto`
- If wanting to experiment get the Jupyter notebook as well: `pip install jupyterlab`
  - To ran on my NAS and have remote access `jupyter notebook --no-browser --ip 0.0.0.0 --port=8888`
- Torchviz for model troubleshooting `pip install torchviz`
  - Graphviz for rendering visualizations https://graphviz.org/download/
- Onnxscript `pip install onnx onnxscript` to export data for Netron visualizer

# Llama architecture steps

## Example model hyper parameters
- training batch size = 7
- vocabulary size = 11
- embedding dimensions = 48
- max sequence length = 5
- query heads = 8
- kv heads per query head = 2
- key heads = value heads = 4
- dimensions per (QKV) head = 6 (want maybe closer to 32,64 or even more, and maybe having less heads, even if it's just 1 or 2 heads)
- feed forward network hidden dimensions = 128 (typically rounded around 8*embedding dimensions/3, but could maybe make 4x or 8x of embedding dimensions if I want to make wide but shallow single transformer layout)
- transformer layers = 9

## Weight matrices
- all are rank-2 matrices
- embeddedding [11, 48] (vocabulary size, embedding dimensions)
- RMS norm weights (for scaling) 
  - attention [9, 48] (transformer layers, embedding dimensions)
  - ffn [9, 48] (transformer layers, embedding dimensions)
  - final [48] (embedding dimensions)
- linear projections 
  - dimensions of the attention = (query heads * dimensions per head) which is the same as embedding dimensions = 48
  - dimensions of kv heads = (key heads * dimensions per head) which is in our case half = 24
  - Wq [48,48] (embedding dimensions, dimensions of the attention) 
  - Wk [48,24] (embedding dimensions, dimensions of kv heads) 
  - Wv [48,24] (embedding dimensions, dimensions of kv heads) 
  - Feed forward network [48,128] (embedding dimensions, feed forward network hidden dimensions)
- Output head
  - Wo [48,48] (dimensions of the attention, embedding dimensions) 


## Input block

### Input tokens
- if inference
  - seq_len = sequence length so far (it grows as we iterate over the tokens)
- if training 
  - seq_len = max_seq_len
  - prepare padding mask [7, 5] (batch_size, max_seq_len)
- tokens shape (batch_size=7, seq_len=5 which depends if we infere/train)

### Embedding
- turning input token from vocabulary to a 48 dimension vector
- X[7,5,48] (batch size, seq_len, embedding dimensions) = ∀batch and ∀token -> embedding[token]

### RMSnorm for Attention
- Normalize the vectors to keep the scale of activations somewhat constant, but then uses WeightsRmsNormAttention to bring back a bit of each vector the power
- Also adding epsilon (1e-8) to 
  - avoid division by zero issue when all Xs would be 0
  - nudge the value have at least some minimum value, which in turn will not cause issues (extremely huge numbers, infinity, NaNs) when we use it to divide X with it
  - but still keeping it is small enough to not affect activations 
- squared_avg = average of all squared X
- X_norm [7,5,48] (batch size, seq_len, embedding dimensions) = WeightsRmsNormAttention * X / sqrt(squared_avg + epsilon)


## Transformer block

### Linear projections of X_norm to Q, K, V (using only weights, no biases)
- they have different dimensionality because they are used in grouped query attention GQA
- Q[7,5,48] = X_norm[7,5,48] · Wq[48,48]
- K[7,5,24] = X_norm[7,5,48] · Wk[48,24]
- V[7,5,24] = X_norm[7,5,48] · Wv[48,24]


### Reshaping and transposing split the dimensions parts of Q,K,V evenly between their heads
- each head will get it's own 6 dimensional chunk
- Qreshaped [7,5,48] reshape to 8 heads with 6 dimensions per head [7,5,8,6] 
  - and use as is (easier to understand)
  - or transpose [7,8,5,6] (batch, q heads, seq, dimension per head) (if i want the attention to be computed in less calls)
- Kreshaped and Vreshaped [7,5,24] reshape to 4 heads with 6 dimensions per head [7,5,4,6] 
  - or transpose [7,4,5,6] (batch, kv heads, seq, dimension per head)
- broadcast probably too?


### RoPE (Rotary Positional Encoding)
- base = 100000 
- dim_index 0 .... (embedding dimensions / 2) -1
- θ[dim_index] = base  ^ {-2 dim_index / dimensions per (QKV) head}

- For each Batch, for each token position, for each attention head:
- Split each head vector into real/imag pairs (2D)
- Apply rotation:
  - each dimension pair (2i, 2i+1) is treathed as  (Q_real, Q_imaginary)
    - Qrotated[seq, 2i]   = Qreshaped[seq, 2i] * cos(θ[i]) - Qreshaped[seq, 2i+1] * sin(θ[i])
    - Qrotated[seq, 2i+1] = Qreshaped[seq, 2i] * sin(θ[i]) + Qreshaped[seq, 2i+1] * cos(θ[i])
    - Krotated .... same

- Shapes unchanged (used the non-transposed simpler layout of Q/K in the Reshaping state):
- Q: [7,5,8,6]
- (batch, seq, q heads, dimension per head)

- K: [7,5,4,6]
- (batch, seq, kv heads, dimension per head) 
- (B,T,H,D_head)


### Prepare for attention if I didn't do it already before(optional)
- bring head forward
- (B,T,H,D_head) transpose -> (B,H,T,D_head)

- Q [7,8,5,6]
- K [7,4,5,6]
- V [7,4,5,6]


### Attention (softmax(QK^T / sqrt(D_head)) * V)
- When inference do single Q (B, H, 1, D_head) againts all previous tooken so far + this one on K and & V (B, H, t+1, D_head)
- When training For each batch B and for each Q head i
  - find equivalent K/V head j (same K/V could be reused with multiple Q)
    - the K/V might have been broadcasted to make this simpler
  - SCOREi = ( Qrotated[B,i] * K[B,j] ^T  ) / sqrt(D_head=6)
  - (T,T)  = (T,D_head)    * (D,T)
  - [5,5]  = [5, 6]        * [6, 5]
  - when adding it back to its batch&head it would look like (B,H,T,T)
  - SCOREi += Masking for future tokens (diagonals)
  - SCOREi += Masking for padded batches (when training same MAX_SEQ_LEN batches, but not every single one inside has same size, so the end token (and after) can produce -inf mask)
    - Also later don't forget to do loss masking
  - ATTENTIONi = Softmax(SCOREi) // per each column, over the last dimension aka normalizes each row T separately (second last dimension) by summing all columns for that row to a 1, example:

  - SCORE[b,h,:,:] =
    ```
      │ 0.1   0.5   0.3   0.2 │
      │ 0.0   0.2   0.4   0.1 │
      │ 0.2   0.1   0.0   0.3 │
      │ 0.3   0.3   0.2   0.0 │
    ```

  - normalized to:
    ```
      │ 0.18  0.36  0.26  0.20 │ <- sums to 1
      │ 0.18  0.24  0.43  0.15 │ <- sums to 1
      │ 0.35  0.26  0.19  0.20 │ <- sums to 1
      │ 0.31  0.31  0.23  0.15 │ <- sums to 1
    ```  

  - each row is normalized independently (still same shape)

  - Describing how much one token T affects the other token Ts?
  - V head weighted sum done either in iteration:
    -for each position s' -> OUTPUT += ATTENTION[s'] * V[s']
      - OUTPUT[i] = ATTENTION[i] * V[B,i]
      - [5,6]     = [5,5]        * [5,6]
      - (T,D_head)= (T,Ts)       * (Ts,D_head)
    -or let the matmul iterate thorugh it internally for us 
      - OUTPUT = ATTENTION * V
  - later when adding it back to its batch&head we get output (B,H,T,D_head)
 
### Join heads 
- transpose (B,H,T,D_head) -> (B,T,H,D_head)
- reshape (B,T,H,D_head) -> (B,T,H * D_head) -> [7,5,48]


### Output projection
- FINAL_ATTENTION = OUTPUT   * Wo
- (B,T,D)         = (B,T,D)  * (H * D_head, D)
- [7,5,48]          [7,5,48] * [48, 48]


### Residual connection with the original input
- Xresidual = X + FINAL_ATTENTION
- (B,T,D) [7,5,48] 


### RMSNorm
- operates over the last dimensions (D)
- XresidualNorm = RMSNorm(Xresidual)
- (B,T,D) [7,5,48] 

### Feedfoward (specifically the Multi-Layer Perceptron (MLP)) block with SwiGLU/SiLU+gating
- FFNinput  (B,T,Dffn):     U = XresidualNorm (B,T,D) * W1 (expand features) (D, Dffn)
- FFNhidden (B,T,Dffn):     V = XresidualNorm (B,T,D) * W2 (gate vector) (D, Dffn)
- SwiGLU (B,T,Dffn):        G = SiLU(V)
   - Gating GLU (B,T,Dffn): H = U element wise multiply with G
- Project back MLP_OUTPUT (B,T,D) = H (B,T,Dffn) * W3 (Dffn,D) to compress back to D


### Residual connection again
- XffnResidual = MLP_OUTPUT + XresidualNorm 
- (B,T,D) [7,5,48] 

### Finished Transformer Block
- If there are more layers feed the current output as input to next layer


## Output block

### Final RMSNorm
- Applies over last D dimension
- Xfinal = RMSNorm(XffnResidual)
- (B,T,D) [7,5,48] 

### Linear projection (LM head)
- map the hidden dimensions into vocab dimensions
- logits                = Xfinal   *  W_vocab^T
- (B,T,vocabulary size)   (B,T,D)  *  (vocabulary size, D)^T is (D, vocabulary size)
- [7,5,11]                [7,5,48]    [11,48]^T is [48,11]
- these do not need to sum to 1 because they are raw scores and not probalities yet

### Softmax - get probabilities
- probs = softmax(logits, dim=-1) (on the last dimension)
- if training the softmax could be indirectly done with cross_entropy inside the loss function
- if inference next_token = sample(probs) # depending on the temperature, top-k, top-p and whatnot, but we can go for the highest for simplicity