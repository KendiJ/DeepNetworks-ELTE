# Deep Network Development Theory Cheat Sheet
## Theory Test 2 — 2026 Spring

---

# 1. METRICS

---

# Regression Metrics

Used when predicting continuous values.

Examples:
- house prices
- temperatures
- salaries

| Metric | Meaning | Used For |
|---|---|---|
| MAE | Mean Absolute Error | Regression |
| MSE | Mean Squared Error | Regression |
| RMSE | Root Mean Squared Error | Regression |

---

# Classification Metrics

Used when predicting categories/classes.

| Metric | Binary Classification | Multi-class Classification |
|---|---|---|
| Accuracy | Yes | Yes |
| Precision | Yes | Yes |
| Recall | Yes | Yes |
| F1-score | Yes | Yes |

---

# Accuracy

Measures:

```text
Correct predictions / Total predictions
```

---

# Precision

Measures:

```text
True Positives / (True Positives + False Positives)
```

Meaning:
> When the model predicts positive, how often is it correct?

---

# Recall

Measures:

```text
True Positives / (True Positives + False Negatives)
```

Meaning:
> How many actual positives were found?

---

# F1-score

Harmonic mean of:
- precision
- recall

Good when:
- dataset is imbalanced

---

# Confusion Matrix

For binary classification:

| | Predicted Positive | Predicted Negative |
|---|---|---|
| Actual Positive | TP | FN |
| Actual Negative | FP | TN |

---

# TP, FP, TN, FN

| Name | Meaning |
|---|---|
| TP | Correctly predicted positive |
| FP | Incorrectly predicted positive |
| TN | Correctly predicted negative |
| FN | Incorrectly predicted negative |

---

# Confusion Matrix Facts

| Question | Answer |
|---|---|
| Size for binary classification | 2×2 |
| Element (i,j) | Number of samples with true class i predicted as j |
| Sum of all elements | Total number of samples |
| Diagonal matrix | Perfect predictions |

---

# 2. DATA AUGMENTATION

---

# What is Data Augmentation?

Artificially increasing dataset diversity by modifying training samples.

Examples:
- rotation
- flipping
- cropping
- noise

---

# Purpose

| Purpose | Explanation |
|---|---|
| Reduce overfitting | Prevent memorization |
| Improve generalization | Better unseen performance |
| Simulate real-world variation | More robust model |

---

# When Should It Be Applied?

| Phase | Apply Augmentation? |
|---|---|
| Training | Yes |
| Validation | No |
| Testing | No |

---

# Cat/Dog Image Transformations

Allowed:
- flip
- crop
- brightness
- rotation

Not ideal:
- transformations changing semantics

---

# Handwritten Digit Transformations

Allowed:
- small rotations
- scaling
- noise

Dangerous:
- horizontal flips (6 may become 9)

---

# Medical Tabular Data

Usually:
- noise injection
- feature masking

NOT:
- image-style transforms

---

# Noise vs Dropout

| Technique | Purpose |
|---|---|
| Noise | Improve robustness |
| Dropout | Prevent overfitting |

---

# Dropout

Randomly removes neurons during training.

---

# Effects of Dropout

| During Training | During Testing |
|---|---|
| Random neurons disabled | All neurons active |

---

# Benefits

- improves generalization
- reduces overfitting

---

# 3. PYTORCH nn.Module

---

# model.train()

Sets:
- training mode

Enables:
- dropout
- batch normalization updates

---

# model.eval()

Sets:
- evaluation mode

Disables:
- dropout randomness
- batchnorm updates

---

# forward()

Defines:
- how input moves through the network

Example:

```python
def forward(self, x):
    return self.net(x)
```

---

# 4. CONVOLUTIONAL NETWORKS (CNNs)

---

# Why MLPs Are Bad for Images

Problems:
- too many parameters
- no spatial awareness
- inefficient for large images

---

# Convolution

A filter slides across input and computes weighted sums.

---

# Filter

Small learnable matrix.

Example:
- edge detector
- texture detector

---

# Heatmap / Feature Map

Output produced by a filter.

Shows:
- where a feature appears

---

# CNN Parameters

For one K×L filter:

```text
K × L × input_channels + bias
```

---

# Pooling Layer

Purpose:
- reduce spatial size
- reduce computation
- improve translation robustness

---

# Max Pooling

Takes maximum value.

---

# Average Pooling

Takes average value.

---

# Padding

Adds borders around image.

Effects:
- larger output size
- preserves edges

---

# Stride > 1

Filter jumps multiple pixels.

Effects:
- smaller outputs
- less computation

---

# Translation Properties

| Property | Meaning |
|---|---|
| Equivariance | Shift input → shifted output |
| Invariance | Shift input → same output |

CNNs are approximately:
- translation equivariant

Pooling helps:
- translation invariance

---

# Flattening CNN Outputs

Input shape:

```python
(B, C, H, W)
```

Flatten:

```python
x = x.reshape(B, C * H * W)
```

---

# CNN Classification Shapes

| Stage | Shape Example |
|---|---|
| Input | (B, 3, 224, 224) |
| Output | (B, num_classes) |

---

# 5. UNSTABLE GRADIENT PROBLEM

---

# Unstable Gradients

Includes:
- vanishing gradients
- exploding gradients

---

# Vanishing Gradients

Gradients become extremely small.

Effects:
- slow learning
- early layers stop learning

---

# Exploding Gradients

Gradients become extremely large.

Effects:
- unstable training
- NaNs

---

# Why Sigmoid Causes Problems

Sigmoid saturates near:
- 0
- 1

Gradients become tiny.

---

# Batch Normalization

Normalizes activations:

```text
mean = 0
variance = 1
```

---

# Benefits

- stabilizes training
- reduces gradient issues
- faster convergence

---

# Residual Networks (ResNets)

Use skip connections:

```text
output = F(x) + x
```

---

# Why ResNets Help

Skip connections:
- improve gradient flow
- reduce vanishing gradients

---

# ResNets Are Usually

- convolutional networks

---

# 6. TRANSFER LEARNING

---

# What is Transfer Learning?

Using pretrained models for new tasks.

Example:
- ImageNet → cat vs dog

---

# Advantages

| Advantage | Explanation |
|---|---|
| Faster training | Already learned useful features |
| Better performance | Especially with small datasets |
| Less data needed | Reuse learned representations |

---

# Similar Tasks Work Better

Transfer learning works best when:
- source task similar to target task

---

# Weight Freezing

Prevent weights from updating.

PyTorch:

```python
param.requires_grad = False
```

---

# Fine-tuning Cat vs Dog

Keep:
- convolution layers

Replace:
- final classification layer

---

# 7. RECURRENT NEURAL NETWORKS (RNNs)

---

# RNN Purpose

Process sequential data.

Examples:
- text
- audio
- time series

---

# Sequence Task Types

| Type | Example |
|---|---|
| N → N | Translation |
| N → 1 | Sentiment analysis |
| 1 → N | Image captioning |
| M → N | Seq2Seq translation |

---

# RNN Parameters

Shared across all time steps.

Changing sequence length:
- does NOT increase parameter count

---

# Computation Cost

Longer sequences:
- more computations

---

# Vanilla RNN Problems

- vanishing gradients
- poor long-term memory

---

# LSTM / GRU Advantages

| Feature | Benefit |
|---|---|
| Gates | Better memory |
| Gradient flow | More stable training |
| Long dependencies | Better handling |

---

# Truncated BPTT

Backpropagation through only part of sequence.

Purpose:
- reduce computation
- reduce memory usage

---

# EOS / END Token

Marks:
- sequence completion

Used in:
- text generation

---

# Sequence Generation Steps

1. Start token
2. Predict next token
3. Feed prediction back
4. Repeat until EOS

---

# Randomness in Generation

Use:
- sampling
- temperature
- top-k sampling

---

# Seq2Seq

Encoder:
- processes input sequence

Decoder:
- generates output sequence

Used for:
- translation
- summarization

---

# 8. ATTENTION NETWORKS

---

# Advantages Over RNNs

| Attention | RNN |
|---|---|
| Parallel processing | Sequential |
| Better long-range dependencies | Harder |
| Faster training | Slower |

---

# Attention Score

Measures:
> how important one token is to another

---

# Self-Attention

Each token attends to:
- all tokens in sequence

---

# Attention Complexity

For sequence length N:

```text
N² attention scores
```

---

# Self-Attention Core Operation

```text
Q · K^T
```

Dot product between:
- query
- key

---

# Attention Weaknesses

| Weakness | Explanation |
|---|---|
| High memory usage | O(N²) |
| Expensive for long sequences | Large attention matrices |

---

# 9. TRANSFORMERS

---

# Why Subword Tokenization?

Compromise between:
- character-level
- word-level

Benefits:
- handles rare words
- smaller vocabulary

---

# Why Transformers Need Causal Masking

Without masking:
- model sees future tokens

RNNs naturally avoid this because:
- sequential processing

---

# Scaled Dot Product Attention

Formula:

```text
(QK^T) / sqrt(d_k)
```

---

# Why Divide by sqrt(d_k)?

Prevents:
- extremely large dot products
- unstable softmax

---

# Positional Encoding

Transformers lack sequence order naturally.

Positional encoding provides:
- token position information

---

# Transformer Types

| Type | Usage |
|---|---|
| Encoder-only | Classification (BERT) |
| Decoder-only | Text generation (GPT) |
| Encoder-decoder | Translation (T5) |

---

# Self-attention vs Cross-attention

| Type | Uses |
|---|---|
| Self-attention | Same sequence |
| Cross-attention | Different sequences |

---

# Vision Transformers (ViT)

Split image into patches.

Reason:
- treat patches like tokens

---

# Patch Size Tradeoff

| Small Patches | Large Patches |
|---|---|
| More detail | Less detail |
| Longer sequences | Shorter sequences |
| More computation | Less computation |

---

# Inductive Bias

Built-in assumptions helping learning.

CNNs:
- strong locality bias

ViTs:
- weaker inductive bias

---

# Why ViTs Need More Data

Less inductive bias means:
- must learn more from data

---

# Swin Transformer

Reintroduces:
- locality

Improves:
- efficiency
- image understanding

---

# 10. UNSUPERVISED LEARNING

---

# Main Tasks

| Task | Purpose |
|---|---|
| Clustering | Group similar samples |
| Compression | Reduce dimensions |
| Denoising | Remove noise |
| Generation | Create new samples |

---

# Clustering

Groups similar data points.

Problem:
- no true labels
- cluster ambiguity

---

# Compression / Dimensionality Reduction

Reduce number of features while preserving information.

---

# Autoencoder

Neural network trained to reconstruct input.

Structure:

```text
Input → Encoder → Bottleneck → Decoder → Reconstruction
```

---

# Preventing Identity Mapping

Use:
- bottleneck layer
- smaller latent representation

---

# Autoencoder Labels

Input itself is target label.

---

# Bottleneck

Compressed representation.

---

# MSE = 0 in Autoencoder

Means:
- perfect reconstruction

---

# Denoising Autoencoder

Input:
- noisy sample

Target:
- clean sample

---

# Sample Generation

Goal:
- create new realistic samples

Examples:
- GANs
- VAEs

---

# LLM Pretraining Tasks

Common unsupervised/self-supervised tasks:
- next-token prediction
- masked token prediction

---

# FINAL EXAM STRATEGY

---

# Focus MOST on:

- metrics
- confusion matrix
- CNN basics
- RNN vs Transformer
- attention
- transfer learning
- unstable gradients
- PyTorch train/eval
- autoencoders

---

