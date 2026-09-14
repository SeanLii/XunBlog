# XunBlog Knowledge Tree v4

本树使用 **historical ownership + conceptual ownership** 组织知识。  
Tree 决定知识归属；模型之间的复用关系由 Navigation Graph 处理。

```text
XunBlog

Mathematics
├── Linear Algebra
│   ├── Vector
│   ├── Vector Norm
│   ├── Matrix
│   ├── Linear Transformation
│   └── Dot Product
│
├── Analysis
│   └── Convolution
│
├── Probability
│   ├── Random Variable
│   ├── Probability Distribution
│   ├── Conditional Probability
│   ├── Bayes' Theorem
│   ├── Expectation
│   ├── Variance
│   ├── Covariance
│   ├── Normal Distribution
│   ├── Multivariate Normal Distribution
│   ├── Latent Variable
│   └── Variational Inference
│       ├── Evidence Lower Bound
│       └── Reparameterization Trick
│
├── Information Theory
│   ├── Entropy
│   ├── Cross-Entropy
│   └── KL Divergence
│
├── Calculus
│   └── Ordinary Differential Equation
│
└── Numerical Methods
    └── Euler Method


Deep Learning
├── Core
│   ├── Linear Layer
│   ├── Multilayer Perceptron
│   ├── Activation Function
│   ├── Embedding
│   ├── Softmax
│   └── Layer Normalization
│
├── Attention
│   ├── Attention
│   ├── Query / Key / Value
│   ├── Self-Attention
│   └── Cross-Attention
│
├── Sequence Modeling
│   ├── Positional Encoding
│   └── Causal Mask
│
├── Transformer
│   ├── Transformer
│   ├── Scaled Dot-Product Attention
│   ├── Multi-Head Attention
│   ├── Position-Wise Feed-Forward Network
│   ├── Transformer Encoder
│   ├── Transformer Decoder
│   └── KV Cache
│
├── BERT
│   ├── BERT
│   └── CLS Token
│
├── DETR
│   ├── DETR
│   └── Object Query
│
├── Representation Learning
│   └── Autoencoder
│
├── Convolutional Neural Networks
│   ├── Convolutional Neural Network
│   └── ResNet
│       └── Residual Connection
│
└── Multimodal Models
    └── Vision-Language Model


Generative Models
├── Variational Autoencoder
├── Conditional Variational Autoencoder
├── Posterior Collapse
└── Flow Matching


Robot Learning
├── Imitation Learning
├── Behavior Cloning
├── DAgger
├── Vision-Language-Action Model
├── Cross-Embodiment Learning
│
├── ACT
│   ├── Action Chunking
│   ├── Temporal Ensemble
│   ├── Architecture
│   ├── CVAE in ACT
│   ├── Vision Pipeline
│   ├── Training
│   ├── Inference
│   ├── Complete Data Flow
│   ├── 为什么 ACT 推理时令 z = 0？
│   └── Paper and Released Implementation
│
└── π0
    ├── Architecture
    ├── Action Expert
    ├── Blockwise Causal Attention Mask in π0
    ├── Flow Matching in π0
    ├── Pre-training and Post-training in π0
    ├── Training
    ├── Inference
    ├── Complete Data Flow
    └── Paper and Released Implementation
```

---

# Ownership decisions that matter

## Transformer does not own all attention knowledge

The following predate Transformer and therefore live outside the Transformer branch:

- Attention
- Query / Key / Value lineage
- Self-Attention
- Cross-Attention
- Positional Encoding
- Causal Mask
- Layer Normalization
- Residual Connection

Transformer retains the designs that are specific to its architecture lineage:

- Scaled Dot-Product Attention
- Multi-Head Attention
- Position-Wise Feed-Forward Network
- Transformer Encoder
- Transformer Decoder

## Model-specific components stay with their source model

- `CLS Token` → BERT
- `Object Query` → DETR
- `Residual Connection` → ResNet residual-learning lineage
- `Action Chunking` / `Temporal Ensemble` → ACT formulation
- `Action Expert` / `Blockwise Causal Attention Mask in π0` → π0

## General theories stay independent

- Convolution → Mathematics / Analysis
- Latent Variable → Probability
- Variational Inference / ELBO / Reparameterization Trick → Probability / Variational Inference
- Flow Matching → Generative Models
- CVAE → Generative Models
- DAgger → Robot Learning

Later reuse creates navigation links, not ownership migration.
