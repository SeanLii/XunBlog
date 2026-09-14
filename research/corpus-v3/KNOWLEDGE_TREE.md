# Knowledge Tree

```text
Mathematics
├── Linear Algebra
│   ├── Vector
│   ├── Matrix
│   └── Dot Product
├── Probability
│   ├── Random Variable
│   ├── Probability Distribution
│   ├── Conditional Probability
│   ├── Expectation
│   ├── Variance
│   ├── Normal Distribution
│   └── Multivariate Normal Distribution
├── Information Theory
│   └── KL Divergence
├── Calculus
│   └── Ordinary Differential Equation
└── Numerical Methods
    └── Euler Method

Deep Learning
├── Core
│   ├── Linear Layer
│   ├── Softmax
│   ├── Embedding
│   ├── Residual Connection
│   └── Layer Normalization
├── Transformer
│   ├── Transformer
│   ├── Attention
│   │   ├── Query / Key / Value
│   │   ├── Scaled Dot-Product Attention
│   │   ├── Self-Attention
│   │   ├── Cross-Attention
│   │   └── Multi-Head Attention
│   ├── Positional Encoding
│   ├── Transformer Encoder
│   ├── Transformer Decoder
│   ├── Feed-Forward Network
│   ├── Causal Mask
│   ├── CLS Token
│   └── Learnable Query Embedding
├── Convolutional Neural Networks
│   ├── Convolution
│   ├── Convolutional Neural Network
│   └── ResNet
└── Multimodal Models
    └── Vision-Language Model

Generative Models
├── Latent Variable
├── Variational Inference
├── Evidence Lower Bound
├── Variational Autoencoder
├── Reparameterization Trick
├── Conditional Variational Autoencoder
├── Posterior Collapse
└── Flow Matching

Robot Learning
├── Imitation Learning
├── Behavior Cloning
├── Vision-Language-Action Model
├── Cross-Embodiment Learning
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
└── π0
    ├── Architecture
    ├── Action Expert
    ├── Blockwise Causal Attention Mask
    ├── Flow Matching in π0
    ├── Pre-training and Post-training in π0
    ├── Training
    ├── Inference
    ├── Complete Data Flow
    └── Paper and Released Implementation
```

## π0 的 canonical ownership

π0 只拥有 π0-specific 的 architecture、action expert、attention mask、training/inference 与数据流。

它依赖但不拥有：

```text
Vision-Language Model        → Deep Learning / Multimodal Models
Vision-Language-Action Model → Robot Learning
Flow Matching                → Generative Models
Ordinary Differential Equation → Mathematics / Calculus
Euler Method                 → Mathematics / Numerical Methods
Action Chunking              → Robot Learning / ACT（现有 canonical page）
Cross-Embodiment Learning    → Robot Learning
```

因此 π0 页面遇到这些概念时通过 canonical link 导航，不在 π0 下重新建立一套通用理论。
