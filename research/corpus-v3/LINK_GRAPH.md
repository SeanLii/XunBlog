# Link Graph

下面记录页面通过 prerequisite、related 与正文 canonical links 形成的 navigation graph。Knowledge Tree 决定归属，本文件只表示可导航关系。

## Convolution

`/deep-learning/cnn/convolution/`

- → `/deep-learning/cnn/convolutional-neural-network/` (related)
- → `/mathematics/linear-algebra/matrix/` (prerequisite)

## Convolutional Neural Network

`/deep-learning/cnn/convolutional-neural-network/`

- → `/deep-learning/cnn/convolution/` (prerequisite, body)
- → `/deep-learning/cnn/resnet/` (related, body)
- → `/robot-learning/act/vision-pipeline/` (related)

## ResNet

`/deep-learning/cnn/resnet/`

- → `/deep-learning/cnn/convolution/` (body)
- → `/deep-learning/cnn/convolutional-neural-network/` (prerequisite)
- → `/deep-learning/core/residual-connection/` (prerequisite)
- → `/robot-learning/act/vision-pipeline/` (related)

## Embedding

`/deep-learning/core/embedding/`

- → `/deep-learning/transformer/cls-token/` (related)
- → `/deep-learning/transformer/learnable-query-embedding/` (related)
- → `/mathematics/linear-algebra/vector/` (prerequisite)

## Layer Normalization

`/deep-learning/core/layer-normalization/`

- → `/deep-learning/transformer/transformer-decoder/` (related)
- → `/deep-learning/transformer/transformer-encoder/` (related)
- → `/mathematics/probability/expectation/` (prerequisite)
- → `/mathematics/probability/variance/` (prerequisite)

## Linear Layer

`/deep-learning/core/linear-layer/`

- → `/deep-learning/core/embedding/` (related)
- → `/mathematics/linear-algebra/matrix/` (prerequisite)
- → `/mathematics/linear-algebra/vector/` (prerequisite)
- → `/robot-learning/act/architecture/` (related)

## Residual Connection

`/deep-learning/core/residual-connection/`

- → `/deep-learning/cnn/resnet/` (related)
- → `/deep-learning/transformer/transformer-encoder/` (related)
- → `/mathematics/linear-algebra/vector/` (prerequisite)

## Softmax

`/deep-learning/core/softmax/`

- → `/deep-learning/transformer/attention/scaled-dot-product-attention/` (related)
- → `/mathematics/linear-algebra/vector/` (prerequisite)

## Vision-Language Model

`/deep-learning/multimodal/vision-language-model/`

- → `/deep-learning/core/embedding/` (prerequisite)
- → `/deep-learning/transformer/` (prerequisite)
- → `/robot-learning/pi0/architecture/` (related)
- → `/robot-learning/vision-language-action-model/` (related, body)

## Transformer

`/deep-learning/transformer/`

- → `/deep-learning/core/layer-normalization/` (body)
- → `/deep-learning/core/residual-connection/` (body)
- → `/deep-learning/transformer/attention/` (prerequisite, body)
- → `/deep-learning/transformer/attention/cross-attention/` (body)
- → `/deep-learning/transformer/attention/qkv/` (body)
- → `/deep-learning/transformer/attention/scaled-dot-product-attention/` (body)
- → `/deep-learning/transformer/attention/self-attention/` (body)
- → `/deep-learning/transformer/learnable-query-embedding/` (body)
- → `/deep-learning/transformer/positional-encoding/` (prerequisite, body)
- → `/deep-learning/transformer/transformer-decoder/` (related)
- → `/deep-learning/transformer/transformer-encoder/` (related)
- → `/robot-learning/act/architecture/` (related)

## Attention

`/deep-learning/transformer/attention/`

- → `/deep-learning/core/softmax/` (body)
- → `/deep-learning/transformer/attention/cross-attention/` (body)
- → `/deep-learning/transformer/attention/multi-head-attention/` (body)
- → `/deep-learning/transformer/attention/qkv/` (related, body)
- → `/deep-learning/transformer/attention/scaled-dot-product-attention/` (related, body)
- → `/deep-learning/transformer/attention/self-attention/` (body)
- → `/deep-learning/transformer/feed-forward-network/` (body)
- → `/mathematics/linear-algebra/vector/` (prerequisite)

## Cross-Attention

`/deep-learning/transformer/attention/cross-attention/`

- → `/deep-learning/transformer/attention/scaled-dot-product-attention/` (prerequisite)
- → `/deep-learning/transformer/transformer-decoder/` (related)
- → `/robot-learning/act/architecture/` (related)

## Multi-Head Attention

`/deep-learning/transformer/attention/multi-head-attention/`

- → `/deep-learning/transformer/` (related)
- → `/deep-learning/transformer/attention/scaled-dot-product-attention/` (prerequisite)

## Query / Key / Value

`/deep-learning/transformer/attention/qkv/`

- → `/deep-learning/core/linear-layer/` (prerequisite)
- → `/deep-learning/transformer/attention/` (prerequisite)
- → `/deep-learning/transformer/attention/cross-attention/` (related)
- → `/deep-learning/transformer/attention/self-attention/` (related)
- → `/mathematics/linear-algebra/dot-product/` (body)

## Scaled Dot-Product Attention

`/deep-learning/transformer/attention/scaled-dot-product-attention/`

- → `/deep-learning/core/softmax/` (prerequisite)
- → `/deep-learning/transformer/attention/multi-head-attention/` (related)
- → `/deep-learning/transformer/attention/qkv/` (prerequisite)
- → `/deep-learning/transformer/causal-mask/` (body)
- → `/mathematics/linear-algebra/dot-product/` (prerequisite)

## Self-Attention

`/deep-learning/transformer/attention/self-attention/`

- → `/deep-learning/transformer/attention/cross-attention/` (related)
- → `/deep-learning/transformer/attention/scaled-dot-product-attention/` (prerequisite)
- → `/deep-learning/transformer/causal-mask/` (body)
- → `/deep-learning/transformer/positional-encoding/` (body)
- → `/deep-learning/transformer/transformer-encoder/` (related)

## Causal Mask

`/deep-learning/transformer/causal-mask/`

- → `/deep-learning/transformer/attention/scaled-dot-product-attention/` (prerequisite)
- → `/deep-learning/transformer/transformer-decoder/` (related)

## CLS Token

`/deep-learning/transformer/cls-token/`

- → `/deep-learning/core/embedding/` (prerequisite)
- → `/deep-learning/transformer/transformer-encoder/` (prerequisite)
- → `/robot-learning/act/cvae-in-act/` (related)

## Feed-Forward Network

`/deep-learning/transformer/feed-forward-network/`

- → `/deep-learning/core/linear-layer/` (prerequisite)
- → `/deep-learning/transformer/transformer-decoder/` (related)
- → `/deep-learning/transformer/transformer-encoder/` (related)

## Learnable Query Embedding

`/deep-learning/transformer/learnable-query-embedding/`

- → `/deep-learning/core/embedding/` (prerequisite)
- → `/deep-learning/transformer/attention/cross-attention/` (prerequisite)
- → `/deep-learning/transformer/transformer-decoder/` (related)
- → `/robot-learning/act/architecture/` (related)

## Positional Encoding

`/deep-learning/transformer/positional-encoding/`

- → `/deep-learning/core/embedding/` (prerequisite)
- → `/deep-learning/transformer/` (related)
- → `/robot-learning/act/vision-pipeline/` (related)

## Transformer Decoder

`/deep-learning/transformer/transformer-decoder/`

- → `/deep-learning/transformer/attention/cross-attention/` (prerequisite)
- → `/deep-learning/transformer/attention/self-attention/` (prerequisite)
- → `/deep-learning/transformer/causal-mask/` (related)
- → `/deep-learning/transformer/feed-forward-network/` (prerequisite)
- → `/deep-learning/transformer/learnable-query-embedding/` (related)
- → `/robot-learning/act/architecture/` (related)

## Transformer Encoder

`/deep-learning/transformer/transformer-encoder/`

- → `/deep-learning/core/layer-normalization/` (prerequisite)
- → `/deep-learning/core/residual-connection/` (prerequisite)
- → `/deep-learning/transformer/attention/self-attention/` (prerequisite, body)
- → `/deep-learning/transformer/feed-forward-network/` (prerequisite, body)
- → `/robot-learning/act/architecture/` (related)

## Conditional Variational Autoencoder

`/generative-models/conditional-variational-autoencoder/`

- → `/generative-models/variational-autoencoder/` (prerequisite)
- → `/mathematics/probability/conditional-probability/` (prerequisite)
- → `/robot-learning/act/cvae-in-act/` (related, body)
- → `/robot-learning/act/why-z-zero-at-inference/` (body)

## Evidence Lower Bound

`/generative-models/evidence-lower-bound/`

- → `/generative-models/variational-autoencoder/` (related)
- → `/generative-models/variational-inference/` (prerequisite)
- → `/mathematics/information-theory/kl-divergence/` (prerequisite)
- → `/mathematics/probability/expectation/` (prerequisite)

## Flow Matching

`/generative-models/flow-matching/`

- → `/mathematics/calculus/ordinary-differential-equation/` (prerequisite)
- → `/mathematics/numerical-methods/euler-method/` (related)
- → `/mathematics/probability/normal-distribution/` (prerequisite)
- → `/robot-learning/pi0/flow-matching-in-pi0/` (related, body)

## Latent Variable

`/generative-models/latent-variable/`

- → `/generative-models/conditional-variational-autoencoder/` (related)
- → `/generative-models/variational-autoencoder/` (related)
- → `/generative-models/variational-inference/` (body)
- → `/mathematics/probability/random-variable/` (prerequisite)

## Posterior Collapse

`/generative-models/posterior-collapse/`

- → `/generative-models/variational-autoencoder/` (prerequisite)
- → `/mathematics/information-theory/kl-divergence/` (prerequisite)
- → `/robot-learning/act/cvae-in-act/` (related)

## Reparameterization Trick

`/generative-models/reparameterization-trick/`

- → `/generative-models/variational-autoencoder/` (related)
- → `/mathematics/probability/multivariate-normal-distribution/` (prerequisite)
- → `/mathematics/probability/normal-distribution/` (prerequisite)

## Variational Autoencoder

`/generative-models/variational-autoencoder/`

- → `/generative-models/conditional-variational-autoencoder/` (related, body)
- → `/generative-models/evidence-lower-bound/` (prerequisite, body)
- → `/generative-models/latent-variable/` (prerequisite)
- → `/generative-models/posterior-collapse/` (related)
- → `/generative-models/reparameterization-trick/` (prerequisite, body)
- → `/generative-models/variational-inference/` (prerequisite)

## Variational Inference

`/generative-models/variational-inference/`

- → `/generative-models/evidence-lower-bound/` (related, body)
- → `/generative-models/variational-autoencoder/` (related)
- → `/mathematics/information-theory/kl-divergence/` (prerequisite, body)
- → `/mathematics/probability/conditional-probability/` (prerequisite)

## Ordinary Differential Equation

`/mathematics/calculus/ordinary-differential-equation/`

- → `/generative-models/flow-matching/` (related, body)
- → `/mathematics/numerical-methods/euler-method/` (related, body)

## KL Divergence

`/mathematics/information-theory/kl-divergence/`

- → `/generative-models/evidence-lower-bound/` (related)
- → `/generative-models/posterior-collapse/` (body)
- → `/generative-models/variational-autoencoder/` (related)
- → `/mathematics/probability/expectation/` (prerequisite)
- → `/mathematics/probability/probability-distribution/` (prerequisite)

## Dot Product

`/mathematics/linear-algebra/dot-product/`

- → `/deep-learning/transformer/attention/scaled-dot-product-attention/` (related, body)
- → `/mathematics/linear-algebra/vector/` (prerequisite)

## Matrix

`/mathematics/linear-algebra/matrix/`

- → `/deep-learning/core/linear-layer/` (related)
- → `/deep-learning/transformer/attention/scaled-dot-product-attention/` (related)
- → `/mathematics/linear-algebra/vector/` (prerequisite)

## Vector

`/mathematics/linear-algebra/vector/`

- → `/mathematics/linear-algebra/dot-product/` (related, body)
- → `/mathematics/linear-algebra/matrix/` (related, body)

## Euler Method

`/mathematics/numerical-methods/euler-method/`

- → `/generative-models/flow-matching/` (related)
- → `/mathematics/calculus/ordinary-differential-equation/` (prerequisite)
- → `/robot-learning/pi0/inference/` (related)

## Conditional Probability

`/mathematics/probability/conditional-probability/`

- → `/generative-models/conditional-variational-autoencoder/` (related)
- → `/mathematics/probability/probability-distribution/` (prerequisite)

## Expectation

`/mathematics/probability/expectation/`

- → `/mathematics/information-theory/kl-divergence/` (related)
- → `/mathematics/probability/probability-distribution/` (prerequisite)
- → `/mathematics/probability/random-variable/` (prerequisite)
- → `/mathematics/probability/variance/` (related)

## Multivariate Normal Distribution

`/mathematics/probability/multivariate-normal-distribution/`

- → `/generative-models/variational-autoencoder/` (related)
- → `/mathematics/linear-algebra/matrix/` (prerequisite)
- → `/mathematics/linear-algebra/vector/` (prerequisite)
- → `/mathematics/probability/normal-distribution/` (prerequisite)
- → `/robot-learning/act/cvae-in-act/` (related)

## Normal Distribution

`/mathematics/probability/normal-distribution/`

- → `/generative-models/variational-autoencoder/` (related)
- → `/mathematics/probability/expectation/` (prerequisite)
- → `/mathematics/probability/multivariate-normal-distribution/` (related, body)
- → `/mathematics/probability/variance/` (prerequisite)

## Probability Distribution

`/mathematics/probability/probability-distribution/`

- → `/mathematics/probability/conditional-probability/` (related, body)
- → `/mathematics/probability/normal-distribution/` (related, body)
- → `/mathematics/probability/random-variable/` (prerequisite)

## Random Variable

`/mathematics/probability/random-variable/`

- → `/generative-models/latent-variable/` (related)
- → `/mathematics/probability/probability-distribution/` (related, body)

## Variance

`/mathematics/probability/variance/`

- → `/mathematics/probability/expectation/` (prerequisite)
- → `/mathematics/probability/normal-distribution/` (related)

## ACT

`/robot-learning/act/`

- → `/deep-learning/cnn/resnet/` (body)
- → `/deep-learning/transformer/` (body)
- → `/generative-models/conditional-variational-autoencoder/` (body)
- → `/robot-learning/act/action-chunking/` (related, body)
- → `/robot-learning/act/architecture/` (related, body)
- → `/robot-learning/act/complete-data-flow/` (body)
- → `/robot-learning/act/cvae-in-act/` (related, body)
- → `/robot-learning/act/inference/` (body)
- → `/robot-learning/act/temporal-ensemble/` (related, body)
- → `/robot-learning/act/training/` (body)
- → `/robot-learning/behavior-cloning/` (prerequisite, body)

## Action Chunking

`/robot-learning/act/action-chunking/`

- → `/robot-learning/act/inference/` (related)
- → `/robot-learning/act/temporal-ensemble/` (related, body)
- → `/robot-learning/behavior-cloning/` (prerequisite)

## Architecture

`/robot-learning/act/architecture/`

- → `/robot-learning/act/` (prerequisite)
- → `/robot-learning/act/cvae-in-act/` (related)
- → `/robot-learning/act/training/` (related)
- → `/robot-learning/act/vision-pipeline/` (related, body)

## Complete Data Flow

`/robot-learning/act/complete-data-flow/`

- → `/robot-learning/act/architecture/` (prerequisite)
- → `/robot-learning/act/cvae-in-act/` (related)
- → `/robot-learning/act/inference/` (prerequisite)
- → `/robot-learning/act/training/` (prerequisite)
- → `/robot-learning/act/vision-pipeline/` (related)

## CVAE in ACT

`/robot-learning/act/cvae-in-act/`

- → `/generative-models/conditional-variational-autoencoder/` (prerequisite, body)
- → `/robot-learning/act/training/` (related)
- → `/robot-learning/act/why-z-zero-at-inference/` (related, body)

## Inference

`/robot-learning/act/inference/`

- → `/robot-learning/act/action-chunking/` (prerequisite)
- → `/robot-learning/act/architecture/` (prerequisite)
- → `/robot-learning/act/temporal-ensemble/` (related)
- → `/robot-learning/act/why-z-zero-at-inference/` (related, body)

## Paper and Released Implementation

`/robot-learning/act/paper-and-released-implementation/`

- → `/robot-learning/act/` (prerequisite)
- → `/robot-learning/act/architecture/` (related)
- → `/robot-learning/act/training/` (related)

## Temporal Ensemble

`/robot-learning/act/temporal-ensemble/`

- → `/robot-learning/act/action-chunking/` (prerequisite)
- → `/robot-learning/act/inference/` (related)

## Training

`/robot-learning/act/training/`

- → `/robot-learning/act/architecture/` (prerequisite)
- → `/robot-learning/act/cvae-in-act/` (prerequisite)
- → `/robot-learning/act/inference/` (related)
- → `/robot-learning/act/paper-and-released-implementation/` (related)

## Vision Pipeline

`/robot-learning/act/vision-pipeline/`

- → `/deep-learning/cnn/resnet/` (prerequisite, body)
- → `/deep-learning/transformer/positional-encoding/` (prerequisite)
- → `/robot-learning/act/architecture/` (related)

## 为什么 ACT 推理时令 z = 0？

`/robot-learning/act/why-z-zero-at-inference/`

- → `/mathematics/probability/normal-distribution/` (prerequisite)
- → `/robot-learning/act/cvae-in-act/` (prerequisite)
- → `/robot-learning/act/inference/` (related)

## Behavior Cloning

`/robot-learning/behavior-cloning/`

- → `/robot-learning/act/` (related)
- → `/robot-learning/act/action-chunking/` (related, body)
- → `/robot-learning/imitation-learning/` (prerequisite)

## Cross-Embodiment Learning

`/robot-learning/cross-embodiment-learning/`

- → `/robot-learning/imitation-learning/` (prerequisite)
- → `/robot-learning/pi0/pretraining-and-posttraining/` (related)
- → `/robot-learning/vision-language-action-model/` (related)

## Imitation Learning

`/robot-learning/imitation-learning/`

- → `/robot-learning/act/` (related)
- → `/robot-learning/act/action-chunking/` (body)
- → `/robot-learning/act/cvae-in-act/` (body)
- → `/robot-learning/behavior-cloning/` (related, body)

## π0

`/robot-learning/pi0/`

- → `/deep-learning/multimodal/vision-language-model/` (body)
- → `/generative-models/flow-matching/` (prerequisite, body)
- → `/robot-learning/act/action-chunking/` (prerequisite, body)
- → `/robot-learning/pi0/action-expert/` (related, body)
- → `/robot-learning/pi0/architecture/` (related, body)
- → `/robot-learning/pi0/blockwise-causal-attention-mask/` (body)
- → `/robot-learning/pi0/complete-data-flow/` (body)
- → `/robot-learning/pi0/flow-matching-in-pi0/` (body)
- → `/robot-learning/pi0/inference/` (related, body)
- → `/robot-learning/pi0/pretraining-and-posttraining/` (body)
- → `/robot-learning/pi0/training/` (related, body)
- → `/robot-learning/vision-language-action-model/` (prerequisite)

## Action Expert

`/robot-learning/pi0/action-expert/`

- → `/deep-learning/transformer/attention/self-attention/` (prerequisite)
- → `/robot-learning/pi0/architecture/` (prerequisite)
- → `/robot-learning/pi0/blockwise-causal-attention-mask/` (related)
- → `/robot-learning/pi0/flow-matching-in-pi0/` (related)

## Architecture

`/robot-learning/pi0/architecture/`

- → `/deep-learning/multimodal/vision-language-model/` (prerequisite)
- → `/deep-learning/transformer/attention/self-attention/` (prerequisite)
- → `/mathematics/numerical-methods/euler-method/` (body)
- → `/robot-learning/pi0/` (prerequisite)
- → `/robot-learning/pi0/action-expert/` (related, body)
- → `/robot-learning/pi0/blockwise-causal-attention-mask/` (related, body)
- → `/robot-learning/pi0/complete-data-flow/` (related)

## Blockwise Causal Attention Mask

`/robot-learning/pi0/blockwise-causal-attention-mask/`

- → `/deep-learning/transformer/attention/self-attention/` (prerequisite)
- → `/deep-learning/transformer/causal-mask/` (prerequisite)
- → `/robot-learning/act/action-chunking/` (body)
- → `/robot-learning/pi0/action-expert/` (related)
- → `/robot-learning/pi0/architecture/` (prerequisite)
- → `/robot-learning/pi0/inference/` (related)

## Complete Data Flow

`/robot-learning/pi0/complete-data-flow/`

- → `/robot-learning/pi0/action-expert/` (related)
- → `/robot-learning/pi0/architecture/` (prerequisite)
- → `/robot-learning/pi0/flow-matching-in-pi0/` (related)
- → `/robot-learning/pi0/inference/` (prerequisite)
- → `/robot-learning/pi0/training/` (prerequisite)

## Flow Matching in π0

`/robot-learning/pi0/flow-matching-in-pi0/`

- → `/generative-models/flow-matching/` (prerequisite)
- → `/mathematics/numerical-methods/euler-method/` (related)
- → `/robot-learning/pi0/architecture/` (prerequisite)
- → `/robot-learning/pi0/inference/` (related)
- → `/robot-learning/pi0/paper-and-released-implementation/` (body)
- → `/robot-learning/pi0/training/` (related)

## Inference

`/robot-learning/pi0/inference/`

- → `/mathematics/numerical-methods/euler-method/` (prerequisite)
- → `/robot-learning/act/temporal-ensemble/` (body)
- → `/robot-learning/pi0/blockwise-causal-attention-mask/` (related, body)
- → `/robot-learning/pi0/complete-data-flow/` (related)
- → `/robot-learning/pi0/flow-matching-in-pi0/` (prerequisite)

## Paper and Released Implementation

`/robot-learning/pi0/paper-and-released-implementation/`

- → `/robot-learning/pi0/` (prerequisite)
- → `/robot-learning/pi0/action-expert/` (related)
- → `/robot-learning/pi0/inference/` (related)
- → `/robot-learning/pi0/training/` (related)

## Pre-training and Post-training in π0

`/robot-learning/pi0/pretraining-and-posttraining/`

- → `/robot-learning/cross-embodiment-learning/` (prerequisite, body)
- → `/robot-learning/pi0/` (prerequisite)
- → `/robot-learning/pi0/training/` (related)

## Training

`/robot-learning/pi0/training/`

- → `/robot-learning/pi0/architecture/` (prerequisite)
- → `/robot-learning/pi0/complete-data-flow/` (related)
- → `/robot-learning/pi0/flow-matching-in-pi0/` (prerequisite)
- → `/robot-learning/pi0/inference/` (related)
- → `/robot-learning/pi0/pretraining-and-posttraining/` (related)

## Vision-Language-Action Model

`/robot-learning/vision-language-action-model/`

- → `/deep-learning/multimodal/vision-language-model/` (prerequisite, body)
- → `/robot-learning/cross-embodiment-learning/` (related)
- → `/robot-learning/pi0/` (related)
