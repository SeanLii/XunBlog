---
title: "Transformer"
kind: "canonical"
domain: "Deep Learning / Transformer"
parent: "Transformer"
canonical: "/deep-learning/transformer/"
prerequisites:
  - "/deep-learning/attention/"
  - "/deep-learning/sequence-modeling/positional-encoding/"
related:
  - "/deep-learning/transformer/transformer-encoder/"
  - "/deep-learning/transformer/transformer-decoder/"
  - "/robot-learning/act/architecture/"
standard: "XunBlog Content & Knowledge Architecture v1.0"
rebuilt: "2026-09-15"
---
# Transformer

Transformer 是一种以 attention 为主要 token-interaction mechanism，并结合 position information、position-wise nonlinear transformation、residual connection 与 normalization 构成的神经网络架构。

给定一组 input representations

\[
X=(x_1,\ldots,x_N),
\qquad x_i\in\mathbb R^{d_{model}},
\]

Transformer 产生新的 representations

\[
H=(h_1,\ldots,h_N),
\]

其中每个 $h_i$ 可以通过 attention 聚合其他 positions 的信息，因此不再只由 $x_i$ 独立决定。

## Transformer Layer

典型 Transformer layer 包含两类主要计算：

1. attention sublayer：在 positions 之间交换信息；
2. position-wise feed-forward sublayer：对每个 position 独立进行 nonlinear feature transformation。

Encoder layer 的基本结构为

```text
input representations
        ↓
Multi-Head Self-Attention
        ↓
Residual + LayerNorm
        ↓
Position-Wise Feed-Forward Network
        ↓
Residual + LayerNorm
        ↓
output representations
```

原始 Transformer 使用 post-norm；许多后续模型采用 pre-norm 或其他 normalization placement。

## Attention Computation

Transformer 使用 [Scaled Dot-Product Attention](/deep-learning/transformer/scaled-dot-product-attention/)：

\[
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)V.
\]

其中 [Query / Key / Value](/deep-learning/attention/qkv/) 分别承担读取请求、匹配索引和内容传输的计算角色。

Transformer 又通过 [Multi-Head Attention](/deep-learning/transformer/multi-head-attention/) 并行建立多个 projection spaces，再将各 heads 的输出合并回 model dimension。

Attention 是 Transformer 的核心组件，但 attention computation 本身不等于完整 Transformer architecture。

## Positional Information

标准 self-attention 对 token permutation 具有对应的 permutation-equivariance，因此单独使用 content vectors 时并不编码 sequence order。

Transformer 必须额外加入 [Positional Encoding](/deep-learning/sequence-modeling/positional-encoding/) 或其他 position mechanism，使 representation 同时包含内容与位置关系。

原始 Transformer 使用 sinusoidal positional encoding；后续模型发展出 learned absolute positions、relative positions、rotary position embeddings 等不同设计。

## Position-Wise Nonlinear Transformation

Attention 完成跨位置的信息交换后，[Position-Wise Feed-Forward Network](/deep-learning/transformer/position-wise-feed-forward-network/) 对每个 token 独立执行同一组 MLP parameters：

\[
\operatorname{FFN}(x)=W_2\phi(W_1x+b_1)+b_2.
\]

因此 Transformer layer 将两类操作分离：

- token mixing：attention；
- feature transformation：position-wise FFN。

多层堆叠使 representations 反复进行跨位置读取与局部 nonlinear transformation。

## Residual Connection and Layer Normalization

每个主要 sublayer 周围通常使用 [Residual Connection](/deep-learning/cnn/resnet/residual-connection/) 与 [Layer Normalization](/deep-learning/core/layer-normalization/)。

Residual path 提供 identity information path；LayerNorm 调整 feature statistics。它们共同影响深层 Transformer 的 optimization stability，但都不是 attention score computation 的一部分。

## Transformer Encoder

[Transformer Encoder](/deep-learning/transformer/transformer-encoder/) 由多层 encoder layers 堆叠而成。原始结构中的 encoder self-attention 对输入 positions 使用全局可见性，因此每个位置可以形成 contextual representation。

输入

\[
X\in\mathbb R^{N\times d_{model}}
\]

经过 encoder stack 后通常仍保持

\[
H\in\mathbb R^{N\times d_{model}}.
\]

sequence length 可以不变，但每个位置所编码的信息已经发生变化。

## Transformer Decoder

原始 [Transformer Decoder](/deep-learning/transformer/transformer-decoder/) 包含：

1. masked self-attention；
2. cross-attention to encoder memory；
3. position-wise FFN。

在 machine translation 中，causal mask 保证 target position 不能读取未来 target tokens；cross-attention 则让 target states 读取 source encoder representations。

Decoder 这一 architecture 后来被扩展到不同信息流：GPT-style decoder-only models 删除 encoder cross-attention；DETR 与 ACT 则使用 learned query slots 并行读取 encoder memory，而不是逐 token autoregressive generation。

## Encoder-Only, Decoder-Only, and Encoder–Decoder Models

Transformer 并不要求固定使用原始 encoder–decoder 组合。

- BERT：encoder-only stack；
- GPT-style language models：causal decoder-style stack；
- T5 等 sequence-to-sequence models：encoder–decoder；
- DETR：encoder–decoder with learned object queries；
- ACT：encoder–decoder with learned action queries。

这些模型共享部分 Transformer building blocks，但 mask、query source、training objective 与 output structure 不同。

## Tensor Shapes

设

\[
X\in\mathbb R^{B\times N\times d_{model}}.
\]

标准 encoder layer 通常保持 shape：

\[
X\rightarrow H,
\qquad
H\in\mathbb R^{B\times N\times d_{model}}.
\]

在 attention 内部，multi-head projection 会暂时拆分 head dimension；FFN 会将 feature dimension扩展到 $d_{ff}$ 后再投影回 $d_{model}$。因此 layer 的外部 shape 可以保持不变，而内部 representation 与 pairwise information flow 持续变化。

## Computational Characteristics

Full self-attention 的 pairwise score matrix 大小为

\[
N\times N,
\]

因此对长 sequence 会带来 quadratic time / memory cost。另一方面，self-attention 中任意两个 positions 可以在一层内直接建立 interaction path，这也是 Transformer 与固定局部 receptive field architecture 的重要差异之一。

Transformer 的整体计算成本还包括 FFN、projection、KV cache 等部分；在不同 sequence length 与 model dimension 下，瓶颈可能不同。

## Relation to ACT

ACT 将视觉 features、proprioception 与 latent-conditioned information 编码为 memory，再使用 learned action queries 经过 Transformer decoder 并行产生 future action representations。

其中 learned output-query pattern 直接继承自 [DETR](/deep-learning/detr/) 风格的 decoder queries，而不是原始 Transformer 的固定标准输入。

Transformer 在 ACT 中提供的是一种信息交互架构；Action Chunking、CVAE branch、Temporal Ensemble 等仍属于 ACT 自身的 policy design。

## Sources

- Vaswani et al. *Attention Is All You Need*. 2017. https://arxiv.org/abs/1706.03762
- Devlin et al. *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. 2018.
- Carion et al. *End-to-End Object Detection with Transformers*. 2020.
