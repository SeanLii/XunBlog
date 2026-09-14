---
title: "Transformer Encoder"
kind: "canonical"
domain: "Deep Learning / Transformer"
parent: "Transformer"
canonical: "/deep-learning/transformer/transformer-encoder/"
prerequisites:
  - "/deep-learning/transformer/attention/self-attention/"
  - "/deep-learning/transformer/feed-forward-network/"
  - "/deep-learning/core/residual-connection/"
  - "/deep-learning/core/layer-normalization/"
related:
  - "/robot-learning/act/architecture/"
---

# Transformer Encoder

Transformer Encoder 是一组重复堆叠的 encoder layers。每层用 self-attention 让各输入位置交换信息，再用 position-wise feed-forward network 独立处理每个位置的特征。

## 单层结构

原始 Transformer encoder layer 包含两个主要子层：

1. Multi-Head Self-Attention；
2. Position-wise Feed-Forward Network。

每个子层外都有 residual connection，并在原始论文中采用 post-norm：

\[
y=\operatorname{LayerNorm}(x+\operatorname{Sublayer}(x)).
\]

后续模型常见 pre-norm 变体，因此“encoder”这个概念应与某一具体 normalization ordering 区分开。

## Self-Attention 的信息融合

若输入有 $N$ 个 tokens，self-attention 会产生 $N\times N$ 的 attention score structure。第 $i$ 个 token 可以从其他 tokens 的 values 中得到加权信息，因此经过一层后，每个位置的表示都可能依赖整个可见序列。

## Feed-Forward 的局部变换

Attention 负责 token 间的信息交换，feed-forward network 则对每个 token 分别应用相同的非线性映射。两者承担不同角色。

## Stack

将 encoder layer 重复 $L$ 次后，得到更深的 contextualization。层数 $L$ 是模型超参数，不属于 Transformer Encoder 的定义。

## ACT 中的两套 Encoder

ACT 内部有两个不同用途的 Transformer encoders：

- CVAE encoder：训练时读取 `[CLS]`、当前 joint positions 与 demonstration action chunk，生成 latent posterior parameters；
- Policy observation encoder：读取经过 ResNet 处理的多相机视觉 features，再与 proprioception 和 latent information 一起形成 observation memory。

这两者都使用 Transformer Encoder 的通用机制，但输入语义和训练角色不同。

## Sources

- [Attention Is All You Need — Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)
- [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware — Zhao et al., 2023](https://arxiv.org/abs/2304.13705)
