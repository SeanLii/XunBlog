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

Transformer Encoder 把一组输入 representations 反复进行“相互读取 + 各自变换”，得到带有上下文的信息表示。

输入和输出通常保持相同的位置数量：

\[
X\in\mathbb R^{n\times d}
\rightarrow
H\in\mathbb R^{n\times d}.
\]

改变的是每个位置包含的信息，而不是必须把序列压缩成一个向量。

## 一个 Encoder Layer

原始 Transformer encoder layer 由两个核心 sublayers 组成：

```text
X
│
├─ Multi-Head Self-Attention
│
├─ Residual + LayerNorm
│
├─ Feed-Forward Network
│
└─ Residual + LayerNorm
↓
H
```

[Self-Attention](/deep-learning/transformer/attention/self-attention/) 让位置之间交换信息；[Feed-Forward Network](/deep-learning/transformer/feed-forward-network/) 对每个位置独立做非线性变换。

## 多层堆叠

Encoder 通常堆叠多个相同结构的 layers：

\[
H^{(0)}=X,
\]

\[
H^{(l+1)}=\operatorname{EncoderLayer}(H^{(l)}).
\]

每一层都可以重新计算 attention，因此“谁读取谁”不是第一层确定后就固定不变。

## Encoder Memory

在 encoder-decoder 架构中，最后一层输出常称为 encoder memory：

```text
input → encoder stack → memory
                          ↑
                          │
                  decoder reads it
```

memory 不是一个特殊的数据类型，只是“encoder 已经处理好的 representations”，供后续 decoder cross-attention 读取。

## ACT 中的两种 Encoder

ACT 有两个容易混淆的 encoder 角色：

1. policy Transformer 中对 observation features 的 encoder；
2. training-only CVAE branch 中对 `[CLS, qpos, action sequence]` 的 Transformer encoder。

它们都使用 Transformer encoder 机制，但输入、目标和生命周期不同。

## Sources

- Vaswani et al., **Attention Is All You Need**, 2017. https://arxiv.org/abs/1706.03762
- Zhao et al., **Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware**, 2023. https://arxiv.org/abs/2304.13705
