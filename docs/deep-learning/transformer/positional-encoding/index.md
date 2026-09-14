---
title: "Positional Encoding"
kind: "canonical"
domain: "Deep Learning / Transformer"
parent: "Transformer"
canonical: "/deep-learning/transformer/positional-encoding/"
prerequisites:
  - "/deep-learning/core/embedding/"
related:
  - "/deep-learning/transformer/"
  - "/robot-learning/act/vision-pipeline/"
---

# Positional Encoding

Positional Encoding 用来给 Transformer representation 加入“位置”信息。

标准 self-attention 可以比较 token contents，但如果只给一组 vectors，本身并没有一个内置的循环结构告诉模型哪个是第 1 个、哪个是第 7 个。

因此输入通常写成

\[
h_i=x_i+p_i,
\]

其中 $x_i$ 是内容表示，$p_i$ 是位置表示。

## 内容信息不足以表示顺序

考虑两个序列：

```text
A B C
C B A
```

如果 A、B、C 的 token embeddings 完全相同，而模型没有任何位置输入，self-attention 只知道“这里有 A/B/C”，没有独立信号表示它们的排列顺序。

加入 positional representation 后：

```text
A + position 1
B + position 2
C + position 3
```

位置成为模型可读取的信息。

## Sinusoidal Positional Encoding

原始 Transformer 使用固定的 sine/cosine functions。对位置 $pos$ 和 embedding 维度索引 $i$：

\[
PE(pos,2i)=\sin\left(pos/10000^{2i/d_{model}}\right),
\]

\[
PE(pos,2i+1)=\cos\left(pos/10000^{2i/d_{model}}\right).
\]

不同维度具有不同频率，因此一个位置会得到一整组周期不同的数值。

## Learned Positional Embedding

也可以直接把每个位置的向量当作参数学习：

\[
p_i\in\mathbb R^{d_{model}}.
\]

这时位置表示不由固定公式决定，而由训练优化。

所以“Transformer 使用 positional encoding”不等于“Transformer 必须使用 sinusoidal encoding”。

## 一维序列与二维图像

文本通常是一维位置。视觉 feature map 则有 row/column 两个空间维度，因此位置表示需要编码二维坐标或等价结构。

DETR 与 ACT 的视觉 backbone 会为 spatial feature map 提供位置表示，让 Transformer 能区分不同图像区域。

## 位置表示与 Attention 的边界

Positional encoding 不负责决定注意力权重，它只是把位置相关信息加入 representations。真正的读取关系仍由 Q/K dot products 与 softmax 学习。

## Sources

- Vaswani et al., **Attention Is All You Need**, 2017. https://arxiv.org/abs/1706.03762
- Carion et al., **End-to-End Object Detection with Transformers**, 2020. https://arxiv.org/abs/2005.12872
