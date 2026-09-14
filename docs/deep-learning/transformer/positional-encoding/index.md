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

Positional Encoding 给序列或空间特征加入位置信息。标准 attention 根据输入表示计算关系，但不会自动把“第 1 个位置”和“第 10 个位置”当成不同的绝对位置，因此需要额外位置表示。

## 一维正弦位置编码

原始 Transformer 使用固定的 sinusoidal encoding。对 position $pos$ 和通道索引 $i$：

\[
PE(pos,2i)=\sin\left(\frac{pos}{10000^{2i/d_{model}}}\right),
\]

\[
PE(pos,2i+1)=\cos\left(\frac{pos}{10000^{2i/d_{model}}}\right).
\]

不同通道具有不同频率，因此一个位置会对应一组跨尺度的正弦/余弦值。位置编码与 token embedding 具有相同的 $d_{model}$ 维度，可以直接相加。

## 固定与可学习

位置表示不只有一种实现。常见选择包括：

- 固定 sinusoidal encoding；
- learned positional embedding；
- 二维或更高维位置编码；
- 相对位置或旋转式位置编码等后续方法。

这些方案的共同目标是让模型获得结构中的位置信息，但具体数学形式并不相同。

## 图像特征中的二维位置

CNN 输出的 feature map 仍保留空间网格。若把 $H\times W$ 网格 flatten 成 $HW$ 个 tokens，仅靠 flatten 后的序号会丢失显式二维结构。

ACT 论文对每个 ResNet18 feature map 加入 2D sinusoidal positional encoding，再送入 Transformer。这样每个视觉 token 同时携带内容特征与其图像位置。

## ACT 中另一类位置

ACT 的 CVAE encoder 处理 `[CLS]`、当前 joint position 和 action sequence 时，官方实现使用固定 sinusoidal table。Policy decoder 的 action query slots 则是 learned embeddings。二者都涉及“位置/槽位身份”，但来源与功能不同，不应合并成同一个参数。

## Sources

- [Attention Is All You Need — Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)
- [ACT official implementation](https://github.com/tonyzhaozh/act)
