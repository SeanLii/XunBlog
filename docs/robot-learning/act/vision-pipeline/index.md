---
title: "Vision Pipeline"
kind: "act-topic"
domain: "Robot Learning / ACT"
parent: "ACT"
canonical: "/robot-learning/act/vision-pipeline/"
prerequisites:
  - "/deep-learning/cnn/resnet/"
  - "/deep-learning/sequence-modeling/positional-encoding/"
related:
  - "/robot-learning/act/architecture/"
---

# Vision Pipeline

ACT 的视觉部分负责把多路 RGB images 转成 Transformer 可以处理的 spatial features。

它不是“把整张图压成一个向量再交给 Transformer”，而是保留 feature map 的空间结构，让后续 attention 仍能读取不同图像位置。

## 从一张 RGB 图像开始

输入图像可写成

\[
I\in\mathbb R^{3\times H\times W}.
\]

先经过 [ResNet](/deep-learning/cnn/resnet/) backbone：

\[
I\rightarrow F,
\qquad
F\in\mathbb R^{C_b\times H'\times W'}.
\]

这里 $F$ 是 feature map。每个 spatial location 都对应一个 feature vector，而不是一个最终分类结果。

## 投影到 Transformer hidden dimension

ResNet 输出 channel 数 $C_b$ 不一定等于 Transformer hidden dimension $d$。官方实现使用 $1\times1$ convolution 做 channel projection：

\[
F' = \operatorname{Conv}_{1\times1}(F),
\qquad
F'\in\mathbb R^{d\times H'\times W'}.
\]

这一步主要改变 channel dimension，不需要把 spatial grid 混在一起。

## 位置表示

Transformer attention 只看 feature 内容时，不天然知道 feature 来自图像哪个位置。因此 backbone 同时提供 positional representation。

可以把每个视觉 token 想成：

```text
视觉内容 feature
+
它在 feature map 中的位置
```

这样 attention 才能区分左上角和右下角即使视觉 feature 相似的两个位置。

## 多相机怎样合并

假设有多个 camera：

```text
cam 1 → feature map F1
cam 2 → feature map F2
cam 3 → feature map F3
cam 4 → feature map F4
```

released code 对每个 camera 使用同一个 backbone module，然后把得到的 features 收集起来，并沿 width 维拼接：

```text
F1 | F2 | F3 | F4
```

对应 positional features 也按同样方式拼接。

所以从 Transformer 的角度，它接收到的是一个更宽的 spatial feature grid；不同 camera 的内容被放进同一个 memory source 中。

## ResNet 与 Transformer 的视觉分工

原始 RGB pixels 维度高，而且局部结构非常强。CNN/ResNet 已经擅长把边缘、纹理、物体局部等视觉信息逐层提取成更紧凑的 representations。

ACT 的分工因此是：

```text
ResNet:
raw image → useful visual features

Transformer:
visual features + robot state + latent
→ global information interaction
→ action representations
```

这不是说 Transformer 不能直接处理 image patches，而是 ACT 的具体架构选择了 CNN backbone + Transformer 的组合。

## 视觉 feature 最终怎样影响 action query

视觉 features 进入 Transformer 后成为 encoder memory 的一部分。Decoder 中每个 action query 可以通过 attention 从这份 memory 中读取与当前未来动作槽位有关的视觉信息。

例如同一个 action chunk 中：

- 较早动作可能主要依赖当前 gripper 与物体位置；
- 较后动作可能读取另一片区域的信息。

模型并没有人为指定哪个 query 必须看哪里，这些 attention patterns 由训练学习。

## Sources

- Zhao et al., **Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware**, 2023. https://arxiv.org/abs/2304.13705
- Official ACT model: https://github.com/tonyzhaozh/act/blob/main/detr/models/detr_vae.py
- He et al., **Deep Residual Learning for Image Recognition**, 2015/2016. https://arxiv.org/abs/1512.03385
