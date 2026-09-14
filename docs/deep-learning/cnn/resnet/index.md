---
title: "ResNet"
kind: "canonical"
domain: "Deep Learning / Convolutional Neural Networks"
parent: "Convolutional Neural Networks"
canonical: "/deep-learning/cnn/resnet/"
prerequisites:
  - "/deep-learning/cnn/convolutional-neural-network/"
  - "/deep-learning/core/residual-connection/"
related:
  - "/robot-learning/act/vision-pipeline/"
---

# ResNet

ResNet（Residual Network）是一类通过 residual connection 构建深层神经网络的架构。它最重要的变化可以只看一条式子：

普通 block 想直接学习

\[
y=H(x),
\]

而 residual block 改成

\[
y=x+F(x).
\]

这里 $F(x)$ 不是最终输出，而是“在输入 $x$ 基础上还需要补充什么变化”。

## 一个 residual block 的形状

```text
x ────────────────┐
│                 │
↓                 │
Conv → ... → Conv │
│                 │
↓                 │
F(x)              │
│                 │
└────── + ←───────┘
        │
        ↓
      y=x+F(x)
```

右边这条直接把输入送到后面的路径通常称为 shortcut / skip connection。

如果最合适的变换接近 identity mapping，网络不需要重新学习完整的 $H(x)=x$；只需要让

\[
F(x)\approx 0.
\]

这使很深的网络更容易优化，也是原始 ResNet 论文的核心动机。

## ResNet 仍然是 CNN

ResNet 没有替代 convolution。它的视觉 feature extraction 仍主要由 convolutional blocks 组成。Residual connection 改变的是这些 layers 的组织方式。

所以知识关系是：

```text
Convolution
    ↓
Convolutional Neural Network
    ↓
ResNet
```

[Convolution](/deep-learning/cnn/convolution/) 负责在图像局部区域提取模式；ResNet 则解决“怎样把很多 convolutional layers 组织成更深、仍然能有效训练的网络”。

## Feature map 会逐渐改变

一张 RGB 图像输入 ResNet 后，会经历多层 convolution 和下采样。空间尺寸通常逐渐减小，而 channel 数增加：

```text
H × W × 3
   ↓
H/4 × W/4 × C1
   ↓
H/8 × W/8 × C2
   ↓
H/16 × W/16 × C3
   ↓
...
```

后面的 feature map 不再直接表示像素颜色，而是学习到更高层的视觉 patterns。

具体任务可以选择：

- 最终 global feature vector；
- 中间 spatial feature map；
- 多尺度 features。

因此“使用 ResNet”并不意味着一定拿 ImageNet classification 的最终 logits。

## ResNet 在 ACT 中的作用

ACT 要从多路相机图像中提取视觉信息，但 Transformer 本身不会直接替代所有图像特征提取步骤。官方 ACT architecture 使用 ResNet backbone 先把每个 camera image 变成 spatial visual features，再把这些 features 投影到 Transformer hidden dimension。

可以先看成：

```text
RGB image
   │
   ↓
ResNet backbone
   │
   ↓
spatial feature map
   │
   ↓
1×1 projection
   │
   ↓
Transformer tokens
```

因此 ACT 使用的是 ResNet 的**视觉表征能力**，不是它原本的 image-classification head。

## Sources

- He et al., **Deep Residual Learning for Image Recognition**, 2015/2016. https://arxiv.org/abs/1512.03385
- Official ACT implementation. https://github.com/tonyzhaozh/act
