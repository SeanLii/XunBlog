---
title: "ResNet"
kind: "canonical"
domain: "Deep Learning / Convolutional Neural Networks"
parent: "Convolutional Neural Networks"
canonical: "/deep-learning/cnn/resnet/"
prerequisites:
  - "/deep-learning/cnn/convolutional-neural-network/"
related:
  - "/deep-learning/cnn/resnet/residual-connection/"
---

# ResNet

ResNet（Residual Network）是 He 等人在 2015 年提出的 deep convolutional network family。它的核心设计是把网络 block 从“直接学习目标 mapping”改成“在 identity path 上学习 residual correction”。

最基本形式：

\[
y=x+F(x).
\]

ResNet 的重要性不只是发明了 skip connection，而是它让非常深的 CNN 在实践中变得明显更容易优化。

## Depth 与 Degradation Problem

直觉上，更深网络至少应该能模拟较浅网络：多出来的 layers 如果什么都不做，保持 identity 就行。

但 He 等人在 plain networks 中观察到 degradation problem：网络变深后 training error 反而可能升高。

这不是单纯 overfitting，因为更深模型连 training set 都优化得更差。

因此问题落在 optimization：即使理论上存在一个不差的 solution，普通 parameterization 也不容易找到它。

## Residual Learning

假设想学习目标 mapping：

\[
H(x).
\]

ResNet 改为让 learned branch 学：

\[
F(x)=H(x)-x.
\]

于是：

\[
H(x)=x+F(x).
\]

如果理想 mapping 接近 identity，只需让 $F(x)\approx0$。

这就是 residual learning 的核心。

## Basic Block

ResNet-18 / ResNet-34 常使用 Basic Block：

```text
x ────────────────────┐
│                     │
Conv 3×3               │
 ↓                     │
Norm + ReLU            │
 ↓                     │
Conv 3×3               │
 ↓                     │
Norm                    │
│                     │
└──────── add ─────────┘
          ↓
         ReLU
```

简化写成：

\[
y=\operatorname{ReLU}(x+F(x)).
\]

## Bottleneck Block

更深的 ResNet-50 / 101 / 152 使用 Bottleneck Block：

```text
1×1 Conv   reduce / transform channels
   ↓
3×3 Conv   spatial processing
   ↓
1×1 Conv   expand channels
```

然后与 shortcut 相加。

1×1 convolutions 让 block 在控制计算量的同时使用较宽的 intermediate representation strategy。

## Projection Shortcut

如果 spatial size 或 channel count 改变：

\[
\operatorname{shape}(x)
\ne
\operatorname{shape}(F(x)),
\]

不能直接 addition。

于是使用 learned projection：

\[
y=W_sx+F(x),
\]

常由 1×1 convolution 完成，同时可能带 stride。

## Stage Structure

典型 ResNet 将 blocks 分成多个 stages。

随着 stage 向后：

- spatial resolution 下降；
- channel count 增加；
- receptive field 增大；
- representation 更适合高层 visual task。

例如 ResNet-50 常见主干可以概括成：

```text
stem
 ↓
conv2_x
 ↓
conv3_x
 ↓
conv4_x
 ↓
conv5_x
 ↓
global pooling / downstream head
```

## ResNet-18、34、50 的数字表示什么

这些数字大致对应有 learnable weights 的 network depth convention。

ResNet-18 / 34 使用 Basic Blocks；ResNet-50 / 101 / 152 使用 Bottleneck Blocks。

所以 “ResNet-50 比 ResNet-34 多 16 层” 不是简单在同一 block 后面继续复制几次，而是 block type 和 stage configuration 也不同。

## Residual Path 的 Optimization Effect

对：

\[
y=x+F(x),
\]

Jacobian 包含 identity term：

\[
\frac{\partial y}{\partial x}
=I+\frac{\partial F}{\partial x}.
\]

这为 forward information 与 backward gradient 提供直接路径。

但不要把它简化成“skip connection 彻底解决 vanishing gradient”。ResNet 的训练优势来自 parameterization、normalization、architecture design 等共同作用。

## ResNet 作为 Backbone

ResNet 不只用于 ImageNet classification。

去掉最终 classifier 后，中间 feature maps 可以作为 backbone output，供：

- object detection；
- segmentation；
- multimodal model；
- robot policy；
- metric learning。

ACT 使用 ResNet 作为视觉特征提取器只是其中一个 downstream application。

## Sources

- He et al. *Deep Residual Learning for Image Recognition*. 2015/2016.
