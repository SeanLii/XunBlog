---
title: "Convolution"
kind: "canonical"
domain: "Deep Learning / Convolutional Neural Networks"
parent: "Convolutional Neural Networks"
canonical: "/deep-learning/cnn/convolution/"
prerequisites:
  - "/mathematics/linear-algebra/matrix/"
related:
  - "/deep-learning/cnn/convolutional-neural-network/"
---

# Convolution

Convolution 在 CNN 中用一个小型可学习 kernel 在输入的局部区域上重复计算。它让同一组参数能够在不同空间位置检测相似的局部模式。

## 二维离散计算

对单通道输入 $X$ 和 kernel $K$，深度学习库实际常实现 cross-correlation：

\[
Y_{i,j}
=\sum_{u=0}^{k_h-1}\sum_{v=0}^{k_w-1}
K_{u,v}X_{i+u,j+v}.
\]

严格数学卷积会翻转 kernel，而 CNN 文献和框架通常仍把上述运算称为 convolution。因为 kernel 参数是通过学习得到的，这个翻转差异不会改变“局部共享线性滤波器”的核心作用。

## 多通道

若输入有 $C_{in}$ 个通道，一个输出 channel 的 kernel 会跨越所有输入 channels：

\[
K\in\mathbb R^{C_{in}\times k_h\times k_w}.
\]

若要得到 $C_{out}$ 个输出 channels，则需要 $C_{out}$ 组 kernels。权重整体 shape 可写成

\[
C_{out}\times C_{in}\times k_h\times k_w.
\]

## Stride 与 Padding

Stride 决定 kernel 每次移动多少像素。较大 stride 会降低 feature map 的空间分辨率。Padding 在输入边缘补值，使输出尺寸按设计保留或缩小。

对单个空间维度，常见输出大小为

\[
\left\lfloor
\frac{N+2P-K}{S}
\right\rfloor+1,
\]

其中 $N$ 是输入大小，$P$ 是 padding，$K$ 是 kernel size，$S$ 是 stride。

## 局部连接与参数共享

全连接层为每个输入位置单独学习权重；convolution 在所有空间位置复用同一个 kernel。这样既减少参数量，也把“某种局部模式可以出现在图像不同位置”这一结构先验加入模型。

ACT 的视觉 backbone 使用 ResNet18，而 ResNet 的主体由 convolutional layers 与 residual blocks 组成。
