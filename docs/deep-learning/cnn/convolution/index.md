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

Convolution 是在输入的局部窗口上重复应用同一组权重，从而得到 feature map 的运算。

对图像来说，可以先想象一个小 kernel 在二维平面上滑动：

```text
image
┌─────────────┐
│ [3×3 area]  │ → dot product with kernel → one output value
│             │
└─────────────┘

kernel 向右/向下移动，重复同一计算
```

## 单通道二维形式

设 kernel $K\in\mathbb R^{r\times s}$。在某个输出位置 $(i,j)$：

\[
Y_{ij}
=
\sum_{u=1}^{r}\sum_{v=1}^{s}
K_{uv}X_{i+u,j+v}.
\]

深度学习框架通常实现的是 cross-correlation 形式（不翻转 kernel），但习惯上仍称 convolution。

## 多通道图像

RGB input 有 3 个 channels。一个 convolution filter 会同时跨所有 input channels：

\[
K\in\mathbb R^{C_{in}\times r\times s}.
\]

如果有 $C_{out}$ 个 filters，就产生 $C_{out}$ 个 output feature maps。

## 参数共享

同一个 kernel 在所有 spatial positions 重复使用。因此“在左上角检测某种局部 pattern”的参数也能在右下角使用。

这就是 CNN 比 fully connected image model 更强的结构先验之一：局部关系与空间平移中的 pattern reuse 被直接写入网络计算方式。

## Stride、Padding 与输出尺寸

- stride：kernel 每次移动多少格；
- padding：在边缘补额外值；
- dilation：kernel 元素之间留间隔。

它们共同决定 output spatial size 与 receptive field。

常见输出高度：

\[
H_{out}
=
\left\lfloor
\frac{H+2P-D(K-1)-1}{S}+1
\right\rfloor.
\]

## 多层 Convolution

第一层只看很小局部；后面的 layer 又在前一层 features 上卷积，因此它的 receptive field 对应原图更大区域。

这让 CNN 逐层形成从局部纹理到更复杂 visual features 的表示。

## 与 ACT 的连接

ACT 不直接把 raw camera pixels 送进 policy Transformer，而先用 ResNet（深层 CNN）提取 spatial features。Convolution 因此是 ACT vision pipeline 最底层的视觉运算之一。
