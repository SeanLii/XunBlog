---
title: "Convolutional Neural Network"
kind: "canonical"
domain: "Deep Learning / Convolutional Neural Networks"
parent: "Convolutional Neural Networks"
canonical: "/deep-learning/cnn/convolutional-neural-network/"
prerequisites:
  - "/deep-learning/cnn/convolution/"
related:
  - "/deep-learning/cnn/resnet/"
  - "/robot-learning/act/vision-pipeline/"
---

# Convolutional Neural Network

Convolutional Neural Network（CNN）是以 convolution 为核心处理网格数据的神经网络。图像是最典型的二维网格，因此 CNN 长期作为视觉特征提取器使用。

## 特征层级

一层 convolution 只直接读取局部邻域。随着多层堆叠，后层单元对应到原图上的 receptive field 逐渐扩大，因此网络可以从局部边缘、纹理逐步组合出更大范围的视觉结构。

这不是说每一层都必然对应固定的人类语义。层级结构来自局部计算与深度堆叠，具体特征由训练目标决定。

## Feature Map

对输入图像

\[
X\in\mathbb R^{C\times H\times W},
\]

CNN 某一层输出可以写成

\[
F\in\mathbb R^{C'\times H'\times W'}.
\]

$C'$ 是 feature channels；$H',W'$ 是新的空间尺寸。每个空间位置不再直接存 RGB，而是存模型学习到的 feature vector。

## 下采样

CNN 常通过 stride、pooling 等方式降低 $H,W$，同时增加 channel dimension。这减少后续计算，也扩大高层特征对应的感受范围。

## 从 Feature Map 到 Transformer Tokens

如果一个 feature map 为

\[
F\in\mathbb R^{C\times H\times W},
\]

可以把空间维 flatten 成

\[
HW\times C
\]

个 token-like features，再加入二维位置编码交给 Transformer。ACT 的 vision pipeline 正是这种 CNN-to-Transformer 接口。

## ResNet

[ResNet](/deep-learning/cnn/resnet/) 不是“另一种非 CNN 模型”，而是一类加入 residual connections 的深层 CNN。ACT 论文使用 ResNet18 将每个相机图像压缩成较低分辨率的 feature map，再让 Transformer 处理多视角信息。
