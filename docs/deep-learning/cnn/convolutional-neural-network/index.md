---
title: "Convolutional Neural Network"
kind: "canonical"
domain: "Deep Learning / Convolutional Neural Networks"
parent: "Convolutional Neural Networks"
canonical: "/deep-learning/cnn/convolutional-neural-network/"
prerequisites:
  - "/mathematics/analysis/convolution/"
  - "/deep-learning/core/activation-function/"
related:
  - "/deep-learning/cnn/resnet/"
---

# Convolutional Neural Network

Convolutional Neural Network（CNN / ConvNet）是一类围绕 [Convolution](/mathematics/analysis/convolution/) 构建的 neural network architecture，尤其适合处理具有 spatial structure 的数据，例如 images。

它的核心不是“用了一个 convolution layer”，而是利用：

- local connectivity；
- weight sharing；
- hierarchical feature extraction；
- spatial resolution 与 channel dimension 的逐层变化。

把一张图像送进 CNN，可以先得到 local feature maps，再逐层组合成更抽象的 representation。

## 从 Image 到 Feature Map

RGB image：

\[
X\in\mathbb R^{3\times H\times W}.
\]

第一层 convolution 可能输出：

\[
F_1\in\mathbb R^{C_1\times H_1\times W_1}.
\]

这里的 channel 不再是 RGB，而是 learned feature channels。

每个 channel 对某种局部 pattern 产生响应，但不要过度把每个 channel 固定解释成“边缘检测器”或某个可命名语义；feature meaning 是训练结果，并且会随网络层级与任务变化。

## 一个典型 CNN Block

常见结构：

```text
Convolution
    ↓
Normalization (optional)
    ↓
Activation
    ↓
Downsampling (optional)
```

多个 blocks 堆起来：

```text
image
  ↓
low-level feature maps
  ↓
mid-level feature maps
  ↓
high-level feature maps
  ↓
head / decoder / downstream module
```

## Channel Growth 与 Spatial Downsampling

很多 CNN 随深度增加会：

- 降低 $H,W$；
- 增加 channels $C$。

低层保留细空间位置；高层每个 feature 对应更大 receptive field，并用更多 channels 表示不同 learned patterns。

这种设计在计算成本与 representation capacity 之间折中。

## Pooling 与 Strided Convolution

早期 CNN 常用 max pooling：

\[
2\times2\text{ window}
\rightarrow
\max\text{ value}
\]

降低 spatial resolution。

现代 CNN 也常使用 strided convolution 完成 downsampling，同时让 downsampling weights 可学习。

两者都不是 CNN 定义本身的必要条件。

## Hierarchical Representation

CNN 的一个关键特点是 receptive field 随层数增大。

低层 unit 只直接读取小 patch；更高层 unit 通过前层 features 间接读取更大区域。

因此可以形成：

```text
small local pattern
      ↓
combination of local patterns
      ↓
larger spatial structure
      ↓
task-relevant representation
```

这是一种 hierarchy，而不是保证网络一定按照“边缘 → 纹理 → 物体”这种固定人工语义顺序学习。

## Classification CNN

经典 image classifier 可能是：

```text
image
 ↓
CNN backbone
 ↓
feature tensor
 ↓
global pooling / flatten
 ↓
Linear Layer
 ↓
class logits
```

但 CNN 并不只用于 classification。

Detection、segmentation、robot perception、audio spectrogram processing 等也可以把 CNN 当 feature extractor 或 backbone。

## 从 Plain CNN 到 ResNet

当网络越来越深，plain CNN 会遇到 optimization difficulties。2015 年的 [ResNet](/deep-learning/cnn/resnet/) 通过 residual connections 让 deep networks 更容易训练。

所以 ResNet 是 CNN architecture family 中的重要发展，而不是 CNN 的定义。

CNN 经历了从早期 ConvNet 到大规模深度视觉 backbone 的发展，但本文重点放在 convolutional architecture 本身的计算结构，而不是模型历史。

## Sources

- LeCun et al. *Gradient-Based Learning Applied to Document Recognition*. 1998.
- Krizhevsky, Sutskever, Hinton. *ImageNet Classification with Deep Convolutional Neural Networks*. 2012.
