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

Convolutional Neural Network（CNN）是一类特别适合处理网格数据的神经网络。对图像来说，最重要的特点是：**同一个小型 filter 会在整张图像上重复使用，从局部区域提取模式。**

先看一张图像：

\[
X\in\mathbb R^{H\times W\times C}.
\]

普通 fully-connected layer 如果直接连接所有像素，会忽略“附近像素之间有强局部关系”这一结构。CNN 使用 [Convolution](/deep-learning/cnn/convolution/) 只观察局部窗口，并共享参数。

## 从像素到 feature map

一个卷积层可以先想成很多个可学习 detector：

```text
image
  │
  ├─ filter 1 → feature map 1
  ├─ filter 2 → feature map 2
  ├─ filter 3 → feature map 3
  └─ ...
```

某个 filter 可以逐渐学会响应边缘、纹理或其他局部模式。多层 CNN 继续在前一层 feature maps 上做 convolution，于是 receptive field 逐渐变大，表示也逐渐从像素级模式变成更高层结构。

## 参数共享

如果“检测竖直边缘”在图像左上角有意义，那么在右下角通常也有意义。CNN 不需要为每个位置单独学习一套完全不同的权重，而是把同一 kernel 滑过不同位置。

这带来两个结果：

- 参数量显著小于把所有像素完全连接；
- 学到的局部 pattern 可以在不同位置复用。

## 多层 CNN 的表征层级

一层 convolution 的感受野很小。多层叠加后，后面的一个 feature 会间接依赖更大的图像区域：

```text
pixels
  ↓
local edges / textures
  ↓
larger motifs
  ↓
object parts / semantic features
```

这不是每层都有人为规定“这一层必须检测什么”，而是训练目标通过 gradient 让网络自动形成有用的层级 features。

## CNN 与 ResNet

当 CNN 变得非常深时，优化会变困难。[ResNet](/deep-learning/cnn/resnet/) 在 CNN blocks 之间加入 residual connections：

\[
y=x+F(x),
\]

使深层 convolutional network 更容易训练。

## 在 ACT 中的位置

ACT 的相机图像先经过 ResNet backbone 得到 spatial features，然后这些 features 才进入 Transformer。也就是说：

```text
image → CNN/ResNet → visual tokens → Transformer → action chunk
```

CNN 负责把原始像素转成更适合后续推理的视觉表示；Transformer 再负责跨位置、跨相机、与机器人状态之间的信息融合。

## Sources

- LeCun et al., **Gradient-Based Learning Applied to Document Recognition**, 1998.
- He et al., **Deep Residual Learning for Image Recognition**, 2015/2016. https://arxiv.org/abs/1512.03385
