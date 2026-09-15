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
standard: "XunBlog Content & Knowledge Architecture v1.0"
rebuilt: "2026-09-15"
---
# Convolutional Neural Network

> **知识边界**：本文的 canonical 对象是 **Convolutional Neural Network**。依赖机制由 [Convolution](/mathematics/analysis/convolution/)、[Activation Function](/deep-learning/core/activation-function/) 的 canonical page 定义；本文只在当前语境中调用其接口。


Convolutional Neural Network（CNN / ConvNet）是一类利用局部连接与参数共享处理具有空间或网格结构数据的 neural network architecture。图像是最典型的输入：

\[
X\in\mathbb R^{C_{in}\times H\times W}.
\]

CNN 通过多层 convolution、nonlinearity 与 spatial-resolution transformation，把原始像素逐步变成适合分类、检测、分割或其他下游任务的 feature representations。

CNN 的主要 inductive biases 包括：

- **locality**：单个 kernel 首先读取局部 neighborhood；
- **weight sharing**：同一组 kernel parameters 在不同空间位置重复使用；
- **translation equivariance**：在边界与 sampling 等条件允许时，输入平移会对应地产生 feature-map 平移；
- **hierarchical composition**：多层局部计算可以形成越来越大的 effective receptive field。

完整的 convolution 数学定义见 [Convolution](/mathematics/analysis/convolution/)。

## Convolutional Layer

二维 convolutional layer 接收

\[
X\in\mathbb R^{C_{in}\times H\times W}
\]

并使用一组 learnable kernels：

\[
W\in\mathbb R^{C_{out}\times C_{in}\times K_h\times K_w}.
\]

第 $c_{out}$ 个 output channel 由所有 input channels 的局部加权组合得到：

\[
Y_{c_{out},i,j}
=
\sum_{c=1}^{C_{in}}
\sum_{u,v}
W_{c_{out},c,u,v}
X_{c,i+u,j+v}
+b_{c_{out}}.
\]

实际 deep-learning libraries 通常实现 cross-correlation 形式而不显式翻转 kernel，但训练中的 learnable-kernel interpretation 不受这一 convention 影响。

输出为

\[
Y\in\mathbb R^{C_{out}\times H_{out}\times W_{out}}.
\]

因此 convolutional layer 同时改变两类结构：

- spatial dimensions $H,W$；
- feature-channel dimension $C$。

## Spatial Output Size

以单个 spatial dimension 为例，给定 input size $H$、kernel size $K$、padding $P$、stride $S$ 与 dilation $D$，常见 output-size 公式为

\[
H_{out}
=
\left\lfloor
\frac{H+2P-D(K-1)-1}{S}+1
\right\rfloor.
\]

这几个参数控制不同结构：

- **padding** 决定边界如何处理，并可帮助保持 spatial size；
- **stride** 决定 kernel 的采样步长，也可实现 downsampling；
- **dilation** 增大 kernel samples 之间的间隔，在不同比例地增加参数数量的情况下扩大 receptive field。

## Local Connectivity and Weight Sharing

Fully connected layer 对不同 input coordinates 通常拥有独立 weights。Convolution 则让同一个 kernel 在所有空间位置滑动。

若 kernel size 为 $K_h\times K_w$，一个 output unit 只直接读取该局部区域，但 kernel parameters 在不同 $(i,j)$ 位置共享。

这种参数共享使 CNN 不需要为图像每个绝对位置单独学习一套局部 detector，同时显著降低参数数量。

例如一个普通 convolutional layer 的参数量约为

\[
C_{out}C_{in}K_hK_w+C_{out},
\]

与 image spatial size $H\times W$ 无直接乘法关系。

## Translation Equivariance

理想离散卷积满足 translation equivariance：若 $T_\Delta$ 表示空间平移，则

\[
\operatorname{Conv}(T_\Delta X)
=
T_\Delta\operatorname{Conv}(X).
\]

这表示某个局部 pattern 从图像左侧移动到右侧时，同一个 kernel 仍可在新位置产生相应 response。

需要区分：

- equivariance 表示“输入变换后，输出以对应方式变换”；
- invariance 表示“输入变换后，最终输出保持不变”。

Pooling、global aggregation、data augmentation 等机制可以进一步增强某些任务所需的 approximate invariance，但 convolution 本身首先提供的是 equivariance。

实际网络中的 padding、stride、finite boundaries 与 nonlinear processing 会使这一性质不再严格成立于所有情况。

## Feature Channels

RGB image 的三个 input channels 有明确物理含义，但深层 CNN 中的 channels 是 learned feature dimensions。

第一层可能输出

\[
F_1\in\mathbb R^{C_1\times H_1\times W_1},
\]

后续层继续形成

\[
F_2,F_3,\ldots,F_L.
\]

单个 channel 不保证对应可被稳定命名的人类语义。网络学习的是一组对训练 objective 有用的 distributed features。

## Nonlinearity and Normalization

如果只连续堆叠线性 convolution operations，而不加入 nonlinear activation，整个 stack 仍可合并为一个线性 transformation。

因此 CNN block 通常包含：

```text
Convolution
    ↓
Normalization (optional)
    ↓
Activation
    ↓
Downsampling (optional)
```

常用 activation 包括 ReLU、GELU、SiLU 等，详见 [Activation Function](/deep-learning/core/activation-function/)。

Normalization 并不是 CNN 定义的一部分，但 Batch Normalization、Group Normalization 等经常用于改善 optimization 或适应不同 batch regimes。

## Spatial Downsampling

随着网络加深，很多 CNN architecture 会逐渐：

- 减小 $H,W$；
- 增大 channel 数 $C$。

Downsampling 可以由 max pooling、average pooling 或 strided convolution 完成。

例如 stride-2 convolution 可以近似把 spatial resolution 减半：

\[
H\times W
\rightarrow
\frac H2\times\frac W2.
\]

较低 spatial resolution 减少后续计算量，同时允许高层 representation 使用更多 channels。

Downsampling 也会丢失空间细节，因此 dense prediction 任务通常需要 multi-scale features、skip connections 或 decoder / upsampling structures 恢复高分辨率信息。

## Receptive Field

一个 unit 的 **receptive field** 是能够影响它的 input region。

单个 $3\times3$ convolution 只直接读取局部 neighborhood，但多层堆叠后 receptive field 会逐步扩大。

例如 stride 1、无 dilation 的连续 $3\times3$ convolutions：

```text
1 layer  → 3×3 receptive field
2 layers → 5×5 effective receptive field
3 layers → 7×7 effective receptive field
```

因此 CNN 可以通过局部 operations 逐层整合更大范围的 spatial context。

实际 receptive-field size 还受到 stride、dilation、pooling 与 network topology 影响。

## Hierarchical Representation

多层 CNN 通常形成空间尺度逐渐扩大的 representation hierarchy：

```text
image
  ↓
local features
  ↓
mid-level spatial patterns
  ↓
high-level task-relevant features
```

这种 hierarchy 来自 receptive-field growth 与 repeated nonlinear transformations。

它不意味着每一层都必然对应“边缘 → 纹理 → 部件 → 物体”这种固定人工语义顺序；具体 features 由 architecture、data 和 training objective 共同决定。

## Output Heads

CNN backbone 的输出可以按任务连接不同 heads。

### Classification

典型结构为：

```text
image
 ↓
CNN backbone
 ↓
feature tensor
 ↓
global pooling
 ↓
Linear Layer
 ↓
class logits
```

### Dense Prediction

Detection、segmentation、depth estimation 等任务需要保留空间信息，因此通常使用 spatial feature maps、多尺度 representations 或 decoder structures，而不是只保留一个 global vector。

CNN 也可作为其他 architecture 的视觉 backbone。例如 DETR 使用 CNN 提取 spatial image features，再交给 Transformer；ACT 使用 ResNet backbone 获取多相机 visual features。

## Architectural Extensions

CNN family 包含许多对基本 convolutional architecture 的扩展，例如：

- residual connections；
- grouped convolution；
- depthwise-separable convolution；
- dilated convolution；
- feature pyramids；
- encoder–decoder CNNs。

这些结构并不是 CNN 定义所必需，但分别改变 optimization、parameter efficiency 或 multi-scale processing。

[ResNet](/deep-learning/cnn/resnet/) 是其中影响最大的 architecture families 之一，它通过 residual learning 改善 very-deep CNN 的 optimization。

## 由机制产生的边界

CNN 的局部性和参数共享带来有效 inductive bias，但也形成相应限制：

- long-range interaction 需要多层传播或大 receptive field；
- aggressive downsampling 会损失 spatial precision；
- translation-related inductive bias 并不适合所有数据结构；
- fixed grid operations 对几何变化、视角变化和非欧式结构没有天然完全不变性。

这些限制促使现代视觉系统结合 attention、multi-scale architectures、geometric priors 与更大规模 pretraining。

## Sources

- LeCun et al. *Gradient-Based Learning Applied to Document Recognition*. 1998.
- Krizhevsky, Sutskever, Hinton. *ImageNet Classification with Deep Convolutional Neural Networks*. 2012.
- Dumoulin, Visin. *A Guide to Convolution Arithmetic for Deep Learning*. 2016.
