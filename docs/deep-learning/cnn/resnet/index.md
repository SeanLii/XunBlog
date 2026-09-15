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

ResNet（Residual Network）是 He 等人提出的 deep convolutional network family。它通过 **residual learning** 将一个 block 的目标 mapping 写成

\[
H(x)=x+F(x),
\]

使 identity mapping 成为 architecture 中的显式 shortcut path，并让很深的 convolutional networks 更容易优化。

## Degradation Problem

在 plain deep networks 中，增加层数并不总会降低 training error。ResNet 论文观察到：当网络变得更深时，training error 可能反而升高，即使新增层理论上可以表示 identity mapping。

这类 **degradation problem** 与单纯 overfitting 不同，因为更深网络在训练集上本身就优化得更差。

ResNet 的设计目标因此不是只增加 model capacity，而是改变深层 network 的 parameterization，使接近 identity 的 mapping 更容易表达和优化。

## Residual Learning

若目标 mapping 为

\[
H(x),
\]

ResNet 让 learned branch 学习 residual

\[
F(x)=H(x)-x.
\]

于是

\[
H(x)=x+F(x).
\]

当目标 mapping 接近 identity 时，只需学习

\[
F(x)\approx0.
\]

这与要求一组 nonlinear layers 直接逼近 identity mapping 的 optimization geometry 不同。

## Basic Block

ResNet-18 / ResNet-34 使用 Basic Block。典型结构为

```text
x ──────────────────────────┐
│                           │
├→ 3×3 Conv → Norm → ReLU   │
│              ↓            │
│          3×3 Conv → Norm  │
│                           │
└──────────── add ───────────┘
              ↓
             ReLU
```

简化表示为

\[
y=\operatorname{ReLU}(x+F(x)).
\]

原论文使用 Batch Normalization；不同后续实现可能改变 normalization 与 activation placement。

## Bottleneck Block

ResNet-50 / 101 / 152 使用 Bottleneck Block：

```text
1×1 Conv
   ↓
3×3 Conv
   ↓
1×1 Conv
```

第一层 1×1 convolution 调整 channel dimension，中间 3×3 convolution 处理 spatial interaction，最后 1×1 convolution 恢复 / 扩展 output channels。

这种设计在增加 depth 的同时控制 3×3 convolution 的计算成本。

## Identity and Projection Shortcuts

当

\[
\operatorname{shape}(x)=\operatorname{shape}(F(x)),
\]

可以直接使用 identity shortcut：

\[
y=x+F(x).
\]

当 spatial resolution 或 channel dimension 改变时，需要 projection：

\[
y=W_sx+F(x),
\]

常由带 stride 的 1×1 convolution 实现。

更详细的数学结构见 [Residual Connection](/deep-learning/cnn/resnet/residual-connection/)。

## Stage Structure

ResNet 通常由 stem 与多个 residual stages 组成。随着 stage 向后：

- spatial resolution 降低；
- channel dimension 增加；
- receptive field 增大；
- representation 逐渐从局部视觉 features 转向更高层 task-relevant features。

典型 ResNet-50 主干可概括为

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

stage transition 通常通过 stride 与 projection shortcut 同时完成 spatial downsampling 和 channel change。

## ResNet Depth Naming

ResNet-18、34、50、101、152 的数字遵循原论文对有 learnable weights layers 的计数 convention。

ResNet-18 / 34 主要使用 Basic Blocks；ResNet-50 及更深版本主要使用 Bottleneck Blocks。因此不同深度并不只是简单复制同一种 block 更多次，block type 与 stage configuration 也会改变。

## Optimization Effect

对 residual mapping

\[
y=x+F(x),
\]

Jacobian 为

\[
\frac{\partial y}{\partial x}
=I+\frac{\partial F}{\partial x}.
\]

identity term 为 forward information 与 backward gradient 提供直接路径。它改变了深层网络的 optimization landscape，但不应简化为“完全解决 vanishing gradient”。Initialization、normalization、activation、optimizer 与整体 architecture 仍然共同决定训练稳定性。

## ResNet as a Backbone

去掉 classification head 后，ResNet 的 intermediate feature maps 可以作为通用视觉 backbone，服务于：

- object detection；
- semantic / instance segmentation；
- multimodal models；
- robot perception；
- metric learning。

ACT 使用 ResNet 提取视觉 features 是其中一个 downstream application，而不是 ResNet 的定义范围。

## Sources

- He et al. *Deep Residual Learning for Image Recognition*. 2015/2016.
- He et al. *Identity Mappings in Deep Residual Networks*. 2016.
