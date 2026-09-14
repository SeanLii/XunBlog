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

ResNet（Residual Network）是一类用 residual blocks 构建的卷积神经网络。它的核心不是单纯“把网络加深”，而是把每个 block 要学习的映射改写为 residual form。

## Residual Block

设理想映射为 $H(x)$。ResNet 让一组 layers 学习

\[
F(x)=H(x)-x,
\]

并输出

\[
y=F(x)+x.
\]

这条 identity shortcut 为输入提供直接路径。ResNet 论文的主要动机来自 degradation problem：普通深层网络增加层数后，训练误差反而可能变差；residual formulation 让更深网络更容易优化。

## Basic Block

ResNet18 使用 basic blocks。典型 basic block 包含两层 $3\times3$ convolution，并把 block 输入通过 shortcut 加到输出上。

如果空间尺寸或 channel dimension 改变，shortcut 不能直接使用 identity，需要 projection 使 shape 匹配。

## ResNet18

“18”指论文定义下计入的带权重层数配置。它由初始 convolution、四个 residual stages 和最终分类头构成。作为视觉 backbone 使用时，通常移除或绕过最终分类用途，而取中间或最后卷积 feature map。

## ACT 中的角色

ACT 论文使用 ResNet18 对每个 $480\times640$ RGB image 提取 $15\times20\times512$ feature map。空间网格 flatten 后变为 300 个 512 维视觉 features；再加二维 sinusoidal position encoding。

论文描述四个相机各自产生这样的 features，合计 1200 个 visual features。Released implementation 在相机循环里复用同一个 `backbones[0]`，把各相机 feature maps 沿 width dimension 拼接后送给 Transformer。这个实现细节属于 [Paper and Released Implementation](/robot-learning/act/paper-and-released-implementation/)，不改变 ResNet 本身的定义。

## Sources

- [Deep Residual Learning for Image Recognition — He et al., 2015](https://arxiv.org/abs/1512.03385)
- [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware — Zhao et al., 2023](https://arxiv.org/abs/2304.13705)
- [ACT official implementation](https://github.com/tonyzhaozh/act)
