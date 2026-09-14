---
title: "Vector"
kind: "canonical"
domain: "Mathematics / Linear Algebra"
parent: "Linear Algebra"
canonical: "/mathematics/linear-algebra/vector/"
prerequisites:
  []
related:
  - "/mathematics/linear-algebra/dot-product/"
  - "/mathematics/linear-algebra/matrix/"
---

# Vector

向量（vector）是一组有顺序的数。它既可以表示几何空间中的方向与长度，也可以表示机器学习中的一组特征。两种解释使用的是同一个数学对象。

## 定义

一个 $d$ 维实向量写作

\[
\mathbf{x}=(x_1,x_2,\ldots,x_d)^\top\in\mathbb{R}^d.
\]

这里 $d$ 是维数，$x_i$ 是第 $i$ 个分量，$\mathbb{R}^d$ 表示由 $d$ 个实数组成的向量空间。上标 $\top$ 表示转置，因此这里把 $\mathbf{x}$ 写成列向量。

在 AI 中，一个 token 的 hidden state、机器人 14 个关节的位置、VAE 的 latent $z$，都可以由向量表示。向量本身并不规定每个分量的物理含义；含义来自建模方式。

## 基本运算

同维向量可以逐分量相加：

\[
\mathbf{x}+\mathbf{y}=(x_1+y_1,\ldots,x_d+y_d)^\top.
\]

标量 $c$ 与向量相乘时，每个分量都乘以 $c$：

\[
c\mathbf{x}=(cx_1,\ldots,cx_d)^\top.
\]

向量的二范数为

\[
\lVert\mathbf{x}\rVert_2=\sqrt{\sum_{i=1}^{d}x_i^2}.
\]

它在几何上对应向量长度，在模型中也常用来衡量向量的大小。

## 向量与表示

神经网络经常把离散对象或连续状态转换成固定维数的向量。此时“一个概念被表示成向量”不意味着某一个分量天然对应某种人类可解释属性。模型学习的是整个向量空间中的结构。

后续的 [Dot Product](/mathematics/linear-algebra/dot-product/) 会利用两个向量的分量共同计算相似方向上的重合程度；Transformer 的 attention score 正是建立在这个运算之上。
