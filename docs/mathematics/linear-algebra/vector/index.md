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

Vector 是一组有顺序的数。在线性代数中常写成

\[
x=
\begin{bmatrix}
x_1\\x_2\\\vdots\\x_d
\end{bmatrix}
\in\mathbb R^d.
\]

它可以表示空间中的方向和长度，也可以更一般地表示一组 features。

## Vector 在 AI 中的表示角色

一个向量的每个维度可以存一个数值特征。例如：

```text
机器人关节状态：
q = [q1, q2, ..., q14]

一个 token hidden state：
h ∈ R^512

VAE latent：
z ∈ R^32
```

它们都叫 vector，但每一维的语义取决于具体模型。

## 向量加法

两个同维 vectors：

\[
x,y\in\mathbb R^d
\]

可以逐维相加：

\[
x+y=
[x_1+y_1,\ldots,x_d+y_d].
\]

Transformer 的 residual connection

\[
y=x+F(x)
\]

就是对相同 hidden dimension 的 vectors / tensors 做这种逐元素加法。

## 标量乘法

一个 scalar $c$ 乘 vector：

\[
cx=[cx_1,\ldots,cx_d].
\]

这会统一缩放整个向量。

Attention 中的 weighted sum

\[
y=\sum_j\alpha_jv_j
\]

就是把多个 value vectors 分别乘 scalar weights，再相加。

## 长度

Euclidean norm：

\[
\|x\|_2
=\sqrt{\sum_i x_i^2}.
\]

它给出向量在欧氏空间中的长度。

## Vector 与 Matrix

Vector 是一维有序数列；[Matrix](/mathematics/linear-algebra/matrix/) 可以看成很多 vectors 按行或列排在一起。

例如 Transformer 一组 $n$ 个 token vectors：

\[
X\in\mathbb R^{n\times d}.
\]

每一行就是一个 $d$ 维 vector。

## 与 Dot Product 的连接

两个同维 vectors 可以通过 [Dot Product](/mathematics/linear-algebra/dot-product/) 变成一个 scalar：

\[
x^\top y=\sum_i x_iy_i.
\]

Transformer attention 使用的 query-key score 正是这种运算。
