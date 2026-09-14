---
title: "Matrix"
kind: "canonical"
domain: "Mathematics / Linear Algebra"
parent: "Linear Algebra"
canonical: "/mathematics/linear-algebra/matrix/"
prerequisites:
  - "/mathematics/linear-algebra/vector/"
related:
  - "/mathematics/linear-algebra/linear-transformation/"
  - "/mathematics/linear-algebra/dot-product/"
---

# Matrix

Matrix 是按行和列排列的一组数：

\[
A=
\begin{bmatrix}
a_{11}&a_{12}&\cdots&a_{1n}\\
a_{21}&a_{22}&\cdots&a_{2n}\\
\vdots&\vdots&\ddots&\vdots\\
a_{m1}&a_{m2}&\cdots&a_{mn}
\end{bmatrix}
\in\mathbb R^{m\times n}.
\]

矩阵的两个最重要角色是：

1. 把很多 [Vector](/mathematics/linear-algebra/vector/) 组织在一起；
2. 表示一个 [Linear Transformation](/mathematics/linear-algebra/linear-transformation/) 在选定 basis 下的计算规则。

理解这两个角色，比把 matrix 当作“二维数组”更重要。

## Shape 决定了它能与谁运算

若

\[
A\in\mathbb R^{m\times n},
\qquad
x\in\mathbb R^n,
\]

那么

\[
Ax\in\mathbb R^m.
\]

矩阵接收一个 $n$ 维 vector，输出一个 $m$ 维 vector。

因此 shape 不只是编程中的报错信息，它直接表达了一个 mapping 的输入空间和输出空间。

## Matrix Multiplication

若

\[
A\in\mathbb R^{m\times n},
\qquad
B\in\mathbb R^{n\times p},
\]

则

\[
C=AB\in\mathbb R^{m\times p}.
\]

其元素：

\[
C_{ij}=
\sum_{k=1}^{n}A_{ik}B_{kj}.
\]

也就是说，$C_{ij}$ 是 $A$ 的第 $i$ 行与 $B$ 的第 $j$ 列的 dot product。

所以 matrix multiplication 不是“对应位置相乘”；它是在把两个线性变换复合，或者一次计算许多行列之间的线性组合。

## Matrix 与一组 vectors

若有 $N$ 个 $d$ 维 vectors，可以按行堆成：

\[
X\in\mathbb R^{N\times d}.
\]

如果右乘

\[
W\in\mathbb R^{d\times h},
\]

则：

\[
XW\in\mathbb R^{N\times h}.
\]

这里同一个 transformation $W$ 同时作用在 $N$ 个 vectors 上。

这解释了为什么神经网络里一个 [Linear Layer](/deep-learning/core/linear-layer/) 可以高效处理整个 batch 或 token sequence：本质上是把许多 vector transformation 合并成一次 matrix multiplication。

## Transpose

Transpose 把行列交换：

\[
A^\top_{ij}=A_{ji}.
\]

如果

\[
A\in\mathbb R^{m\times n},
\]

则

\[
A^\top\in\mathbb R^{n\times m}.
\]

在 [Dot Product](/mathematics/linear-algebra/dot-product/) 中，column vectors 常写成：

\[
x^\top y.
\]

在 attention 中，$QK^\top$ 也是通过 transpose 让所有 query 与 key 的 pairwise dot products 一次完成。

但这些只是 matrix 的应用；matrix 本身的意义不依赖 Transformer。

## Identity Matrix

Identity matrix：

\[
I=
\begin{bmatrix}
1&0&\cdots&0\\
0&1&\cdots&0\\
\vdots&&\ddots&\vdots\\
0&0&\cdots&1
\end{bmatrix}
\]

满足：

\[
Ix=x,
\qquad
AI=A,
\qquad
IA=A
\]

在维度匹配时成立。

它对应“什么都不改变”的 linear transformation，也是 residual connection 中 identity path 的线性代数原型。

## Matrix 不只是数据容器

在代码里，matrix 经常表现为二维 tensor。但数学上它更强的意义在于：

> **它可以表示坐标、批量数据，也可以表示空间之间的线性映射。**

后续的 Linear Layer、convolution 展开、attention、least squares 等大量计算，都建立在这一点上。
