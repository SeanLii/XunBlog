---
title: "Linear Transformation"
kind: "canonical"
domain: "Mathematics / Linear Algebra"
parent: "Linear Algebra"
canonical: "/mathematics/linear-algebra/linear-transformation/"
prerequisites:
  - "/mathematics/linear-algebra/vector/"
  - "/mathematics/linear-algebra/matrix/"
related:
  - "/deep-learning/core/linear-layer/"
---

# Linear Transformation

Linear Transformation 是保持 vector addition 与 scalar multiplication 的映射。

设

\[
T:\mathbb R^n\rightarrow\mathbb R^m.
\]

如果对任意 vectors $x,y$ 和 scalars $a,b$ 都有：

\[
T(ax+by)=aT(x)+bT(y),
\]

那么 $T$ 是 linear transformation。

这一定义意味着：linear transformation 不会破坏空间中的线性组合关系。

## Matrix 是 Linear Transformation 的坐标表示

在有限维实向量空间并选定 basis 后，每个 linear transformation 都可以写成：

\[
T(x)=Ax,
\]

其中

\[
A\in\mathbb R^{m\times n}.
\]

因此 matrix 与 linear transformation 紧密相连，但两个概念并不完全相同：

- transformation 是映射本身；
- matrix 是这个映射在某组 basis 下的表示。

换 basis 后 matrix 可以改变，而底层 transformation 可以保持不变。

## 一列一列理解 Matrix

若

\[
A=[a_1\ a_2\ \cdots\ a_n],
\]

其中 $a_i$ 是第 $i$ 列，那么对

\[
x=[x_1,\ldots,x_n]^\top
\]

有：

\[
Ax=x_1a_1+x_2a_2+\cdots+x_na_n.
\]

所以 matrix-vector multiplication 可以理解为：用输入坐标 $x_i$ 对 matrix 的 columns 做线性组合。

这比“按照公式做乘法”更直接地说明了 transformation 在做什么。

## 几何上可以做什么

二维或三维 linear transformation 可以实现：

- rotation；
- scaling；
- reflection；
- shear；
- projection（某些 projection）；
- 多种操作的组合。

但纯 linear transformation 必须把原点映射到原点：

\[
T(0)=0.
\]

如果加入一个 bias：

\[
f(x)=Ax+b,
\]

得到的是 affine transformation，而严格来说不再是 linear transformation。

这也是为什么神经网络里的 `Linear` layer 通常从数学上更准确地说是 affine layer：它通常包含 bias。

## Composition

如果先做：

\[
y=Bx,
\]

再做：

\[
z=Ay,
\]

则：

\[
z=A(Bx)=(AB)x.
\]

Matrix multiplication 因此对应 transformation composition。

这解释了为什么多个**没有非线性激活**的 linear layers 仍然可以合并成一个线性/仿射 mapping。神经网络需要 [Activation Function](/deep-learning/core/activation-function/) 打破这种可合并性，才能表示更复杂的 nonlinear functions。

## 与神经网络的关系

[Linear Layer](/deep-learning/core/linear-layer/) 直接建立在这个概念上。

神经网络里常见：

\[
y=xW+b.
\]

它把表示从一个 feature space 映射到另一个 feature space。Q/K/V projections、MLP layers、classification heads 都会使用这种结构。

这些是 linear transformation 的应用；linear transformation 本身是独立的线性代数对象。
