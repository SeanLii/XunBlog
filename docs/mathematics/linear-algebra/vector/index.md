---
title: "Vector"
kind: "canonical"
domain: "Mathematics / Linear Algebra"
parent: "Linear Algebra"
canonical: "/mathematics/linear-algebra/vector/"
prerequisites: []
related:
  - "/mathematics/linear-algebra/vector-norm/"
  - "/mathematics/linear-algebra/dot-product/"
  - "/mathematics/linear-algebra/matrix/"
---

# Vector

Vector 是同时具有加法与 scalar multiplication 结构的数学对象。在线性代数中，向量属于某个 vector space；在选定 basis 后，有限维向量可以用一组 coordinates 表示。

最常见的实数向量写成：

\[
x=
\begin{bmatrix}
x_1\\
x_2\\
\vdots\\
x_n
\end{bmatrix}
\in\mathbb R^n.
\]

这里 $n$ 是 vector space 的 dimension，$x_i$ 是相对于当前 basis 的 coordinates。

## Vector Space Structure

一个 real vector space $V$ 允许两种基本运算：

- vector addition：$u+v$；
- scalar multiplication：$\alpha v$。

并满足加法交换律、结合律、零向量存在、加法逆元存在以及 scalar multiplication 的分配律等公理。

这些公理保证了 linear combination：

\[
\alpha_1v_1+\cdots+\alpha_kv_k
\]

仍然属于同一 vector space。

## Coordinates 与 Basis

Vector 本身与它的 coordinates 不是同一个概念。

给定 basis：

\[
B=(b_1,\ldots,b_n),
\]

任意 $v\in V$ 都可以唯一表示为：

\[
v=x_1b_1+\cdots+x_nb_n.
\]

coordinate vector 为：

\[
[v]_B=
\begin{bmatrix}
x_1\\
\vdots\\
x_n
\end{bmatrix}.
\]

更换 basis 后，同一个抽象 vector 的 coordinate values 会改变，但 vector 本身不变。

## Linear Combination 与 Span

给定 vectors $v_1,\ldots,v_k$，它们所有可能 linear combinations 构成：

\[
\operatorname{span}(v_1,\ldots,v_k)
=
\left\{
\sum_{i=1}^{k}\alpha_iv_i
:\alpha_i\in\mathbb R
\right\}.
\]

Span 描述这些 vectors 能生成的全部 directions / subspace。

若一组 vectors 中不存在非零 coefficients 使：

\[
\sum_i\alpha_iv_i=0,
\]

则它们 linearly independent。

## Length 与 Direction

在 $\mathbb R^n$ 中，最常用长度是 Euclidean norm：

\[
\|x\|_2
=
\sqrt{\sum_{i=1}^{n}x_i^2}.
\]

非零 vector 的 normalized direction 为：

\[
\hat x=\frac{x}{\|x\|_2}.
\]

一般 norm 的定义与性质见 [Vector Norm](/mathematics/linear-algebra/vector-norm/)。

## Dot Product 与 Geometry

Euclidean dot product 定义为：

\[
x^\top y
=
\sum_i x_i y_i.
\]

它与长度和夹角满足：

\[
x^\top y
=
\|x\|_2\|y\|_2\cos\theta.
\]

因此 dot product 同时编码 algebraic combination 与 geometric alignment。更完整内容见 [Dot Product](/mathematics/linear-algebra/dot-product/)。

## Vector 与 Matrix

Matrix-vector multiplication 定义从一个 coordinate vector 到另一个 coordinate vector 的映射：

\[
y=Ax.
\]

当 $A\in\mathbb R^{m\times n}$ 时：

\[
x\in\mathbb R^n,
\qquad
y\in\mathbb R^m.
\]

从 abstract linear algebra 的角度，matrix 是某个 [Linear Transformation](/mathematics/linear-algebra/linear-transformation/) 在选定 bases 下的 coordinate representation。

## Random Vector

若 vector 的各分量是 random variables，可写成 random vector：

\[
X=
\begin{bmatrix}
X_1\\
\vdots\\
X_n
\end{bmatrix}.
\]

它可以具有 mean vector：

\[
\mu=\mathbb E[X]
\]

和 covariance matrix：

\[
\Sigma=
\mathbb E[(X-\mu)(X-\mu)^\top].
\]

这把线性代数结构与 multivariate probability 连接起来。

## Vectors in Machine Learning

Machine learning 中大量对象都以 vector coordinates 表示，例如：

- feature vector；
- embedding；
- neural hidden state；
- gradient；
- model parameter block；
- robot joint state；
- action vector。

这些对象的语义不同，但都可以使用同一套线性代数运算。Vector 本身不属于某个特定模型，它是这些表示方式共享的基础数学结构。
