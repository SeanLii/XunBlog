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

Vector 是线性代数最基本的对象之一。它不是“AI 里的特征数组”，也不等同于 Python list。更准确地说，vector 是一个同时支持**相加**和**数乘**的数学对象；在有限维实向量空间里，我们通常把它写成一列数字：

\[
x=
\begin{bmatrix}
x_1\\
x_2\\
\vdots\\
x_d
\end{bmatrix}
\in \mathbb{R}^d.
\]

这里的 $d$ 是 vector 的维度。每一个分量 $x_i$ 都是在某一组坐标基底下的坐标值。

一个二维 vector 可以画成平面上的箭头；三维 vector 可以画成空间中的箭头。高维以后我们无法直接画出来，但运算规则没有改变。

## Vector space 中的两个基本操作

对于同一空间中的两个 vectors $x,y\in\mathbb R^d$，可以逐分量相加：

\[
x+y=
\begin{bmatrix}
x_1+y_1\\
\vdots\\
x_d+y_d
\end{bmatrix}.
\]

也可以乘一个 scalar $c$：

\[
cx=
\begin{bmatrix}
cx_1\\
\vdots\\
cx_d
\end{bmatrix}.
\]

几何上，向量相加可以理解为位移的合成；数乘会改变长度，并在 $c<0$ 时翻转方向。

这两个操作之所以重要，是因为后面的 [Linear Transformation](/mathematics/linear-algebra/linear-transformation/) 必须保持它们：

\[
T(x+y)=T(x)+T(y),
\qquad
T(cx)=cT(x).
\]

## 坐标不是 vector 本身

写成

\[
[2,1]
\]

时，我们看到的是 vector 在某组 basis 下的坐标，而不是 vector 的全部数学身份。

同一个几何 vector 换一组 basis 后，坐标可以改变；vector 本身没有改变。这一点在以后理解 basis change、eigenvector 或不同 feature space 时很重要。

在很多机器学习问题中，我们直接固定标准 basis，因此常把“vector”和“坐标数组”放在一起说。工程上这样通常没有问题，但概念上要知道两者不是完全同一件事。

## 方向与大小是两类不同信息

一个 vector 可以同时携带：

- **方向**；
- **大小**。

大小通常用 [Vector Norm](/mathematics/linear-algebra/vector-norm/) 描述，例如 Euclidean norm：

\[
\|x\|_2=\sqrt{x_1^2+\cdots+x_d^2}.
\]

把非零 vector 除以自己的长度：

\[
\hat x=\frac{x}{\|x\|_2},
\]

得到 unit vector，长度为 1，只保留方向信息。

这个区分会直接影响 [Dot Product](/mathematics/linear-algebra/dot-product/) 与 cosine similarity：dot product 同时受到长度和方向影响，而归一化后的 cosine 主要比较方向。

## 一组 vectors 可以组成更高层结构

多个同维 vectors 可以按行或按列堆成 [Matrix](/mathematics/linear-algebra/matrix/)。

例如三个二维 vectors：

\[
x_1=[1,2],\quad x_2=[3,4],\quad x_3=[5,6]
\]

按行组成：

\[
X=
\begin{bmatrix}
1&2\\
3&4\\
5&6
\end{bmatrix}.
\]

这一步在机器学习里极其常见：一个 token、一个样本、一个关节状态都可能用 vector 表示，而一批这样的 vectors 则自然形成 matrix 或更高阶 tensor。

## Vector 在机器学习中的角色

Vector 本身不属于 AI，但 AI 大量使用 vector 作为统一表示形式。

例如：

- 一个样本的 feature vector；
- 一个 token 的 embedding；
- Transformer 的 hidden state；
- 机器人某时刻的 joint state；
- VAE 的 latent code。

这些对象语义完全不同，但都可以落到同一个数学空间 $\mathbb R^d$ 中，因此可以使用同一套线性代数运算。

这也是 vector 在现代机器学习里如此核心的原因：它提供了一种统一的、可计算的表示空间，而不是因为“神经网络天生只认识向量”。
