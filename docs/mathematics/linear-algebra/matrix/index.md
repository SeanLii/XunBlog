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
standard: "XunBlog Content & Knowledge Architecture v1.0"
rebuilt: "2026-09-15"
---
# Matrix

Matrix 是按 rows 与 columns 排列的二维标量数组：

\[
A=
\begin{bmatrix}
a_{11}&\cdots&a_{1n}\\
\vdots&\ddots&\vdots\\
a_{m1}&\cdots&a_{mn}
\end{bmatrix}
\in\mathbb R^{m\times n}.
\]

Matrix 可以作为数据表、线性方程系数、双线性形式或 linear transformation 的 coordinate representation。在线性代数中最核心的角色是表示线性映射。

## Shape

\[
A\in\mathbb R^{m\times n}
\]

表示 $A$ 有 $m$ 行、$n$ 列。

若：

\[
x\in\mathbb R^n,
\]

则 matrix-vector product：

\[
y=Ax
\]

得到：

\[
y\in\mathbb R^m.
\]

因此 matrix shape 直接编码 input dimension 与 output dimension。

## Matrix-Vector Multiplication

第 $i$ 个 output coordinate 为：

\[
y_i
=
\sum_{j=1}^{n}A_{ij}x_j.
\]

也就是说，$A$ 的第 $i$ 行与 $x$ 做 dot product。

另一种等价视角是按 columns 展开：

\[
Ax
=
x_1a_1+x_2a_2+\cdots+x_na_n,
\]

其中 $a_j$ 是 $A$ 的第 $j$ 列。

这说明 matrix-vector multiplication 也是 matrix columns 的 linear combination。

## Matrix Multiplication

若：

\[
A\in\mathbb R^{m\times n},
\qquad
B\in\mathbb R^{n\times p},
\]

则：

\[
C=AB\in\mathbb R^{m\times p},
\]

且：

\[
C_{ij}
=
\sum_{k=1}^{n}A_{ik}B_{kj}.
\]

Matrix multiplication 对应 linear transformations 的 composition。

一般情况下：

\[
AB\neq BA.
\]

因此 matrix multiplication 不满足交换律。

## Transpose

Transpose 把 rows 与 columns 交换：

\[
(A^\top)_{ij}=A_{ji}.
\]

若：

\[
A\in\mathbb R^{m\times n},
\]

则：

\[
A^\top\in\mathbb R^{n\times m}.
\]

并满足：

\[
(AB)^\top=B^\top A^\top.
\]

## Identity Matrix

Identity matrix：

\[
I_n=
\begin{bmatrix}
1&&0\\
&\ddots&\\
0&&1
\end{bmatrix}
\]

满足：

\[
I_nx=x,
\qquad
AI=A,
\qquad
IA=A
\]

（在 shapes 合法时）。

## Inverse

对 square matrix $A$，若存在 $A^{-1}$ 使：

\[
A^{-1}A=AA^{-1}=I,
\]

则 $A$ invertible。

可逆性意味着对应 linear transformation 是 bijection。不是所有 square matrices 都可逆。

## Rank

Matrix rank 可以定义为 column space 的 dimension：

\[
\operatorname{rank}(A)
=
\dim(\operatorname{Col}(A)).
\]

它描述 linear transformation 实际能够产生多少独立 output directions。

若：

\[
A\in\mathbb R^{m\times n},
\]

则：

\[
\operatorname{rank}(A)
\le
\min(m,n).
\]

低 rank matrix 会把 input space 压缩到更低维 subspace。

## Symmetric Matrix

若：

\[
A=A^\top,
\]

则 $A$ symmetric。

Real symmetric matrices 具有 real eigenvalues，并可由 orthonormal eigenvectors diagonalize：

\[
A=Q\Lambda Q^\top.
\]

Covariance matrix 是重要的 symmetric positive-semidefinite matrix。

## Matrix 与 Linear Transformation

设 linear transformation：

\[
T:\mathbb R^n\rightarrow\mathbb R^m.
\]

在 standard basis 下，总存在 matrix $A\in\mathbb R^{m\times n}$ 使：

\[
T(x)=Ax.
\]

Matrix 因此不是 linear transformation 的抽象定义本身，而是选定 bases 后的数值表示。详见 [Linear Transformation](/mathematics/linear-algebra/linear-transformation/)。

## Batched Data

Machine learning 常把多个 vectors 堆成 matrix：

\[
X
=
\begin{bmatrix}
x_1^\top\\
\vdots\\
x_B^\top
\end{bmatrix}
\in\mathbb R^{B\times d}.
\]

这只是同时存储 $B$ 个 $d$-dimensional vectors。进一步的 tensor dimensions 可以表示 sequence、heads、spatial grids 等结构。

## Connections

- [Vector](/mathematics/linear-algebra/vector/)：matrix 作用的基本对象。
- [Linear Transformation](/mathematics/linear-algebra/linear-transformation/)：matrix 的核心线性代数含义。
- [Dot Product](/mathematics/linear-algebra/dot-product/)：matrix multiplication 的局部计算结构。
