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

设：

\[
T:V\rightarrow W,
\]

若对任意 $u,v\in V$ 和 scalar $\alpha,\beta$ 都有：

\[
T(\alpha u+\beta v)
=
\alpha T(u)+\beta T(v),
\]

则 $T$ 是 linear transformation。

等价地，它同时满足：

\[
T(u+v)=T(u)+T(v),
\]

\[
T(\alpha u)=\alpha T(u).
\]

## Immediate Consequences

Linear transformation 必须满足：

\[
T(0)=0.
\]

并且对任意 finite linear combination：

\[
T\left(\sum_i\alpha_iv_i\right)
=
\sum_i\alpha_iT(v_i).
\]

因此只要知道 transformation 对一组 basis vectors 的作用，就可以确定它对整个 space 的作用。

## Matrix Representation

设：

\[
T:\mathbb R^n\rightarrow\mathbb R^m.
\]

在 standard bases 下，存在唯一 matrix：

\[
A\in\mathbb R^{m\times n}
\]

使：

\[
T(x)=Ax.
\]

Matrix 的第 $j$ 列就是：

\[
T(e_j),
\]

即第 $j$ 个 basis vector 经过 transformation 后的 coordinates。

## Basis Dependence

Abstract transformation $T$ 不依赖某个固定 coordinate system，但 matrix representation 依赖 input 与 output bases。

若更换 basis，同一个 $T$ 会由不同 matrix 表示。

因此：

> linear transformation 是映射；matrix 是该映射在选定 basis 下的 representation。

## Kernel

Kernel 定义为：

\[
\ker T
=
\{x\in V:T(x)=0\}.
\]

它包含所有被 transformation 映射到 zero vector 的 directions。

若：

\[
\ker T=\{0\},
\]

则 $T$ injective。

## Image

Image（或 range）定义为：

\[
\operatorname{Im}T
=
\{T(x):x\in V\}.
\]

它表示 transformation 能够产生的所有 outputs。

若 $T(x)=Ax$，则：

\[
\operatorname{Im}T
=
\operatorname{Col}(A).
\]

其 dimension 就是 matrix rank。

## Rank-Nullity Theorem

对有限维 $V$：

\[
\dim(V)
=
\dim(\ker T)
+
\dim(\operatorname{Im}T).
\]

写成 matrix language：

\[
n
=
\operatorname{nullity}(A)
+
\operatorname{rank}(A).
\]

它把 information lost directions 与 preserved output dimensions 联系起来。

## Injective, Surjective, Invertible

- **injective**：不同 inputs 不会被映射成同一个 output；
- **surjective**：output space 中每个元素都有 preimage；
- **bijective**：同时 injective 与 surjective。

若 $T:V\to V$ bijective，则存在 inverse linear transformation：

\[
T^{-1}.
\]

在 matrix representation 中对应 invertible matrix。

## Composition

若：

\[
T_1:U\rightarrow V,
\qquad
T_2:V\rightarrow W,
\]

则 composition：

\[
T_2\circ T_1
\]

仍是 linear transformation。

若 matrix representations 为 $A$ 和 $B$，则 composition 对应：

\[
BA.
\]

这解释了 matrix multiplication 与多层 linear mappings 的关系。

## Affine Transformation

Neural network 中常见：

\[
y=Wx+b.
\]

当 $b\neq0$ 时，这不是严格意义上的 linear transformation，因为：

\[
T(0)=b\neq0.
\]

它属于 affine transformation。

Deep-learning frameworks 常把这种 layer 命名为 `Linear`，见 [Linear Layer](/deep-learning/core/linear-layer/)。

## Examples

典型 linear transformations 包括：

- rotation；
- reflection；
- scaling；
- projection；
- dimensionality reduction 到 subspace；
- coordinate transformations（在适当表示下）。

它们虽然几何效果不同，但都保持 linear combinations。
