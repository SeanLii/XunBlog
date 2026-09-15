---
title: "Position-Wise Feed-Forward Network"
kind: "canonical"
domain: "Deep Learning / Transformer"
parent: "Transformer"
canonical: "/deep-learning/transformer/position-wise-feed-forward-network/"
prerequisites:
  - "/deep-learning/core/multilayer-perceptron/"
related:
  - "/deep-learning/transformer/"
standard: "XunBlog Content & Knowledge Architecture v1.0"
rebuilt: "2026-09-15"
---
# Position-Wise Feed-Forward Network

Position-Wise Feed-Forward Network 是原始 Transformer layer 中的一个子层。它对 sequence 中**每个 position 独立应用同一个 MLP**。

原论文写成：

\[
\operatorname{FFN}(x)
=
\max(0,xW_1+b_1)W_2+b_2.
\]

如果输入：

\[
X=
\begin{bmatrix}
x_1\\
x_2\\
\vdots\\
x_N
\end{bmatrix}
\in\mathbb R^{N\times d_{model}},
\]

则：

\[
y_i=\operatorname{FFN}(x_i)
\]

对所有 positions 使用同一组 parameters。

## Role in a Transformer Layer

Attention 负责在 positions 之间交换信息。

FFN 则处理每个 position 已经收集到的 feature representation：

```text
Self / Cross Attention
        ↓
不同位置的信息已经被汇总到当前 token
        ↓
Position-Wise FFN
        ↓
在当前 token 的 feature dimensions 内做 nonlinear transformation
```

因此这两个子层的职责不同：

- Attention：cross-position communication；
- Position-Wise FFN：per-position nonlinear feature processing。

## Position-Wise 的准确含义

“Position-wise”不是说每个 position 有自己的 MLP。

恰恰相反，所有 positions 共享同一个 function：

\[
f_\theta:\mathbb R^{d_{model}}
\rightarrow
\mathbb R^{d_{model}}.
\]

只是这个 function 被分别作用在：

\[
x_1,x_2,\ldots,x_N.
\]

因此 FFN 本身不会让 $x_i$ 直接读取 $x_j$。

## Expansion 与 Projection Back

原始 Transformer 使用：

\[
d_{model}=512,\qquad d_{ff}=2048.
\]

所以第一层：

\[
\mathbb R^{512}
\rightarrow
\mathbb R^{2048},
\]

再经过 ReLU，第二层：

\[
\mathbb R^{2048}
\rightarrow
\mathbb R^{512}.
\]

这形成：

```text
model dimension
      ↓ expand
larger hidden dimension
      ↓ nonlinearity
      ↓ project back
model dimension
```

较大的 intermediate dimension 给每个 token 更大的 nonlinear feature capacity。

## 与单个 Linear Layer 的区别

如果没有 activation：

\[
W_2(W_1x+b_1)+b_2
\]

仍然只是一个 affine mapping。

加入 ReLU 后：

\[
W_2\operatorname{ReLU}(W_1x+b_1)+b_2
\]

构成 nonlinear mapping。

完整的一般结构见 [Multilayer Perceptron](/deep-learning/core/multilayer-perceptron/)。

## Modern Variants

后续 Transformer family 经常替换原始 ReLU MLP，例如：

- GELU FFN；
- GEGLU；
- SwiGLU；
- gated MLP。

抽象形式常写成：

\[
\operatorname{FFN}(x)
=
W_o\left[
\phi(xW_g)\odot(xW_v)
\right].
\]

这些实现改变 activation / gating / parameter allocation，但仍然保留 position-wise nonlinear transformation 的基本角色。

## 在 Transformer 中的位置

MLP 并不是 Transformer 发明的，因此 [Multilayer Perceptron](/deep-learning/core/multilayer-perceptron/) 独立存在。

但“Position-Wise Feed-Forward Network”是 Transformer architecture 中定义的具体子层：同一 FFN 被独立应用到每个 sequence position，并与 attention、residual、normalization 一起组成 Transformer layer。

所以：

```text
MLP
└── 通用 neural-network structure

Transformer
└── Position-Wise Feed-Forward Network
    └── MLP 在 Transformer layer 中的具体组织方式
```

## Sources

- Vaswani et al. *Attention Is All You Need*. 2017, Section 3.3.
