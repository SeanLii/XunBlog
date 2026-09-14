---
title: "Entropy"
kind: "canonical"
domain: "Mathematics / Information Theory"
parent: "Information Theory"
canonical: "/mathematics/information-theory/entropy/"
prerequisites:
  - "/mathematics/probability/probability-distribution/"
related:
  - "/mathematics/information-theory/cross-entropy/"
  - "/mathematics/information-theory/kl-divergence/"
---

# Entropy

Entropy 描述一个 probability distribution 的不确定性。直觉上，如果结果几乎已经确定，entropy 很低；如果许多结果都差不多可能，entropy 更高。

对于 discrete distribution $p(x)$，Shannon entropy 定义为：

\[
H(p)
=-\sum_x p(x)\log p(x).
\]

也常写作：

\[
H(X)=-\mathbb E[\log p(X)].
\]

## 从 Information Content 开始

单个事件 $x$ 的 self-information 定义为：

\[
I(x)=-\log p(x).
\]

一个非常可能发生的事件，$p(x)$ 大，因此 $-\log p(x)$ 小；一个罕见事件带来更多“意外信息”，所以 information content 更大。

Entropy 就是对所有可能事件的 information content 取 expectation：

\[
H(X)=\mathbb E[I(X)].
\]

所以 entropy 可以理解为：

> **从这个 distribution 采样一次，平均会带来多少不确定信息。**

## 一个 Bernoulli 例子

若硬币正面概率为 $p$，则：

\[
H(p)
=-p\log p-(1-p)\log(1-p).
\]

当 $p=0$ 或 $1$ 时，结果完全确定：

\[
H=0.
\]

当 $p=0.5$ 时，两种结果同样可能，entropy 最大。

因此 entropy 不是“值越随机越好”，而是对 probability mass 分散程度的一种严格度量。

## Log Base 决定单位

若使用 $\log_2$，单位是 bits；使用 natural logarithm $\ln$，单位常称 nats。

改变 log base 只会按常数比例缩放 entropy，不改变 distribution 之间的比较关系。

## Maximum Entropy

对具有 $K$ 个可能结果的 categorical distribution，entropy 在 uniform distribution 时最大：

\[
p(x)=\frac1K.
\]

此时：

\[
H=\log K.
\]

因为在没有偏好的情况下，每个结果同样可能，不确定性最高。

## Entropy 与编码长度

Information theory 中，$-\log p(x)$ 与理想编码长度直接相关。

更可能的 symbol 可以分配更短 code，更罕见的 symbol 需要更长 code。Entropy 给出在理想条件下平均编码长度的基本极限。

这也是为什么 entropy 不只是“概率曲线的一个统计量”，而是信息量与压缩之间的桥梁。

## 从 Entropy 到 Cross-Entropy

Entropy 使用真实 distribution $p$ 自己来评估平均信息量。

如果我们改用另一个 distribution $q$ 给来自 $p$ 的数据编码，就得到 [Cross-Entropy](/mathematics/information-theory/cross-entropy/)：

\[
H(p,q)=-\mathbb E_{x\sim p}[\log q(x)].
\]

这会进一步连接到 [KL Divergence](/mathematics/information-theory/kl-divergence/)。
