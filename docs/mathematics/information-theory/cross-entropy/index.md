---
title: "Cross-Entropy"
kind: "canonical"
domain: "Mathematics / Information Theory"
parent: "Information Theory"
canonical: "/mathematics/information-theory/cross-entropy/"
prerequisites:
  - "/mathematics/information-theory/entropy/"
related:
  - "/mathematics/information-theory/kl-divergence/"
  - "/deep-learning/core/softmax/"
standard: "XunBlog Content & Knowledge Architecture v1.0"
rebuilt: "2026-09-15"
---
# Cross-Entropy

Cross-Entropy 衡量使用 model distribution $q$ 描述来自 target distribution $p$ 的 samples 时的 expected negative log-probability。

Discrete case：

\[
H(p,q)
=
-\sum_x p(x)\log q(x).
\]

Expectation form：

\[
H(p,q)
=
-\mathbb E_{x\sim p}[\log q(x)].
\]

## Continuous Case

若 $p(x)$ 与 $q(x)$ 是相对于同一基准测度定义的 probability densities，则 continuous cross-entropy 定义为

\[
H(p,q)
=
-\int p(x)\log q(x)\,dx
=
-\mathbb E_{x\sim p}[\log q(x)].
\]

这里的 $q(x)$ 是 **density**，不是单点 probability。连续变量的单点通常具有 probability 0，但 density 可以大于 1，因此 continuous cross-entropy 与 differential entropy 一样，不保证非负。

如果存在一个在 $p$ 下具有正测度的区域，而 $q(x)=0$ 在该区域成立，则 $\log q(x)=-\infty$，cross-entropy 发散到 $+\infty$。因此要得到有限 cross-entropy，$p$ 不能在 $q$ 为零的区域保留正质量；更正式地说，需要 $p$ 对 $q$ 绝对连续，并满足相应可积性条件。

## Relation to Entropy and KL Divergence

在 discrete case 中，在定义式里加入并减去 $\log p(x)$ 可得

\[
H(p,q)
=
H(p)+D_{KL}(p\|q).
\]

其中

\[
H(p)
=
-\sum_xp(x)\log p(x),
\]

\[
D_{KL}(p\|q)
=
\sum_xp(x)\log\frac{p(x)}{q(x)}.
\]

在 continuous case 中，同样有

\[
H(p,q)=h(p)+D_{KL}(p\|q),
\]

其中 $h(p)=-\int p(x)\log p(x)\,dx$ 是 differential entropy。

因为

\[
D_{KL}(p\|q)\ge0,
\]

所以在 $p$ 固定、相关量有限时，cross-entropy 相对于 $q$ 的最小值在 $q=p$（几乎处处）取得。Discrete case 中这也给出

\[
H(p,q)\ge H(p).
\]

Continuous case 中 $h(p)$ 本身可以为负，因此“cross-entropy 非负”并不是一般结论；真正稳定成立的是它与 $h(p)$ 的差等于非负 KL divergence。

## Cross-Entropy as Log Loss

若 dataset 来自 unknown data distribution $p_{data}$，model 为 $q_\theta$，经验 negative log-likelihood：

\[
\mathcal L(\theta)
=
-\frac1N\sum_{i=1}^{N}
\log q_\theta(x_i)
\]

是 population cross-entropy：

\[
H(p_{data},q_\theta)
\]

的 sample estimate。

因此 maximum likelihood 与 cross-entropy minimization 密切对应。

## Categorical Classification

对 $K$-class classification，target distribution $y$ 与 predicted probabilities $p$ 的 cross-entropy：

\[
\mathcal L
=
-\sum_{k=1}^{K}y_k\log p_k.
\]

若 target 是 one-hot，真实 class 为 $c$：

\[
\mathcal L=-\log p_c.
\]

因此分类 cross-entropy 只取 model 对 correct class 分配的 log-probability，但 softmax normalization 使所有 logits 都参与这个 probability。

## Logits 与 Softmax

设 logits：

\[
z\in\mathbb R^K.
\]

Softmax probability：

\[
p_k
=
\frac{e^{z_k}}{\sum_j e^{z_j}}.
\]

One-hot cross-entropy：

\[
-\log p_c
=
-z_c+
\log\sum_j e^{z_j}.
\]

实际实现常直接使用 logits 计算 log-softmax / log-sum-exp，以提高 numerical stability，而不是先显式计算 probabilities 再取 log。

## Gradient with Softmax

Softmax + cross-entropy 的一个重要结果是：

\[
\frac{\partial\mathcal L}{\partial z_k}
=
p_k-y_k.
\]

这使 classification gradient 具有清晰形式：predicted probability 与 target probability 的差。

## Soft Targets

Cross-entropy 不要求 $p$ 是 one-hot distribution。

例如 label smoothing 或 knowledge distillation 可以使用：

\[
y_k\in[0,1],
\qquad
\sum_ky_k=1.
\]

loss 仍然是：

\[
-\sum_ky_k\log p_k.
\]

这允许 target 本身表达 uncertainty 或 teacher distribution。

## Binary Cross-Entropy

对 Bernoulli target $y\in\{0,1\}$ 与 predicted probability $p$：

\[
\mathcal L
=
-y\log p
-(1-y)\log(1-p).
\]

这就是 Bernoulli cross-entropy，也等价于 negative log-likelihood。

## Support and Infinite Loss

若：

\[
p(x)>0
\]

但：

\[
q(x)=0,
\]

则：

\[
-\log q(x)=\infty.
\]

因此 cross-entropy 强烈惩罚 model 对真实可能事件分配零概率。

## Connections

- [Entropy](/mathematics/information-theory/entropy/)：target distribution 自身的 uncertainty。
- [KL Divergence](/mathematics/information-theory/kl-divergence/)：$H(p,q)=H(p)+D_{KL}(p\|q)$。
- [Softmax](/deep-learning/core/softmax/)：常把 logits 转为 categorical probabilities。
