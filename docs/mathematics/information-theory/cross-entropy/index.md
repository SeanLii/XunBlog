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
---

# Cross-Entropy

Cross-Entropy 衡量：数据实际上来自 distribution $p$，但我们用另一个 distribution $q$ 来表示或编码这些数据时，平均需要付出多大的 negative log-probability。

定义：

\[
H(p,q)
=-\sum_x p(x)\log q(x).
\]

也就是：

\[
H(p,q)
=-\mathbb E_{x\sim p}[\log q(x)].
\]

注意 expectation 是对真实 distribution $p$ 取的，但 log 中使用的是 $q$。

## 与 Entropy 的差别

Entropy：

\[
H(p)=-\sum_x p(x)\log p(x)
\]

问的是：如果使用正确的 distribution $p$ 自己编码数据，平均不确定性是多少。

Cross-entropy：

\[
H(p,q)=-\sum_x p(x)\log q(x)
\]

问的是：如果实际数据来自 $p$，却使用 $q$ 来预测或编码，会付出多少代价。

如果 $q=p$：

\[
H(p,q)=H(p).
\]

否则通常更大。

## 与 KL Divergence 的关系

展开：

\[
D_{KL}(p\|q)
=\sum_x p(x)
\log\frac{p(x)}{q(x)}.
\]

可以得到：

\[
\boxed{
H(p,q)=H(p)+D_{KL}(p\|q)
}
\]

因为：

\[
D_{KL}(p\|q)\ge0,
\]

所以：

\[
H(p,q)\ge H(p).
\]

也就是说，用错误的 distribution $q$ 描述来自 $p$ 的数据，不会比知道真实 distribution 更省信息代价。

## Classification 中的 Cross-Entropy

假设真实类别是 class $k$，one-hot target distribution 为：

\[
p_i=
\begin{cases}
1,&i=k\\
0,&i\ne k.
\end{cases}
\]

模型通过 [Softmax](/deep-learning/core/softmax/) 给出 prediction $q_i$。

Cross-entropy：

\[
H(p,q)
=-\sum_i p_i\log q_i
=-\log q_k.
\]

因此 one-hot classification 中，cross-entropy 就是在惩罚模型给真实类别分配的 probability 太低。

## 从 Logits 直接计算的数值稳定性

工程实现通常不会先显式算 softmax 再取 log，而是把 `log-softmax` 与 negative log-likelihood 合并。

原因是：

- 大 logits 的 exponentiation 可能 overflow；
- 很小 probability 再取 log 可能 underflow；
- `logsumexp` 可以更稳定地完成同一计算。

所以许多库中的 `CrossEntropyLoss` 接收 raw logits，而不是已经 softmax 后的 probabilities。

## Cross-Entropy 不是“分类专属 Loss”

Classification 是最常见应用之一，但 cross-entropy 本身是 information-theoretic quantity。

Language modeling、density estimation、teacher-student distributions 等场景都可以使用同一结构：真实或目标 distribution 提供 expectation，模型 distribution 提供 log probability。
