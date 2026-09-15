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

Entropy 衡量 probability distribution 的平均 information content 或 uncertainty。

对 discrete random variable $X\sim p$，定义：

\[
H(X)
=
-\sum_x p(x)\log p(x).
\]

也可以写成 expectation：

\[
H(X)
=
\mathbb E_{X\sim p}
[-\log p(X)].
\]

## Self-Information

单个 event $x$ 的 self-information 定义为：

\[
I(x)=-\log p(x).
\]

Probability 越小，$I(x)$ 越大；probability 越大，information content 越小。

Entropy 就是 self-information 的 expectation。

## Log Base 与 Units

如果使用：

\[
\log_2,
\]

entropy 单位是 bits。

如果使用 natural logarithm：

\[
\ln,
\]

单位是 nats。

改变 log base 只改变一个 constant scale factor。

## Basic Properties

Discrete entropy 满足：

\[
H(X)\ge0.
\]

如果 $X$ 是 deterministic：

\[
P(X=x_0)=1,
\]

则：

\[
H(X)=0.
\]

对于具有 $K$ 个可能 outcomes 的 discrete variable：

\[
H(X)\le\log K,
\]

等号在 uniform distribution：

\[
p(x)=\frac1K
\]

时成立。

## Joint Entropy

对 joint distribution：

\[
p(x,y),
\]

joint entropy 为：

\[
H(X,Y)
=
-\sum_{x,y}p(x,y)\log p(x,y).
\]

它描述 pair $(X,Y)$ 的总 uncertainty。

## Conditional Entropy

Conditional entropy：

\[
H(X\mid Y)
=
-\sum_{x,y}
p(x,y)
\log p(x\mid y).
\]

等价地：

\[
H(X\mid Y)
=
\mathbb E_Y[H(X\mid Y=y)].
\]

它表示在已经知道 $Y$ 后，$X$ 还剩多少平均 uncertainty。

## Chain Rule

Entropy 满足：

\[
H(X,Y)
=
H(Y)+H(X\mid Y)
\]

也等于：

\[
H(X)+H(Y\mid X).
\]

对 sequence：

\[
H(X_1,\ldots,X_n)
=
\sum_{i=1}^{n}
H(X_i\mid X_1,\ldots,X_{i-1}).
\]

它与 probability chain rule：

\[
p(x_{1:n})
=
\prod_i p(x_i\mid x_{<i})
\]

直接对应。

## Independence

若 $X,Y$ independent，则：

\[
H(X,Y)=H(X)+H(Y).
\]

并且：

\[
H(X\mid Y)=H(X).
\]

知道 $Y$ 不减少对 $X$ 的 uncertainty。

## Coding Interpretation

Shannon source coding theorem 将 entropy 与 lossless coding 联系起来。

对于来自 distribution $p$ 的长 i.i.d. sequence，最佳 prefix coding / block coding 的 average code length 不能系统性低于 entropy，并可以在适当 coding scheme 下逼近 entropy。

因此：

\[
H(p)
\]

它给出该 information source 在无损编码条件下的理论平均码长下界。

## Differential Entropy

对 continuous random variable，定义 differential entropy：

\[
h(X)
=
-\int f(x)\log f(x)\,dx.
\]

它与 discrete entropy 有重要区别：

- differential entropy 可以为负；
- 它依赖 coordinate scale；
- 不应直接解释为离散 code length。

KL divergence、mutual information 等 quantities 在连续情况下通常具有更稳定的 invariant interpretation。

## Cross-Entropy 与 KL Divergence

对 true distribution $p$ 与 model $q$：

\[
H(p,q)
=
-\mathbb E_{x\sim p}\log q(x).
\]

它与 entropy、KL divergence 满足：

\[
H(p,q)
=
H(p)+D_{KL}(p\|q).
\]

因此当 $p$ 固定时，minimize cross-entropy 等价于 minimize forward KL divergence。

## Connections

- [Cross-Entropy](/mathematics/information-theory/cross-entropy/)：用另一个 distribution 的 log-probability 衡量编码/预测代价。
- [KL Divergence](/mathematics/information-theory/kl-divergence/)：两个 distributions 之间的 relative-information quantity。
