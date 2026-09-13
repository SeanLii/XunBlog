---
title: KL Divergence
description: 理解 KL divergence 如何约束 latent distribution。
---

# KL Divergence

<NoteMeta status="Seed" difficulty="Intermediate" updated="2026-09" />

## 直觉

KL divergence 衡量：如果真实数据来自分布 $q$，却用分布 $p$ 去描述它，会多付出多少信息代价。

$$
D_{KL}(q\|p)=\mathbb{E}_{z\sim q}\left[\log\frac{q(z)}{p(z)}\right]
$$

在 CVAE 中，它让 encoder 学到的 $q(z\mid x,c)$ 不要任意散开，而是接近一个简单先验 $p(z)$。这样推理时才能从先验中选择或采样合理的 $z$。

::: warning 常见误解
KL divergence 不是对称距离。一般情况下，$D_{KL}(q\|p) \neq D_{KL}(p\|q)$。
:::

