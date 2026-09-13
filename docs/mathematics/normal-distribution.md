---
title: Normal Distribution
description: 从直觉到公式理解正态分布。
---

# Normal Distribution

<NoteMeta status="Seed" difficulty="Foundation" updated="2026-09" />

## 为什么需要它

正态分布经常被用来表示围绕某个中心连续变化的不确定性。在 VAE / CVAE 中，我们希望 latent space 不只是离散标签，而是一个可以采样、插值且结构连续的空间。

## 数学表达

若随机变量 $z$ 服从均值为 $\mu$、方差为 $\sigma^2$ 的正态分布：

$$
z \sim \mathcal{N}(\mu, \sigma^2)
$$

标准正态分布对应 $\mu=0$、$\sigma=1$。

::: tip 正在继续整理
下一步将补充均值、方差、标准差的直觉，以及它们在 CVAE encoder 输出中的具体含义。
:::

