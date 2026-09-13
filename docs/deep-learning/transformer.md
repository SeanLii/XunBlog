---
title: Transformer
description: Transformer 如何建模序列内的依赖关系。
status: learning
pageType: concept
canonical: /deep-learning/transformer
difficulty: intermediate
updated: "2026-09"
---

# Transformer

<NoteMeta />

## 核心问题

对于一个序列，当前位置应该关注哪些其他位置？Self-attention 让每个 token 根据内容动态聚合整个序列的信息。

$$
\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

在语言模型里 token 可以是词元；在 ACT 中，token 可以来自图像、机器人关节状态、latent variable 或动作序列。

## 与 ACT 的连接

ACT 使用 Transformer 不是为了生成文本，而是为了同时读取当前观测，并一次解码未来一段动作。
