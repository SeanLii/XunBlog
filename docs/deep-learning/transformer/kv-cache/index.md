---
title: "KV Cache"
kind: "canonical"
domain: "Deep Learning / Transformer"
parent: "Transformer"
canonical: "/deep-learning/transformer/kv-cache/"
prerequisites:
  - "/deep-learning/attention/qkv/"
  - "/deep-learning/sequence-modeling/causal-mask/"
related:
  - "/robot-learning/pi0/inference/"
standard: "XunBlog Content & Knowledge Architecture v1.0"
rebuilt: "2026-09-15"
---
# KV Cache

KV Cache 是 Transformer inference 中复用已经计算过的 Key / Value tensors 的机制。

它解决的问题很直接：如果一部分 context 在后续计算中不会改变，就没有必要每一步都重新把这部分 tokens 投影成 K 和 V。

## 重复计算从哪里来

在一层 attention 中：

\[
Q=XW_Q,\qquad
K=XW_K,\qquad
V=XW_V.
\]

自回归生成第 $t$ 个 token 时，过去 tokens：

\[
x_1,\ldots,x_{t-1}
\]

已经在上一轮出现过。

如果每一步都把整个 prefix 重新 forward，一样的 past K/V 会被反复计算。

KV Cache 保存：

\[
K_{1:t-1},\qquad V_{1:t-1},
\]

下一步只需要计算新 token 的：

\[
k_t,\quad v_t,
\]

再把它们追加到 cache。

## Prefill 与 Decode

典型推理可以分成两阶段。

### Prefill

一次处理已有 prompt：

\[
x_{1:n}
\]

并建立各层：

\[
K_{1:n}^{(l)},V_{1:n}^{(l)}.
\]

### Decode

之后每生成一个新 token，只计算新位置的 hidden state 和新 K/V，再读取缓存的历史 K/V。

因此 KV Cache 主要减少的是**重复计算**，不是让 attention 完全摆脱读取历史 context 的成本。

## Cache 的 Shape

对某层某个 head，可以抽象成：

\[
K_{\text{cache}}\in\mathbb R^{T\times d_k},
\qquad
V_{\text{cache}}\in\mathbb R^{T\times d_v}.
\]

实际实现还包含：

- batch dimension；
- layer；
- attention heads；
- possibly grouped / multi-query KV heads。

sequence 越长，cache memory 通常也随 $T$ 增长。

所以 KV Cache 用显存换取 inference compute。

## 与 Causal Attention 的关系

在标准 causal generation 中，新 query 只能读取过去与当前位置。

缓存刚好利用了这一结构：过去 tokens 的 representation 不需要因为未来 token 出现而反向更新。

如果 attention pattern 允许旧 token 读取后来才出现的 tokens，那么旧 K/V 对应的 hidden states 可能需要重算，缓存逻辑就不能直接照搬。

因此能否安全 cache，与 attention dependency structure 有直接关系。

## π0 中的 KV Cache

π0 inference 并不是逐 token 生成文本，但它有另一个重复结构：同一个 robot timestep 内会进行多次 Flow Matching integration。

Images、language 和 robot state 在这些 flow steps 中保持不变，只有 noisy action block 在变化。

因此 π0 可以先对固定 prefix 做一次 forward，保存其 K/V；之后每个 flow step 只重新计算会变化的 action suffix。

这与语言模型 KV Cache 的核心思想相同：

> **不变的 context 不重复计算。**

具体 block dependency 见 π0 的 [Inference](/robot-learning/pi0/inference/) 与 [Blockwise Causal Attention Mask](/robot-learning/pi0/blockwise-causal-attention-mask/)。

## Sources

- Vaswani et al. *Attention Is All You Need*. 2017 — Q/K/V attention foundation.
- Physical Intelligence `openpi` implementation — prefix KV caching in π0 inference.
