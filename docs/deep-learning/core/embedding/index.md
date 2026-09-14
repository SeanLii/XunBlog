---
title: "Embedding"
kind: "canonical"
domain: "Deep Learning / Core"
parent: "Deep Learning"
canonical: "/deep-learning/core/embedding/"
prerequisites:
  - "/mathematics/linear-algebra/vector/"
related:
  - "/deep-learning/core/linear-layer/"
  - "/deep-learning/sequence-modeling/positional-encoding/"
---

# Embedding

Embedding 是把 discrete object 或 structured identity 映射到 continuous vector space 的表示方法。

最典型的情况是 token ID：

```text
"cat" → token id 3917 → vector in R^d
```

模型不能直接对“3917”这个整数做有意义的几何运算，因为 ID 只表示身份，不表示大小关系。Embedding 把这种离散 identity 变成 learned vector：

\[
e_i\in\mathbb R^d.
\]

## Embedding Table

如果 vocabulary size 为 $V$，embedding dimension 为 $d$，可以维护：

\[
E\in\mathbb R^{V\times d}.
\]

token id $i$ 查表得到：

\[
e_i=E_{i,:}.
\]

训练时，这些 rows 与模型其他 parameters 一起通过 gradient 更新。

所以 embedding lookup 并不是“把数字编码成向量的固定公式”，而是在学习一个 representation table。

## Continuous Representation Space

离散 IDs 之间没有自然几何：

```text
id 100 与 id 101
```

并不比

```text
id 100 与 id 9000
```

更相似。

Embedding space 则允许模型通过位置关系表达 learned structure：

- 相似对象可能得到相近表示；
- downstream layers 可以做 dot product、linear projection、attention 等连续计算；
- 同一个 object 可以根据训练目标形成任务相关 representation。

注意“embedding 接近 = 语义相似”不是定义，而是某些训练目标下可能形成的性质。

## One-Hot + Linear Layer 的等价视角

把 token $i$ 写成 one-hot vector：

\[
o_i\in\mathbb R^V.
\]

则：

\[
o_i^\top E=E_{i,:}=e_i.
\]

所以 embedding lookup 可以理解为对 one-hot input 做 matrix multiplication，只是实际实现不需要构造巨大 sparse one-hot vector，直接取对应 row 更高效。

## Embedding 不只属于语言

任何 discrete identity 都可以使用 embedding：

- words / subword tokens；
- user IDs / item IDs；
- categorical variables；
- graph node IDs；
- positions；
- learned query slots；
- task IDs。

区别在于“这个 identity 是什么”，不是 embedding 机制本身改变了。

## Static 与 Contextual Representation

Embedding table 中一个 token ID 通常有固定 base vector。

但经过 neural network 后，同一 token 在不同 context 下可以得到不同 hidden state。

例如 Transformer 中：

```text
token embedding
      ↓
+ positional information
      ↓
Transformer layers
      ↓
contextual hidden state
```

因此 token embedding 和最终 contextual representation 不是同一个概念。

## Positional / Query Embedding 是同一机制的不同身份

[Positional Encoding](/deep-learning/sequence-modeling/positional-encoding/) 中，learned position embeddings 表示 position identity；learnable queries 则用 learned vectors 表示 output slots 或 requests。

这些都利用 continuous vector 来表示某种 discrete role，但它们的语义来自模型结构和训练目标。

Embedding 的核心因此不是“词向量”，而是：

> **把离散身份映射成可以被神经网络连续计算的 learned representation。**
