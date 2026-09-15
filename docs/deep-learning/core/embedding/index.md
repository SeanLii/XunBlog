---
title: "Embedding"
kind: "canonical"
domain: "Deep Learning / Core"
parent: "Deep Learning"
canonical: "/deep-learning/core/embedding/"
prerequisites:
  - "/mathematics/linear-algebra/vector/"
  - "/mathematics/linear-algebra/matrix/"
related:
  - "/deep-learning/core/linear-layer/"
  - "/deep-learning/sequence-modeling/positional-encoding/"
standard: "XunBlog Content & Knowledge Architecture v1.0"
rebuilt: "2026-09-15"
---
# Embedding

> **知识边界**：本文的 canonical 对象是 **Embedding**。依赖机制由 [Vector](/mathematics/linear-algebra/vector/)、[Matrix](/mathematics/linear-algebra/matrix/) 的 canonical page 定义；本文只在当前语境中调用其接口。


Embedding 是把离散对象、类别或结构中的实体映射到连续向量空间的表示方法。对离散集合

\[
\mathcal V=\{1,\ldots,V\},
\]

embedding 可以写成映射

\[
f:\mathcal V\rightarrow\mathbb R^d.
\]

对对象 $i$，其表示为

\[
e_i=f(i)\in\mathbb R^d.
\]

Embedding 的价值在于：离散 identity 本身没有适合梯度优化的连续几何，而向量表示可以进入 Linear Layer、dot product、Attention、距离函数等连续计算。

## Embedding Matrix

最常见的 learned embedding 使用参数矩阵

\[
E\in\mathbb R^{V\times d}.
\]

对象 $i$ 的 embedding 是第 $i$ 行：

\[
e_i=E_{i,:}.
\]

训练时，$E$ 与模型其他 parameters 一起通过 gradient descent 更新。Embedding 因此不是固定编码规则，而是由 training objective 决定的 representation parameters。

## One-Hot Representation and Lookup

令 $o_i\in\mathbb R^V$ 为对象 $i$ 的 one-hot vector，则

\[
o_i^\top E=E_{i,:}=e_i.
\]

因此 embedding lookup 与 one-hot vector 乘 embedding matrix 在数学上等价。实际实现直接索引对应 row，避免构造高维 sparse one-hot vector。

## Learned Geometry

Embedding space 中可以定义距离、角度和 dot product，但这些几何关系的含义由训练目标产生。

若训练 objective 鼓励相关对象得到相似表示，则可能出现

\[
\operatorname{sim}(e_i,e_j)
\]

较高的现象；但“距离近必然语义相似”不是 embedding 的定义。不同任务对同一对象可能学习出完全不同的空间结构。

## Static Embedding and Contextual Representation

Static embedding 为一个 identity 提供固定 base vector。例如一个 token ID 在 embedding table 中只有一个 row。

Contextual representation 则依赖输入上下文。Transformer 中常见数据流为

```text
token ID
  ↓
token embedding
  ↓
+ positional / segment information
  ↓
contextual network
  ↓
context-dependent hidden state
```

因此 embedding 与 contextual hidden state 是不同层次的表示。BERT 中同一个 subword token 在不同句子中共享初始 token embedding，但经过 encoder 后会得到不同 contextual representations。

## Pretrained Embeddings

Embedding 可以与整个模型共同训练，也可以先在大规模数据上预训练，再用于下游任务。

经典 word embeddings 如 Word2Vec、GloVe 学习静态词表示；现代语言模型则更强调 contextual representation。两者都属于 continuous representation learning，但训练目标与表示使用方式不同。

## Embedding Beyond Language

Embedding 并不局限于词或 token。常见对象包括：

- categorical variables；
- user / item IDs；
- graph nodes；
- task IDs；
- discrete actions；
- positions；
- learned output slots / queries。

关键条件是：输入对象具有离散 identity，而模型需要把它转换成可学习的连续表示。

## Positional and Query Embeddings

Learned positional embedding 使用 vector 表示 position identity；learnable query embedding 使用 vector 表示 output slot 或 persistent query identity。

它们在计算形式上都可以由参数表或 learned vectors 实现，但语义不同。位置 embedding 表示“在哪里”，query embedding 表示“哪个可学习槽位 / 请求”。具体角色由 architecture 决定。

## Input–Output Weight Tying

在语言模型中，输入 token embedding matrix 有时会与输出 vocabulary projection 共享参数。若 hidden state $h$ 通过

\[
z=Eh
\]

或等价形式产生 vocabulary logits，这种 **weight tying** 会减少参数量，并把输入与输出 token representation 置于相关参数空间中。它是 embedding 的一种 architecture design，并非 embedding 的必要条件。

## 由机制产生的边界

Embedding table 本身存在若干边界：

- 未见过或未分配 ID 的对象无法直接 lookup；
- 大词表会带来显著参数量；
- learned geometry 可能继承训练数据偏差；
- static embedding 无法单独表达上下文依赖的多义性。

Subword tokenization、hash embedding、factorized embedding 与 contextual models 都是在不同场景下处理这些问题的方法。

## Sources

- Mikolov et al. *Efficient Estimation of Word Representations in Vector Space*. 2013.
- Pennington, Socher, Manning. *GloVe: Global Vectors for Word Representation*. 2014.
- Press, Wolf. *Using the Output Embedding to Improve Language Models*. 2017.
