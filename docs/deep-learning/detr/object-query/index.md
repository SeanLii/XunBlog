---
title: "Object Query"
kind: "canonical"
domain: "Deep Learning / DETR"
parent: "DETR"
canonical: "/deep-learning/detr/object-query/"
prerequisites:
  - "/deep-learning/detr/"
  - "/deep-learning/attention/cross-attention/"
related:
  - "/robot-learning/act/architecture/"
standard: "XunBlog Content & Knowledge Architecture v1.0"
rebuilt: "2026-09-15"
---
# Object Query

Object Query 是 [DETR](/deep-learning/detr/) decoder 中的一组 learned output slots。它们作为模型参数被学习，而不是由某个输入 token 直接产生。

设 DETR 使用 $N$ 个 queries：

\[
Q_0=
\begin{bmatrix}
q_1\\
q_2\\
\vdots\\
q_N
\end{bmatrix}
\in\mathbb R^{N\times d}.
\]

训练开始时，这些 vectors 没有预先规定“哪一个负责什么类别”。它们的功能由 decoder interaction、matching loss 与 prediction objective 一起塑造。

## Object Query as Prediction Slot

每个 object query 对应一个 learned prediction slot：

```text
query 1 ─┐
query 2 ─┤
...      ├─ read image memory ─→ object representations
query N ─┘
```

decoder 结束后，每个 slot 进入 prediction heads，输出：

- class；
- bounding box；
- 或 no-object。

这和普通 token embedding 的角色不同：object query 的主要职责是定义输出 slots。

## Reading Image Memory

DETR decoder 中，object queries 为固定数量的 prediction slots 提供 learned query identity，decoder hidden states 再通过 cross-attention 读取 encoder memory。

从 architecture 层面可以简写为

\[
H=\operatorname{Decoder}(Q_0,M),
\]

其中 $Q_0$ 表示 learned object-query information，$M$ 是 image memory。但这个写法只是概念级简写；**原始 DETR 官方实现并不是把 $Q_0$ 直接当作普通 decoder content state。**

官方实现先建立

\[
T_0=0,
\]

再把 learned query embeddings 作为 `query_pos` 传入 decoder：

\[
H=\operatorname{Decoder}(T_0,M;Q_{pos}=Q_0).
\]

在 decoder layer 的 self-attention 中，query / key 由当前 content state 加上 query position 得到；在 cross-attention 中，decoder query 也加入 $Q_0$，encoder key 则加入 image positional encoding。以 cross-attention 为例，可抽象写成

\[
Q=(T+Q_0)W_Q,\qquad
K=(M+P_M)W_K,\qquad
V=MW_V.
\]

因此需要区分三个层级：

- **Object Query embedding $Q_0$**：learned slot identity，在原始官方实现中作为 `query_pos` 使用；
- **Decoder content state $T$**：第一层从零开始，随后被 attention 与 feed-forward updates 逐层写入 image/object information；
- **Attention query matrix $Q$**：某一 attention layer 内部经过 projection 后真正参与 dot-product attention 的计算向量。

三者相关，但不是同一个对象。尤其不能把“object query”与 attention 公式中的 $Q$ 直接视为同义词。

## 一对一 Set Prediction

Object queries 与 Hungarian matching 配合，使模型学会把不同 prediction slots 分配给不同 objects。

没有匹配到真实 object 的 queries 会学习 no-object class。

因此 object query 的意义不能脱离 DETR 的 set-prediction objective：它不是一般 Transformer 的标准组件，而是 DETR architecture 的核心设计之一。

## 从 Object Query 到 ACT Action Query

ACT 官方实现继承 DETR-style decoder queries，但把 $N$ 个 object slots 改成 $k$ 个 future action slots：

\[
q_1,\ldots,q_k.
\]

每个 slot 最终对应 action chunk 中一个 temporal position。

这是 DETR 机制在机器人 action prediction 中的具体改造，因此 ACT 页面应该说明这种继承关系，而不是把“Learnable Query Embedding”错误写成 Transformer 自身提出的通用组件。

## Sources

- Carion et al. *End-to-End Object Detection with Transformers*. ECCV 2020.
- Official DETR implementation: facebookresearch/detr.
- Zhao et al. *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware*. 2023.
