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
---

# Object Query

Object Query 是 DETR decoder 中的一组 learned output slots。它们作为模型参数被学习，而不是由某个输入 token 直接产生。

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

## Query 是 Output Slot

可以先把每个 object query 理解成一个待填写的预测槽位：

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

## Query 如何读取 Image Memory

DETR decoder 中，queries 通过 cross-attention 读取 encoder memory。

抽象写成：

\[
H=
\operatorname{Decoder}(Q_0,M),
\]

其中 $M$ 是 image memory。

attention 内部真正的 query vectors 仍由 hidden states 投影得到：

\[
Q=HW_Q.
\]

所以要区分：

- **Object Query embedding**：decoder 的 learned input / positional slot；
- **Attention Q**：每层通过 projection 得到的计算向量。

它们不是同一层级概念。

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
