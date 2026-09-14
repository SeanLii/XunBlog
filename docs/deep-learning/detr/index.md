---
title: "DETR"
kind: "canonical"
domain: "Deep Learning / DETR"
parent: "Deep Learning"
canonical: "/deep-learning/detr/"
prerequisites:
  - "/deep-learning/cnn/resnet/"
  - "/deep-learning/transformer/"
related:
  - "/deep-learning/detr/object-query/"
  - "/robot-learning/act/architecture/"
---

# DETR

DETR（DEtection TRansformer）把 object detection 改写成一个直接的 set prediction 问题：给定一张图像，模型一次输出固定数量的 object predictions，而不是先产生大量 anchors / proposals，再经过多阶段筛选。

最小数据流是：

```text
image
  ↓
CNN backbone
  ↓
spatial feature sequence
  ↓
Transformer encoder
  ↓
image memory
  ↑
object queries
  ↓
Transformer decoder
  ↓
fixed set of object predictions
```

每个预测通常包含 object class 与 bounding box。

## 从 Dense Candidates 到 Set Prediction

传统 detector 往往产生大量候选框，再依赖 proposal、anchor matching、NMS 等步骤。

DETR 的目标是直接学习：

\[
\{\hat y_1,\ldots,\hat y_N\}
\]

这样一个无序 prediction set。

关键难点是：ground-truth objects 同样没有天然顺序，所以训练时不能简单规定“第 1 个 query 对第 1 个 object”。

## Bipartite Matching

DETR 使用 Hungarian matching 在 predictions 与 ground-truth objects 之间寻找一对一 assignment。

设 ground truth 为：

\[
Y=\{y_1,\ldots,y_M\},
\]

predictions 为：

\[
\hat Y=\{\hat y_1,\ldots,\hat y_N\}.
\]

训练先寻找代价最小的 matching，再对匹配结果计算 classification 与 box losses。

这使不同 queries 被鼓励输出不同 objects，并避免传统 NMS 的重复检测处理。

## Object Queries

Decoder 接收一组固定数量的 learned [Object Query](/deep-learning/detr/object-query/)。

它们不是“car query”“person query”这样的类别模板，而是模型学习出来的 output slots。每个 query 通过 decoder 与 image memory 交互，形成一个 object prediction representation。

因此 query 数量决定模型一次最多维护多少 prediction slots。

## Encoder 与 Decoder 的分工

Encoder 让 image feature positions 彼此交换信息，形成带有全局上下文的 image memory。

Decoder 则让 object queries：

1. 彼此交互；
2. 从 image memory 读取信息；
3. 逐层形成 object-level predictions。

DETR 因此是 Transformer encoder-decoder 在 vision set prediction 中的一种具体使用方式。

## DETR 与 ACT 的连接

ACT 的官方实现直接修改自 DETR codebase，并借用了 learned decoder query slots。

但 ACT 把 query 的语义从“object prediction slot”改成“future action position”。这是一种架构复用：

```text
DETR object query
→ object output slot

ACT action query
→ future action output slot
```

所以 ACT 中的 learned action-query slots 更直接继承自 DETR 的 Object Query 思路，而不是原始 Transformer 自带的标准输出 token。

## Sources

- Carion et al. *End-to-End Object Detection with Transformers*. ECCV 2020.
- Official DETR repository: facebookresearch/detr.
