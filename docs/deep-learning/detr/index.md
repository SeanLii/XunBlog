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

DETR（DEtection TRansformer）把 object detection 表述为 **direct set prediction**：模型从一张图像直接输出一个固定大小的预测集合，每个 slot 给出类别与 bounding box，而不再依赖 anchor generation、proposal refinement 与 non-maximum suppression（NMS）构成多阶段检测流程。

整体数据流为

```text
image
  ↓
CNN backbone
  ↓
spatial feature map + positional encoding
  ↓
Transformer encoder
  ↓
image memory
  ↑
learned object queries
  ↓
Transformer decoder
  ↓
N object representations
  ↓
class head + box head
  ↓
set of detections
```

DETR 的关键组成是 **set-based bipartite matching loss** 与 **Transformer encoder–decoder + learned object queries**。

## Set Prediction Formulation

给定图像 $x$，真实 objects 构成无序集合

\[
Y=\{y_1,\ldots,y_M\}.
\]

DETR 产生固定数量 $N$ 的 predictions：

\[
\hat Y=\{\hat y_1,\ldots,\hat y_N\},
\qquad N\ge M.
\]

每个 prediction 包含：

\[
\hat y_j=(\hat p_j,\hat b_j),
\]

其中 $\hat p_j$ 是类别分布，$\hat b_j$ 是 normalized bounding box。

因为 object set 没有天然顺序，training 不能预先规定第 $j$ 个 query 必须对应某个固定 ground-truth object。

## CNN Backbone and Feature Sequence

原始 DETR 使用 CNN backbone（例如 ResNet）提取低分辨率 feature map：

\[
f\in\mathbb R^{C\times H'\times W'}.
\]

通过 1×1 convolution 投影到 Transformer hidden dimension，再把 spatial grid flatten 成 sequence：

\[
X\in\mathbb R^{H'W'\times d_{model}}.
\]

2D positional encoding 被加入这些 spatial features，使 Transformer 保留图像位置结构。

## Transformer Encoder

Encoder 对所有 spatial feature positions 做 global self-attention，形成带有全局 context 的 image memory。

与只依赖 local convolution receptive field 的交互不同，encoder 中任意两个 spatial positions 可以通过 attention 直接建立关系。

Encoder 输出不会直接一一对应 objects；它仍然是 image feature memory。

## Object Queries and Decoder

Decoder 接收固定数量的 learned [Object Query](/deep-learning/detr/object-query/)。每个 query 对应一个 prediction slot，而不是固定 object category。

Decoder layers 让 queries：

1. 通过 self-attention 彼此交互；
2. 通过 cross-attention 读取 image memory；
3. 逐层形成 object-level hidden representations。

最终每个 query slot 通过两个 prediction heads 输出 class 与 box。

## Bipartite Matching

DETR 使用 Hungarian algorithm 在 ground-truth objects 与 predictions 之间寻找一对一 assignment。

设一个 permutation / assignment 为 $\sigma$，matching cost 综合类别与 box information。训练先求

\[
\hat\sigma
=
\arg\min_\sigma
\sum_{i=1}^{N}
\mathcal C_{match}(y_i,\hat y_{\sigma(i)}).
\]

Ground-truth set 会 padding 到 $N$ 个 entries，其中未对应真实 object 的 slots 标记为 no-object class。

一对一 assignment 的作用是防止多个 query slots 同时被奖励为同一个 ground-truth object，从而把“每个 object 只预测一次”的约束直接放入 training objective。

## Set Prediction Loss

在得到最优 matching 后，DETR 对 matched pairs 计算 Hungarian loss。

分类部分使用 negative log-likelihood；box 部分结合 L1 loss 与 generalized IoU（GIoU）loss：

\[
\mathcal L_{box}
=
\lambda_{L1}\lVert b-\hat b\rVert_1
+
\lambda_{giou}\mathcal L_{GIoU}(b,\hat b).
\]

L1 直接约束坐标误差，GIoU 提供与 box overlap 更一致的几何 signal。两者结合避免只依赖某一种坐标尺度或 overlap criterion。

## No-Object Class

因为输出 slot 数 $N$ 固定，而图像中的真实 objects 数 $M$ 可变，未匹配到真实 object 的 queries 被训练为特殊的 no-object 类别。

因此 DETR 不需要动态改变 decoder queries 数量：固定 prediction slots + no-object class 就能表示不同 object count。

## Auxiliary Decoder Losses

原始 DETR 在 decoder 的中间 layers 上也加入 prediction heads 与 auxiliary losses。这样每层 decoder 都获得直接 supervision，有助于 optimization。

这些 auxiliary heads 在参数上可以共享，并不改变最终 set prediction definition；它们主要属于 training strategy。

## Why NMS Is Not Required

传统 detectors 经常产生大量重叠候选框，再通过 NMS 删除重复 detections。

DETR 的 one-to-one bipartite matching 直接要求 prediction slots 与 ground-truth objects 建立唯一对应，因此模型在训练中就受到“不要重复预测同一 object”的结构性约束。最终 inference 可以直接读取 slot predictions，不需要标准 NMS pipeline。

## Training and Convergence

原始 DETR 在 COCO 上取得与当时 Faster R-CNN baseline 可比的整体表现，但存在训练收敛较慢、对 small objects 表现较弱等问题。后续 Deformable DETR 等工作主要针对 attention sparsity、multi-scale features 与 convergence efficiency 进行改进。

这些后续方法说明 direct set prediction framework 很有扩展性，但原始 DETR 本身的定义仍是 fixed queries + Transformer + bipartite set loss。

## Relation to ACT

ACT 官方实现基于 DETR codebase 改造，并继承 learned decoder query slots 的 architecture pattern。

DETR 中：

```text
object query
→ object prediction slot
```

ACT 中：

```text
action query
→ future action position
```

这种复用说明 learned queries 可以作为并行输出 slots，但 ACT 的 Action Chunking、CVAE latent branch 与 Temporal Ensemble 都不属于 DETR。

## Sources

- Carion et al. *End-to-End Object Detection with Transformers*. ECCV 2020. https://arxiv.org/abs/2005.12872
- Rezatofighi et al. *Generalized Intersection over Union: A Metric and A Loss for Bounding Box Regression*. 2019.
- Official DETR repository: https://github.com/facebookresearch/detr
