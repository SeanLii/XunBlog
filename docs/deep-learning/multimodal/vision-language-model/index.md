---
title: "Vision-Language Model"
kind: "canonical"
domain: "Deep Learning / Multimodal Models"
parent: "Multimodal Models"
canonical: "/deep-learning/multimodal/vision-language-model/"
prerequisites:
  - "/deep-learning/core/embedding/"
  - "/deep-learning/transformer/"
related:
  - "/robot-learning/vision-language-action-model/"
---

# Vision-Language Model

Vision-Language Model（VLM）是一类同时处理 visual information 与 natural language 的模型。它的核心不是简单“把图片和文字都塞进 Transformer”，而是建立两种 modality 之间可以共同学习、对齐或交互的 representation 与 prediction mechanism。

一个 VLM 可能完成：

- image-text matching；
- image captioning；
- visual question answering；
- multimodal dialogue；
- visual grounding；
- zero-shot visual classification。

因此 VLM 是一个 model family，而不是某一种固定 architecture。

## 两种 Modality 最开始并不在同一个表示空间

文本通常先变成 token embeddings：

\[
t_1,\ldots,t_N\in\mathbb R^{d_t}.
\]

图像则可能经过 CNN 或 Vision Transformer 得到 visual features：

\[
v_1,\ldots,v_M\in\mathbb R^{d_v}.
\]

问题是：这些 features 的数量、dimension、统计性质和语义来源都可能不同。

所以 multimodal model 需要解决至少两件事：

1. 怎样把 vision / language 表示成模型可联合处理的形式；
2. 怎样让两种 representations 建立语义对应关系。

## Dual-Encoder 路线

CLIP 是典型 dual-encoder 思路：

```text
image → image encoder → image embedding
text  → text encoder  → text embedding
```

训练目标让 matching image-text pairs 的 embeddings 更接近，不匹配 pairs 更远。

这类模型非常适合 retrieval 与 zero-shot classification，因为 image / text 可以分别编码再比较。

但两个 modalities 在 encoder 内部并没有进行 token-level deep interaction。

## Fusion 路线

另一类模型会让 visual tokens 与 language tokens 在 joint Transformer 或 cross-attention 中直接交互：

```text
visual tokens ─┐
               ├→ multimodal fusion → output
text tokens ───┘
```

这样语言可以针对具体 visual regions 读取信息，视觉表示也可以被语言 context 调整。

现代 multimodal LLM 常把 visual encoder output 通过 projector 映射到 language model hidden space，再与 text tokens 一起处理。

## Training Objectives

不同 VLM 使用不同 objectives，例如：

- contrastive image-text alignment；
- image-text matching；
- masked language / masked image modeling；
- caption generation；
- next-token prediction on multimodal sequences。

因此不能用一个 loss 定义全部 VLM。

更稳定的定义是：

> **模型同时接收或学习 vision 与 language，并通过联合训练使两种 modality 在同一个任务中建立可利用的关系。**

## VLM 输出不一定是 Text

有些 VLM 输出 text tokens；有些输出 similarity score；有些产生 multimodal embedding；还有些输出 region grounding 或其他 structured predictions。

所以“VLM = 看图说话模型”太窄。

## 从 VLM 到 VLA

[Vision-Language-Action Model](/robot-learning/vision-language-action-model/) 进一步要求模型不仅理解图像和语言，还要产生 robot actions。

VLM 已经提供：

- visual semantics；
- language grounding；
- cross-modal representation。

但这不自动等于 motor control。VLA 还需要处理：

- robot state；
- continuous / discrete action space；
- temporal control；
- embodiment differences；
- closed-loop interaction。

因此 VLA 可以建立在 VLM backbone 上，但 VLM 自身是更广泛的 multimodal learning topic。

## Sources

- Radford et al. *Learning Transferable Visual Models From Natural Language Supervision (CLIP)*. 2021.
- Dosovitskiy et al. *An Image Is Worth 16×16 Words*. 2021.（视觉 tokenization 的重要基础之一）
