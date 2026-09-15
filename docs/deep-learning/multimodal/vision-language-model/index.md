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

Vision-Language Model（VLM）是一类联合建模 visual input 与 natural language 的模型。它们学习两种 modality 之间的对应关系，使图像与文本能够被对齐、融合、检索或用于条件生成。

VLM 不是单一 architecture。不同模型可以采用完全不同的信息流：dual encoders、cross-attention fusion、vision encoder + language decoder、multimodal decoder-only Transformer 等都属于这一范围。

## Modal Representations

图像通常先由 visual encoder 转换成 feature vectors：

\[
V=(v_1,\ldots,v_M),
\qquad v_i\in\mathbb R^{d_v}.
\]

文本由 tokenizer 与 text encoder / language model 转换成

\[
T=(t_1,\ldots,t_N),
\qquad t_j\in\mathbb R^{d_t}.
\]

两种 modality 在 token 数量、hidden dimension、统计分布与语义结构上都可能不同。VLM architecture 必须规定它们如何进入可联合学习的 representation space。

## Dual-Encoder Models

Dual-encoder architecture 分别编码 image 与 text：

```text
image → image encoder → image embedding
text  → text encoder  → text embedding
```

CLIP 是代表性模型。它对一个 batch 中 matching image-text pairs 提高 similarity，同时降低 mismatched pairs 的 similarity。

设 normalized image embedding 为 $u_i$，text embedding 为 $v_j$，相似度 logits 可以写成

\[
s_{ij}=\frac{u_i^\top v_j}{\tau},
\]

再对 image-to-text 与 text-to-image 两个方向使用 cross-entropy objective。

Dual encoders 的优势是 image / text 可以离线独立编码，因此适合 large-scale retrieval 与 zero-shot classification；限制是两种 modality 在 encoder 内部没有进行细粒度 token-level interaction。

## Fusion Models

Fusion architecture 允许 visual tokens 与 text tokens 在网络内部直接交互。常见方式包括：

- concatenation 后使用 joint self-attention；
- text queries cross-attend visual features；
- visual features 通过 gated cross-attention 注入 language model；
- 用 multimodal connector 把视觉表示转换为 language-model-compatible tokens。

这类模型能够根据语言 context 动态读取不同 image regions，更适合 visual question answering、captioning 与 multimodal dialogue。

## Visual Encoder and Projector

现代 generative VLM 常把预训练 visual encoder 与 language model 连接起来。若 visual output dimension 与 language-model hidden size 不同，需要 projector / adapter：

\[
V' = g_\phi(V),
\qquad
V'\in\mathbb R^{M\times d_{LM}}.
\]

Projector 可以是 Linear Layer、MLP、cross-attention resampler 或更复杂 connector。它的作用是建立 compatible representation interface，而不是单独完成全部 visual-language alignment。

## Training Objectives

VLM 的 objective 取决于 architecture 与任务。

### Contrastive Alignment

学习 matching image-text pairs 的 shared embedding geometry，例如 CLIP。

### Image–Text Matching

给定 image 与 text，预测它们是否匹配。

### Captioning / Autoregressive Language Modeling

以 image features 为 condition，最大化 text sequence likelihood：

\[
\log p(y_1,\ldots,y_T\mid I)
=
\sum_t
\log p(y_t\mid y_{<t},I).
\]

### Masked Multimodal Objectives

对文本或视觉 representations 做 masked prediction，使模型利用 cross-modal context 恢复缺失信息。

没有单一 loss 可以定义全部 VLM；关键是 objective 是否使 vision 与 language 建立可用于下游任务的联合结构。

## Generative and Non-Generative VLMs

VLM 的输出不一定是 text。

非生成式模型可以输出：

- image-text similarity；
- retrieval embedding；
- classification score；
- grounding score。

生成式 VLM 则通常通过 language decoder / LLM 产生自然语言 tokens，用于 captioning、VQA、dialogue 等任务。

因此 “VLM = 看图说话模型” 只覆盖其中一类。

## Language Grounding in Vision

VLM 的核心能力之一是让 language concepts 与 visual evidence 建立对应关系。这个 grounding 可以是 global-level（整图与文本对齐），也可以是 region / token-level interaction。

Global contrastive alignment 能支持 zero-shot classification 与 retrieval，但不保证模型具有精确 object-level grounding；细粒度 grounding 通常需要更强的 token interaction、region supervision 或专门 objective。

## Pretraining and Transfer

大规模 image-text pairs 为 VLM 提供弱监督。预训练可以学习跨任务可迁移的 visual-language representations，再通过 prompting、linear probing 或 fine-tuning 适配下游任务。

CLIP 展示了 natural-language supervision 对 zero-shot visual classification 的可迁移性；Flamingo 等模型进一步展示了 pretrained vision / language components 与 multimodal cross-attention 在 few-shot multimodal generation 中的能力。

## Evaluation

VLM 的 evaluation 必须与输出类型对应，例如：

- image-text retrieval recall；
- zero-shot classification accuracy；
- VQA accuracy；
- caption metrics；
- grounding accuracy；
- multimodal reasoning benchmarks。

单一 benchmark 无法完整表示 VLM 的视觉识别、语言生成、grounding 与 reasoning 能力。

## Limitations

VLM 的主要限制包括：

- image-text web data 中的噪声与偏差；
- global alignment 不等于细粒度 grounding；
- generative VLM 可能产生与视觉证据不一致的 hallucination；
- high-resolution images 与 long multimodal context 带来显著计算成本；
- language prior 可能在视觉证据不足时主导输出。

因此“接入一个视觉 encoder”并不自动得到可靠的 multimodal reasoning system。

## From VLM to VLA

[Vision-Language-Action Model](/robot-learning/vision-language-action-model/) 在 vision-language modeling 之外增加 robot state、action representation、temporal control 与 embodiment constraints。

VLM 可以提供 visual semantics 与 language grounding，但 motor control 还需要学习

\[
p(a_{t:t+H}\mid o_t,l,\ldots),
\]

或其他 action-generation objective。VLA 因此是建立在 multimodal representation 上、面向 embodied control 的模型 family，而不是 VLM 的同义词。

## Sources

- Radford et al. *Learning Transferable Visual Models From Natural Language Supervision*. 2021. https://arxiv.org/abs/2103.00020
- Alayrac et al. *Flamingo: a Visual Language Model for Few-Shot Learning*. 2022. https://arxiv.org/abs/2204.14198
- Li et al. *BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation*. 2022.
