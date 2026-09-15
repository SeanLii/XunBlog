---
title: "BERT"
kind: "canonical"
domain: "Deep Learning / BERT"
parent: "Deep Learning"
canonical: "/deep-learning/bert/"
prerequisites:
  - "/deep-learning/transformer/transformer-encoder/"
  - "/deep-learning/attention/self-attention/"
  - "/deep-learning/sequence-modeling/positional-encoding/"
related:
  - "/deep-learning/bert/cls-token/"
---

# BERT

BERT（Bidirectional Encoder Representations from Transformers）是一个基于 Transformer Encoder 的预训练语言表示模型。它通过在大规模无标注文本上进行预训练，学习能够同时利用左右上下文的 token-level representations，再通过少量 task-specific parameters 与 fine-tuning 适配分类、序列标注、自然语言推断和问答等任务。

BERT 的整体结构可以写成

```text
WordPiece tokens
      ↓
token + position + segment embeddings
      ↓
stacked Transformer Encoders
      ↓
contextual token representations
      ↓
task-specific prediction head
```

因此 BERT 的核心创新不在于重新定义 Transformer Encoder，而在于 **bidirectional pre-training objective、input representation 与统一 fine-tuning framework** 的组合。

## Input Tokenization

原始 BERT 使用 WordPiece tokenization。一个词可以被拆成多个 subword units，因此 vocabulary 不需要为所有完整单词分别分配独立 token。

输入 sequence 最多由 512 个 tokens 构成。BERT 使用若干 special tokens：

- `[CLS]`：放在 sequence 开头；
- `[SEP]`：分隔 sentence pairs 或标记 sequence boundary；
- `[MASK]`：用于 Masked Language Modeling。

[CLS Token](/deep-learning/bert/cls-token/) 是 BERT 的具体输入设计，不是 Transformer architecture 的通用必要组件。

## Input Representation

位置 $i$ 的输入表示由三部分相加：

\[
e_i=e_i^{token}+e_i^{position}+e_i^{segment}.
\]

其中：

- token embedding 表示 WordPiece identity；
- position embedding 表示 absolute position；
- segment embedding 表示 token 属于 sentence A 还是 sentence B。

三部分具有相同 hidden dimension，因此可以逐元素相加后送入 encoder stack。

## Bidirectional Encoder

BERT 使用非 causal 的 Transformer Encoder self-attention。对某个 token position $i$，其 hidden state 可以同时依赖左侧和右侧 context：

\[
h_i=f(x_1,\ldots,x_N).
\]

这与 left-to-right autoregressive language model 的条件结构不同。BERT 的目标是学习上下文表示，而不是直接定义一个从左到右的 generative language model。

## Masked Language Modeling

BERT 的主要预训练目标是 Masked Language Modeling（MLM）。原论文随机选择 15% 的 token positions 作为 prediction targets。

对于被选中的位置：

- 80% 替换为 `[MASK]`；
- 10% 替换为一个随机 token；
- 10% 保持原 token 不变。

无论实际输入如何替换，这些 selected positions 都用于预测原始 token identity。

设被选位置集合为 $\mathcal M$，MLM loss 可以概括为

\[
\mathcal L_{MLM}
=-
\sum_{i\in\mathcal M}
\log p_\theta(x_i\mid \tilde x),
\]

其中 $\tilde x$ 是被 corruption 后的输入 sequence。

这种设计减少了 pre-training 中 `[MASK]` token 与 downstream fine-tuning 输入之间的完全不一致，因为一部分 prediction targets 并不会真的被替换为 `[MASK]`。

## Next Sentence Prediction

原始 BERT 还使用 Next Sentence Prediction（NSP）。训练样本由两段 text A 与 B 构成：

- 50% 情况下 B 是 A 在原文中的真实下一段；
- 50% 情况下 B 从 corpus 中随机采样。

模型使用 `[CLS]` representation 预测 `IsNext` / `NotNext`。

原论文把 NSP 与 MLM 一起用于 pre-training。后续工作发现 NSP 并非所有预训练方案都必需，因此不能把 NSP 当成现代 bidirectional encoder pretraining 的普遍定义；它属于原始 BERT training recipe。

## Pre-Training Corpus

原始 BERT 使用 BooksCorpus 与 English Wikipedia 进行预训练。预训练数据只需要原始文本，不要求人工标注下游任务 labels。

这种范式将学习过程拆成两阶段：

```text
large unlabeled text
      ↓
pre-training
      ↓
general contextual representations
      ↓
task-specific fine-tuning
```

## BERT Base and BERT Large

原论文给出两个主要配置。

BERT Base：

\[
L=12,
\qquad
H=768,
\qquad
A=12,
\]

约 110M parameters。

BERT Large：

\[
L=24,
\qquad
H=1024,
\qquad
A=16,
\]

约 340M parameters。

其中 $L$ 是 Transformer layers 数，$H$ 是 hidden size，$A$ 是 attention heads 数。

## Fine-Tuning

预训练后的 BERT 通常保留全部 encoder parameters，并根据下游任务添加轻量 prediction head，再联合 fine-tune。

### Sequence Classification

使用 `[CLS]` final representation：

\[
h_{CLS}
\rightarrow
\text{classifier}
\rightarrow
\text{label logits}.
\]

适用于 sentiment classification、natural language inference 等 sequence-level tasks。

### Token Classification

对每个 token hidden state 独立预测 label：

\[
h_i\rightarrow y_i.
\]

可用于 named entity recognition 等 sequence labeling tasks。

### Extractive Question Answering

对 passage 中每个 token 预测 answer span 的 start / end scores。模型主体仍是同一个 BERT encoder，只改变 output head 与 loss。

BERT 的重要贡献之一正是：多个 NLP tasks 可以共享同一套 pretrained architecture，而无需为每个任务重新设计复杂 task-specific network。

## BERT and Transformer

BERT 使用 Transformer Encoder，但二者不是同一个概念。

Transformer 定义 attention-based architecture；BERT 则定义了一个具体 encoder-only language representation model，包括 WordPiece input、learned position / segment embeddings、MLM、原始 NSP objective 与 fine-tuning procedure。

## Limitations

原始 BERT 存在几个明确边界：

- MLM 只在部分 positions 产生 token prediction signal；
- `[MASK]` 主要出现在 pre-training，而下游输入通常没有该 token；
- absolute position embedding 限制了预训练最大 sequence length；
- full self-attention 的计算成本随 sequence length 二次增长；
- BERT 本身不是自然的 left-to-right generative model。

后续 RoBERTa、ALBERT、DeBERTa 等模型分别修改了 data scale、training objective、parameter sharing、position modeling 等部分，但这些变体不属于 BERT 的定义本身。

## Relation to ACT

ACT 的 training-only latent encoder 使用一个 learnable CLS-like token 汇总 joint state 与 action sequence，再由该位置预测 latent distribution parameters。这借用了 summary token 的 architecture pattern，但 ACT 并不使用完整 BERT model，也不使用 BERT 的 MLM / NSP objectives。

## Sources

- Devlin et al. *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. 2018. https://arxiv.org/abs/1810.04805
