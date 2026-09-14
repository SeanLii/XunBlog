---
title: "Architecture"
kind: "act-topic"
domain: "Robot Learning / ACT"
parent: "ACT"
canonical: "/robot-learning/act/architecture/"
prerequisites:
  - "/robot-learning/act/"
related:
  - "/robot-learning/act/cvae-in-act/"
  - "/robot-learning/act/vision-pipeline/"
  - "/robot-learning/act/training/"
---

# Architecture

ACT 的 architecture 可以分成两个相互连接但用途不同的部分：training-only CVAE encoder，以及在训练和推理中都使用的 policy network。前者从 demonstration action chunk 推断 latent distribution；后者从当前 observation 与 latent $z$ 预测 future action chunk。

## Tensor Conventions

以下使用论文默认设置说明 shape：

\[
k=100,
\qquad
d_{model}=512,
\qquad
d_a=d_q=14.
\]

其中 $k$ 是 chunk size，$d_a$ 是 action dimension，$d_q$ 是 proprioceptive joint-position dimension。

为了看清单样本的数据流，正文省略 batch dimension；实现中所有 tensor 都还有 batch axis。

## Training-only CVAE Encoder

训练时有 current joint positions

\[
q_t\in\mathbb R^{14}
\]

和 ground-truth future action chunk

\[
A_t\in\mathbb R^{k\times14}.
\]

它们分别经过 linear projections 变成 $d_{model}$ 维 features，并在最前面加一个 learned [CLS Token](/deep-learning/transformer/cls-token/)。因此 sequence length 为

\[
1+1+k=k+2.
\]

也就是

\[
X_{cvae}\in\mathbb R^{(k+2)\times512}.
\]

加上 sinusoidal positional encoding 后，sequence 进入 BERT-like [Transformer Encoder](/deep-learning/transformer/transformer-encoder/)。只取最终 CLS position 的 hidden state：

\[
h_{CLS}\in\mathbb R^{512}.
\]

然后用 projection 输出 diagonal Gaussian parameters。Released implementation 固定

\[
d_z=32,
\]

因此得到

\[
\mu,\log\sigma^2\in\mathbb R^{32}.
\]

通过 [Reparameterization Trick](/generative-models/reparameterization-trick/) 得到

\[
z\in\mathbb R^{32}.
\]

这个 encoder 在 inference 时完全不用。

## Vision Backbone

当前 observation 含四张

\[
480\times640\times3
\]

RGB images。论文中的 ResNet18 将每张图转换为

\[
15\times20\times512
\]

feature map。Flatten 空间维后，每个 camera 得到

\[
300\times512.
\]

四个 camera 共

\[
1200\times512
\]

visual features，并加入 2D sinusoidal positional encoding。完整图像处理属于 [Vision Pipeline](/robot-learning/act/vision-pipeline/)。

## Observation Transformer Encoder

Current qpos 与 sampled $z$ 分别经过 linear projection 到 512 维：

\[
e_q\in\mathbb R^{512},
\qquad
e_z\in\mathbb R^{512}.
\]

论文把它们与 1200 个 visual features 合并，所以 encoder input 为

\[
X_{obs}\in\mathbb R^{1202\times512}.
\]

[Transformer Encoder](/deep-learning/transformer/transformer-encoder/) 对这些 features 做 self-attention，使来自不同 camera、proprioception 与 latent 的信息进入同一 memory。

## Action Queries 与 Transformer Decoder

ACT 为 $k$ 个 action slots 定义 [Learnable Query Embedding](/deep-learning/transformer/learnable-query-embedding/)：

\[
E_{query}\in\mathbb R^{k\times512}.
\]

它们不是过去的 action tokens，也不是 ground-truth future actions。它们是一组固定数量的可学习输出槽位。

Transformer decoder 使用这些 query positions，并通过 [Cross-Attention](/deep-learning/transformer/attention/cross-attention/) 读取 observation encoder memory。Decoder 输出

\[
H\in\mathbb R^{k\times512}.
\]

再投影成

\[
\hat A_t\in\mathbb R^{k\times14}.
\]

这 $k$ 行分别对应未来 action chunk 中的各 action slots。

## Non-autoregressive Action Output

ACT 并不是先预测 $\hat a_t$，再把它作为输入预测 $\hat a_{t+1}$。$k$ 个 action slots 在一次 forward 中产生，并在 Transformer decoder 内通过 query self-attention 与 cross-attention 建立关系。

因此它虽然使用 Transformer Decoder，却不应该被解释成语言模型式的 token-by-token autoregressive decoder。

## Training 与 Inference 的结构差异

训练：

```text
(q_t, A_t) → CVAE encoder → μ, logσ² → z
(images_t, q_t, z) → policy → Â_t
```

推理：

```text
CVAE encoder removed
z = 0
(images_t, q_t, z) → policy → Â_t
```

这种不对称是 CVAE training 的设计结果，而不是漏掉了一个 inference module。

## Paper 与 Code 的边界

论文给出概念 architecture 和默认 shapes；released code 决定实际 tensor ordering、shared backbone、output head 等实现细节。两者并非所有地方完全一致。具体差异单独记录在 [Paper and Released Implementation](/robot-learning/act/paper-and-released-implementation/)，避免把 implementation behavior 偷换成论文定义。

## Sources

- [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware — Zhao et al., 2023](https://arxiv.org/abs/2304.13705)
- [ACT official implementation — tonyzhaozh/act](https://github.com/tonyzhaozh/act)
