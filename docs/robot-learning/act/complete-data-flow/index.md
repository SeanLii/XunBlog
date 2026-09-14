---
title: "Complete Data Flow"
kind: "act-topic"
domain: "Robot Learning / ACT"
parent: "ACT"
canonical: "/robot-learning/act/complete-data-flow/"
prerequisites:
  - "/robot-learning/act/architecture/"
  - "/robot-learning/act/training/"
  - "/robot-learning/act/inference/"
related:
  - "/robot-learning/act/cvae-in-act/"
  - "/robot-learning/act/vision-pipeline/"
---

# Complete Data Flow

Complete Data Flow 把 ACT 的 training 与 inference 放在同一条数据链上。它的目的不是重复各模块理论，而是明确每个 tensor 从哪里来、经过什么模块、只在哪个阶段存在。

## Training Data Flow

以论文默认 $k=100$、hidden dimension 512、四相机、14-DoF action 为例。

### Demonstration sample

从 trajectory 的 timestep $t$ 取得：

\[
q_t\in\mathbb R^{14},
\]

\[
I_t^{1:4}\in\mathbb R^{4\times3\times480\times640},
\]

\[
A_t\in\mathbb R^{100\times14}.
\]

不足 100 个 future actions 时补 padding，并生成 `is_pad`。

### Normalization

\[
q_t
\rightarrow
\tilde q_t,
\qquad
A_t
\rightarrow
\tilde A_t.
\]

Images 缩放到 $[0,1]$，进入 policy 时再做 channel normalization。

### Training-only latent encoder

```text
[CLS]                       -> 512
q_t -> Linear               -> 512
a_1 ... a_100 -> Linear     -> 100 × 512
```

拼成

\[
102\times512
\]

sequence，加 sinusoidal positional encoding 后经过 Transformer encoder。

取 CLS output：

\[
h_{CLS}\in\mathbb R^{512}.
\]

投影得到

\[
\mu,\log\sigma^2\in\mathbb R^{32}.
\]

采样：

\[
\epsilon\sim\mathcal N(0,I),
\qquad
z=\mu+\sigma\odot\epsilon.
\]

### Vision features

每张 image：

\[
3\times480\times640
\xrightarrow{ResNet18}
512\times15\times20.
\]

空间 flatten 后每 camera 为

\[
300\times512.
\]

四 camera 合计

\[
1200\times512.
\]

并携带二维 positional encoding。

### Observation memory

qpos 和 latent 分别投影：

\[
\tilde q_t\rightarrow e_q\in\mathbb R^{512},
\]

\[
z\rightarrow e_z\in\mathbb R^{512}.
\]

与 visual features 合并为论文描述的

\[
1202\times512
\]

encoder input。Transformer encoder 输出 observation memory。

### Action queries

100 个 learnable query embeddings：

\[
E_q\in\mathbb R^{100\times512}.
\]

Transformer decoder 让这些 queries 读取 observation memory，得到

\[
H\in\mathbb R^{100\times512}.
\]

Action head 投影成

\[
\hat A_t\in\mathbb R^{100\times14}.
\]

### Loss

Released path：

\[
\mathcal L_{reconst}
=
\operatorname{masked\ L1}(\hat A_t,\tilde A_t),
\]

\[
\mathcal L_{KL}
=
D_{KL}
(q_\phi(z|A_t,q_t)\|\mathcal N(0,I)),
\]

\[
\mathcal L
=
\mathcal L_{reconst}
+\beta\mathcal L_{KL}.
\]

Gradient 同时更新 CVAE encoder、ResNet、observation Transformer、decoder queries 和 action head。

## Inference Data Flow

Inference 中整条 training-only left branch 消失：

```text
NO ground-truth future action chunk
NO CVAE encoder
NO μ / logσ²
NO reparameterized posterior sample
```

直接设置

\[
z=0.
\]

当前 observation 走 policy branch：

```text
4 RGB images ─> ResNet18 ─> visual features ─┐
q_t ─> normalize ─> Linear ─────────────────┤
z=0 ─> Linear ──────────────────────────────┤
                                             v
                                  Transformer Encoder
                                             |
100 learned action queries ------------------|
                                             v
                                  Transformer Decoder
                                             |
                                             v
                                    100 × 14 actions
```

## Execution Branch

如果 temporal ensemble 关闭：

```text
query once -> execute chunk slots sequentially -> query again
```

如果 temporal ensemble 开启：

```text
every timestep:
current observation
    -> predict new chunk
    -> collect all predictions for current physical timestep
    -> exponential weighted average
    -> denormalize
    -> execute one 14-D target action
```

随后环境返回下一 observation，闭环继续。

## Training/Inference Asymmetry

最容易混淆的一点是：训练时 action chunk 同时是 **target** 和 CVAE recognition encoder 的输入；推理时 action chunk 只能是 **model output**，不存在 target branch。

所以不能在 inference diagram 中继续保留“把 future action 输入 encoder 得到 $z$”这条路径。那会要求提前知道模型本来应该预测的答案，破坏实际推理条件。

## Module Ownership

这条数据流调用了 Transformer、ResNet、CVAE、Normal Distribution、KL Divergence 等知识，但它们的通用理论分别属于自己的 canonical pages。这里仅说明 ACT 在一次完整 forward/rollout 中怎样连接它们。

## Sources

- [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware — Zhao et al., 2023](https://arxiv.org/abs/2304.13705)
- [ACT official implementation — tonyzhaozh/act](https://github.com/tonyzhaozh/act)
