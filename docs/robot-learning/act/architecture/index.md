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

ACT architecture 包含两条职责不同的数据流：

1. **policy predictor**：根据当前 observation 与 latent condition 预测 future action chunk；
2. **training-only latent encoder**：根据 current qpos 与 ground-truth action chunk 构造 approximate posterior $q_\phi(z\mid\cdot)$。

部署时只保留第一条主干，第二条只参与 training。

## Policy Predictor

```text
multi-camera RGB images
          ↓
     ResNet backbone
          ↓
spatial visual features ────────┐
                                │
current qpos → projection ──────┤
                                │
latent z → projection ──────────┤
                                ↓
                     Transformer encoder
                                ↓
                             memory
                                ↑
                        action queries
                                ↓
                     Transformer decoder
                                ↓
                         action head
                                ↓
                 [a_t ... a_t+k-1]
```

Observation-side features 构成 encoder memory，$k$ 个 learned action queries 则定义 decoder 的 $k$ 个 future output slots。

## Visual Features

每个 camera image 由 [ResNet](/deep-learning/cnn/resnet/) backbone 提取 feature map，再用 1×1 convolution 投影到 Transformer hidden dimension。

多相机 features 与相应 positional representations 在 released code 中沿 spatial width 方向拼接。Transformer 因此接收具有空间结构的 visual feature grid，而不是单个 global image vector。

完整视觉路径见 [Vision Pipeline](/robot-learning/act/vision-pipeline/)。

## Proprioception and Latent Conditioning

Current joint positions $q_t$ 经 Linear Layer 投影到 hidden dimension。Latent $z$ 也由独立 projection 映射到相同 dimension。

官方实现为 proprioception feature 与 latent feature 配置额外 learned position embeddings，使 Transformer 能区分这两类非图像 input tokens 的结构身份。

Policy encoder 的信息源可以抽象为

\[
\{\text{visual features},\text{proprioceptive feature},\text{latent feature}\}.
\]

## Action Queries

ACT 继承 [DETR](/deep-learning/detr/) 的 learned output-query pattern。设 chunk length 为 $k$，模型维护

\[
Q_{action}\in\mathbb R^{k\times d_{model}}.
\]

这些 embeddings 不是 future action values，而是 decoder 的 learned output slots。经过 decoder 后得到

\[
H_{dec}\in\mathbb R^{B\times k\times d_{model}},
\]

再通过 action head 映射到

\[
\hat A\in\mathbb R^{B\times k\times d_a}.
\]

第 $j$ 个 output slot 对应 action chunk 中第 $j$ 个 temporal position。

## Training-Only Latent Encoder

训练时 ground-truth future action chunk 已知。ACT 将以下 sequence 输入另一套 Transformer encoder：

```text
[CLS] , current qpos , a_t , a_t+1 , ... , a_t+k-1
```

这些 inputs 先分别投影到 latent encoder hidden dimension。Encoder 最终取 [CLS Token](/deep-learning/bert/cls-token/) representation，并通过 Linear Layer 输出

\[
(\mu,\log\sigma^2).
\]

由此定义 diagonal Gaussian approximate posterior：

\[
q_\phi(z\mid q_t,A_t)
=
\mathcal N(\mu,\operatorname{diag}(\sigma^2)).
\]

released code 的 latent dimension 为 32。该数值属于具体 implementation configuration，不是 CVAE 的理论要求。

## Two Transformer Modules

ACT 中存在两套不同 Transformer computations：

- **latent encoder Transformer**：只在训练时运行，用 qpos 与 ground-truth actions 推断 latent posterior；
- **policy Transformer encoder–decoder**：训练和 inference 都运行，根据 observation 与 latent condition 预测 action chunk。

二者 parameters 与职责不同。Deployment graph 删除 latent encoder，只保留 policy predictor，并设置 $z=0$。

## Architecture Boundaries

ACT 复用了 ResNet、Transformer、CLS-like summary token 与 DETR-style output queries，但这些通用机制不是 ACT 的独立理论贡献。

ACT architecture 的特定组合关系是：

\[
\text{demonstration latent modeling}
+
\text{observation memory}
+
\text{learned action queries}
\rightarrow
\text{future action chunk}.
\]

Action Chunking 与 Temporal Ensemble 又分别规定了 prediction unit 与 deployment-time aggregation，因此 architecture 只是完整 ACT policy 的一个层面。

## Sources

- Zhao et al. *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware*. 2023. https://arxiv.org/abs/2304.13705
- Official model implementation: https://github.com/tonyzhaozh/act/blob/main/detr/models/detr_vae.py
- Carion et al. *End-to-End Object Detection with Transformers*. 2020.
