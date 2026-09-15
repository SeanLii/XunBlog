---
title: "Training"
kind: "act-topic"
domain: "Robot Learning / ACT"
parent: "ACT"
canonical: "/robot-learning/act/training/"
prerequisites:
  - "/robot-learning/act/architecture/"
  - "/robot-learning/act/cvae-in-act/"
related:
  - "/robot-learning/act/inference/"
  - "/robot-learning/act/paper-and-released-implementation/"
standard: "XunBlog Content & Knowledge Architecture v1.0"
rebuilt: "2026-09-15"
---
# Training

ACT training 同时优化 action reconstruction 与 latent regularization。每个样本包含当前 observation 和 demonstration 中从该时刻开始的 future action chunk；ground-truth chunk 既是 prediction target，也参与 training-only latent posterior inference。

## Training Sample

从 demonstration trajectory 选择时刻 $t$，构造：

- current images $I_t$；
- current joint positions $q_t$；
- future action chunk $A_t=a_{t:t+k-1}$；
- padding mask $m_t$。

可写成

\[
(I_t,q_t,A_t,m_t).
\]

当 episode 尾部剩余动作不足 $k$ 时，dataset 将 sequence pad 到固定长度，并记录哪些 positions 是无效 padding。

## Posterior Inference

Training-only latent encoder 根据 qpos 与 ground-truth chunk 预测 diagonal Gaussian parameters：

\[
(\mu_t,\log\sigma_t^2)
=f_\phi(q_t,A_t).
\]

随后

\[
\epsilon\sim\mathcal N(0,I),
\qquad
z_t=\mu_t+\sigma_t\odot\epsilon.
\]

Ground-truth future action 进入该分支是 variational training 的一部分；inference 阶段因为未来动作未知，这条 branch 会被移除。

## Action-Chunk Prediction

Visual images 经过 backbone，qpos 与 latent 经过 projection 后进入 policy Transformer：

\[
\hat A_t=g_\theta(I_t,q_t,z_t).
\]

输出 shape 为

\[
\hat A_t\in\mathbb R^{B\times k\times d_a}.
\]

ALOHA released configuration 中 state / action dimension 为 14。

## Reconstruction Loss

Released `policy.py` 对有效 positions 使用 element-wise L1 loss：

\[
\mathcal L_{recon}
=
\operatorname{masked\;L1}(A_t,\hat A_t).
\]

Padding positions 不参与 reconstruction objective。

论文 Algorithm 1 中 reconstruction term 的伪代码写成 MSE，而方法正文说明最终使用 L1；released code 也实现 L1。该差异单独记录在 [Paper and Released Implementation](/robot-learning/act/paper-and-released-implementation/)。

## KL Loss

Approximate posterior 被约束接近 standard normal prior：

\[
\mathcal L_{KL}
=
D_{KL}
\left(
q_\phi(z\mid q_t,A_t)
\|\mathcal N(0,I)
\right).
\]

总 objective 为

\[
\mathcal L
=
\mathcal L_{recon}
+
\beta\mathcal L_{KL}.
\]

论文主要 ACT configuration 使用 $\beta=10$。

## Padding Mask

固定 chunk length 要求 episode 尾部补齐 missing positions。若 padding values 参与 loss，模型会被迫拟合人工填充值，因此 mask 必须同时影响 reconstruction 与 latent sequence processing。

在 latent Transformer encoder 中，padding mask 也防止 padded action tokens 被当作真实 future sequence information。

## Optimization Configuration

Released ACT training 使用 ResNet18 backbone，并为 backbone 设置较小 learning rate。Optimizer build path 继承自 DETR codebase，使用 AdamW-style parameter groups。

这些属于具体 released implementation 的 optimization choices，而不是 ACT 算法定义。论文对 optimizer 的文字描述与 released code 也存在差异，见对应 discrepancy 页面。

## Training Computation Graph

```text
                 ground-truth future chunk A_t
                     │                 │
                     │                 └──────── target
                     ↓
q_t ───────────→ latent encoder
                     ↓
                 μ , logσ²
                     ↓
                  sample z
                     │
images → ResNet ─────┤
q_t ─────────────────┤
                     ↓
              policy Transformer
                     ↓
                 predicted A_t
                     │
             ┌───────┴────────┐
             ↓                ↓
          L1 loss           KL loss
             └───────┬────────┘
                     ↓
                 total loss
```

Training 的关键结构是：future action chunk 同时参与 approximate posterior inference 与 reconstruction supervision，而 deployment 只保留 policy predictor。

## Sources

- Zhao et al. *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware*. 2023. https://arxiv.org/abs/2304.13705
- Official loss implementation: https://github.com/tonyzhaozh/act/blob/main/policy.py
- Official model implementation: https://github.com/tonyzhaozh/act/blob/main/detr/models/detr_vae.py
