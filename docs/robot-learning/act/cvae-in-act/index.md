---
title: "CVAE in ACT"
kind: "act-topic"
domain: "Robot Learning / ACT"
parent: "ACT"
canonical: "/robot-learning/act/cvae-in-act/"
prerequisites:
  - "/generative-models/conditional-variational-autoencoder/"
related:
  - "/robot-learning/act/why-z-zero-at-inference/"
  - "/robot-learning/act/training/"
standard: "XunBlog Content & Knowledge Architecture v1.0"
rebuilt: "2026-09-15"
---
# CVAE in ACT

ACT 在训练阶段使用 Conditional Variational Autoencoder 结构，为 action-chunk predictor 引入 latent variable $z$。该 latent 用于表示在当前 observation 条件下，demonstration action 中仍然存在的变化。

通用 CVAE 的 conditional ELBO、prior、posterior 与 generation 结构见 [Conditional Variational Autoencoder](/generative-models/conditional-variational-autoencoder/)。本页只描述这些变量在 ACT 中的具体对应。

## Variable Correspondence

通用 CVAE 可以写成

\[
q_\phi(z\mid x,y),
\qquad
p_\theta(y\mid x,z).
\]

在 ACT 中：

- condition $x$：当前 robot observation；
- output $y$：future action chunk $A_t$；
- latent $z$：demonstration variation 的隐藏表示。

ACT released model 的 recognition encoder 实际使用 current qpos 与 ground-truth actions：

\[
q_\phi(z\mid q_t,A_t).
\]

Camera images 不进入这条 training-only latent encoder。

## Recognition Encoder

Ground-truth action sequence 与 qpos 先被投影到 latent Transformer hidden dimension，并与 learned CLS-like token 组成 sequence：

```text
[CLS]  q_t  a_t  a_t+1  ...  a_t+k-1
                    ↓
          Transformer Encoder
                    ↓
               h_CLS
                    ↓
             Linear projection
             ↙              ↘
            μ              log σ²
```

由此定义

\[
q_\phi(z\mid q_t,A_t)
=
\mathcal N(\mu,\operatorname{diag}(\sigma^2)).
\]

随后通过 reparameterization

\[
z=\mu+\sigma\odot\epsilon,
\qquad
\epsilon\sim\mathcal N(0,I).
\]

## Latent Conditioning of the Policy

Sampled $z$ 不直接表示 action vector。released code 先将 latent projection 到 policy Transformer hidden dimension，再与 proprioception feature、visual memory 一起条件化 action prediction：

```text
z
↓ latent projection
latent feature ────────┐
qpos feature ──────────┤
visual features ───────┤
                       ↓
               policy Transformer
                       ↓
                  action chunk
```

因此 $z$ 是 policy condition，而不是 future trajectory 本身。

## KL Regularization

ACT 使用 standard normal prior

\[
p(z)=\mathcal N(0,I)
\]

并在训练 objective 中加入

\[
D_{KL}
\left(
q_\phi(z\mid q_t,A_t)
\|\mathcal N(0,I)
\right).
\]

这一项限制 training-time posterior 与 prior 的偏离程度，使 inference 阶段可以在没有 ground-truth future actions 时从 prior side 选择 latent condition。

## Deterministic Inference Choice

通用 CVAE generation 可以从 prior 随机采样 $z$，产生多个可能输出。ACT 的 released inference 没有采用这种 stochastic policy；它固定

\[
z=0,
\]

即 standard normal prior 的 mean。

因此 ACT 使用 CVAE-style training 并不意味着 deployment policy 必然随机。固定 latent 的具体含义见 [为什么 ACT 推理时令 z = 0？](/robot-learning/act/why-z-zero-at-inference/)。

## Training and Inference Graphs

Training：

```text
ground-truth action chunk ─┐
current qpos ──────────────┤
                           ↓
                    latent encoder
                           ↓
                           z
                           │
images + qpos ─────────────┤
                           ↓
                      ACT policy
                           ↓
                 reconstructed chunk
```

Inference：

```text
z = 0 ───────────────┐
images + qpos ───────┤
                     ↓
                ACT policy
                     ↓
              predicted chunk
```

Ground-truth future action 只属于 training graph；deployment 时该变量尚未发生，因此不会进入 inference computation。

## Sources

- Zhao et al. *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware*. 2023. https://arxiv.org/abs/2304.13705
- Official implementation: https://github.com/tonyzhaozh/act/blob/main/detr/models/detr_vae.py
