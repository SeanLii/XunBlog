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
---

# CVAE in ACT

ACT 使用 CVAE 的目的不是“生成随机机器人动作”，而是在训练阶段给 action-chunk predictor 加一个 latent variable $z$，让它能够表示 demonstration 中没有被当前 observation 完全决定的变化。

这页只解释 **CVAE 在 ACT 里具体对应什么**。通用的 conditional ELBO、prior、posterior 与 generation 关系见 [Conditional Variational Autoencoder](/generative-models/conditional-variational-autoencoder/)。

## ACT 中的 x、y、z 对应关系

把通用 CVAE 写成

\[
q_\phi(z\mid x,y),
\qquad
p_\theta(y\mid x,z).
\]

在 ACT 中可以对应为：

- condition $x$：当前 robot observation，尤其包括 image 与 current qpos；
- output $y$：未来 action chunk；
- latent $z$：用于表示 demonstration style / variation 的隐藏变量。

因此 training-time recognition path 近似为

\[
q_\phi(z\mid q_t,a_{t:t+k-1}).
\]

注意官方 latent encoder 并没有把 camera images 送进去；它使用 qpos 与 ground-truth action sequence 推断 $z$。

## Training branch 怎样得到 z

官方模型先把 action sequence 与 qpos 投影到 Transformer hidden dimension，并在最前面加入一个 [CLS Token](/deep-learning/bert/cls-token/)：

```text
[CLS]  q_t  a_t  a_t+1  ...  a_t+k-1
  │      │    │     │             │
  └──────┴────┴─────┴─────────────┘
                    ↓
          Transformer Encoder
                    ↓
             CLS representation
                    ↓
              Linear layer
              ↙          ↘
             μ            log σ²
```

于是 approximate posterior 写成

\[
q_\phi(z\mid q_t,a_{t:t+k-1})
=
\mathcal N(\mu,\operatorname{diag}(\sigma^2)).
\]

然后使用 reparameterization：

\[
z=\mu+\sigma\odot\epsilon,
\qquad
\epsilon\sim\mathcal N(0,I).
\]

## z 怎样进入 action predictor

Sampled $z$ 不直接当作 32 维 action。它先经过 linear projection 变成 Transformer hidden dimension 的 feature，再和 proprioception、visual features 一起进入 policy Transformer。

```text
z (latent_dim=32 in released code)
          │
          ↓ linear projection
latent feature (hidden_dim)
          │
          ├─────────────┐
qpos feature ───────────┤
visual features ────────┤
                        ↓
                policy Transformer
                        ↓
                 action chunk
```

因此 $z$ 是一个 condition signal，而不是未来轨迹本身。

## KL 对 latent posterior 的约束

ACT 使用 standard normal prior：

\[
p(z)=\mathcal N(0,I).
\]

训练时加入

\[
D_{KL}\left(q_\phi(z\mid q_t,a_{t:t+k-1})\|\mathcal N(0,I)\right).
\]

它推动 training-time posterior 不要任意远离 standard normal。

这件事对 inference 很重要，因为部署时没有真实 future action，不能再运行 recognition encoder。模型只能从 prior 侧决定 $z$。

## ACT 的确定性推理选择

通用 CVAE 在 generation 时完全可以从 prior 采样不同 $z$，得到多样输出。

ACT 的 released inference 不是这么做。它直接令

\[
z=0,
\]

也就是 standard normal 的 mean，并得到 deterministic action prediction。

所以 ACT 使用 CVAE training，不意味着部署时 robot policy 必然是 stochastic 的。

更完整解释见 [为什么 ACT 推理时令 z = 0？](/robot-learning/act/why-z-zero-at-inference/)。

## 这条支路什么时候存在

最容易画错的一点可以用两张图固定下来。

Training：

```text
true action chunk ─┐
current qpos ──────┤→ latent encoder → z
                   │
images + qpos ─────┴──────────────→ policy → reconstructed chunk
```

Inference：

```text
z = 0 ─────────────┐
images + qpos ─────┴──────────────→ policy → predicted chunk
```

部署时 ground-truth future action 根本不存在，因此任何把它放进 inference graph 的解释都是错误的。

## Sources

- Zhao et al., **Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware**, 2023. https://arxiv.org/abs/2304.13705
- Official implementation: https://github.com/tonyzhaozh/act/blob/main/detr/models/detr_vae.py
