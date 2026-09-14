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
---

# Training

ACT training 的核心可以用一句话表示：

> 给定当前 observation 和 demonstration 中接下来的真实 action chunk，训练模型重建这段 action chunk，同时约束 latent posterior 接近 standard normal prior。

真正理解 training 时，必须同时看清 **target branch** 和 **latent branch**。

## 一个训练样本

从 demonstration trajectory 中选一个时刻 $t$。模型得到：

- current images $I_t$；
- current joint positions $q_t$；
- ground-truth future action chunk $A_t=a_{t:t+k-1}$；
- padding mask（如果 episode 尾部不足 $k$ 步）。

所以一条 training sample 不是只有 $(o_t,a_t)$，而更接近：

\[
(I_t,q_t,A_t,m_t).
\]

## 第一步：future action 先进入 latent encoder

Training-time CVAE branch 使用 current qpos 和真实 action chunk 推断 posterior parameters：

\[
(\mu_t,\log\sigma_t^2)
=f_\phi(q_t,A_t).
\]

然后

\[
\epsilon\sim\mathcal N(0,I),
\]

\[
z_t=\mu_t+\sigma_t\odot\epsilon.
\]

这里 $A_t$ 进入网络并不是“泄露答案给 inference”。训练阶段本来就允许用 ground truth 构造 latent posterior；部署阶段这条 encoder 会被删掉。

## 第二步：policy 根据 observation 与 z 预测整个 chunk

视觉图像经过 backbone，qpos 与 $z$ 被投影后进入 policy Transformer：

\[
\hat A_t
=g_\theta(I_t,q_t,z_t).
\]

输出 shape 可以写成

\[
\hat A_t\in\mathbb R^{B\times k\times d_a},
\]

其中 $d_a$ 是 action dimension；ALOHA released code 中 state/action dimension 为 14。

## 第三步：reconstruction loss

官方 released `policy.py` 对 action prediction 使用 element-wise L1 loss，并用 padding mask 去掉 padded action positions：

\[
\mathcal L_{recon}
=
\operatorname{masked\;L1}(A_t,\hat A_t).
\]

它要求每个有效 future timestep 的 joint target 接近 demonstration。

## 第四步：KL loss

Approximate posterior 被约束接近

\[
p(z)=\mathcal N(0,I).
\]

因此加入

\[
\mathcal L_{KL}
=
D_{KL}\left(
q_\phi(z\mid q_t,A_t)
\|\mathcal N(0,I)
\right).
\]

released policy 的总 loss 为

\[
\mathcal L
=
\mathcal L_{L1}
+\beta\mathcal L_{KL}.
\]

论文主要配置使用 $\beta=10$。

## Padding Mask 的作用

若时刻 $t$ 接近 episode 结尾，未来可能只剩 17 步，但模型仍固定输出 $k=100$ 个 slots。Dataset 需要把不足的部分 pad 到固定长度。

如果这些 padding 也被当作真实 action 计算 loss，模型会被迫学习人为填充值。因此 reconstruction 只应统计有效位置。

Latent encoder 也需要 padding mask，避免 attention 把 padded actions 当成真实 sequence information。

## 优化器与 backbone learning rate

released training code 的 ACT configuration 使用 ResNet18 backbone，并给 backbone 较小 learning rate。模型构建沿用了 DETR 风格的 parameter groups 与 AdamW optimizer。

这些属于 released implementation 的 optimization choice，不是 ACT 算法定义。换数据、backbone 或训练规模后可以改变。

## Training graph

把所有步骤放到一张图：

```text
                 ground-truth future actions A_t
                         │
                         ├─────────────── target ───────────────┐
                         │                                      │
q_t ─────────────────────┤                                      │
                         ↓                                      │
                  latent encoder                                │
                         ↓                                      │
                    μ , logσ²                                   │
                         ↓                                      │
                     sample z                                   │
                         │                                      │
images ─→ ResNet ────────┤                                      │
q_t ─────────────────────┤                                      │
                         ↓                                      │
                  policy Transformer                            │
                         ↓                                      │
                    predicted A_hat_t ── L1 ────────────────────┘
                         │
             KL(q(z|q,A) || N(0,I))
```

这张图中 future action 同时扮演两种角色：**latent encoder 的输入**和**最终 reconstruction target**。

## Sources

- Zhao et al., **Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware**, 2023. https://arxiv.org/abs/2304.13705
- Official loss implementation: https://github.com/tonyzhaozh/act/blob/main/policy.py
- Official model implementation: https://github.com/tonyzhaozh/act/blob/main/detr/models/detr_vae.py
