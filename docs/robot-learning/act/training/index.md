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

ACT Training 从 demonstration trajectory 中随机选择一个当前时刻，取该时刻的 observation 和后续 action sequence，训练 policy 重建 future action chunk，同时用 KL regularization 约束 training-only CVAE encoder 的 latent distribution。

## Training Sample

一条 demonstration trajectory 可以写成

\[
(o_0,a_0),(o_1,a_1),\ldots,(o_{T-1},a_{T-1}).
\]

官方 dataset loader 随机选择 `start_ts`。当前 observation 只取这个时刻：

\[
o_t=(I_t,q_t).
\]

Action target 则从当前附近开始一直取到 episode 末尾，再用 padding 补齐固定长度。进入 ACTPolicy 后只保留前 `num_queries = k` 个 actions，因此最终 supervision 对应一个固定长度 action chunk。

对于 simulation，代码从 `action[start_ts:]` 开始；对于 real data，released loader 使用

```text
action[max(0, start_ts - 1):]
```

并在注释中标为用于时间对齐的 `hack`。这是实现中的数据对齐细节，不是论文算法定义。

## Padding Mask

靠近 episode 末尾时剩余 action 数不足 $k$。Loader 用零值补齐，并生成 Boolean `is_pad` mask：真实 action positions 为 false，padding positions 为 true。

CVAE encoder 使用这个 mask，避免把 padded action tokens 当作真实 demonstration sequence 内容。Reconstruction loss 也会用 `is_pad` 屏蔽 padding positions。

## Dataset Normalization

官方实现对整个 dataset 统计 action 与 qpos 的 mean/std：

\[
\tilde a=\frac{a-\mu_a}{\sigma_a},
\qquad
\tilde q=\frac{q-\mu_q}{\sigma_q}.
\]

标准差最小被 clip 到 $10^{-2}$，防止某个几乎不变化的维度导致除以极小数。

图像先从 uint8 缩放到 $[0,1]$，进入 policy 后再做 ImageNet-style channel normalization。Normalization 是 training/inference 必须保持一致的数据接口，而不是模型理论的一部分。

## CVAE Forward Path

训练时，current normalized qpos 与 ground-truth normalized action chunk 进入 training-only CVAE encoder：

\[
(q_t,A_t)
\rightarrow
(\mu,\log\sigma^2).
\]

随后

\[
\epsilon\sim\mathcal N(0,I),
\qquad
z=\mu+\sigma\odot\epsilon.
\]

Images、qpos 与 sampled $z$ 再进入 policy network：

\[
\hat A_t
=
\pi_\theta(o_t,z).
\]

这个顺序非常重要：ground-truth future actions 只通过 CVAE encoder 影响 latent sample；policy decoder 的任务仍是从 current observation 与 $z$ 预测 action chunk。

## Reconstruction Loss

这里必须区分论文内部描述与 released code。

### Paper description

Algorithm 1 写作

\[
\mathcal L_{reconst}
=
\operatorname{MSE}(\hat A_t,A_t).
\]

但同一篇论文 IV-C 正文随后明确说明：实际使用 L1 reconstruction loss，并称 L1 比 L2 对 action sequence 建模更精确。

### Released implementation

官方 `policy.py` 使用逐元素 L1：

\[
|A_t-\hat A_t|,
\]

将 padded positions 乘 0 后再 `.mean()`。因此 released implementation 与论文 IV-C 的文字说明一致，而与 Algorithm 1 中的 MSE 记载不一致。

需要进一步注意：代码是在 mask 后对完整 tensor 做 `.mean()`，分母仍包含被置零的 padded elements；它并不是先只抽取 valid elements 再对 valid count 求平均。这个区别在 episode 尾部 padding 比例较高时会改变 loss scale。

## KL Loss

对 diagonal Gaussian posterior 与 standard normal prior，代码逐维计算

\[
KL_j
=-\frac12
\left(
1+\log\sigma_j^2-
\mu_j^2-
\sigma_j^2
\right),
\]

先对 latent dimensions 求和，再对 batch 求平均：

\[
\mathcal L_{KL}
=
\frac1B\sum_{b=1}^{B}
\sum_{j=1}^{d_z} KL_{b,j}.
\]

最终 released loss 为

\[
\mathcal L
=
\mathcal L_{L1}
+
\beta\mathcal L_{KL}.
\]

论文默认

\[
\beta=10.
\]

## Optimization

论文 Algorithm 1 写“ADAM”。当前 released implementation 实际构造 `torch.optim.AdamW`，并将 backbone parameters 放到独立 parameter group，可使用不同 learning rate。

论文默认超参数为：

| Hyperparameter | Value |
|---|---:|
| learning rate | $10^{-5}$ |
| batch size | 8 |
| encoder layers | 4 |
| decoder layers | 7 |
| feed-forward dimension | 3200 |
| hidden dimension | 512 |
| attention heads | 8 |
| chunk size | 100 |
| $\beta$ | 10 |
| dropout | 0.1 |

官方 README 的示例训练命令与其中的主要 ACT hyperparameters 一致，并示例训练 2000 epochs；README 还建议 real-world data 可能需要更长训练。Epoch 数属于 repository usage guidance，不是论文算法定义。

## Validation and Checkpoint Selection

Released training loop每个 epoch 先运行 validation，记录平均 validation loss，并保存最低 validation loss 对应的 state dict。最终 `policy_best.ckpt` 就是这个 best-validation checkpoint。

论文也说明 test-time rollout 使用最低 validation loss 的 policy。这使 model selection 与 rollout evaluation 分开：机器人环境中的 success rate 不直接参与 training gradient。

## Sources

- [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware — Zhao et al., 2023](https://arxiv.org/abs/2304.13705)
- [ACT official implementation — tonyzhaozh/act](https://github.com/tonyzhaozh/act)
