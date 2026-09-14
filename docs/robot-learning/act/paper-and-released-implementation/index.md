---
title: "Paper and Released Implementation"
kind: "act-topic"
domain: "Robot Learning / ACT"
parent: "ACT"
canonical: "/robot-learning/act/paper-and-released-implementation/"
prerequisites:
  - "/robot-learning/act/"
related:
  - "/robot-learning/act/training/"
  - "/robot-learning/act/architecture/"
---

# Paper and Released Implementation

ACT 的论文与官方 released implementation 大体对应，但并非逐行一致。这里把能够从原论文和当前官方仓库直接核对出的差异与实现细节集中记录，避免在其他页面中把其中一个版本悄悄改写成另一个。

本文所称“当前官方仓库 / current main”指 **2026-09-15 核对到的 GitHub main branch 状态**。仓库代码会继续变化，因此复现时还应记录自己实际使用的 commit。

## Reconstruction Loss

### Paper description

论文 Algorithm 1 第 9 行写的是

\[
\mathcal L_{reconst}
=
\operatorname{MSE}(\hat A,A).
\]

但同一篇论文 IV-C 的正文又明确写道实际使用 **L1 loss instead of the more common L2 loss**，并称 L1 对 action sequence 建模更精确。

因此这里首先是论文内部的记载不一致，而不只是“paper versus code”。

### Released implementation

当前官方 `policy.py` 使用

```python
F.l1_loss(actions, a_hat, reduction='none')
```

并结合 padding mask。因此 released implementation 与论文 IV-C 的 L1 文字描述一致，与 Algorithm 1 的 MSE 公式不一致。

XunBlog 在 ACT Training 页面以 released behavior 为实现说明，同时保留 Algorithm 1 的原始 MSE 记载。

## Optimizer

### Paper description

Algorithm 1 写“Update $\theta,\phi$ with ADAM”。

### Released implementation

`detr/main.py` 构建的是

```python
torch.optim.AdamW(...)
```

并给 backbone parameter group 单独设置 learning rate。

因此“论文算法写 Adam”和“当前官方代码使用 AdamW”应同时保留。

## Action Output Head

### Paper description

论文 IV-C 说 Transformer decoder 的 $k\times512$ output 会通过一个 MLP down-project 成

\[
k\times14.
\]

### Released implementation

当前 `DETRVAE` 定义

```python
self.action_head = nn.Linear(hidden_dim, state_dim)
```

也就是一个 linear projection。它在宽泛意义上可以被视为最简单的 feed-forward projection，但代码并不是一个带隐藏层的多层 MLP。

因此实现页面应写“linear action head”，同时保留论文原文的 MLP 描述。

## Multi-Camera Backbone

### Paper description

论文从每个 camera image 获得 ResNet18 feature map，并将四个视角形成总计 1200 个 visual features。

### Released implementation

ACT model construction 只建立一个 backbone，并在 camera loop 中始终调用

```python
self.backbones[0](image[:, cam_id])
```

所以 ACT branch 的多个 cameras 共享同一套 ResNet weights。随后各 camera feature maps 沿 width dimension concatenate，再进入 Transformer flatten。

这不改变“每个 camera 都被 ResNet18 处理”的高层描述，但明确了 released code 并非四套独立 ResNet parameter sets。

## Latent Parameterization

### Paper description

论文说 CVAE encoder predicts mean and variance of a diagonal Gaussian style variable。

### Released implementation

代码实际输出

\[
\mu
\quad\text{and}\quad
\log\sigma^2,
\]

并通过

\[
\sigma=\exp(\tfrac12\log\sigma^2)
\]

完成 reparameterization。Latent dimension 被硬编码为 32。

这里并不构成理论冲突：`logvar` 是实现 Gaussian variance 的常见数值参数化；但准确阅读代码时应该写 `logvar`，而不是声称网络直接输出 variance tensor。

## Real-Data Temporal Alignment

### Paper description

论文把训练样本表述为 current observation $o_t$ 与 future action sequence $a_{t:t+k}$。

### Released implementation

Dataset loader 对 simulation 使用 `action[start_ts:]`，但 real data 使用

```python
action[max(0, start_ts - 1):]
```

并标注 `hack, to make timesteps more aligned`。

这是 released data pipeline 的工程对齐规则。它说明复现实验时仅照论文符号配对 observation/action 还不足以完全复刻官方数据加载行为。

## Padding-Aware L1 Reduction

### Paper description

论文没有给出 padding reduction 的代码级定义。

### Released implementation

代码先把 padded action positions 的逐元素 L1 乘 0，然后对整个 tensor `.mean()`：

```python
l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
```

因此 padding positions 虽然不贡献 numerator，却仍在 mean 的总 element count 中。这与“只对 valid elements 求平均”不是完全相同的 reduction。

这是实现细节，不应改写为论文提出的理论设计。

## Decoder-Layer Output in the Current Repository

### Paper description

论文描述一个 7-layer Transformer decoder，并把其 output 用于 action prediction，没有说明只使用第一 decoder layer 的输出。

### Released implementation

当前 `transformer.py` 设置

```python
return_intermediate_dec=True
```

所以 decoder 返回所有 decoder layers 的 intermediate outputs。`Transformer.forward` 直接返回这个 stacked tensor；随后 `detr_vae.py` 调用

```python
hs = self.transformer(...)[0]
```

按当前 tensor layout，`[0]` 选择的是第一个 decoder layer 的 output，而不是最后一层。

官方 GitHub 目前存在未关闭 issue #52，提出这是从原始 DETR code 修改返回值时遗留的错误，并建议使用最后一层。由于该 issue 的 maintainer resolution 目前并未写进 released main branch，XunBlog 不把“这是作者确认的 bug”当成既定事实，而记录为：

> **Observed released-code behavior:** current main selects the first returned decoder-layer output. An open repository issue argues this is unintended. Authoritative intent is unresolved in the current released source.

复现研究若修改这一行，应明确说明已经偏离当前 main branch，而不是悄悄把修正后的行为当成官方原版。

## Documentation Rule

以上差异采用三层标记：

1. **Paper description**：原论文实际写了什么；
2. **Released implementation**：官方仓库当前 main 实际做了什么；
3. **Observed issue / interpretation**：只有在代码行为需要解释时才单独标明，不把社区推断冒充作者结论。

这种分层比强行整理成一套“干净版本”更重要，因为 ACT 的可复现性正依赖这些细节。

## Sources

- [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware — Zhao et al., 2023](https://arxiv.org/abs/2304.13705)
- [ACT official implementation — tonyzhaozh/act](https://github.com/tonyzhaozh/act)
- [ACT GitHub Issue #52 — Critical BUG in transformer decoder output](https://github.com/tonyzhaozh/act/issues/52)
- [Original DETR implementation](https://github.com/facebookresearch/detr)
