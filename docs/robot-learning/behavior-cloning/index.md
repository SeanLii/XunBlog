---
title: "Behavior Cloning"
kind: "canonical"
domain: "Robot Learning"
parent: "Robot Learning"
canonical: "/robot-learning/behavior-cloning/"
prerequisites:
  - "/robot-learning/imitation-learning/"
related:
  - "/robot-learning/act/"
  - "/robot-learning/act/action-chunking/"
---

# Behavior Cloning

Behavior Cloning（BC）把 imitation learning 转化为 supervised learning：从 expert demonstrations 中取 observation-action pairs，训练 policy 在相同 observation 上预测 expert action。

## Definition

给定 expert dataset

\[
\mathcal D
=\{(o_i,a_i)\}_{i=1}^{N},
\]

学习参数化 policy $\pi_\theta$，使它在 demonstration data 上尽量复现 expert actions。

对于连续动作，一种常见 deterministic objective 是

\[
\min_\theta
\frac1N\sum_{i=1}^{N}
\ell(\pi_\theta(o_i),a_i),
\]

其中 $\ell$ 可以是 L1、L2 或由概率模型得到的 negative log-likelihood。Behavior Cloning 本身并不规定必须使用哪一种神经网络或哪一种具体 loss。

## Expert Demonstration Distribution

Supervised training 看到的 observations 来自 expert rollout。可以把 expert 在环境中诱导出的 state/observation distribution 记作

\[
d_{\pi^*}(o).
\]

训练风险近似为

\[
\mathbb E_{o\sim d_{\pi^*}}
[\ell(\pi_\theta(o),\pi^*(o))].
\]

这里的关键限制是：模型主要在 expert 会到达的 observations 上接受监督。

## Closed-Loop Deployment

部署后，执行的是 learned policy $\pi_\theta$。它自己的动作会改变下一步 observation，因此 rollout 分布变成

\[
d_{\pi_\theta}(o).
\]

只要 learned policy 与 expert 不完全相同，就可能逐渐访问 demonstration 中少见的位置。训练误差即使在 expert distribution 上很小，也不能自动保证在 $d_{\pi_\theta}$ 上同样小。

## Policy-Induced Distribution Shift

这种分布变化不是普通 train/test split 的静态差异。Policy 本身参与生成下一步输入：

\[
o_t
\xrightarrow{\pi_\theta}
a_t
\xrightarrow{\text{environment}}
o_{t+1}.
\]

如果 $a_t$ 有小误差，$o_{t+1}$ 就可能偏离 expert trajectory。随后 policy 在更陌生的 $o_{t+1}$ 上再预测，误差可能继续积累。

## Compounding Error

早期 imitation-learning 分析表明，纯 supervised imitation 在有限 horizon $T$ 下可能出现随 horizon 更快增长的 rollout cost；经典结果常用 $O(T^2\epsilon)$ 形式表达最坏情况下的误差累积，其中 $\epsilon$ 是在 expert-like training distribution 上的单步错误率。

这个式子不是说任何 BC 机器人都必然精确产生 $T^2\epsilon$ 次错误，而是在说明 sequential feedback 会把单步 supervised error 放大。它揭示了“training accuracy 很高”与“closed-loop rollout 很稳”之间没有简单等价关系。

## DAgger

DAgger 的核心做法是让当前 policy 实际 rollout，在它真正访问到的 states 上询问 expert action，然后把这些新 labeled states 加回 dataset：

\[
\mathcal D
\leftarrow
\mathcal D\cup
\{(o,\pi^*(o)):o\sim d_{\pi_\theta}\}.
\]

反复聚合后，训练数据会覆盖更多 learned policy 自己会遇到的状态。这样直接针对 policy-induced distribution shift，而不是只在原始 expert trajectories 上优化。

DAgger 的代价是需要新的 on-policy interaction 和 expert labeling。机器人远程操作场景中，这可能昂贵、危险或不自然。

## Recovery and Data Coverage

BC 是否容易从错误中恢复，很大程度取决于 demonstration data 是否覆盖 recovery states。若数据中只有理想轨迹，policy 偏离后可能从未见过应该怎样回到成功轨迹。

因此“模型容量更大”不能单独解决 distribution coverage 问题。更强模型可以更好拟合已有数据，却不能凭空获得未观察状态下的正确 expert behavior。

## Relationship to Action Chunking

[Action Chunking](/robot-learning/act/action-chunking/) 没有像 DAgger 那样直接收集 policy-induced states。ACT 采取的是另一个方向：一次预测未来 $k$ 个动作，把决策层面的 effective horizon 缩短，并结合频繁重预测与 temporal ensemble 保持反馈。

因此 action chunking 可以缓解 compounding error，但它没有从理论上消除 Behavior Cloning 的 distribution shift。ACT 仍然是 offline demonstration learning，其性能仍依赖 demonstration coverage。

## Limitations

Behavior Cloning 的主要优势是简单、稳定、可以直接使用现有 demonstrations。它的主要限制来自 sequential deployment：训练数据由 expert 产生，而部署数据由 learned policy 自己产生。

这也是为什么机器人 imitation learning 的许多方法，不论使用 Transformer、diffusion model 还是其他模型，都必须额外考虑 temporal structure、multi-modality、recovery behavior 或 on-policy coverage。模型架构改变了 policy class，却不会自动让 closed-loop distribution shift 消失。

## Sources

- [A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning — Ross, Gordon, Bagnell, 2011](https://proceedings.mlr.press/v15/ross11a.html)
- [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware — Zhao et al., 2023](https://arxiv.org/abs/2304.13705)
