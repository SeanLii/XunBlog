---
title: "Behavior Cloning"
kind: "canonical"
domain: "Robot Learning"
parent: "Robot Learning"
canonical: "/robot-learning/behavior-cloning/"
prerequisites:
  - "/robot-learning/imitation-learning/"
related:
  - "/robot-learning/dagger/"
  - "/robot-learning/act/"
standard: "XunBlog Content & Knowledge Architecture v1.0"
rebuilt: "2026-09-15"
---
# Behavior Cloning

> **知识边界**：本文的 canonical 对象是 **Behavior Cloning**。依赖机制由 [Imitation Learning](/robot-learning/imitation-learning/) 的 canonical page 定义；本文只在当前语境中调用其接口。


Behavior Cloning（BC）是把 expert demonstrations 直接转化为 supervised policy-learning problem 的 imitation-learning 方法。

给定 demonstration dataset：

\[
\mathcal D
=
\{(o_i,a_i^*)\}_{i=1}^{N},
\]

学习 policy：

\[
\pi_\theta(a\mid o),
\]

使其在 expert observations 上对 expert actions 分配较高概率。

最一般的 maximum-likelihood objective 为：

\[
\theta^*
=
\arg\min_\theta
\mathbb E_{(o,a^*)\sim\mathcal D}
[-\log \pi_\theta(a^*\mid o)].
\]

BC 的核心不是某一种 network 或 loss，而是直接从 expert observation-action supervision 学习 policy，不在训练目标中显式求解环境 reward 或 transition dynamics。

## Deterministic 与 Stochastic Policies

若 policy 是 deterministic regression：

\[
\hat a=f_\theta(o),
\]

可以使用：

\[
\mathcal L_{\text{MSE}}
=
\|a^*-\hat a\|_2^2
\]

或：

\[
\mathcal L_{\text{L1}}
=
\|a^*-\hat a\|_1.
\]

若 policy 显式定义 action distribution：

\[
\pi_\theta(a\mid o),
\]

则可以直接优化 negative log-likelihood。

因此 Behavior Cloning 不等于 MSE regression。BC 描述监督来源与学习范式，而不是固定的 output distribution。

## Training Distribution

demonstrations 来自 expert-induced distribution：

\[
(o,a^*)\sim d_{\pi^*}.
\]

训练时，模型只需读取已有 dataset，就可以计算 supervised loss，不必在环境中执行自己的预测。这使 BC 适合大规模 offline training，但也意味着 loss 主要约束 expert visited states 上的 behavior。

## Closed-Loop Deployment

部署时 learner 自己决定 action：

\[
a_t\sim\pi_\theta(\cdot\mid o_t),
\]

环境随后产生新的 state：

\[
s_{t+1}\sim P(\cdot\mid s_t,a_t).
\]

observation distribution 因而变为：

\[
d_{\pi_\theta},
\]

而不再是 training distribution $d_{\pi^*}$。这构成 BC 中最重要的 train-test mismatch。

## Covariate Shift

若 learner 在某个 expert state 上产生小误差：

\[
a_t\neq a_t^*,
\]

系统可能进入 demonstrations 中很少出现的 state：

\[
\tilde s_{t+1}.
\]

此时 policy 需要在缺少监督的数据区域做 extrapolation。后续错误又会继续改变 trajectory，形成 sequential distribution shift。

## Compounding Error

在经典 sequential imitation analysis 中，若单步分类错误率约为 $\epsilon$，vanilla supervised imitation 的 worst-case performance degradation 可能出现与：

\[
T^2\epsilon
\]

同阶的 horizon dependence。

其机制是：

\[
\text{action error}
\rightarrow
\text{future state changes}
\rightarrow
\text{future input distribution changes}.
\]

[DAgger](/robot-learning/dagger/) 通过直接收集 learner-induced states 上的 expert labels 来处理这一问题。

## Multimodal Actions

BC dataset 中同一 observation 可能对应多个合理 expert behaviors。若：

\[
p(a\mid o)
\]

是 multimodal，而 model 只输出单一 conditional mean，则 regression loss 可能得到不同 modes 之间的平均值。

因此更表达性的 BC policy 可以使用：

- mixture density；
- latent-variable model；
- autoregressive policy；
- diffusion / flow policy；
- sequence-level prediction。

这些方法仍然属于 Behavior Cloning，只要监督目标直接来自 expert demonstrations。

## One-Step 与 Sequence-Level Behavior Cloning

BC 不要求输出必须是单个 action。

单步 policy：

\[
\pi_\theta(a_t\mid o_t)
\]

可以扩展为 sequence policy：

\[
\pi_\theta(a_{t:t+k-1}\mid o_t).
\]

训练 target 从一个 expert action 变成 expert action chunk：

\[
A_t^*
=
(a_t^*,\ldots,a_{t+k-1}^*).
\]

[ACT](/robot-learning/act/) 使用的 action-chunk prediction 仍属于 Behavior Cloning；变化的是 policy output representation 与 architecture。

## Partial Observability

若 observation 缺失决策需要的信息，即使 dataset 很大，也可能出现：

\[
p(a\mid o)
\]

本身具有不可消除的不确定性。

常见处理是让 policy 条件化于 observation history、recurrent state、multi-view images、proprioception 或 task/language condition。

## Dataset Quality 与 Coverage

对 BC 来说，dataset 不只是训练样本数量，还需要考虑：

- expert quality；
- state-space coverage；
- task balance；
- action noise；
- recovery examples；
- sensor synchronization；
- label consistency；
- temporal frequency。

某些 rare-but-critical states 即使只占很小比例，也可能决定 closed-loop policy 是否稳定。

## Regularization 与 Generalization

BC 仍然是 supervised learning，因此 weight decay、augmentation、dropout、early stopping、balanced sampling 等常见 regularization 都可以使用。

但普通 supervised regularization 不能从根本上消除 policy-induced distribution shift，因为后者来自 sequential interaction structure。

## Evaluation

BC 应同时评估两类指标。

### Supervised metrics

例如 action error、negative log-likelihood 与 validation loss。

### Closed-loop metrics

例如 task success、completion rate、time-to-failure、recovery rate 与 trajectory quality。

对于 control policy，closed-loop metrics 通常更接近最终目标。

## Strengths and Limitations

Behavior Cloning 的主要优势包括：

- objective 简单；
- 可完全 offline；
- 易于利用大规模 demonstration data；
- 可以与多种 architecture / output distribution 组合；
- 不需要显式设计 reward。

主要限制包括：

- learner-induced distribution shift；
- dataset coverage dependence；
- multimodal action averaging；
- partial observability；
- expert errors / inconsistency；
- rare failure-state supervision 不足。

## Connections

- [Imitation Learning](/robot-learning/imitation-learning/)：BC 所属的更广泛学习范式。
- [DAgger](/robot-learning/dagger/)：通过 interactive data aggregation 缓解 distribution shift。
- [ACT](/robot-learning/act/)：sequence-level Behavior Cloning 的代表性 robot policy。

## Sources

- Pomerleau. *ALVINN: An Autonomous Land Vehicle in a Neural Network*. 1989.
- Ross, Gordon, Bagnell. *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning*. AISTATS, 2011.
