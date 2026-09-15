---
title: "Imitation Learning"
kind: "canonical"
domain: "Robot Learning"
parent: "Robot Learning"
canonical: "/robot-learning/imitation-learning/"
prerequisites: []
related:
  - "/robot-learning/behavior-cloning/"
  - "/robot-learning/dagger/"
  - "/robot-learning/act/"
standard: "XunBlog Content & Knowledge Architecture v1.0"
rebuilt: "2026-09-15"
---
# Imitation Learning

Imitation Learning（模仿学习）研究如何利用 expert demonstrations 学习决策策略。监督信息来自 expert behavior，而不是由 agent 通过 reward trial-and-error 独立发现策略。

在 sequential decision process 中，一条 demonstration trajectory 可以写成：

\[
\tau
=
(o_1,a_1,o_2,a_2,\ldots,o_T,a_T),
\]

其中：

- $o_t$ 是时刻 $t$ 的 observation；
- $a_t$ 是 expert action；
- $T$ 是 trajectory horizon。

目标是学习 policy：

\[
\pi_\theta(a_t\mid o_{\le t}),
\]

使其在环境中执行时产生与 expert 相近的行为。

## Demonstration Distribution

设 expert policy 为 $\pi^*$。当 expert 与环境交互时，会诱导出自己的 state / observation distribution：

\[
d_{\pi^*}.
\]

demonstration dataset 是从这一 distribution 上得到的 trajectories。因此 imitation learning 的监督不仅包含“正确 action 是什么”，还隐含了 expert 通常访问哪些 states。

## Policy-Induced Distribution

部署 learned policy $\pi_\theta$ 后，环境中的 observation distribution 由 learner 自己的行为决定：

\[
d_{\pi_\theta}.
\]

环境 transition 可写为：

\[
s_{t+1}\sim P(\cdot\mid s_t,a_t).
\]

当前 action 会影响未来 state，未来 state 又成为 policy 的新输入。如果：

\[
d_{\pi_\theta}\neq d_{\pi^*},
\]

learner 可能进入 demonstrations 很少覆盖的区域。这是 imitation learning 中 distribution shift 与 compounding error 的来源。

## Offline 与 Interactive Imitation Learning

根据训练过程中是否还能访问 expert，可以区分两类数据条件。

### Offline imitation

训练只使用预先收集好的 demonstrations：

\[
\mathcal D=\{\tau_i\}_{i=1}^{N}.
\]

训练阶段不能针对 learner 新访问的 state 再查询 expert。这种设置适合 real-world robotics 中 expert interaction 昂贵、数据已经离线收集的情况。

### Interactive imitation

learner rollout 过程中可以继续查询 expert：

\[
o_t\sim d_{\pi_\theta}
\quad\Rightarrow\quad
a_t^*=\pi^*(o_t).
\]

这样可以直接获得 learner-induced states 上的 supervision。[DAgger](/robot-learning/dagger/) 是代表性的 interactive imitation algorithm。

## Behavior Cloning

[Behavior Cloning](/robot-learning/behavior-cloning/) 是最直接的 imitation-learning 方法。它把 expert observation-action pairs 作为 supervised data，优化：

\[
\theta^*
=
\arg\min_\theta
\mathbb E_{(o,a^*)\sim\mathcal D}
[\ell(\pi_\theta(o),a^*)].
\]

若 policy 输出概率分布，则常用 negative log-likelihood：

\[
\mathcal L_{\text{BC}}
=
-\mathbb E_{(o,a^*)\sim\mathcal D}
\log \pi_\theta(a^*\mid o).
\]

Behavior Cloning 的优势是简单、可并行、可以完全 offline；主要限制是训练 distribution 与部署 distribution 可能不同。

## Sequential Error Propagation

单步 prediction error 在 sequential control 中可能改变后续输入。若某次 action 使系统偏离 expert trajectory：

\[
s_t\rightarrow \tilde s_{t+1},
\]

后续 policy 不再面对典型 expert state，而是在新的 distribution 上继续预测。

因此 imitation-learning quality 不能只用 per-step validation error 判断，还需要 closed-loop rollout evaluation，例如 task success、trajectory completion、recovery ability 与 long-horizon stability。

## Demonstration Coverage

Policy 能否在部署中稳定工作，很大程度上取决于 dataset 覆盖哪些 situations。重要维度包括：

- task diversity；
- initial-state diversity；
- object / environment variation；
- recovery trajectories；
- failure-adjacent states；
- observation quality；
- action calibration；
- temporal synchronization。

对于 offline imitation，dataset coverage 是无法通过训练算法完全消除的限制。

## Multimodal Expert Behavior

同一 observation 下可能存在多个合理 actions，因此：

\[
p(a\mid o)
\]

可以是 multimodal distribution。

若 model 只输出单一 regression mean，可能产生不属于任何 expert mode 的中间 action。更表达性的 imitation policies 可以使用 mixture distributions、latent-variable models、autoregressive models、diffusion / flow models 或 sequence-level action generation。

这些方法改变 policy distribution 的表示，但不改变 imitation learning 的监督来源。

## Partial Observability

若 observation $o_t$ 不包含决策所需的全部 state，则：

\[
\pi(a_t\mid o_t)
\]

可能无法唯一确定正确 action。

常见处理包括使用 observation history：

\[
\pi(a_t\mid o_{t-L:t}),
\]

或引入 recurrent / state-estimation mechanism。在 robotics 中，多相机视觉、proprioception、force information 与 temporal context 都可能减少 partial observability。

## Action Representation

Imitation learning 不规定 policy 必须输出哪种 action representation。可能包括：

- discrete actions；
- continuous joint / Cartesian commands；
- probability distributions；
- action chunks；
- tokenized actions；
- continuous generative trajectories。

不同 representation 改变 optimization、latency、multimodality 与 closed-loop behavior，但监督仍可以来自 demonstrations。

## Main Method Families

在当前知识范围内，可以区分三类常见方法：

### Direct supervised imitation

直接学习 expert mapping，例如 Behavior Cloning。

### Interactive dataset aggregation

在 learner-induced states 上继续获得 expert labels，例如 DAgger。

### Structured / generative policy modeling

保持 demonstration supervision，但改变 policy output distribution 与 temporal representation，例如 action chunking、latent-variable policy 与 diffusion/flow policy。

更广义的 imitation learning 还包括从 demonstrations 推断 reward / objective 的方法，例如 inverse reinforcement learning；它与直接 policy cloning 的建模目标不同。

## Evaluation

Imitation policy 的评价通常需要同时考虑：

- per-step prediction loss；
- closed-loop task success；
- long-horizon reliability；
- recovery behavior；
- distribution generalization；
- sample efficiency；
- control latency；
- safety constraints。

训练 loss 降低并不自动意味着 rollout performance 提升。

## Connections

- [Behavior Cloning](/robot-learning/behavior-cloning/)：offline supervised imitation 的基本形式。
- [DAgger](/robot-learning/dagger/)：通过 learner-induced states 减少 distribution mismatch。
- [ACT](/robot-learning/act/)：使用 action chunking 与 Transformer 的 robot imitation policy。
- [Vision-Language-Action Model](/robot-learning/vision-language-action-model/)：将 visual-language representation 与 robot action policy 结合。

## Sources

- Pomerleau. *ALVINN: An Autonomous Land Vehicle in a Neural Network*. 1989.
- Ross, Gordon, Bagnell. *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning*. AISTATS, 2011.
