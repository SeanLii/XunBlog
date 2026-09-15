---
title: "Vision-Language-Action Model"
kind: "canonical"
domain: "Robot Learning"
parent: "Robot Learning"
canonical: "/robot-learning/vision-language-action-model/"
prerequisites:
  - "/deep-learning/multimodal/vision-language-model/"
  - "/robot-learning/imitation-learning/"
related:
  - "/robot-learning/cross-embodiment-learning/"
  - "/robot-learning/pi0/"
---

# Vision-Language-Action Model

Vision-Language-Action Model（VLA）是一类同时利用 visual observations、language conditions 与 robot-control supervision 学习 action policy 的模型。

可以用一般 conditional policy 表示：

\[
p_\theta(
a_{t:t+H-1}
\mid
I_{\le t},\, l,\, s_{\le t}
),
\]

其中：

- $I$ 表示 image / visual observations；
- $l$ 表示 language instruction；
- $s$ 表示 proprioceptive robot state；
- $a$ 表示 action 或 action sequence；
- $H$ 是 action horizon。

VLA 的定义性特征是输出与 robot behavior 建立直接训练关系，而不只是产生 text 或 semantic representation。

## Vision-Language Representation 与 Control

[Vision-Language Model](/deep-learning/multimodal/vision-language-model/) 可以学习 object semantics、image-text alignment、language grounding 与 multimodal representation。

Robot policy 还需要输出具有明确 physical semantics 的 actions，例如 joint commands、end-effector deltas、gripper commands 或 action chunks。

因此 VLA 需要在 vision-language representation 与 control space 之间建立 policy mapping。

## VLA 是 Model Family

VLA 不是单一 architecture。不同模型可以在以下方面采用不同设计：

- visual encoder；
- language model / multimodal backbone；
- state representation；
- action representation；
- action decoder；
- temporal horizon；
- training objective；
- pretraining / post-training strategy。

因此“把 action 当成 text token”只描述部分 VLA，而不是类别定义。

## Discrete Action Tokenization

RT-2 的代表性设计是把 robot actions 离散化并编码成 tokens，使 action prediction 可以复用 autoregressive vision-language model 的 token-generation interface。

抽象地：

\[
a\rightarrow z_a\in\{1,\ldots,K\},
\]

然后学习：

\[
p(z_a\mid I,l).
\]

这种方式可以直接复用 language-model output machinery，但 continuous actions 需要 quantization，action precision 与 discretization resolution 存在 trade-off，多 token autoregressive decoding 也会影响 control latency。

OpenVLA 延续 action-tokenization 路线，并在大规模机器人 demonstrations 上训练开放的 generalist VLA。

## Continuous Action Generation

另一类 VLA 保持 action 为连续变量：

\[
a_t\in\mathbb R^d.
\]

policy 可以通过 direct regression、mixture distributions、diffusion 或 flow matching 预测 continuous action / action chunk。

[π0](/robot-learning/pi0/) 使用 VLM backbone 与 Action Expert，并通过 flow matching 生成 continuous action chunks。

因此 VLA 的核心不是 action 的离散或连续形式，而是：

\[
\text{vision + language (+ state)}
\rightarrow
\text{robot policy}.
\]

## Pretraining 与 Robot Data

VLA 常利用两类作用不同的数据。

### Vision-language data

提供 broad visual concepts、language semantics、image-text grounding 与 web-scale semantic diversity。

### Robot trajectories

提供 observation-action relationship、motor-control semantics、embodiment-specific interfaces 与 physical interaction experience。

仅有 vision-language pretraining 不会自动产生 robot-control ability；robot action supervision 必须把 representation 对齐到 control task。

## Co-Training 与 Fine-Tuning

VLA 可以采用不同训练策略：

- 先训练 VLM，再在 robot data 上 fine-tune；
- 同时混合 vision-language 与 robot objectives；
- 冻结部分 backbone，只训练 action modules；
- pretrain generalist robot policy，再针对 downstream task fine-tune。

RT-2 使用 web-scale vision-language tasks 与 robot trajectory data 的 co-fine-tuning；OpenVLA 则展示开放 backbone 上的大规模 robot pretraining 与 downstream fine-tuning。

## Robot State

实际 VLA 往往不仅需要 images 和 language，还使用 proprioception：

\[
s_t=[\text{joint state},\text{gripper},\ldots].
\]

language 描述任务条件，vision 提供 external scene，robot state 描述当前 embodiment configuration。三类 condition 的角色不同。

## Temporal Modeling

Robot control 是 sequential problem。policy 可以预测单步 action：

\[
p(a_t\mid I_t,l,s_t),
\]

也可以预测 action chunk：

\[
p(a_{t:t+H-1}\mid I_t,l,s_t).
\]

action chunk 可以降低 inference frequency、建立短期 temporal consistency，但也改变 observation feedback cadence。不同 VLA 对 horizon 与 replanning interval 的选择不同。

## Closed-Loop Execution

无论 action representation 如何，VLA 最终都处于 closed loop：

\[
o_t
\rightarrow
\pi_\theta
\rightarrow
a_t
\rightarrow
\text{environment}
\rightarrow
o_{t+1}.
\]

因此 model quality 不应只用 offline action-prediction loss 评价，还需要考虑 task success、latency、observation refresh、action smoothness、long-horizon error、recovery 与 safety。

## Generalist Training

Generalist VLA 希望一个 model 覆盖 many tasks、objects、environments，并可能覆盖 many embodiments。

这需要同时处理 task diversity、visual diversity、language diversity 与 robot-interface diversity。[Cross-Embodiment Learning](/robot-learning/cross-embodiment-learning/) 专门研究最后一个维度。

## Transfer from Vision-Language Pretraining

VLA 的重要研究目标之一，是让大规模 vision-language knowledge 改善 robot generalization。

RT-2 展示了将 VLM 与 robot trajectories co-fine-tune 后，对未在 robot training data 中直接出现的 semantic instructions 和 object concepts 获得更强 generalization。

但 semantic generalization 与 motor generalization 并不等价。识别 object category 或理解 instruction 不自动保证稳定的 grasp、contact 与 trajectory execution。

因此 VLA performance 需要同时分析：

\[
\text{semantic competence}
+
\text{physical control competence}.
\]

## Action-Space 与 Embodiment Constraints

不同 robots 具有不同 action dimensions、units 与 kinematics。跨 embodiment VLA 需要处理 action normalization、state schema、robot identity、embodiment-specific heads/adapters 或 common action representations。

这些问题属于 model interface 与 data representation 的一部分，而不是简单扩大 backbone 就会自动消失的问题。

## Limitations

当前 VLA 仍受到多方面限制：

- robot data 规模远小于 web data；
- long-horizon planning 与 recovery 仍然困难；
- precise manipulation 需要低 latency、高频 continuous control；
- cross-embodiment action spaces 难统一；
- language understanding 与 physical success 并不等价；
- closed-loop failures 可能快速累积；
- inference compute 直接影响控制频率。

因此 VLA 应理解为持续发展的 model family，而不是已经收敛到单一 architecture 的方法。

## Connections

- [Vision-Language Model](/deep-learning/multimodal/vision-language-model/)：提供 multimodal representation / semantic pretraining。
- [Imitation Learning](/robot-learning/imitation-learning/)：大量 VLA 的 robot action supervision 来源。
- [Cross-Embodiment Learning](/robot-learning/cross-embodiment-learning/)：处理不同 robot interfaces 的共享学习。
- [π0](/robot-learning/pi0/)：continuous action + flow matching 的 VLA 实例。

## Sources

- Brohan et al. / Zitkovich et al. *RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control*. 2023.
- Kim et al. *OpenVLA: An Open-Source Vision-Language-Action Model*. 2024.
- Open X-Embodiment Collaboration et al. *Open X-Embodiment*. 2023/2024.
