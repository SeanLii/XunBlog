---
title: "Pre-training and Post-training in π0"
kind: "pi0-topic"
domain: "Robot Learning / π0"
parent: "π0"
canonical: "/robot-learning/pi0/pretraining-and-posttraining/"
prerequisites:
  - "/robot-learning/pi0/"
  - "/robot-learning/cross-embodiment-learning/"
related:
  - "/robot-learning/pi0/training/"
  - "/robot-learning/vision-language-action-model/"
standard: "XunBlog Content & Knowledge Architecture v1.0"
rebuilt: "2026-09-15"
---
# Pre-training and Post-training in π0

π0 的 generalist behavior 不是单靠网络结构产生的。模型先把 Internet-scale 视觉语言表征与大规模、多任务机器人轨迹结合，再用更集中、更高质量的数据强化困难能力。pre-training 扩大支持域，post-training 改变该域中的数据权重；两者共同决定最终策略分布。

## 两层 Pre-training 提供不同信息

第一层来自 VLM：图像与文本预训练提供物体、场景、属性和指令的语义对应。第二层来自 robot pre-training：轨迹数据把这些语义表示连接到状态变化与连续动作。

若把数据集写为

\[
\mathcal D_{pre}=\bigcup_{e=1}^{E}\bigcup_{m=1}^{M_e}\mathcal D_{e,m},
\]

$e$ 表示 embodiment，$m$ 表示任务。模型在混合分布上最小化条件 flow objective：

\[
\min_\theta\ \mathbb E_{(o,A)\sim\mathcal D_{pre}}\mathcal L_{FM}(o,A;\theta).
\]

VLM 知道“杯子”是什么，并不等于知道某台机械臂应怎样移动；robot pre-training 提供的正是从语义条件到动作分布的连接。

## 跨 Embodiment 需要统一接口

不同机器人具有不同相机布局、关节数量、state definition 和 control convention。[Cross-Embodiment Learning](/robot-learning/cross-embodiment-learning/) 不能靠把原始向量直接拼到同一批次实现。π0 使用 dataset transforms 做归一化、维度适配与 action representation 对齐：

\[
(q^{(e)},A^{(e)})\xrightarrow{T_e}(\tilde q,\tilde A).
\]

共享模型在 $(\tilde q,\tilde A)$ 上训练，执行时再通过 $T_e^{-1}$ 返回对应机器人接口。统一表示创造参数共享，也会隐藏 embodiment 差异；不相容的控制语义或错误 normalization 会直接污染共享目标。

## 数据混合决定模型实际关注什么

大数据集并不自动得到均衡能力。若数据源 $j$ 的 sampling weight 为 $w_j$，训练分布为

\[
p_{mix}(o,A)=\sum_j w_j p_j(o,A).
\]

高频平台、短任务或重复轨迹可能支配梯度。权重、质量过滤和任务覆盖因此与 model scale 一样重要。π0 纳入多种 single-arm、dual-arm、mobile manipulation 数据以及 Open X-Embodiment 数据，但覆盖范围仍不等于任意新机器人上的零样本可执行性。

## Post-training 重新分配能力

generalist pre-training 需要兼顾大量行为，困难的精细任务在混合数据中的概率可能很低。post-training 使用更集中或更高质量的数据继续优化：

\[
\theta_{post}=\arg\min_\theta
\mathbb E_{(o,A)\sim\mathcal D_{post}}
\mathcal L_{FM}(o,A;\theta),
\qquad \theta\leftarrow\theta_{pre}.
\]

它不是重新定义架构，而是让参数更贴近目标任务分布。能力提升的同时，过窄的 post-training distribution 可能损害其他任务，因此数据比例和训练长度决定 specialization 与 retention 的折中。

## 泛化边界来自数据支持域

视觉语言先验有助于识别未在机器人数据中充分标注的语义，但精确 motor control 仍需要相容的动力学、视角、state/action interface 与轨迹质量。π0 的训练范式提高了跨任务复用的上限，并没有消除 distribution shift。面对新 embodiment 时，是否需要 adaptation 取决于 $T_e$ 能否保留动作含义，以及训练数据是否覆盖相近的视觉与接触模式。

## Sources

- Black et al. *π0: A Vision-Language-Action Flow Model for General Robot Control*. 2024, data and training sections. https://arxiv.org/abs/2410.24164
- Physical Intelligence. *π0: Our First Generalist Policy*. https://www.pi.website/blog/pi0
- Open X-Embodiment Collaboration. *Open X-Embodiment*. 2023. https://robotics-transformer-x.github.io/
