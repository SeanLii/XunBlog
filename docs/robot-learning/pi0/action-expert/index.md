---
title: "Action Expert"
kind: "pi0-topic"
domain: "Robot Learning / π0"
parent: "π0"
canonical: "/robot-learning/pi0/action-expert/"
prerequisites:
  - "/robot-learning/pi0/architecture/"
  - "/deep-learning/attention/self-attention/"
related:
  - "/robot-learning/pi0/flow-matching-in-pi0/"
  - "/robot-learning/pi0/blockwise-causal-attention-mask/"
standard: "XunBlog Content & Knowledge Architecture v1.0"
rebuilt: "2026-09-15"
---
# Action Expert

π0 的预训练 Vision-Language Model 已能表示图像与语言，却没有处理机器人关节状态、连续动作和生成时间的参数接口。Action Expert 补上了这一部分：它是一组面向机器人 token 的 Transformer 参数，与 VLM 参数共同完成一次注意力计算，而不是接在 VLM 输出后的独立控制器。理解它之前，应先把握 [π0 的整体架构](/robot-learning/pi0/architecture/) 中三类输入 token 的位置。

## 从模态缺口到双 Expert

设图像与语言 token 为 $X_v$，机器人状态与动作 token 为 $X_a$。如果让同一套参数处理两者，模型既要保留视觉语言预训练能力，又要从有限的机器人数据中学习连续控制，两个目标会争用相同容量。π0 因而把每层的变换写成按 token 类型路由的形式：

\[
H_v^{(l+1)}=F_v^{(l)}(H^{(l)}),\qquad
H_a^{(l+1)}=F_a^{(l)}(H^{(l)}).
\]

$F_v^{(l)}$ 使用 VLM expert 参数，$F_a^{(l)}$ 使用 Action Expert 参数；二者看到的注意力上下文由同一个结构化 mask 决定。因此，参数路由分开了模态变换，却没有切断信息交换。

## 连续状态如何成为 Token

机器人状态 $q_t\in\mathbb R^{d_q}$ 不是离散词表中的 token。模型先通过线性投影把它送入 Action Expert 的 hidden space：

\[
e_q=W_q q_t+b_q\in\mathbb R^{d_h}.
\]

不同 embodiment 的状态维数并不一定相同。released openpi 在数据变换阶段把它们映射或 padding 到统一接口，所以配置里的 action dimension 32 表示模型接口上限，不等于每台机器人都有 32 个物理执行器。

## 动作 Token 同时携带数值与生成时间

[Flow Matching in π0](/robot-learning/pi0/flow-matching-in-pi0/) 在每个生成时刻 $	au$ 都向模型提供中间动作 $a^	au$。同一个数值出现在纯噪声附近和数据附近时意义不同，因此动作 embedding 必须同时编码二者：

\[
e_a=g_\phi(W_a a^\tau,\,\varphi(\tau)),
\]

其中 $\varphi(\tau)$ 是 sinusoidal time embedding，$g_\phi$ 是 learned transformation。长度为 $H$ 的动作块由 $H$ 个这样的 token 表示，从而保留每个未来位置，而不是把整段轨迹压成单一向量。

## 注意力把语义条件写入动作

Action Expert 内部仍执行 [Self-Attention](/deep-learning/attention/self-attention/) 与逐位置非线性变换。动作 token 可以读取图像、语言和机器人状态，也能在动作块内部彼此读取。于是某个位置的表示不仅包含“第几步怎么动”，还包含物体位置、指令语义和相邻动作之间的协调关系。

这种交互解释了为何 Action Expert 不能被理解成普通 MLP action head：action head 只把最终 hidden state 投影到动作维度，而跨模态条件融合发生在更早的多层注意力中。

## 输出是速度场而不是最终动作

取出 $H$ 个动作位置的 hidden states：

\[
H_a\in\mathbb R^{H\times d_h},\qquad
v_\theta=W_{out}H_a+b_{out}\in\mathbb R^{H\times d_a}.
\]

$v_\theta$ 描述当前动作样本在 flow path 上应沿哪个方向移动。数值积分器用它更新 $A^\tau$；只有多次更新结束后的 $A$ 才是交给控制器的 action chunk。

## 较小参数规模来自重复计算

论文中的 Action Expert 约为 300M 参数，hidden width 约 1024，MLP dimension 约 4096，小于 VLM expert。原因直接来自推理过程：固定 observation prefix 可以缓存，而动作 token 会在每个 flow step 改变并重新计算。缩小反复执行的 expert 能降低延迟，同时把大模型容量留给只需编码一次的视觉语言语义。

代价是动作侧容量受到更强约束。复杂控制能力不会仅由一个大 VLM 自动产生，仍取决于机器人数据覆盖、状态与动作归一化、expert 容量以及 flow objective 是否为不同 embodiment 提供一致的学习信号。

## Sources

- Black et al. *π0: A Vision-Language-Action Flow Model for General Robot Control*. 2024, Appendix B. https://arxiv.org/abs/2410.24164
- Physical Intelligence. *openpi*, `pi0.py` and `pi0_config.py`. https://github.com/Physical-Intelligence/openpi
