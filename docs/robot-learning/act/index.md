---
title: "ACT"
kind: "canonical"
domain: "Robot Learning / ACT"
parent: "ACT"
canonical: "/robot-learning/act/"
prerequisites:
  - "/robot-learning/behavior-cloning/"
related:
  - "/robot-learning/act/action-chunking/"
  - "/robot-learning/act/temporal-ensemble/"
  - "/robot-learning/act/architecture/"
  - "/robot-learning/act/cvae-in-act/"
---

# ACT

ACT（Action Chunking with Transformers）是一种从机器人当前观测直接预测未来一段动作的 imitation-learning policy。它最容易被理解成下面这个系统：

```text
当前时刻 t

多路相机图像 ───────┐
                    │
当前关节位置 q_t ───┤
                    ↓
                   ACT
                    ↓
        未来一段连续动作
[a_t, a_{t+1}, ..., a_{t+k-1}]
```

如果机器人有 14 个需要控制的关节，并且 chunk size 为 100，那么 ACT 一次输出的不是一个 14 维动作，而是一个大约为

\[
100\times 14
\]

的动作序列。每一行对应未来一个 timestep 的 joint target。

这件事——**一次预测一段动作，而不是只预测下一步**——是理解 ACT 的起点。Transformer、CVAE、ResNet 都是帮助它完成这件事的组件，不是 ACT 本身的定义。

## 从单步控制到动作片段

普通 [Behavior Cloning](/robot-learning/behavior-cloning/) 可以把机器人 policy 写成

\[
\hat a_t=\pi_\theta(o_t),
\]

其中 $o_t$ 是当前观测，$\hat a_t$ 是下一步动作。

这意味着一段 500 步的任务，policy 要连续做大约 500 次彼此衔接的决策。前面某一步只要稍微偏离 demonstration，后面看到的状态就可能越来越不像训练数据，误差也会继续传下去。

ACT 改成

\[
(\hat a_t,\hat a_{t+1},\ldots,\hat a_{t+k-1})
=\pi_\theta(o_t).
\]

模型现在一次表达的是一个短时间内完整、连贯的动作片段。例如“手向前移动、夹住物体、开始抬起”可以由同一次预测覆盖，而不必在每一个控制周期重新独立决定下一步。

这就是 [Action Chunking](/robot-learning/act/action-chunking/)。

## ACT 运行起来是什么样

“预测 100 步”不等于“预测一次以后 100 步都不再看相机”。ACT 的实际执行仍然可以保持闭环：在下一个 timestep，机器人拿到新的图像和关节状态，再次调用 policy，又得到一个新的 action chunk。

于是会出现重叠：

```text
t 时刻预测：      a_t   a_t+1 a_t+2 a_t+3 ...
t+1 时刻预测：          a_t+1 a_t+2 a_t+3 ...
t+2 时刻预测：                a_t+2 a_t+3 ...
```

对同一个未来时刻，例如 $t+2$，模型可能已经给出多次预测。ACT 的 [Temporal Ensemble](/robot-learning/act/temporal-ensemble/) 会把这些预测加权融合，而不是只保留其中一个。

因此 ACT 同时保留了两件事：

- **chunk-level prediction**：一次预测较长的动作结构；
- **closed-loop replanning**：每一步仍能利用最新观测修正后续动作。

## 整个模型先看一遍

先忽略内部细节，ACT policy 可以画成：

```text
Camera images
     │
     ↓
   ResNet
     │
     ├─────────────┐
     │             │
Joint state ───────┤
                   ↓
          Transformer Encoder
                   │
                   ↓
                Memory
                   │
          Action Queries
                   │
                   ↓
          Transformer Decoder
                   │
                   ↓
           Future action chunk
```

视觉图像先经过 [ResNet](/deep-learning/cnn/resnet/) 变成 feature map；当前 joint state 也被映射到同一个 hidden dimension。它们一起进入 Transformer，使模型能够把多个相机区域、机器人自身状态与未来动作槽位联系起来。

这里的 Transformer 并不是为了生成文字。它只是一个能够让不同信息相互读取、再并行产生多个输出位置的网络结构。

## Training-only latent z

到目前为止，这个模型看起来像一个普通的确定性网络：给 observation，输出 action chunk。

但人类 demonstration 常常不是完全一致的。即使面对相近状态，不同示范也可能有不同的速度、微小轨迹、抓取姿态或操作习惯。ACT 因此在训练阶段再加入一个 latent variable $z$：

```text
                     ┌─ current joint state
future action chunk ─┤
                     ↓
            training-only encoder
                     ↓
                     z

camera + joint state + z
           │
           ↓
        ACT decoder
           │
           ↓
 reconstructed action chunk
```

这个结构来自 [Conditional Variational Autoencoder](/generative-models/conditional-variational-autoencoder/)。但在 ACT 中，CVAE 的作用很具体：训练时让 latent $z$ 吸收 demonstration 中难以单靠当前 observation 决定的变化。

因此训练时完整关系可以先记成：

\[
(o_t, a_{t:t+k-1})
\rightarrow z
\rightarrow \hat a_{t:t+k-1}.
\]

这里真实的 future action chunk 一方面是监督目标，另一方面也只在训练阶段用于得到 $z$。

## 推理时模型会变简单

机器人真正部署时没有“真实未来动作”可以提供给 encoder。因此 training-only CVAE encoder 会被拿掉。

ACT 论文与官方实现采用 standard normal prior，并在推理时使用其均值：

\[
z=0.
\]

于是推理时真正运行的数据流重新变成：

```text
当前图像 + 当前关节状态
            │
            ↓
       ACT policy
            │
            ↓
      future action chunk
            │
            ↓
     Temporal Ensemble
            │
            ↓
       当前执行动作
```

所以不要把 ACT 理解成“部署时先用未来动作算 z，再预测未来动作”。那条支路只属于 training。

## ACT 组件的职责分工

现在可以把整个系统拆成几块：

| 部分 | 在 ACT 中的作用 |
|---|---|
| [Behavior Cloning](/robot-learning/behavior-cloning/) | 从 demonstration 学习 robot policy 的基本训练方式 |
| [Action Chunking](/robot-learning/act/action-chunking/) | 一次预测未来一段动作，缩短 policy 层面的决策链 |
| [ResNet](/deep-learning/cnn/resnet/) | 把相机图像转成视觉 features |
| [Transformer](/deep-learning/transformer/) | 融合 observation，并让多个 action query 从 memory 中读取信息 |
| [CVAE in ACT](/robot-learning/act/cvae-in-act/) | 训练时建模 demonstration 中的 latent variation |
| [Temporal Ensemble](/robot-learning/act/temporal-ensemble/) | 融合同一执行时刻来自多个 overlapping chunks 的预测 |

这些模块之间的关系比模块名称本身更重要。ACT 的创新不是“发明 Transformer”或“发明 CVAE”，而是把它们围绕 action-chunk prediction 组织成一个适合细粒度机器人操作的 policy。

## 输入和输出的具体含义

在 ALOHA 论文的双臂设置中，observation 包含多路 RGB 图像和双臂关节位置。可以抽象写成

\[
o_t=(I_t^1, I_t^2,\ldots,I_t^C,q_t).
\]

其中：

- $I_t^c$：第 $c$ 个相机在时刻 $t$ 的图像；
- $q_t$：当前 proprioception，在论文设置中主要是 joint positions；
- $a_t$：机器人需要执行的 action，在该系统中是 absolute joint-position target。

ACT 学到的是

\[
\pi_\theta(a_{t:t+k-1}\mid o_t),
\]

更准确地说，训练阶段还通过 latent $z$ 构造一个 conditional generative model，而部署时使用固定 $z=0$ 得到确定性预测。

## 从这里继续深入

如果现在只记住一句话，可以记成：

> **ACT 是一个每一步都重新观察机器人状态、但每次一次预测未来一段动作的 imitation-learning policy。训练阶段再用 CVAE latent 建模示范差异，部署阶段通过 temporal ensemble 融合重叠的 action chunks。**

接下来可以按模型内部结构继续：

- [Action Chunking](/robot-learning/act/action-chunking/)：一次预测一段动作具体改变了什么；
- [Temporal Ensemble](/robot-learning/act/temporal-ensemble/)：重叠预测怎样融合；
- [Architecture](/robot-learning/act/architecture/)：ACT 网络内部每条数据流；
- [CVAE in ACT](/robot-learning/act/cvae-in-act/)：latent branch 在 ACT 中如何工作；
- [Training](/robot-learning/act/training/) 与 [Inference](/robot-learning/act/inference/)：训练和部署为什么不是同一张图；
- [Complete Data Flow](/robot-learning/act/complete-data-flow/)：把整个过程按 tensor/data flow 重新串起来。

## Sources

- Zhao et al., **Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware**, 2023. https://arxiv.org/abs/2304.13705
- Official ACT implementation. https://github.com/tonyzhaozh/act
