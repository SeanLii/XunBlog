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

ACT 的论文描述与官方 released code 大体对应，但有几处值得单独记录。这个页面不重新解释 ACT，只记录“论文写了什么、公开实现实际做了什么”。

## Reconstruction loss

### Paper description

论文 Algorithm 1 的伪代码把 reconstruction term 写成 MSE；但正文方法部分又说明最终使用 L1 loss，并指出 L1 在实验中比 L2 更好。

因此单看 Algorithm 1 会得到不完整印象。

### Released implementation

官方 `policy.py` 明确使用

\[
\operatorname{L1Loss}(actions,a\_hat)
\]

并用 padding mask 去掉 padded timesteps。总 loss 为

\[
\mathcal L=L1+\beta KL.
\]

所以如果目标是复现 released ACT，应该以实际 L1 implementation 为准，同时保留论文内部表述差异。

## Optimizer

### Paper description

论文实验配置文字中把 optimizer 写成 Adam。

### Released implementation

官方 repository 的 model/optimizer building 基于 DETR code path，使用 AdamW-style optimizer parameter groups，并对 backbone 使用单独较小 learning rate。

因此论文中的“Adam”和 released code 的“AdamW”不应被静默合并。

## Latent dimension

### Paper description

论文描述 CVAE latent design，但很多读者容易把某个代码常量误认为算法定义。

### Released implementation

`detr_vae.py` 直接设置：

```text
latent_dim = 32
```

并把 CLS representation linear-project 成 $2\times32$ 个值，分别作为 $\mu$ 与 $\log\sigma^2$。

32 是 released implementation choice，不是 CVAE 或 ACT 的理论必需值。

## Multi-camera backbone handling

released model 对不同 cameras 调用 `self.backbones[0]`，也就是共享同一个 backbone module 来提取每路图像 feature；随后把 camera feature maps 沿 width dimension 拼接。

这类细节在高层 architecture figure 中通常不会完全展开，但复现时会影响 tensor shape 与参数共享方式。

## Temporal Ensemble implementation

论文提出对同一 timestep 的多个 overlapping predictions 使用 exponential weighting。

released evaluation code：

1. 每个 query time 产生整个 action chunk；
2. 写入 `all_time_actions[query_time, target_time]`；
3. 当前 $t$ 取 `all_time_actions[:, t]` 的已填充 rows；
4. 使用 $\exp(-0.01 i)$ 归一化权重；
5. 对候选 action vectors 加权求和。

因此阅读权重公式时应同时确认候选 predictions 的排列顺序。

## Paper configuration 与 repository defaults

论文 Table III 给出主要实验的 model hyperparameters，例如 4 encoder layers、7 decoder layers、8 heads、chunk size 100、KL weight $\beta=10$ 等。

repository 中部分参数是固定值，部分来自 command-line arguments。复现时应明确自己是在复现“论文某个实验配置”还是“repository 默认代码路径”，两者不能只靠一个表格概括。

## 保留 Paper / Code 差异的意义

论文负责描述研究方法与实验；代码负责把某个版本的方法具体实现出来。两者出现小差异并不罕见。

面对 paper 与 released code 的差异，不应该为了让叙述更整齐而挑一个版本隐藏另一个：

```text
paper claim / description
          与
released implementation behavior
```

分别记录，并说明哪一个结论来自哪里。

## Sources

- Zhao et al., **Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware**, 2023. https://arxiv.org/abs/2304.13705
- Official ACT repository. https://github.com/tonyzhaozh/act
- `policy.py`: https://github.com/tonyzhaozh/act/blob/main/policy.py
- `detr_vae.py`: https://github.com/tonyzhaozh/act/blob/main/detr/models/detr_vae.py
- `imitate_episodes.py`: https://github.com/tonyzhaozh/act/blob/main/imitate_episodes.py
