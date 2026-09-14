---
title: "Vision Pipeline"
kind: "act-topic"
domain: "Robot Learning / ACT"
parent: "ACT"
canonical: "/robot-learning/act/vision-pipeline/"
prerequisites:
  - "/deep-learning/cnn/resnet/"
  - "/deep-learning/transformer/positional-encoding/"
related:
  - "/robot-learning/act/architecture/"
---

# Vision Pipeline

ACT 的 Vision Pipeline 把多个 camera RGB images 转换成带二维位置的 visual feature sequence，再交给 observation Transformer。它的作用是保留足够的空间视觉信息，同时把二维 feature maps 转成 attention 可以处理的 token sequence。

## Paper Input

原论文 ALOHA observation 使用四个 RGB cameras：front、top 和两个 wrist cameras。每张图像分辨率为

\[
480\times640\times3.
\]

控制与数据记录频率为 50 Hz。

## ResNet18 Feature Extraction

论文使用 [ResNet](/deep-learning/cnn/resnet/)18 把每张图像转换为

\[
15\times20\times512
\]

feature map。

这里 $15\times20$ 仍是二维空间网格，512 是每个网格位置的 feature dimension。

Flatten spatial dimensions 后：

\[
15\times20=300,
\]

所以单 camera 变成

\[
300\times512.
\]

四个 cameras 合计

\[
1200\times512.
\]

## 2D Positional Encoding

如果只把 feature map flatten，Transformer 会得到一组 visual vectors，但需要额外知道这些 vectors 原本位于图像哪里。论文因此加入二维 sinusoidal position encoding。

位置编码与 feature vector 相加或在 attention 中共同使用，使视觉内容与空间坐标同时进入 Transformer。其一般原理属于 [Positional Encoding](/deep-learning/transformer/positional-encoding/)。

## Joining Other Modalities

Current joint positions 与 latent $z$ 分别投影到 512 维，再作为两个额外 features 加入 visual sequence：

\[
1200+2=1202.
\]

论文因此得到 observation encoder input

\[
1202\times512.
\]

视觉 feature 不需要先被压缩成一个全局 image vector。Transformer 可以直接在 1200 个空间位置与两个 non-visual features 之间做信息融合。

## Released Implementation

当前官方代码先把 image tensor 变为 `[batch, num_cam, channel, height, width]`，像素除以 255。进入 `ACTPolicy` 后再使用 ImageNet-style normalization：

\[
\text{mean}=(0.485,0.456,0.406),
\]

\[
\text{std}=(0.229,0.224,0.225).
\]

模型只构建一个 ACT backbone，并在 camera loop 中对每个 camera 都调用 `self.backbones[0]`，因此各 camera 共享同一 backbone weights。各 camera feature maps 随后沿 width dimension concatenate；Transformer 内部再 flatten spatial dimensions。

如果每个 camera 的 feature shape 相同，这种 concatenate-then-flatten 仍产生与把所有 camera spatial tokens 放入同一 sequence 等价的 token count，但代码层面的布局与论文的概念描述不同。

## End-to-End Training

ACT 不是先离线提取固定视觉 features 再训练控制头。论文把 ResNet visual encoder 与 policy 作为整体训练，因此 perception representation 可以针对 action prediction objective 调整。

这也是 ACT 与论文中一些使用 separately trained/frozen visual encoder 的 baselines 的重要区别之一。

## Sources

- [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware — Zhao et al., 2023](https://arxiv.org/abs/2304.13705)
- [ACT official implementation — tonyzhaozh/act](https://github.com/tonyzhaozh/act)
