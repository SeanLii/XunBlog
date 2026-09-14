---
title: "ACT Vision Pipeline：4 张 RGB 图像到底怎样变成 1200 个 Transformer Tokens？"
description: "从 ACT 原论文与官方代码出发，完整拆解 480×640 RGB 图像经过 ImageNet normalization、ResNet18、15×20×512 feature map、1×1 projection、2D sinusoidal positional encoding、多相机拼接与 Transformer flatten 后，如何变成 1200 个视觉 tokens，并最终与 qpos、z 组成 1202×512 的 Policy Encoder 输入。"
status: reviewed
pageType: application
canonical: /robot-learning/act/vision-pipeline
updated: "2026-09-15"
---

# ACT Vision Pipeline：4 张 RGB 图像到底怎样变成 1200 个 Transformer Tokens？

我们前面已经把 ACT 的 Transformer、CVAE、Action Query 都拆开了。

现在还有一个非常重要的问题：

> **Transformer 到底看见了什么图像信息？**

ACT 的输入并不是直接把：

\[
480\times640\times3
\]

个 RGB 数字塞进 Transformer。

原论文给出的视觉链路是：

\[
\boxed{
480\times640\times3
\rightarrow
15\times20\times512
\rightarrow
300\times512
}
\]

每个 camera 产生：

\[
300
\]

个 feature vectors。

四个 camera：

\[
4\times300
=
\boxed{
1200
}
\]

个视觉 features。

再加：

- 当前 joint-position feature；
- latent \(z\) feature；

得到：

\[
\boxed{
1202\times512
}
\]

作为 Policy Transformer Encoder 的输入。

看上去只是几行 shape conversion，

但里面其实藏着非常多值得理解的问题：

1. 为什么 \(480\times640\) 会变成 \(15\times20\)？
2. 为什么刚好缩小32倍？
3. 15×20 的每个格子是不是对应原图一个 32×32 patch？
4. 如果不是，它到底看了多大区域？
5. 512 个 channel 到底是什么？
6. 一个 feature vector 是“一个物体”吗？
7. 为什么不用 ResNet 最后的 classification output？
8. 为什么保留 spatial feature map？
9. 为什么还要一个 \(1\times1\) Conv？
10. 明明 ResNet18 已经输出512维，为什么还做 512→512？
11. Flatten 之后空间结构不是没了吗？
12. 2D sinusoidal positional encoding到底怎样把空间位置补回来？
13. ACT 代码是不是直接把 positional embedding加进 feature？
14. 四个 camera是各自用一个 ResNet，还是共享同一个？
15. 四个 camera怎样拼接？
16. 为什么代码写：
   ```python
   torch.cat(..., axis=3)
   ```
   是沿 width 拼？
17. 拼完为什么是 \(15\times80\)，最后又等价于1200 tokens？
18. ACT 有没有显式 camera-ID embedding？
19. 如果没有，Transformer怎么知道某个 token来自 wrist camera 还是 top camera？
20. Policy Encoder里的第一个 token到底是视觉还是 \(z\)？
21. joint token和 \(z\) token有没有自己的 positional embedding？
22. ResNet 是 frozen 的吗？
23. ACT 的 perception 和 policy 是不是 end-to-end jointly optimized？
24. ResNet feature token和 ViT patch token有什么区别？
25. Action Query最终怎样读取这些视觉 tokens？

这一篇把整条视觉数据流彻底走完。

---

# 1. 从最原始的输入开始

ACT 原论文中的真实 ALOHA setup：

\[
\boxed{
4\text{ 个 RGB cameras}
}
\]

每张图：

\[
\boxed{
480\times640\times3
}
\]

四个相机包括：

- 两个 wrist cameras；
- 一个 front camera；
- 一个 top camera。

论文中说明 ALOHA 使用4个 Logitech C922x webcams。

---

# 2. Tensor 中的形状

对一个 batch：

\[
B
\]

官方 model forward 期望：

\[
\boxed{
image:
[B,N_{cam},3,H,W]
}
\]

例如：

\[
[B,4,3,480,640]
\]

---

# 3. 为什么 PyTorch 常用 C×H×W？

图像文件通常：

\[
H\times W\times C
\]

ACT dataset loader 会做：

```python
image_data =
    torch.einsum(
        'k h w c -> k c h w',
        image_data
    )
```

于是：

\[
[k,H,W,3]
\rightarrow
[k,3,H,W]
\]

---

# 4. 第一步：把 uint8 Pixel 转成 [0,1]

原始 camera pixel：

\[
0,\ldots,255
\]

官方 dataset code：

```python
image_data =
    image_data / 255.0
```

所以变成：

\[
\boxed{
[0,1]
}
\]

范围的 float tensor。

---

# 5. 第二步：ImageNet Normalization

ACT 使用 ImageNet-pretrained ResNet18。

因此在 Policy 入口处继续：

```python
normalize =
    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )

image =
    normalize(image)
```

---

# 6. 公式是什么？

对于 RGB channel \(c\)：

\[
\boxed{
x'_c
=
\frac{x_c-\mu_c}{\sigma_c}
}
\]

其中：

\[
\mu=
[
0.485,
0.456,
0.406
]
\]

\[
\sigma=
[
0.229,
0.224,
0.225
]
\]

---

# 7. 为什么要这么 Normalize？

因为 backbone 初始化自：

> ImageNet pretrained weights。

这些 convolution filters是在相应 image normalization convention 下训练的。

如果输入 scale/distribution突然完全不同，

早期 feature activations也会：

> 大幅偏离预训练阶段。

---

# 8. Normalization 不会改变 Shape

仍然：

\[
[B,4,3,480,640]
\]

它只改变：

> 数值尺度。

---

# 9. 接下来逐 Camera 处理

官方：

```python
for cam_id, cam_name in enumerate(
    self.camera_names
):
    features, pos =
        self.backbones[0](
            image[:, cam_id]
        )
```

对第 \(c\) 个 camera：

\[
image[:,c]
\]

shape：

\[
\boxed{
[B,3,480,640]
}
\]

---

# 10. 一个重要实现事实：Released ACT 共享同一个 Backbone

注意官方代码是：

```python
self.backbones[0](
    image[:, cam_id]
)
```

不是：

```python
self.backbones[cam_id]
```

对于 ACT policy，

`build()` 只创建一个：

```python
backbone = build_backbone(args)
backbones.append(backbone)
```

然后所有 camera重复使用：

\[
\boxed{
\text{同一个 ResNet18 参数集合}
}
\]

---

# 11. 所以不是 4 套独立 ResNet18

更准确：

\[
I^{(1)}
\xrightarrow{\text{same ResNet}}
F^{(1)}
\]

\[
I^{(2)}
\xrightarrow{\text{same ResNet}}
F^{(2)}
\]

\[
I^{(3)}
\xrightarrow{\text{same ResNet}}
F^{(3)}
\]

\[
I^{(4)}
\xrightarrow{\text{same ResNet}}
F^{(4)}
\]

---

# 12. 为什么共享 Backbone 很合理？

四路输入都是：

> RGB images。

卷积网络需要学习的：

- edge；
- texture；
- shape；
- object appearance；

很多视觉 feature可以共享。

这也减少参数量。

---

# 13. 但 Viewpoint 不同怎么办？

虽然参数共享，

不同 camera图像内容不同。

所以：

\[
F^{(1)}
\neq
F^{(2)}
\]

模型仍然可以得到不同 feature values。

---

# 14. 现在进入 ResNet18

ACT 使用：

\[
\boxed{
ResNet18
}
\]

而且不是最终分类器输出。

官方 backbone 只返回：

\[
\boxed{
layer4
}
\]

feature map。

---

# 15. 为什么不用 ResNet 的最后 Classification Logits？

标准 ImageNet ResNet 最后：

```text
layer4
↓
global average pool
↓
FC
↓
1000 class logits
```

如果拿：

\[
1000
\]

类 logits，

大量 spatial structure已经：

> 被 global pooling压掉。

机器人控制需要知道：

- 物体在哪；
- gripper在哪；
- 左右关系；
- 接触位置；

所以必须保留：

\[
\boxed{
\text{spatial feature map}
}
\]

---

# 16. ACT Backbone 截止在 layer4

官方 backbone：

```python
return_layers =
    {'layer4': '0'}
```

通过：

```python
IntermediateLayerGetter
```

拿：

> ResNet layer4 output。

没有执行：

- avgpool；
- fc；

作为视觉 policy feature。

---

# 17. 为什么输出是 15×20？

这来自 ResNet18 的：

\[
\boxed{
\text{output stride}=32
}
\]

---

# 18. ResNet18 的 Downsampling Path

标准 torchvision ResNet18：

```text
conv1:
7×7, stride 2

maxpool:
3×3, stride 2

layer1:
stride 1

layer2:
stride 2

layer3:
stride 2

layer4:
stride 2
```

总 stride：

\[
2\times2\times1\times2\times2\times2
=
\boxed{
32
}
\]

---

# 19. Height

\[
480/32
=
\boxed{
15
}
\]

---

# 20. Width

\[
640/32
=
\boxed{
20
}
\]

所以：

\[
\boxed{
480\times640
\rightarrow
15\times20
}
\]

---

# 21. 完整 Shape 表

单张图片：

\[
[3,480,640]
\]

大致经过：

```text
Input
3 × 480 × 640

conv1, stride 2
64 × 240 × 320

maxpool, stride 2
64 × 120 × 160

layer1
64 × 120 × 160

layer2, stride 2
128 × 60 × 80

layer3, stride 2
256 × 30 × 40

layer4, stride 2
512 × 15 × 20
```

---

# 22. 所以 ACT Paper 的

\[
480\times640\times3
\rightarrow
15\times20\times512
\]

并不是神秘 magic。

就是：

\[
\boxed{
\text{ResNet18 layer4 feature map}
}
\]

---

# 23. 为什么 Channels 最终是 512？

ResNet18 的 stage channels：

\[
64
\rightarrow
64
\rightarrow
128
\rightarrow
256
\rightarrow
512
\]

layer4输出：

\[
\boxed{
512
}
\]

channels。

官方 ACT backbone也明确：

```python
num_channels =
    512
    if name in (
        'resnet18',
        'resnet34'
    )
    else 2048
```

---

# 24. 一个 Feature Map Cell 到底是什么？

现在有 tensor：

\[
F
\in
\mathbb R^{512\times15\times20}
\]

选择空间位置：

\[
(r,c)
\]

得到：

\[
\boxed{
F[:,r,c]
\in
\mathbb R^{512}
}
\]

这就是一个：

> 512维视觉 feature vector。

---

# 25. 这个向量不是“512个像素”

它是：

> 多层卷积经过学习后形成的 512 个 feature channels 的 activation。

每一维：

\[
F_j(r,c)
\]

表示第 \(j\) 个 learned channel：

> 对当前 spatial location附近视觉模式的响应。

---

# 26. Channel 不是人工定义语义

不能说：

```text
channel 0 = cup
channel 1 = gripper
channel 2 = table
```

网络没有这样的人工标签。

更准确：

\[
\boxed{
\text{512 channels form a learned feature basis}
}
\]

---

# 27. 某些 Channel 可能对某种 Pattern 更敏感

例如：

- edge；
- material；
- object part；
- shape；
- texture；
- pose feature；

但 deeper representation通常是：

> distributed。

一个概念可能由很多 channels共同表示。

---

# 28. 15×20 的一个格子是不是等于 32×32 Pixel Patch？

这是一个非常常见的误解。

因为：

\[
\text{stride}=32
\]

所以相邻 feature centers相隔大约：

\[
32
\]

个 input pixels。

但这不代表：

> 每个 feature只看独立的32×32区域。

---

# 29. Stride 和 Receptive Field 是两个不同概念

### Stride / Jump

相邻 feature位置在原图坐标上的中心间隔。

ACT ResNet18 layer4：

\[
\boxed{
j=32
}
\]

---

### Receptive Field

一个 output feature理论上可以受到多大 input区域影响。

它通常：

> 远大于32×32。

---

# 30. ResNet18 的 Theoretical Receptive Field

用标准 receptive-field recursion：

\[
j_l
=
j_{l-1}s_l
\]

\[
r_l
=
r_{l-1}
+
(k_l-1)j_{l-1}
\]

其中：

- \(j\)：input-coordinate jump；
- \(r\)：receptive field size；
- \(s\)：stride；
- \(k\)：kernel size。

---

# 31. 从 Input 开始

\[
r_0=1
\]

\[
j_0=1
\]

---

# 32. conv1：7×7, stride 2

\[
r=1+(7-1)\times1=7
\]

\[
j=2
\]

---

# 33. maxpool：3×3, stride 2

\[
r=7+(3-1)\times2
=
11
\]

\[
j=4
\]

---

# 34. layer1

ResNet18 layer1有：

> 2个 BasicBlocks，

每个2个 3×3 conv。

一共：

\[
4
\]

个 stride-1 3×3 conv。

每一个增加：

\[
2\times4=8
\]

所以：

\[
r
=
11+4\times8
=
43
\]

\[
j=4
\]

---

# 35. layer2

第一 block第一 conv stride2：

\[
r=43+2\times4=51
\]

\[
j=8
\]

后面3个3×3 conv每个增加：

\[
2\times8=16
\]

所以：

\[
r
=
51+3\times16
=
99
\]

---

# 36. layer3

第一 conv stride2：

\[
r
=
99+2\times8
=
115
\]

\[
j=16
\]

后面3个 conv：

\[
3\times32
=
96
\]

所以：

\[
r
=
211
\]

---

# 37. layer4

第一 conv stride2：

\[
r
=
211+2\times16
=
243
\]

\[
j=32
\]

后面3个 conv：

\[
3\times64
=
192
\]

所以：

\[
\boxed{
r\approx435
}
\]

---

# 38. 所以一个 layer4 Cell 理论上能“看”多大区域？

大约：

\[
\boxed{
435\times435
}
\]

input-pixel receptive field，

忽略边界/padding差异。

---

# 39. 这说明什么？

虽然 feature grid只有：

\[
15\times20
\]

但每个 feature并不是：

> 只描述局部32×32 patch。

它已经融合：

> 非常大范围的上下文。

---

# 40. 但“理论 Receptive Field 435”也不能过度理解

理论 receptive field：

> 表示哪些 pixels存在 computational path影响该 feature。

真实 effective receptive field：

> 往往并不是区域内所有像素贡献相同。

通常中心区域贡献更大。

所以：

\[
435\times435
\]

是：

> architecture-derived theoretical receptive field，

不是说 feature均匀看完435×435。

---

# 41. 这就是 CNN Token 和 ViT Patch Token 的第一个区别

ViT标准 patch embedding：

> 一开始把固定 patch，例如16×16，直接线性投影成 token。

ResNet feature token：

> 是经过多层 overlapping convolution和downsampling后形成的 feature。

因此一个 token的 receptive field：

> 可以覆盖很多相邻区域。

---

# 42. ACT 的 Visual Token 更准确叫

\[
\boxed{
\text{CNN spatial feature token}
}
\]

而不是：

> raw image patch token。

---

# 43. 一个 Visual Feature Cell 还保留 Spatial Position 吗？

在 tensor中：

\[
F[:,r,c]
\]

当然还有：

\[
r,c
\]

坐标。

但一旦我们 flatten：

\[
15\times20
\rightarrow
300
\]

Transformer只是看到：

\[
300
\]

个 vectors。

---

# 44. Self-Attention 本身并不知道二维几何

如果没有 positional information，

Transformer只处理：

> 一组 token vectors。

交换两个 visual tokens，

模型没有原生的“上/下/左/右”概念。

---

# 45. 所以 ACT 需要 2D Positional Encoding

原论文明确：

> To preserve the spatial information, we add a 2D sinusoidal position embedding.

它来自 DETR-style image positional encoding。

---

# 46. 为什么是 2D 而不是普通 1D？

图像天然有：

\[
(r,c)
\]

两个空间坐标。

如果只用 flatten index：

\[
0,1,\ldots,299
\]

虽然也能区分 token，

但二维结构：

> 没有被显式分开编码。

2D encoding分别表示：

- vertical coordinate；
- horizontal coordinate。

---

# 47. ACT Official PositionEmbeddingSine

对于 feature map：

\[
H=15,\ W=20
\]

代码构造：

\[
y_{r,c}
\]

和：

\[
x_{r,c}
\]

coordinates。

---

# 48. 坐标先 Normalize 到 2π

官方代码：

```python
y_embed =
    y_embed /
    (
        y_embed[:, -1:, :]
        + eps
    ) * 2π

x_embed =
    x_embed /
    (
        x_embed[:, :, -1:]
        + eps
    ) * 2π
```

所以不同 image size下：

> positional phase scale保持类似。

---

# 49. 然后构造多个 Frequency

类似原始 Transformer：

\[
\sin(
x/\tau_i
)
\]

\[
\cos(
x/\tau_i
)
\]

以及：

\[
\sin(
y/\tau_i
)
\]

\[
\cos(
y/\tau_i
)
\]

其中不同维度：

> 使用不同 frequency scale。

---

# 50. 为什么最终 Positional Encoding 是 512维？

ACT：

\[
hidden\_dim=512
\]

代码：

```python
N_steps =
    args.hidden_dim // 2
```

所以：

\[
N_{\text{steps}}
=
256
\]

---

# 51. 256维给 Y

最终：

\[
pos_y
\in
\mathbb R^{256}
\]

---

# 52. 256维给 X

\[
pos_x
\in
\mathbb R^{256}
\]

---

# 53. Concatenate

\[
\boxed{
pos(r,c)
=
[
pos_y(r),
pos_x(c)
]
\in
\mathbb R^{512}
}
\]

所以每一个视觉 token：

> 都有一个512维空间 positional vector。

---

# 54. 这就是“2D”真正的意思

并不是：

> Position Embedding是2维 vector。

而是：

> 编码两个坐标轴。

最终 embedding仍然：

\[
512
\]

维。

---

# 55. Feature 和 Position Shape 一致

Visual feature：

\[
F
\in
\mathbb R^{B\times512\times15\times20}
\]

Position：

\[
P
\in
\mathbb R^{B\times512\times15\times20}
\]

因此空间上：

> 每个 feature cell有一个对应 positional code。

---

# 56. Paper 说“add position embedding”，代码真的直接加了吗？

这是一个有意思的实现细节。

论文概念上说：

\[
feature+position
\]

但是 official DETR-style implementation：

> 并不把 position永久写入 feature tensor。

它把：

```python
src
```

和：

```python
pos
```

分开传给 Transformer。

---

# 57. Transformer Encoder Self-Attention 中

代码：

```python
q = k =
    src + pos

src2 =
    self.self_attn(
        q,
        k,
        value=src
    )[0]
```

所以：

\[
\boxed{
Q/K\text{ source}
=
src+pos
}
\]

而：

\[
\boxed{
V\text{ source}
=
src
}
\]

---

# 58. 为什么这样做？

Positional encoding主要影响：

> token之间“谁和谁匹配”。

它帮助 Attention知道：

- 这个 feature在左边；
- 那个 feature在上方。

真正被读取的内容：

\[
V
\]

仍主要是 visual content representation。

---

# 59. 所以更精确地说

Paper-level：

\[
\boxed{
\text{visual features receive 2D positional information}
}
\]

Code-level：

\[
\boxed{
\text{position is injected into Q/K in DETR-style attention}
}
\]

---

# 60. 现在回到 1×1 Conv

官方 ACT：

```python
self.input_proj =
    nn.Conv2d(
        backbones[0].num_channels,
        hidden_dim,
        kernel_size=1
    )
```

ResNet18：

\[
backbone\ channels=512
\]

ACT：

\[
hidden\_dim=512
\]

所以：

\[
\boxed{
512
\rightarrow
512
}
\]

---

# 61. 为什么一样维度还要 Projection？

因为：

\[
\text{same dimension}
\neq
\text{same representation space}
\]

ResNet的512维：

> 是视觉 backbone feature basis。

Transformer的512维：

> 是 policy hidden embedding space。

一个 learned linear projection可以：

> 重新组合 feature channels。

---

# 62. 1×1 Conv 在每个 Spatial Position 做什么？

对于：

\[
x_{r,c}
\in
\mathbb R^{512}
\]

1×1 Conv本质：

\[
\boxed{
y_{r,c}
=
Wx_{r,c}+b
}
\]

其中：

\[
W
\in
\mathbb R^{512\times512}
\]

---

# 63. 它不会混 Spatial Neighbors

kernel：

\[
1\times1
\]

所以 output \((r,c)\)：

> 只读取同一个 spatial location的512 channels。

它做的是：

\[
\boxed{
\text{channel mixing}
}
\]

不是：

> spatial mixing。

---

# 64. 这为什么和“Flatten 后 Linear”几乎等价？

如果把 feature map变成：

\[
X
\in
\mathbb R^{300\times512}
\]

然后对每个 row共享同一个：

\[
Linear(512,512)
\]

数学上就是：

> 同一个位置独立 channel projection。

这和 \(1\times1\) Conv等价。

---

# 65. Paper Figure 11 有一个值得标记的细节

Figure 11 的图中文字显示：

\[
15\times20\times728
\]

以及：

\[
728\rightarrow512
\]

但：

- 论文正文明确写：
  \[
  15\times20\times512
  \]
- official ResNet18 code明确：
  \[
  num\_channels=512
  \]
- official `input_proj`在canonical hidden_dim=512时：
  \[
  512\rightarrow512
  \]

---

# 66. 所以 Figure 11 的 “728” 与正文和 released code 不一致

最稳妥的知识库写法：

\[
\boxed{
\text{正文与released implementation均支持512 channels；Figure 11中的728应视为图示不一致/疑似标注错误。}
}
\]

不要按照图中的728去实现 canonical code。

---

# 67. 这是读 Paper 时非常典型的一课

不要因为：

> Figure里出现一个数字，

就忽略：

- main text；
- source code；
- architecture definition。

当三者冲突：

> 必须明确记录 discrepancy。

---

# 68. 每个 Camera 现在得到什么？

经过：

```python
features =
    self.input_proj(features)
```

得到：

\[
\boxed{
F^{(c)}
\in
\mathbb R^{B\times512\times15\times20}
}
\]

同时：

\[
\boxed{
P^{(c)}
\in
\mathbb R^{B\times512\times15\times20}
}
\]

---

# 69. Paper 的 Flatten 是什么？

对空间维：

\[
15\times20
=
300
\]

因此：

\[
\boxed{
15\times20\times512
\rightarrow
300\times512
}
\]

---

# 70. 每一个 Spatial Cell 变成一个 Token

可以编号：

\[
v_1,\ldots,v_{300}
\]

其中：

\[
v_i
\in
\mathbb R^{512}
\]

---

# 71. 一个 Camera 就是 300 个 Visual Tokens

```text
camera c

15 × 20 spatial grid
↓
flatten
↓
300 feature vectors
↓
300 visual tokens
```

---

# 72. 四个 Camera

理论描述：

\[
300+300+300+300
=
\boxed{
1200
}
\]

visual tokens。

---

# 73. Official Code 并不是先 flatten 每个 Camera 再 cat

它先：

```python
src =
    torch.cat(
        all_cam_features,
        axis=3
    )
```

axis 3：

> width dimension。

---

# 74. 为什么是 Width？

单 camera：

\[
[B,512,15,20]
\]

四个 camera沿 width拼：

\[
\boxed{
[B,512,15,80]
}
\]

因为：

\[
20\times4
=
80
\]

---

# 75. Positional Tensor 也同样拼

```python
pos =
    torch.cat(
        all_cam_pos,
        axis=3
    )
```

所以：

\[
\boxed{
[B,512,15,80]
}
\]

---

# 76. 然后 Transformer 内部 Flatten

官方：

```python
src =
    src.flatten(2)
       .permute(2,0,1)
```

于是：

\[
[B,512,15,80]
\]

先：

\[
[B,512,1200]
\]

再：

\[
\boxed{
[1200,B,512]
}
\]

---

# 77. 所以“沿 Width 拼”只是 Implementation Trick

最终本质仍是：

\[
\boxed{
1200\text{ visual tokens}
}
\]

而不是说：

> 模型真的认为四张图物理上连成一张80格宽的camera panorama。

---

# 78. 但这里有一个很高级的细节：Position Embedding 会怎样？

每个 camera 的 position encoding：

> 是在自己的15×20 feature map上单独计算的。

所以 camera 1 的：

\[
(r,c)
\]

和 camera 2 的：

\[
(r,c)
\]

会得到：

> 同样形式的 2D sinusoidal position code。

---

# 79. 然后这些 Position Maps 沿 Width 直接拼

因此并不是重新计算一个：

\[
15\times80
\]

全局 coordinate system。

---

# 80. 换句话说

camera 1：

\[
c=0,\ldots,19
\]

camera 2：

> 又重新使用

\[
c=0,\ldots,19
\]

的 positional pattern。

同理 camera 3、4。

---

# 81. Released ACT 有显式 Camera-ID Embedding 吗？

在这条代码路径中：

\[
\boxed{
没有。
}
\]

没有看到：

```python
camera_embed[cam_id]
```

这样的 learnable camera identity token。

---

# 82. Backbone 也共享

所有 camera使用：

```python
self.backbones[0]
```

因此 architecture没有通过：

> 独立 backbone parameters

显式编码 camera ID。

---

# 83. 那 Transformer 怎么区分 Camera？

严格来说：

> **Released code没有给每个 visual token显式附加 camera-ID embedding。**

它能利用的差异主要来自：

1. 不同 camera的实际图像内容；
2. 每个固定 viewpoint形成的不同 visual statistics；
3. 上游 feature values本身。

---

# 84. Camera Content 可以隐式泄露 View Identity

例如 wrist camera常看到：

- gripper占很大比例；
-近距离 object；
- 特定背景。

top camera：

- 全局桌面；
- 俯视 geometry。

所以模型可以根据 content：

> 推断“这大概是哪一路”。

---

# 85. 但从纯 Architecture Symmetry 看

如果把两个 camera feature blocks连同它们重复的 spatial positional codes整体互换，

Transformer并没有一个明确：

> `camera_id=1` / `camera_id=2`

标签来区分。

---

# 86. 这是值得研究的 Design Detail

现代 multi-camera policy常会加：

- camera embeddings；
- modality embeddings；
- view tokens；

来显式告诉 Transformer：

> token来自哪个 camera。

Canonical ACT released code：

> 没有这样做。

---

# 87. 不能因此说 ACT “不知道 Camera”

模型仍可从视觉内容：

> 学会 viewpoint-specific behavior。

但更准确：

\[
\boxed{
\text{camera identity is implicit in content, not explicitly encoded as a separate learned ID in released ACT.}
}
\]

---

# 88. Camera Order 有没有用？

代码有固定：

```python
camera_names
```

顺序，

并按这个顺序 cat。

工程上：

> 必须保持一致顺序。

---

# 89. 但 Attention 理论上并不自动读取“tensor第几段”作为 Camera ID

Transformer需要：

> positional/modality signal

才能显式知道 slot ordering。

由于相机局部2D pos重复，

camera block order本身并没有单独 camera embedding编码。

所以不要把：

> Python list顺序

等同于：

> 网络明确拥有 camera-ID representation。

---

# 90. 现在加入 Joint Token

当前：

\[
qpos
\in
\mathbb R^{14}
\]

官方：

```python
proprio_input =
    self.input_proj_robot_state(
        qpos
    )
```

其中：

```python
Linear(14,512)
```

所以：

\[
\boxed{
qpos:
[B,14]
\rightarrow
[B,512]
}
\]

---

# 91. 加入 z Token

训练：

\[
z\in\mathbb R^{32}
\]

推理：

\[
z=0
\]

官方：

```python
latent_input =
    self.latent_out_proj(
        latent_sample
    )
```

其中：

\[
Linear(32,512)
\]

所以：

\[
\boxed{
z:
[B,32]
\rightarrow
[B,512]
}
\]

---

# 92. Paper 说“append two more features”

概念上：

\[
1200
+
1
+
1
=
\boxed{
1202
}
\]

---

# 93. Released Code 的实际 Sequence Order

Transformer内部：

```python
addition_input =
    torch.stack(
        [
            latent_input,
            proprio_input
        ],
        axis=0
    )

src =
    torch.cat(
        [
            addition_input,
            src
        ],
        axis=0
    )
```

所以实际顺序：

```text
token 0:
z

token 1:
qpos

token 2 ...:
visual tokens
```

---

# 94. 即

\[
\boxed{
[
z,\ qpos,\ v_1,\ldots,v_{1200}
]
}
\]

总长度：

\[
\boxed{
1202
}
\]

---

# 95. 这和“append”这个自然语言描述不完全相同

Paper只是说：

> append two more features。

Released implementation其实：

> prepend到视觉 token sequence前面。

但对 self-attention而言：

> 只要 positional identity设计对应，主要语义不依赖英语里“append”的字面顺序。

---

# 96. z 和 qpos 也有 Position Embeddings 吗？

有。

官方：

```python
self.additional_pos_embed =
    nn.Embedding(
        2,
        hidden_dim
    )
```

所以有两个 learned positional embeddings：

\[
p_z
\]

和：

\[
p_q
\]

---

# 97. Transformer 里

```python
additional_pos_embed =
    additional_pos_embed
    .unsqueeze(1)
    .repeat(1,bs,1)
```

得到：

\[
[2,B,512]
\]

---

# 98. 然后和 Visual Positional Embeddings 拼起来

\[
[2,B,512]
+
[1200,B,512]
\]

得到：

\[
\boxed{
[1202,B,512]
}
\]

position sequence。

---

# 99. 所以完整 Policy Encoder Inputs

Content：

\[
\boxed{
X=
[
z,\ qpos,\ v_1,\ldots,v_{1200}
]
}
\]

Position：

\[
\boxed{
P=
[
p_z,\ p_q,\ p_1,\ldots,p_{1200}
]
}
\]

---

# 100. Encoder Self-Attention 中

\[
Q=(X+P)W_Q
\]

\[
K=(X+P)W_K
\]

\[
V=XW_V
\]

概念上如此。

---

# 101. 这意味着 z Token 也能读 Image

Policy Encoder Self-Attention没有 modality隔离。

所以：

\[
z
\]

token可以 attend：

- qpos；
- visual tokens。

---

# 102. qpos Token 也能读 Image

同理：

\[
qpos
\]

representation经过 self-attention后：

> 可以融合视觉信息。

---

# 103. Visual Token 也能读 z 和 qpos

某 wrist-camera token：

> 可以 attend当前关节姿态和 latent style。

因此 Encoder做的是：

\[
\boxed{
\text{multimodal fusion}
}
\]

---

# 104. 这也是为什么不能把 Encoder 输出简单叫“图像 features”

经过4层 Self-Attention后：

> 视觉、joint、latent已经彼此 contextualized。

Encoder memory是：

\[
\boxed{
\text{observation-conditioned multimodal memory}
}
\]

---

# 105. 1202 个 Token 的计算成本

Self-Attention score matrix：

\[
1202\times1202
\]

每个 head：

\[
\approx1.445\times10^6
\]

pairwise scores。

---

# 106. 8 Heads

每 layer：

\[
8\times1202^2
\]

虽然实际实现通过 batched matrix multiplication完成，

但这说明：

> 多相机 spatial token数量直接影响 Attention cost。

---

# 107. 为什么不用 Raw Pixels 当 Tokens？

如果每个 pixel一个 token：

四张图：

\[
4\times480\times640
=
\boxed{
1,228,800
}
\]

tokens。

Self-Attention：

\[
N^2
\]

会巨大到：

> 完全不现实。

---

# 108. ResNet 做了两件事

### 1. Spatial Compression

\[
480\times640
\rightarrow
15\times20
\]

每 camera：

\[
307,200
\text{ pixels}
\rightarrow
300
\text{ positions}
\]

---

### 2. Semantic Feature Extraction

每个 spatial position不再是：

> 一个 RGB pixel，

而是：

\[
512\text{-D learned visual feature}
\]

---

# 109. 所以 ResNet 是一个 Visual Tokenizer 吗？

作为直觉：

> 可以这么看。

它把 dense RGB：

\[
I
\]

变成：

\[
\boxed{
\{v_i\}_{i=1}^{300}
}
\]

这样的视觉 token set/grid。

---

# 110. 但它不是离散 Tokenizer

这些：

\[
v_i
\]

是 continuous vectors。

不是：

- VQ code；
- token ID；
- integer vocabulary。

---

# 111. 为什么 15×20 是一个合理折中？

更高 resolution：

> spatial precision更强，

但 token数和 attention cost增加。

更低 resolution：

> 计算便宜，

但小物体/精细位置可能丢失。

ACT面对：

> fine manipulation。

15×20仍然保留了一个粗 spatial grid，

并依赖：

- CNN receptive field；
- multiple cameras；

获得控制所需信息。

---

# 112. 但论文也明确承认视觉感知仍是限制

例如 cable tie这样的：

> 低对比、小物体

很难从image observations精确定位。

论文把：

- pretraining；
- more data；
- better perception；

列为未来改进方向。

---

# 113. 15×20 会不会太粗，无法毫米级控制？

这是一个很好的问题。

单看 feature-grid spacing：

\[
32\text{ pixels}
\]

似乎很粗。

但不能把它理解成：

> 位置只能以32 pixels为单位。

---

# 114. CNN Feature 不是 Hard Quantized Coordinate

由于：

- overlapping receptive fields；
- convolution activations；
- channel amplitudes；

物体在cell内部发生小位移时：

> feature values也会连续变化。

所以网络可以：

> 从连续 feature pattern中推断 sub-grid information。

---

# 115. 一个直觉例子

假设小物体从：

\[
x=100
\]

移动到：

\[
x=105
\]

它不一定导致：

> “完全相同token直到跨过32px边界”。

卷积 receptive fields重叠，

邻近 cell activations会：

> 连续变化。

---

# 116. 但 Downsampling 仍然会造成信息损失

这不是说：

> 32× downsampling完全没有 precision cost。

高频细节：

> 确实可能被削弱。

这也是为什么视觉 backbone设计对精细 manipulation非常重要。

---

# 117. Wrist Camera 为什么有价值？

全局 top/front camera：

> 看任务整体。

Wrist cameras：

> 提供 gripper附近的近距离细节。

因此即使同样15×20 feature grid，

wrist view里的目标：

> 在原始图中通常占更大像素区域。

这提高：

> fine manipulation局部视觉分辨率。

---

# 118. 多相机实际上提供 Multi-Scale-by-View

不是传统 image pyramid，

但不同 viewpoint天然提供：

- global context；
- local detail。

所以：

\[
\boxed{
\text{4-camera fusion}
}
\]

是 ACT perception能力的重要部分。

---

# 119. ResNet Backbone 是 Frozen 的吗？

需要区分：

### FrozenBatchNorm

官方 ResNet使用：

```python
norm_layer =
    FrozenBatchNorm2d
```

所以 BatchNorm statistics/affine：

> fixed。

---

# 120. 但 Convolution Weights 呢？

ACT `backbone.py` 原本 inherited DETR 的 freeze logic：

```python
parameter.requires_grad_(False)
```

在 released ACT code里：

> 这一段被注释掉了。

所以 backbone convolution parameters：

> 保持 trainable。

---

# 121. Optimizer 还专门给 Backbone 一个 Learning Rate Group

官方：

```python
{
    "params":
        backbone_params,
    "lr":
        args.lr_backbone
}
```

canonical：

\[
\boxed{
lr_{backbone}
=
10^{-5}
}
\]

---

# 122. 所以 Released ACT 是怎样的？

更准确：

\[
\boxed{
\text{ImageNet-pretrained ResNet18 is fine-tuned jointly with the policy, using a dedicated small backbone learning rate, while BatchNorm is frozen.}
}
\]

---

# 123. 为什么 Backbone Learning Rate 要小？

Pretrained visual features已经有：

> 很有用的通用表示。

如果 learning rate太大：

> 可能快速破坏 pretrained representation。

因此通常用较小：

\[
lr_{\text{backbone}}
\]

fine-tune。

---

# 124. 这和 BeT Baseline 有一个区别

ACT论文比较方法时指出：

> BeT的image observations由 separately trained frozen visual encoder预处理，

perception和control：

> 不是 jointly optimized。

ACT则允许视觉 feature：

> 根据 robot imitation objective一起适配。

---

# 125. 所以 ACT 的 L1 Gradient 可以回到 ResNet

训练 graph：

\[
L_1
\rightarrow
action\ head
\rightarrow
decoder
\rightarrow
encoder
\rightarrow
visual\ token
\rightarrow
input\_proj
\rightarrow
ResNet18
\]

因此视觉 filters可以：

> 学习什么图像信息最有利于动作预测。

---

# 126. 这就是 End-to-End Perception-to-Control

不是：

```text
先固定视觉算法
↓
输出物体坐标
↓
另一个控制器
```

而是：

\[
\boxed{
pixels
\rightarrow
features
\rightarrow
actions
\rightarrow
imitation\ loss
}
\]

整个路径可微。

---

# 127. 但 FrozenBatchNorm 为什么仍然存在？

小机器人 dataset：

> batch较小。

如果 BatchNorm running statistics不断更新，

可能：

- noisy；
- unstable；
- 偏离 ImageNet pretrained calibration。

FrozenBN是 DETR-style backbone design的一部分。

---

# 128. `pretrained=is_main_process()` 是什么？

Released code使用旧 torchvision API：

```python
pretrained =
    is_main_process()
```

在常见单进程训练中：

> main process会加载 pretrained ResNet weights。

这是历史实现细节。

现代 torchvision API通常使用：

```python
weights=...
```

---

# 129. 现在看一个完整 Single-Camera Shape Trace

输入：

\[
[B,3,480,640]
\]

---

ResNet18：

\[
[B,512,15,20]
\]

---

1×1 projection：

\[
[B,512,15,20]
\]

---

2D positional embedding：

\[
[B,512,15,20]
\]

---

Flatten：

\[
[300,B,512]
\]

所以：

\[
\boxed{
300\text{ visual tokens / camera}
}
\]

---

# 130. 四路 Camera

```text
cam 1 → 300
cam 2 → 300
cam 3 → 300
cam 4 → 300
```

总：

\[
\boxed{
1200
}
\]

---

# 131. 再加 qpos 和 z

\[
1200+1+1
=
\boxed{
1202
}
\]

最终：

\[
\boxed{
[1202,B,512]
}
\]

进入 Transformer Encoder。

---

# 132. 这和 Paper 完全对应

论文正文：

\[
4\times
(
480\times640\times3
)
\]

经过 ResNet18：

\[
4\times
(
15\times20\times512
)
\]

flatten：

\[
4\times
(
300\times512
)
\]

concat：

\[
1200\times512
\]

再加入：

\[
qpos,z
\]

得到：

\[
\boxed{
1202\times512
}
\]

---

# 133. 为什么 Transformer Encoder 输出仍是 1202 个 Tokens？

Self-Attention：

> 不把 sequence压成一个 vector。

输入：

\[
[1202,512]
\]

输出：

\[
\boxed{
[1202,512]
}
\]

每个 token变成：

> contextualized representation。

---

# 134. 所以 Encoder 后

原来的 visual token：

> 已经不只是局部 image feature。

它可能整合：

- 其他 camera；
- robot joints；
- latent z；
- 全局视觉 context。

---

# 135. 一个 wrist token可以读取 top-camera token

因为 Encoder Self-Attention是：

> global。

所以 multi-view fusion：

> 不是只靠 camera feature concatenation。

真正信息融合发生在：

\[
\boxed{
Transformer Encoder Self-Attention
}
\]

中。

---

# 136. Transformer 可以学习跨 Camera Correspondence 吗？

Architecture允许。

例如一个 token可能学会：

> wrist view中的gripper附近feature

与：

> top view中的object feature

之间的关系。

---

# 137. 但 ACT 没有显式 Geometry

没有输入：

- camera intrinsics；
- extrinsics；
- depth；
- epipolar constraints；
- 3D coordinates。

所以 cross-view关系：

> 主要从 demonstration data中学习。

---

# 138. 这意味着 Camera Calibration 不是显式模型输入

虽然 physical setup固定，

model可通过：

> stable viewpoint statistics

隐式学习camera geometry。

---

# 139. 这也意味着 Viewpoint Shift 可能比较敏感

如果部署时 camera位置大幅改变：

> model输入 distribution改变。

因为它没有显式3D geometric representation来自动重新对齐。

---

# 140. 现在进入 Decoder Cross-Attention

Encoder产生 memory：

\[
M
\in
\mathbb R^{1202\times B\times512}
\]

Action Query slots：

\[
Q_{slots}
\in
\mathbb R^{k\times B\times512}
\]

---

# 141. Cross-Attention Score

对某个 future action slot \(i\)：

\[
q_i
\]

与所有 memory keys：

\[
k_1,\ldots,k_{1202}
\]

计算：

\[
s_{ij}
=
\frac{
q_i^\top k_j
}{
\sqrt{d_k}
}
\]

---

# 142. Softmax 后

\[
\alpha_{ij}
=
softmax_j(
s_{ij}
)
\]

所以 action slot \(i\)：

> 可以从1202个memory tokens中动态读取。

---

# 143. 这1202个 Memory Locations 包括

- latent token；
- qpos token；
- camera 1 visual positions；
- camera 2 visual positions；
- camera 3 visual positions；
- camera 4 visual positions。

因此 Cross-Attention本质：

\[
\boxed{
\text{future action slot}
\rightarrow
\text{multimodal observation memory retrieval}
}
\]

---

# 144. “Action Query 看图”不是直接看 Pixels

它真正读取的是：

\[
\boxed{
\text{Transformer-encoded CNN visual representations}
}
\]

所以完整路径：

```text
pixel
↓
CNN feature
↓
visual token
↓
encoder contextualization
↓
memory key/value
↓
action-query cross-attention
↓
action representation
```

---

# 145. 为什么不是 Decoder 直接 Cross-Attend ResNet Features？

理论上可以。

但 ACT先加：

> Transformer Encoder。

这样 visual tokens可以：

1. camera间交流；
2. 与 qpos交流；
3. 与 z交流；
4. 建立全局 context；

再交给 Decoder。

---

# 146. 所以 Encoder 和 Decoder 的视觉职责不同

### ResNet18

\[
\boxed{
\text{local-to-mid/high-level visual feature extraction}
}
\]

### Transformer Encoder

\[
\boxed{
\text{global multimodal contextualization}
}
\]

### Transformer Decoder

\[
\boxed{
\text{future action slots retrieve task-relevant memory}
}
\]

---

# 147. 一张图看完整 Vision Pipeline

```text
4 RGB cameras
[B,4,3,480,640]
        │
        ▼
ImageNet normalization
        │
        ├──────────┬──────────┬──────────┐
        ▼          ▼          ▼          ▼
      cam1       cam2       cam3       cam4
        │          │          │          │
        └──── shared ResNet18 weights ───┘
        │          │          │          │
        ▼          ▼          ▼          ▼
[B,512,15,20] × 4
        │
        ▼
shared 1×1 Conv projection
        │
        ▼
[B,512,15,20] × 4
        │
        ├── 2D sine positional map per camera
        │
        ▼
concat camera feature maps along width
        │
        ▼
[B,512,15,80]
        │
        ▼
flatten spatial dimension
        │
        ▼
[1200,B,512]
        │
        ├──────── z: [B,32] → [B,512]
        │
        └───── qpos: [B,14] → [B,512]
        │
        ▼
[z, qpos, 1200 visual tokens]
        │
        ▼
[1202,B,512]
        │
        ▼
4-layer Transformer Encoder
        │
        ▼
multimodal memory
[1202,B,512]
        │
        ▼
7-layer Transformer Decoder
        ▲
        │
k action query slots
        │
        ▼
[k,B,512]
        │
        ▼
shared Linear(512,14)
        │
        ▼
[k,B,14]
```

---

# 148. 一张图理解 “Feature Map ≠ Patch Grid”

```text
raw image
480 × 640

     ↓ deep overlapping convolutions

feature grid
15 × 20

each feature location:
- centers roughly 32 px apart
- theoretical receptive field ~435 px
- represented by 512 learned channels
```

所以不能写：

> “ACT把图片简单切成300个32×32 patch。”

这是：

\[
\boxed{
错误的。
}
\]

---

# 149. ACT 与 ViT 的 Tokenization 对比

## ViT

典型：

\[
16\times16\times3
\]

patch：

\[
\rightarrow
Linear
\rightarrow
d
\]

每个 token：

> 初始主要来自一个固定 raw patch。

---

## ACT + ResNet18

\[
RGB
\rightarrow
deep\ CNN
\rightarrow
15\times20\times512
\]

每个 token：

> 已经经过深层卷积、多尺度downsampling和residual processing。

---

# 150. ViT 先 Tokenize 再做大部分视觉建模

ACT：

> CNN先做大量视觉建模，

Transformer再处理高层 visual tokens。

所以 ACT 属于一种：

\[
\boxed{
CNN\ backbone
+
Transformer
}
\]

hybrid architecture。

---

# 151. 为什么 2023 ACT 不直接用 ViT？

论文没有把它表述为：

> “ViT一定不好”。

更合理的历史/工程背景是：

- ResNet18成熟；
- pretrained；
- 计算成本低；
- DETR架构天然就是 CNN backbone + Transformer。

ACT明显继承了：

> DETR的视觉前端设计。

---

# 152. DETR 的视觉结构就是直接前身

DETR：

```text
image
↓
CNN
↓
spatial feature map
↓
1×1 projection
↓
2D positional encoding
↓
Transformer
```

ACT延续这套 skeleton，

再加入：

- multi-camera；
- qpos；
- z；
- action decoder。

---

# 153. `input_proj` 也是 DETR 血统

DETR通常需要：

\[
C_{backbone}
\rightarrow
d_{model}
\]

例如：

\[
2048
\rightarrow
256
\]

ACT ResNet18恰好：

\[
512
\rightarrow
512
\]

所以看起来像“多余”。

其实 interface仍然保留：

\[
\boxed{
\text{backbone channel space}
\rightarrow
\text{Transformer hidden space}
}
\]

---

# 154. 如果把 hidden_dim 改成256呢？

同一个 code自动变：

\[
512
\rightarrow
256
\]

所以 `input_proj`也承担：

> dimension adapter。

---

# 155. 因此 canonical 512→512 只是一个特殊配置

不要因此删掉：

```python
input_proj
```

否则：

> architecture失去通用性，

也去掉一个 learned basis transformation。

---

# 156. 一个高级问题：为什么 Positional Encoding 不加 Camera ID？

Canonical ACT没有。

这意味着一个自然 extension：

\[
\boxed{
pos
=
spatial\_pos
+
camera\_embed
}
\]

---

# 157. 例如

camera \(c\)：

\[
e_c
\in
\mathbb R^{512}
\]

每个该camera视觉 token：

\[
p_{r,camera}
=
p^{2D}_{r}
+
e_c
\]

这样 Transformer明确知道：

> 同一 \((r,c)\) 位置但来自不同 camera。

---

# 158. 这不是 Original ACT

如果未来做研究/改进，

可以实验：

> camera-ID embeddings是否提升 multi-view fusion。

但知识库要明确：

\[
\boxed{
\text{extension}
\neq
\text{canonical ACT fact}
}
\]

---

# 159. 另一个高级改进：更高 Resolution Feature Pyramid

ACT只使用：

> layer4 single-scale feature map。

但 fine manipulation可能受益于：

- layer2；
- layer3；
- layer4；

multi-scale features。

---

# 160. 为什么？

layer2：

> spatial resolution高，细节多。

layer4：

> semantics强，spatial resolution低。

Feature pyramid可以：

> 同时保留精细位置与高层语义。

---

# 161. Canonical ACT 没这么做

official backbone默认：

```python
return_layers =
    {'layer4': '0'}
```

所以：

\[
\boxed{
\text{single-scale last-stage ResNet feature}
}
\]

---

# 162. 再一个高级方向：ViT / Foundation Visual Encoder

后续 robot policies常采用：

- ViT；
- DINO；
- CLIP；
- SigLIP；
- large pretrained vision-language backbones。

相比 ResNet18：

> 视觉先验和规模更强。

但 ACT 原版刻意保持：

> relatively simple visual backbone。

---

# 163. 为什么 Original ACT 的视觉模块仍然值得学？

因为它非常清楚地展示：

\[
\boxed{
\text{pixels}
\rightarrow
\text{spatial features}
\rightarrow
\text{tokens}
\rightarrow
\text{attention memory}
}
\]

这条基本链路。

以后换成 ViT：

> 只是 visual tokenizer/backbone变了。

Transformer如何消费 visual representations：

> 基本思想仍然相通。

---

# 164. Common Misconception 1：ACT Transformer直接输入 Raw RGB Pixels

**错误。**

先过 ResNet18。

---

# 165. Common Misconception 2：480×640 被切成 15×20 个32×32不重叠 Patch

**错误。**

这是 CNN downsampling feature map，

不是 raw patch partition。

---

# 166. Common Misconception 3：Feature Cell 只看 32×32 pixels

**错误。**

stride约32，

但 theoretical receptive field约：

\[
435\times435
\]

---

# 167. Common Misconception 4：512 Channels 就是512个 Objects

**错误。**

是512维 learned feature basis。

---

# 168. Common Misconception 5：ResNet 最终输出1000 ImageNet Classes给 ACT

**错误。**

ACT取的是：

> layer4 feature map。

---

# 169. Common Misconception 6：15×20 Spatial Grid Flatten 后就彻底没有空间信息

**错误。**

ACT使用2D sinusoidal positional information。

---

# 170. Common Misconception 7：2D Positional Embedding只有2维

**错误。**

它编码二维坐标，

但 canonical output是：

\[
512
\]

维。

---

# 171. Common Misconception 8：Position Encoding 在代码里永久直接加进 Value Feature

不完全准确。

DETR-style ACT中：

> position主要加到 Attention Q/K source，

value仍使用content tensor。

---

# 172. Common Misconception 9：四个 Cameras 各有一套独立 ResNet18

Released ACT policy：

\[
\boxed{
\text{共享同一个 backbone}
}
\]

---

# 173. Common Misconception 10：代码里 `backbones` 是 list，所以一定四个

ACT `build()`：

> 只 append一个 backbone。

变量名是历史接口，

不能只看名字判断。

---

# 174. Common Misconception 11：四个 Camera 是先各自flatten，再在 Python里拼成1200

Paper概念上可以这么理解。

Released code实际：

> 先沿 width拼成15×80，再统一flatten。

最终token数等价。

---

# 175. Common Misconception 12：15×80 代表四张图被几何拼成真实 Panorama

**错误。**

它主要是 tensor packing strategy。

---

# 176. Common Misconception 13：Canonical ACT 有显式 Camera-ID Embedding

Released code：

> 没有单独 camera embedding。

---

# 177. Common Misconception 14：Camera Order 完全不重要

工程上：

> 必须保持与训练一致。

但不要把 list order误认为：

> architecture中已经有显式 camera-ID encoding。

---

# 178. Common Misconception 15：qpos 和 z 是在 Transformer Encoder 后才加进去

**错误。**

它们作为两个512-D tokens：

> 和visual tokens一起进入 Policy Encoder。

---

# 179. Common Misconception 16：Policy Encoder 输入顺序是 visual → qpos → z

Paper自然语言只说增加两个 features。

Released code实际：

\[
\boxed{
[z,\ qpos,\ visual...]
}
\]

---

# 180. Common Misconception 17：z/qpos 没有 Positional Identity

**错误。**

official code有：

```python
additional_pos_embed =
    nn.Embedding(2,512)
```

---

# 181. Common Misconception 18：ResNet18 Completely Frozen

Released ACT code：

> FrozenBatchNorm，

但 convolution parameters参与 optimizer并使用dedicated backbone LR。

---

# 182. Common Misconception 19：FrozenBatchNorm = Frozen Backbone

**错误。**

只冻结 BN statistics/affine behavior：

\[
\neq
\]

冻结全部 conv weights。

---

# 183. Common Misconception 20：ACT Perception 是单独预训练完以后固定的

Released ACT：

> ImageNet-pretrained backbone继续和 policy联合 fine-tune。

---

# 184. Common Misconception 21：1×1 Conv 没用，因为512→512

**错误。**

它仍是 learned channel-space transformation。

---

# 185. Common Misconception 22：1×1 Conv 会融合邻居 Spatial Information

**不会。**

它只混：

> channels。

---

# 186. Common Misconception 23：Paper Figure 11 的728就是 Canonical ResNet18 Output

与正文和released code冲突。

canonical implementation应按：

\[
\boxed{
512
}
\]

理解。

---

# 187. Common Misconception 24：每个 Visual Token 对应一个 Object

**错误。**

每个 token对应：

> 一个 spatial feature location。

一个object可能影响：

> 多个 tokens。

一个 token也可能包含：

> 多个物体/背景的上下文。

---

# 188. Common Misconception 25：Action Query 直接 Cross-Attend Raw ResNet Feature

中间还有：

> 4-layer Transformer Encoder contextualization。

---

# 189. Common Misconception 26：Encoder 只在同一 Camera 内做 Attention

**错误。**

1200 visual tokens已经进入同一个 self-attention sequence。

---

# 190. Common Misconception 27：不同 Cameras 在 Transformer 中绝对隔离

**错误。**

它们可以全局互相attend。

---

# 191. Common Misconception 28：ACT 显式使用 Camera Calibration 做3D Fusion

**没有。**

Canonical architecture没有：

- intrinsics；
- extrinsics；
- depth geometry；

作为显式输入。

---

# 192. Common Misconception 29：15×20 分辨率意味着最多只能定位到32-pixel precision

**错误。**

连续 feature activations可以编码 sub-grid变化。

但 downsampling仍然存在信息损失。

---

# 193. Common Misconception 30：CNN Token 和 ViT Patch Token 是完全一样的东西

都是 visual vectors，

但生成机制不同：

- raw patch projection；
- deep convolutional feature map。

---

# 194. 如果只记一个 Shape Chain

\[
\boxed{
[B,4,3,480,640]
}
\]

\[
\downarrow
\]

shared ResNet18：

\[
\boxed{
4\times[B,512,15,20]
}
\]

\[
\downarrow
\]

1×1 projection：

\[
\boxed{
4\times[B,512,15,20]
}
\]

\[
\downarrow
\]

camera concat：

\[
\boxed{
[B,512,15,80]
}
\]

\[
\downarrow
\]

flatten：

\[
\boxed{
[1200,B,512]
}
\]

\[
\downarrow
\]

prepend \(z,qpos\)：

\[
\boxed{
[1202,B,512]
}
\]

---

# 195. 如果只记一句话理解 ResNet18 的作用

> **ACT 不让 Transformer直接处理一百多万个 RGB pixels，而先用一个 ImageNet-pretrained、随后联合fine-tune的 ResNet18把每张480×640图像压缩成15×20个高层 spatial features；这样每个 camera只产生300个512维视觉向量，在大幅降低 token 数量的同时保留一个粗二维空间网格和丰富的卷积视觉表示。**

---

# 196. 如果只记一句话理解 15×20

> **15×20来自 ResNet18 的总 output stride 32，而不是把原图机械切成300个32×32 patch：相邻 feature locations的中心大约相隔32 pixels，但一个 layer4 feature的理论 receptive field可以覆盖约435×435 pixels，因此每个 token已经融合了广泛上下文。**

---

# 197. 如果只记一句话理解 512 Channels

> **一个 spatial cell 的512维向量不是512个类别，也不是512个物体，而是 ResNet通过动作任务联合微调后形成的512维 learned visual feature basis；Transformer把这个整体向量当成一个 visual token。**

---

# 198. 如果只记一句话理解 Positional Encoding

> **CNN feature本身告诉模型“这里看到了什么”，2D sinusoidal positional encoding则告诉attention“这个feature位于二维feature grid的哪里”；released ACT沿用DETR实现，并不是把position永久混进value，而是在encoder self-attention中主要将 \(src+pos\) 用作Q/K、将 \(src\) 本身作为V。**

---

# 199. 如果只记一句话理解 Multi-Camera

> **Canonical ACT 对所有 camera共享同一个ResNet18，将每路15×20 features沿width维拼接后统一flatten成1200 visual tokens，再让一个全局Transformer Encoder完成跨相机、视觉—关节—latent的信息融合；released code没有单独的camera-ID embedding，因此view identity主要通过固定视角产生的视觉内容统计被隐式学习。**

---

# 200. 如果只记一句话理解为什么是1202

> **1200来自 \(4\times15\times20\) 个视觉空间位置，另外的2个token分别是32维style latent投影成的512维latent token和14维current qpos投影成的512维proprio token，因此Policy Transformer Encoder实际处理的是 \([z,\ qpos,\ 1200\ visual\ tokens]\)，总长度1202。**

---

# 201. 到这里 ACT Architecture 的主干已经几乎闭合

现在我们可以从最原始输入完整说：

```text
4张RGB图
↓
ImageNet normalization
↓
shared ResNet18
↓
15×20×512 per camera
↓
1×1 projection
↓
2D spatial position
↓
1200 visual tokens
+
qpos token
+
z token
↓
1202-token Policy Encoder
↓
multimodal memory
↓
k Action Queries
↓
Decoder Cross-Attention
↓
k×14 future joint targets
```

这就是从：

\[
\boxed{
pixels
}
\]

到：

\[
\boxed{
actions
}
\]

的完整视觉控制链路。

---

# 202. 下一篇建议：ACT 的局限，以及 Diffusion Policy 为什么出现

到这里 ACT 本身最核心的高级模块已经基本补齐：

- Action Chunking；
- Temporal Ensemble；
- CVAE；
- \(z\)；
- Posterior Collapse；
- Transformer；
- Attention；
- DETR Action Queries；
- Behavior Cloning Distribution Shift；
- Vision Pipeline。

接下来最值得做的已经不再是继续拆 ACT 内部某一层，

而是站到更高一级：

> **ACT 到底哪里不够，后来的 robot policy 为什么会走向 Diffusion Policy、VLA 等方向？**

下一篇建议：

> **`act-vs-diffusion-policy.md` —《ACT vs Diffusion Policy：为什么机器人动作后来开始用 Diffusion 生成？》**

重点会讲：

- deterministic regression 的局限；
- CVAE multimodality和Diffusion multimodality有什么根本区别；
- Gaussian latent \(z\) 与 iterative denoising；
- action chunk共同点；
- ACT一次forward vs Diffusion多步采样；
- expressivity；
- inference latency；
- temporal consistency；
- receding-horizon execution；
- observation history；
- ACT的 \(z=0\) deterministic inference；
- Diffusion Policy如何保留multi-modal action distribution；
- 为什么Diffusion Policy不是“比ACT更新所以一定更好”；
- 两类方法各自的工程tradeoff；
- 再自然过渡到 VLA。

---

## Primary Source：ACT

Tony Z. Zhao, Vikash Kumar, Sergey Levine, Chelsea Finn.

**Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware.**  
RSS 2023.

- arXiv: https://arxiv.org/abs/2304.13705
- PDF: https://arxiv.org/pdf/2304.13705
- Project: https://tonyzhaozh.github.io/aloha/
- Official code: https://github.com/tonyzhaozh/act

论文 Section IV-C 明确给出 canonical visual pipeline：

\[
480\times640\times3
\rightarrow
15\times20\times512
\]

flatten：

\[
300\times512
\]

四个 camera：

\[
1200\times512
\]

再加入 joint positions和style variable：

\[
\boxed{
1202\times512
}
\]

作为 Transformer Encoder 输入。

论文也明确说明：

> 使用 ResNet18 image encoders，并加入2D sinusoidal positional embedding来保留spatial information。

---

## ACT Detailed Architecture

同一论文 Appendix C / Figure 11 给出详细结构。

正文和 Appendix prose均支持：

- ResNet18；
- flatten spatial features；
- project to 512；
- 2D sinusoidal position；
- concatenate camera feature sequences；
- append/project joints and \(z\)；
- encoder output作为decoder cross-attention的 keys/values。

### Figure 11 Channel-Count Discrepancy

Figure 11图中可见：

\[
15\times20\times728
\]

和：

\[
728\rightarrow512
\]

但论文正文明确写：

\[
15\times20\times512
\]

而 released ResNet18 implementation同样明确输出：

\[
512
\]

channels。

因此 canonical implementation应以：

\[
\boxed{
512
}
\]

为准，并将 Figure 11中的728记录为图示不一致。

---

## ACT Official Backbone

`detr/models/backbone.py`:

https://github.com/tonyzhaozh/act/blob/main/detr/models/backbone.py

Released code：

```python
return_layers =
    {'layer4': '0'}
```

所以 ACT只使用：

> ResNet最后 convolutional stage `layer4`。

对于 ResNet18：

```python
num_channels = 512
```

并使用：

```python
FrozenBatchNorm2d
```

以及 ImageNet pretrained initialization。

---

## ACT Official Vision Forward

`detr/models/detr_vae.py`:

https://github.com/tonyzhaozh/act/blob/main/detr/models/detr_vae.py

核心：

```python
for cam_id, cam_name in enumerate(
    self.camera_names
):
    features, pos =
        self.backbones[0](
            image[:, cam_id]
        )

    features =
        features[0]

    pos =
        pos[0]

    all_cam_features.append(
        self.input_proj(
            features
        )
    )

    all_cam_pos.append(
        pos
    )
```

这说明：

\[
\boxed{
\text{all cameras share self.backbones[0]}
}
\]

随后：

```python
src =
    torch.cat(
        all_cam_features,
        axis=3
    )

pos =
    torch.cat(
        all_cam_pos,
        axis=3
    )
```

也就是：

> camera dimension被fold进width dimension。

---

## ACT Official Projection

同一文件：

```python
self.input_proj =
    nn.Conv2d(
        backbones[0].num_channels,
        hidden_dim,
        kernel_size=1
    )
```

canonical：

\[
512\rightarrow512
\]

这是：

> 每个spatial位置共享的learned channel projection。

---

## ACT Official Transformer Flatten

`detr/models/transformer.py`:

https://github.com/tonyzhaozh/act/blob/main/detr/models/transformer.py

官方：

```python
bs, c, h, w =
    src.shape

src =
    src.flatten(2)
       .permute(2,0,1)
```

因此：

\[
[B,512,15,80]
\rightarrow
[1200,B,512]
\]

---

## ACT z/qpos Token Insertion

同一 Transformer code：

```python
addition_input =
    torch.stack(
        [
            latent_input,
            proprio_input
        ],
        axis=0
    )

src =
    torch.cat(
        [
            addition_input,
            src
        ],
        axis=0
    )
```

所以 released code实际 Encoder content order：

\[
\boxed{
[z,\ qpos,\ visual_1,\ldots,visual_{1200}]
}
\]

总长度：

\[
1202
\]

---

## ACT Additional Positional Embeddings

`detr_vae.py`：

```python
self.additional_pos_embed =
    nn.Embedding(
        2,
        hidden_dim
    )
```

所以：

- \(z\) token；
- proprio token；

拥有独立 learned positional identities。

---

## ACT 2D Positional Encoding

`detr/models/position_encoding.py`:

https://github.com/tonyzhaozh/act/blob/main/detr/models/position_encoding.py

它继承 DETR-style：

**PositionEmbeddingSine**

并明确说明：

> 是类似 Attention Is All You Need 的 sinusoidal encoding，generalized to images。

canonical：

```python
N_steps =
    hidden_dim // 2
```

hidden_dim=512：

\[
N_{\text{steps}}=256
\]

得到：

- 256-D \(y\) encoding；
- 256-D \(x\) encoding；

拼成：

\[
512
\]

维 2D positional vector。

---

## ACT Positional Information in Attention

`detr/models/transformer.py`：

```python
q = k =
    self.with_pos_embed(
        src,
        pos
    )

src2 =
    self.self_attn(
        q,
        k,
        value=src,
        ...
    )[0]
```

所以released implementation里：

\[
\boxed{
Q/K:\ src+pos
}
\]

\[
\boxed{
V:\ src
}
\]

这是 DETR-style positional injection。

---

## Official Image Preprocessing

`utils.py`:

https://github.com/tonyzhaozh/act/blob/main/utils.py

dataset：

```python
image_data =
    image_data / 255.0
```

然后 `policy.py`：

https://github.com/tonyzhaozh/act/blob/main/policy.py

使用：

```python
transforms.Normalize(
    mean=[
        0.485,
        0.456,
        0.406
    ],
    std=[
        0.229,
        0.224,
        0.225
    ]
)
```

与 ImageNet-pretrained ResNet的标准 normalization一致。

---

## ResNet Primary Source

Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.

**Deep Residual Learning for Image Recognition.**  
CVPR 2016.

- arXiv: https://arxiv.org/abs/1512.03385
- PDF: https://arxiv.org/pdf/1512.03385

ResNet引入 residual learning：

\[
y=
F(x)+x
\]

使深卷积网络更容易优化。

ACT使用其中较轻量的：

\[
\boxed{
ResNet18
}
\]

作为视觉 backbone。

---

## Torchvision ResNet18 Architecture

Official torchvision docs/source:

https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html

https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py

standard ResNet architecture包含：

```text
conv1 stride 2
maxpool stride 2
layer1 stride 1
layer2 stride 2
layer3 stride 2
layer4 stride 2
```

总 output stride：

\[
32
\]

因此：

\[
480/32=15
\]

\[
640/32=20
\]

---

## DETR Primary Source

Nicolas Carion et al.

**End-to-End Object Detection with Transformers.**  
ECCV 2020.

- arXiv: https://arxiv.org/abs/2005.12872
- Official code: https://github.com/facebookresearch/detr

ACT视觉模块明显沿用 DETR-style：

\[
CNN\ feature\ map
\rightarrow
1\times1\ projection
\rightarrow
2D\ positional\ encoding
\rightarrow
Transformer
\]

Released ACT `detr/`目录也直接由 DETR implementation修改而来。

---

## 本文知识连接

### Computer Vision

- Convolution
- CNN
- ResNet
- [Residual Connection](../../deep-learning/residual-connection.md)
- Receptive Field
- Feature Map
- 1×1 Convolution
- ImageNet Pretraining

### Transformer

- [Transformer](../../deep-learning/transformer.md)
- [Self-Attention](../../deep-learning/self-attention.md)
- [Cross-Attention](../../deep-learning/cross-attention.md)
- [Positional Encoding](../../deep-learning/positional-encoding.md)
- [Q / K / V](../../deep-learning/qkv.md)

### ACT

- [ACT Architecture](./architecture.md)
- [从 DETR 到 ACT](./detr-to-act.md)
- [CVAE in ACT](./cvae-in-act.md)
- [ACT Training](./training.md)
- [ACT Inference](./inference.md)
- [ACT Complete Data Flow](./complete-data-flow.md)

### 下一步

- ACT vs Diffusion Policy：为什么机器人动作后来开始用 Diffusion 生成？
