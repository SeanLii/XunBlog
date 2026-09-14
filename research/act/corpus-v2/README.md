# XunBlog ACT Knowledge Corpus — v2

这是对 v1 正文的完整重写。v1 已验证的 Knowledge Tree、canonical ownership 与 URL 结构被保留；正文不沿用旧文章的组织方式。

## 本版最主要的变化

- **先建立整体 mental model，再拆细节。** ACT、Transformer、VAE、CVAE、CNN/ResNet 等总览页先解释“这是什么、输入输出、整条数据流”，再进入模块和数学。
- **不按论文顺序写文章。** 原论文用于事实校验和研究边界，文章顺序按读者理解依赖重新设计。
- **子页是总图的局部放大。** Attention、QKV、Action Chunking、Temporal Ensemble 等不再各自重新起一套背景。
- **Training 与 Inference 明确分图。** ACT 的 ground-truth action → latent encoder 支路只属于训练。
- **数学概念先说明“描述什么”。** Normal Distribution、Variance、KL Divergence 等先建立对象与尺度，再给公式和 AI 连接。

## 文件

- `KNOWLEDGE_TREE.md` — canonical ownership / sidebar 层级
- `LINK_GRAPH.md` — 正文实际导航关系
- `CONTENT_INDEX.md` — 53 个页面索引
- `SOURCE_MAP.md` — 原论文和官方实现依据
- `CONTENT_DESIGN_STANDARD.md` — 本版新增的“理解层级”写作标准
- `QUALITY_AUDIT.md` — 自动检查与人工内容验收结果
- `content/` — 所有 canonical / ACT-specific Markdown 页面

## 当前范围

只覆盖 ACT 以及理解 ACT 所需要的前置知识；没有扩展到 Diffusion Policy、OpenVLA、π0、GR00T、VLA 或 World Model。
