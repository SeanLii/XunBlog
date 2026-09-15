# AI Learning Notes

围绕 ACT（Action Chunking with Transformers）、π0 以及理解它们所需前置知识构建的结构化知识库。

## 内容范围

- Mathematics：Linear Algebra、Probability、Information Theory
- Deep Learning：Core、Transformer、Attention、Convolutional Neural Networks
- Generative Models：Latent Variable、Variational Inference、ELBO、VAE、CVAE
- Robot Learning：Imitation Learning、Behavior Cloning、VLA、Cross-Embodiment Learning、ACT、π0

当前站点正文已完整更新为 XunBlog Knowledge Corpus v6，由 82 个独立 canonical knowledge pages 构成。知识归属由目录树决定，跨页面链接只负责导航。

## Local development

```bash
npm install
npm run docs:dev
```

## Verification

```bash
npm run docs:audit         # canonical URL、内部链接与知识图
npm run docs:audit:math    # 全量公式解析
npm run docs:build         # 生产构建
npm run docs:audit:render  # 构建产物中的公式、图表与样式
```

## Repository structure

```text
docs/
├── mathematics/
├── deep-learning/
├── generative-models/
├── robot-learning/
│   └── act/
└── .vitepress/
```

每个知识页面的目录路径与 frontmatter 中的 canonical URL 一致。VitePress 配置、知识图和视觉主题位于 `docs/.vitepress/`。

## Deployment

`.github/workflows/deploy.yml` 在每次推送到 `main` 后构建并发布 GitHub Pages。项目站点的 `/XunBlog/` base path 由工作流自动注入。

第二版配套索引、知识树、来源说明和原包验收记录保存在 `research/act/corpus-v2/`。站点正文仅将行内公式分隔符转换为现有渲染器支持的 `$...$` 格式。

π0 专题与 6 篇通用前置知识已从 v3 导入，原 53 篇正文不变。v3 配套说明见 `research/corpus-v3/`。

v6 已完整导入 82 篇知识页面，并按新版归属迁移路由。旧路径由构建阶段生成兼容跳转页；完整索引、归属与迁移表见 `research/corpus-v6/`。
