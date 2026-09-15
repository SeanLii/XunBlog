# XunBlog

以 ACT（Action Chunking with Transformers）与 π0 为入口，递归连接理解两者所需的机器人学习、生成模型、深度学习与数学知识。每个概念只拥有一个 canonical page；归属树负责稳定导航，依赖链接负责跨分支复用。

全站内容以 [`standards/XunBlog_知识内容创作与知识架构规范_v1.0.html`](standards/XunBlog_知识内容创作与知识架构规范_v1.0.html) 为唯一创作与架构标准。

当前全站验收结果见 [`CONTENT_ACCEPTANCE.md`](CONTENT_ACCEPTANCE.md)。

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

## 内容结构

```text
docs/
├── mathematics/
├── deep-learning/
├── generative-models/
├── robot-learning/
│   └── act/
└── .vitepress/
```

82 个 canonical knowledge pages 共同构成当前知识系统。每篇文章只展开当前对象的知识增量；依赖概念通过 canonical link 下钻，不在多篇文章中重复定义。

## Deployment

`.github/workflows/deploy.yml` 在每次推送到 `main` 后构建并发布 GitHub Pages。项目站点的 `/XunBlog/` base path 由工作流自动注入。

部署目标为 GitHub Pages；推送到 `main` 后由 GitHub Actions 构建并发布。
