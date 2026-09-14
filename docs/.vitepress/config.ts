import { defineConfig } from 'vitepress'
import { withMermaid } from 'vitepress-plugin-mermaid'
import { useKatexMath } from './markdown/katex.mjs'

const base = process.env.BASE_PATH || '/'

export default withMermaid(defineConfig({
  lang: 'zh-CN',
  title: 'AI Learning Notes',
  description: '从直觉到数学，从数学到实现的个人 AI 知识库。',
  base,
  cleanUrls: true,
  lastUpdated: true,
  head: [
    ['meta', { name: 'theme-color', content: '#5b62e8' }],
    ['link', { rel: 'icon', href: `${base}logo.svg`, type: 'image/svg+xml' }]
  ],
  markdown: {
    config(markdown) {
      useKatexMath(markdown)
    },
    lineNumbers: true,
    image: { lazyLoading: true },
    theme: {
      light: 'github-light',
      dark: 'github-dark'
    }
  },
  mermaid: {
    theme: 'base',
    themeVariables: {
      primaryColor: '#ececff',
      primaryTextColor: '#252633',
      primaryBorderColor: '#777ee9',
      lineColor: '#777b8d',
      secondaryColor: '#f4f4f1',
      tertiaryColor: '#ffffff',
      fontFamily: 'Inter, ui-sans-serif, sans-serif'
    }
  },
  themeConfig: {
    logo: '/logo.svg',
    siteTitle: 'AI Learning Notes',
    nav: [
      { text: '首页', link: '/' },
      {
        text: '学习笔记',
        items: [
          { text: 'Mathematics', link: '/mathematics/' },
          { text: 'Deep Learning', link: '/deep-learning/' },
          { text: 'Generative Models', link: '/generative-models/' },
          { text: 'LLM', link: '/llm/' },
          { text: 'Robot Learning', link: '/robot-learning/' },
          { text: 'Embodied AI', link: '/embodied-ai/' }
        ]
      },
      { text: 'Projects', link: '/projects/' },
      { text: 'About', link: '/about' }
    ],
    sidebar: {
      '/mathematics/': [
        {
          text: 'Mathematics',
          items: [
            { text: '概览', link: '/mathematics/' },
            { text: 'Normal Distribution', link: '/mathematics/normal-distribution' },
            { text: 'KL Divergence', link: '/mathematics/kl-divergence' }
          ]
        }
      ],
      '/deep-learning/': [
        {
          text: 'Neural Network Foundations',
          items: [
            { text: '概览', link: '/deep-learning/' },
            { text: 'Linear Layer', link: '/deep-learning/linear-layer' },
            { text: 'ReLU', link: '/deep-learning/relu' },
            { text: 'MLP', link: '/deep-learning/mlp' },
            { text: 'Backpropagation', link: '/deep-learning/backpropagation' }
          ]
        },
        {
          text: 'Attention',
          items: [
            { text: 'Dot Product', link: '/deep-learning/dot-product' },
            { text: 'Softmax', link: '/deep-learning/softmax' },
            { text: 'Attention', link: '/deep-learning/attention' },
            { text: 'Query / Key / Value', link: '/deep-learning/qkv' },
            { text: 'Self-Attention', link: '/deep-learning/self-attention' },
            { text: 'Cross-Attention', link: '/deep-learning/cross-attention' },
            { text: 'Multi-Head Attention', link: '/deep-learning/multi-head-attention' }
          ]
        },
        {
          text: 'Transformer',
          items: [
            { text: 'Transformer', link: '/deep-learning/transformer' },
            { text: 'Positional Encoding', link: '/deep-learning/positional-encoding' },
            { text: 'Causal Mask', link: '/deep-learning/causal-mask' },
            { text: 'Transformer Encoder', link: '/deep-learning/transformer-encoder' },
            { text: 'Transformer Decoder', link: '/deep-learning/transformer-decoder' },
            { text: 'Feed-Forward Network', link: '/deep-learning/feed-forward-network' },
            { text: 'Residual Connection', link: '/deep-learning/residual-connection' },
            { text: 'Layer Normalization', link: '/deep-learning/layer-normalization' },
            { text: 'Dropout', link: '/deep-learning/dropout' }
          ]
        }
      ],
      '/generative-models/': [
        {
          text: 'Generative Models',
          items: [
            { text: '概览', link: '/generative-models/' },
            { text: 'Latent Variable', link: '/generative-models/latent-variable' },
            { text: 'VAE', link: '/generative-models/vae' },
            { text: 'Reparameterization Trick', link: '/generative-models/reparameterization-trick' },
            { text: 'CVAE', link: '/generative-models/cvae' },
            { text: 'Posterior Collapse', link: '/generative-models/posterior-collapse' }
          ]
        }
      ],
      '/llm/': [
        {
          text: 'Large Language Models',
          items: [{ text: '概览', link: '/llm/' }]
        }
      ],
      '/robot-learning/': [
        {
          text: 'Robot Learning',
          items: [
            { text: '学习地图', link: '/robot-learning/' },
            { text: 'Imitation Learning', link: '/robot-learning/imitation-learning' },
            { text: 'Behavior Cloning & Distribution Shift', link: '/robot-learning/imitation-learning/behavior-cloning-distribution-shift' }
          ]
        },
        {
          text: 'ACT · Action Chunking',
          collapsed: false,
          items: [
            { text: '专题概览', link: '/robot-learning/act/' },
            { text: '01 · ACT 解决什么问题？', link: '/robot-learning/act/act-what-problem-does-it-solve' },
            { text: '02 · Action Chunking', link: '/robot-learning/act/action-chunking' },
            { text: '03 · Temporal Ensemble', link: '/robot-learning/act/temporal-ensemble' },
            { text: '04 · Architecture', link: '/robot-learning/act/architecture' },
            { text: '05 · Vision Pipeline', link: '/robot-learning/act/vision-pipeline' },
            { text: '06 · 从 DETR 到 ACT', link: '/robot-learning/act/detr-to-act' },
            { text: '07 · CVAE in ACT', link: '/robot-learning/act/cvae-in-act' },
            { text: '08 · 为什么推理时 z = 0？', link: '/robot-learning/act/why-z-zero-at-inference' },
            { text: '09 · Training', link: '/robot-learning/act/training' },
            { text: '10 · Inference', link: '/robot-learning/act/inference' },
            { text: '11 · Complete Data Flow', link: '/robot-learning/act/complete-data-flow' }
          ]
        }
      ],
      '/embodied-ai/': [
        {
          text: 'Embodied AI',
          items: [{ text: '概览', link: '/embodied-ai/' }]
        }
      ],
      '/projects/': [
        {
          text: 'Projects',
          items: [{ text: '项目索引', link: '/projects/' }]
        }
      ]
    },
    search: {
      provider: 'local',
      options: {
        translations: {
          button: { buttonText: '搜索笔记', buttonAriaLabel: '搜索笔记' },
          modal: {
            noResultsText: '没有找到相关内容',
            resetButtonTitle: '清除查询',
            footer: {
              selectText: '选择',
              navigateText: '切换',
              closeText: '关闭'
            }
          }
        }
      }
    },
    outline: {
      level: [2, 3],
      label: '本页目录'
    },
    docFooter: {
      prev: '上一篇',
      next: '下一篇'
    },
    lastUpdated: {
      text: '最后更新',
      formatOptions: { dateStyle: 'medium' }
    },
    returnToTopLabel: '返回顶部',
    sidebarMenuLabel: '目录',
    darkModeSwitchLabel: '外观',
    lightModeSwitchTitle: '切换到浅色模式',
    darkModeSwitchTitle: '切换到深色模式',
    footer: {
      message: 'Learning first. Content first. Engineering second.',
      copyright: 'Built as a living AI knowledge system.'
    }
  }
}))
