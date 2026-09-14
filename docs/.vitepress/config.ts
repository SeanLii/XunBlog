import { defineConfig } from 'vitepress'
import { withMermaid } from 'vitepress-plugin-mermaid'
import { useKatexMath } from './markdown/katex.mjs'

const base = process.env.BASE_PATH || '/'

export default withMermaid(defineConfig({
  lang: 'zh-CN',
  title: 'AI Learning Notes',
  description: '围绕 ACT 与必要前置知识建立的结构化 AI 知识库。',
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
        text: '知识树',
        items: [
          { text: 'Mathematics', link: '/mathematics/' },
          { text: 'Deep Learning', link: '/deep-learning/' },
          { text: 'Generative Models', link: '/generative-models/' },
          { text: 'Robot Learning', link: '/robot-learning/' }
        ]
      },
      { text: 'ACT', link: '/robot-learning/act/' }
    ],
    sidebar: {
      '/mathematics/': [
        {
          text: 'Mathematics',
          items: [{ text: '知识地图', link: '/mathematics/' }]
        },
        {
          text: 'Linear Algebra',
          items: [
            { text: 'Vector', link: '/mathematics/linear-algebra/vector/' },
            { text: 'Matrix', link: '/mathematics/linear-algebra/matrix/' },
            { text: 'Dot Product', link: '/mathematics/linear-algebra/dot-product/' }
          ]
        },
        {
          text: 'Probability',
          items: [
            { text: 'Random Variable', link: '/mathematics/probability/random-variable/' },
            { text: 'Probability Distribution', link: '/mathematics/probability/probability-distribution/' },
            { text: 'Conditional Probability', link: '/mathematics/probability/conditional-probability/' },
            { text: 'Expectation', link: '/mathematics/probability/expectation/' },
            { text: 'Variance', link: '/mathematics/probability/variance/' },
            { text: 'Normal Distribution', link: '/mathematics/probability/normal-distribution/' },
            { text: 'Multivariate Normal Distribution', link: '/mathematics/probability/multivariate-normal-distribution/' }
          ]
        },
        {
          text: 'Information Theory',
          items: [
            { text: 'KL Divergence', link: '/mathematics/information-theory/kl-divergence/' }
          ]
        }
      ],
      '/deep-learning/': [
        {
          text: 'Deep Learning',
          items: [{ text: '知识地图', link: '/deep-learning/' }]
        },
        {
          text: 'Core',
          items: [
            { text: 'Linear Layer', link: '/deep-learning/core/linear-layer/' },
            { text: 'Softmax', link: '/deep-learning/core/softmax/' },
            { text: 'Embedding', link: '/deep-learning/core/embedding/' },
            { text: 'Residual Connection', link: '/deep-learning/core/residual-connection/' },
            { text: 'Layer Normalization', link: '/deep-learning/core/layer-normalization/' }
          ]
        },
        {
          text: 'Transformer',
          collapsed: false,
          items: [
            { text: 'Transformer', link: '/deep-learning/transformer/' },
            {
              text: 'Attention',
              collapsed: true,
              items: [
                { text: 'Attention', link: '/deep-learning/transformer/attention/' },
                { text: 'Query / Key / Value', link: '/deep-learning/transformer/attention/qkv/' },
                { text: 'Scaled Dot-Product Attention', link: '/deep-learning/transformer/attention/scaled-dot-product-attention/' },
                { text: 'Self-Attention', link: '/deep-learning/transformer/attention/self-attention/' },
                { text: 'Cross-Attention', link: '/deep-learning/transformer/attention/cross-attention/' },
                { text: 'Multi-Head Attention', link: '/deep-learning/transformer/attention/multi-head-attention/' }
              ]
            },
            { text: 'Positional Encoding', link: '/deep-learning/transformer/positional-encoding/' },
            { text: 'Transformer Encoder', link: '/deep-learning/transformer/transformer-encoder/' },
            { text: 'Transformer Decoder', link: '/deep-learning/transformer/transformer-decoder/' },
            { text: 'Feed-Forward Network', link: '/deep-learning/transformer/feed-forward-network/' },
            { text: 'Causal Mask', link: '/deep-learning/transformer/causal-mask/' },
            { text: 'CLS Token', link: '/deep-learning/transformer/cls-token/' },
            { text: 'Learnable Query Embedding', link: '/deep-learning/transformer/learnable-query-embedding/' }
          ]
        },
        {
          text: 'Convolutional Neural Networks',
          items: [
            { text: 'Convolution', link: '/deep-learning/cnn/convolution/' },
            { text: 'Convolutional Neural Network', link: '/deep-learning/cnn/convolutional-neural-network/' },
            { text: 'ResNet', link: '/deep-learning/cnn/resnet/' }
          ]
        }
      ],
      '/generative-models/': [
        {
          text: 'Generative Models',
          items: [
            { text: '知识地图', link: '/generative-models/' },
            { text: 'Latent Variable', link: '/generative-models/latent-variable/' },
            { text: 'Variational Inference', link: '/generative-models/variational-inference/' },
            { text: 'Evidence Lower Bound', link: '/generative-models/evidence-lower-bound/' },
            { text: 'Variational Autoencoder', link: '/generative-models/variational-autoencoder/' },
            { text: 'Reparameterization Trick', link: '/generative-models/reparameterization-trick/' },
            { text: 'Conditional Variational Autoencoder', link: '/generative-models/conditional-variational-autoencoder/' },
            { text: 'Posterior Collapse', link: '/generative-models/posterior-collapse/' }
          ]
        }
      ],
      '/robot-learning/': [
        {
          text: 'Robot Learning',
          items: [
            { text: '知识地图', link: '/robot-learning/' },
            { text: 'Imitation Learning', link: '/robot-learning/imitation-learning/' },
            { text: 'Behavior Cloning', link: '/robot-learning/behavior-cloning/' }
          ]
        },
        {
          text: 'ACT',
          collapsed: false,
          items: [
            { text: 'ACT', link: '/robot-learning/act/' },
            { text: 'Action Chunking', link: '/robot-learning/act/action-chunking/' },
            { text: 'Temporal Ensemble', link: '/robot-learning/act/temporal-ensemble/' },
            { text: 'Architecture', link: '/robot-learning/act/architecture/' },
            { text: 'CVAE in ACT', link: '/robot-learning/act/cvae-in-act/' },
            { text: 'Vision Pipeline', link: '/robot-learning/act/vision-pipeline/' },
            { text: 'Training', link: '/robot-learning/act/training/' },
            { text: 'Inference', link: '/robot-learning/act/inference/' },
            { text: 'Complete Data Flow', link: '/robot-learning/act/complete-data-flow/' },
            { text: '为什么推理时令 z = 0？', link: '/robot-learning/act/why-z-zero-at-inference/' },
            { text: 'Paper and Released Implementation', link: '/robot-learning/act/paper-and-released-implementation/' }
          ]
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
