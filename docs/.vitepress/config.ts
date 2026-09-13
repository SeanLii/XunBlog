import { defineConfig } from 'vitepress'
import { withMermaid } from 'vitepress-plugin-mermaid'

export default withMermaid(defineConfig({
  lang: 'zh-CN',
  title: 'AI Learning Notes',
  description: '从直觉到数学，从数学到实现的个人 AI 知识库。',
  base: process.env.BASE_PATH || '/',
  cleanUrls: true,
  lastUpdated: true,
  head: [
    ['meta', { name: 'theme-color', content: '#5b62e8' }],
    ['link', { rel: 'icon', href: '/logo.svg', type: 'image/svg+xml' }]
  ],
  markdown: {
    math: true,
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
          text: 'Deep Learning',
          items: [
            { text: '概览', link: '/deep-learning/' },
            { text: 'Transformer', link: '/deep-learning/transformer' }
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
            { text: 'Imitation Learning', link: '/robot-learning/imitation-learning' }
          ]
        },
        {
          text: 'ACT · Action Chunking',
          collapsed: false,
          items: [
            { text: '专题概览', link: '/robot-learning/act/' },
            { text: '01 · ACT 解决什么问题？', link: '/robot-learning/act/why-action-chunking' },
            { text: '02 · Transformer 在做什么', link: '/robot-learning/act/transformer' },
            { text: '03 · CVAE 与 latent z', link: '/robot-learning/act/cvae' },
            { text: '04 · Temporal Ensemble', link: '/robot-learning/act/temporal-ensemble' }
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
