<script setup lang="ts">
import { withBase } from 'vitepress'
const areas = [
  { index: '01', title: 'Mathematics', text: '线性代数、概率与信息论：理解 ACT 所需的数学基础。', link: '/mathematics/', state: 'Foundations' },
  { index: '02', title: 'Deep Learning', text: '核心神经网络、Transformer、Attention 与卷积网络。', link: '/deep-learning/', state: 'Core models' },
  { index: '03', title: 'Generative Models', text: '从潜变量与变分推断，到 VAE、CVAE 与 posterior collapse。', link: '/generative-models/', state: 'Latent models' },
  { index: '04', title: 'Robot Learning', text: 'Imitation Learning、Behavior Cloning 与 ACT 完整知识树。', link: '/robot-learning/', state: 'Current focus' }
]

const path = [
  { label: 'Behavior Cloning', state: 'foundation' },
  { label: 'Transformer', state: 'foundation' },
  { label: 'Conditional VAE', state: 'learning' },
  { label: 'ACT', state: 'focus' },
]
</script>

<template>
  <main class="home-page">
    <section class="home-hero">
      <div class="hero-signal" aria-hidden="true">
        <span></span><span></span><span></span>
        <small>ACT KNOWLEDGE CORPUS · 2026—</small>
      </div>
      <p class="hero-kicker">Personal AI Knowledge Base</p>
      <h1>理解智能，<br><em>从直觉到实现。</em></h1>
      <p class="hero-intro">
        一套围绕 ACT 与必要前置知识建立的结构化知识库：每个概念独立归属，并通过链接递归连接。
      </p>
      <div class="hero-actions">
        <a class="primary-action" :href="withBase('/robot-learning/act/')">进入 ACT 专题 <span>↗</span></a>
        <a class="quiet-action" :href="withBase('/robot-learning/')">查看知识树 <span>→</span></a>
      </div>
      <div class="hero-footnote">
        <span><i></i> Current focus</span>
        <strong>Action Chunking with Transformers</strong>
      </div>
    </section>

    <section class="home-section areas-section">
      <header class="section-heading">
        <div><span>01</span><p>Knowledge Areas</p></div>
        <h2>建立连接，而不是堆积文章。</h2>
      </header>
      <div class="area-grid">
        <a v-for="area in areas" :key="area.title" :href="withBase(area.link)" class="area-card">
          <div class="area-card-top"><span>{{ area.index }}</span><small>{{ area.state }}</small></div>
          <h3>{{ area.title }}</h3>
          <p>{{ area.text }}</p>
          <b>Explore <span>↗</span></b>
        </a>
      </div>
    </section>

    <section class="home-section path-section">
      <header class="section-heading">
        <div><span>02</span><p>Learning Path</p></div>
        <h2>现在，正在理解 ACT。</h2>
      </header>
      <div class="path-panel">
        <div class="path-copy">
          <p class="eyebrow">CURRENT THREAD</p>
          <h3>从分布，到潜变量，<br>再到机器人动作。</h3>
          <p>ACT 不是一个孤立模型。它把 Behavior Cloning、Transformer、CVAE 和机器人控制连接成一条完整路径。</p>
          <a :href="withBase('/robot-learning/act/')">打开专题笔记 <span>→</span></a>
        </div>
        <ol class="path-map" aria-label="ACT 学习路径">
          <li v-for="(item, index) in path" :key="item.label" :class="item.state">
            <span>{{ String(index + 1).padStart(2, '0') }}</span>
            <strong>{{ item.label }}</strong>
            <small>{{ item.state }}</small>
          </li>
        </ol>
      </div>
    </section>

    <section class="home-section latest-section">
      <header class="section-heading">
        <div><span>03</span><p>Start Here</p></div>
        <h2>从一个真实问题开始。</h2>
      </header>
      <a class="featured-note" :href="withBase('/robot-learning/act/')">
        <div>
          <span class="note-label">FEATURED KNOWLEDGE · ACT</span>
          <h3>Action Chunking with Transformers</h3>
          <p>从当前视觉与关节状态出发，一次预测未来一段连续动作，并通过 temporal ensemble 与 CVAE 完成稳定控制。</p>
        </div>
        <span class="featured-arrow">↗</span>
      </a>
    </section>
  </main>
</template>
