<script setup lang="ts">
import { withBase } from 'vitepress'
const areas = [
  { index: '01', title: 'Mathematics', text: '概率、分布与优化：为模型理解打牢数学地基。', link: '/mathematics/', state: 'Foundations' },
  { index: '02', title: 'Deep Learning', text: '从神经网络出发，理解 Attention 与 Transformer。', link: '/deep-learning/', state: 'Core models' },
  { index: '03', title: 'Robot Learning', text: 'Imitation Learning、ACT、VLA 与具身决策。', link: '/robot-learning/', state: 'Current focus' },
  { index: '04', title: 'Embodied AI', text: '连接感知、推理与行动的智能系统。', link: '/embodied-ai/', state: 'Exploring' }
]

const path = [
  { label: 'Probability', state: 'foundation' },
  { label: 'Transformer', state: 'foundation' },
  { label: 'CVAE', state: 'learning' },
  { label: 'ACT', state: 'focus' },
  { label: 'VLA', state: 'next' }
]
</script>

<template>
  <main class="home-page">
    <section class="home-hero">
      <div class="hero-signal" aria-hidden="true">
        <span></span><span></span><span></span>
        <small>LEARNING LOG · 2026—</small>
      </div>
      <p class="hero-kicker">Personal AI Knowledge Base</p>
      <h1>理解智能，<br><em>从直觉到实现。</em></h1>
      <p class="hero-intro">
        一套持续生长的 AI 学习笔记：把问题拆成直觉、数学、模型、代码与实验，记录理解发生的过程。
      </p>
      <div class="hero-actions">
        <a class="primary-action" :href="withBase('/robot-learning/act/')">进入 ACT 专题 <span>↗</span></a>
        <a class="quiet-action" :href="withBase('/robot-learning/')">查看学习地图 <span>→</span></a>
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
          <p>ACT 不是一个孤立模型。它把概率建模、Transformer、CVAE 和机器人控制连接成一条完整的学习路径。</p>
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
      <a class="featured-note" :href="withBase('/robot-learning/act/act-what-problem-does-it-solve')">
        <div>
          <span class="note-label">FEATURED NOTE · ACT 01</span>
          <h3>为什么机器人需要 Action Chunking？</h3>
          <p>单步预测如何积累误差？一次预测一段动作，为什么能让策略更连贯、更稳定？</p>
        </div>
        <span class="featured-arrow">↗</span>
      </a>
    </section>
  </main>
</template>
