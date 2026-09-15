<script setup lang="ts">
import KnowledgeTree from './KnowledgeTree.vue'
import { formatUpdated, useKnowledge } from '../composables/knowledge'
const { model, href } = useKnowledge()
const openSearch = () => window.dispatchEvent(new Event('xunblog:search'))
</script>
<template>
  <div class="home-page">
    <header class="home-intro"><h1>XunBlog</h1><p>AI Knowledge Base</p>
      <button class="home-search" @click="openSearch"><span>Search knowledge...</span><kbd>⌘K / Ctrl K</kbd></button>
    </header>
    <section class="home-section" aria-labelledby="start-title"><h2 id="start-title">从这里开始</h2>
      <p class="start-intro">先进入目标模型；遇到依赖概念时沿 canonical link 下钻，掌握后返回主线。</p>
      <div class="starting-points">
        <article v-for="entry in model.startingPoints" :key="entry.route" class="starting-card">
          <p class="starting-label">Learning Path</p>
          <h3><a :href="href(entry.route)">{{ model.nodes[entry.route].title }}</a></h3>
          <p>{{ entry.description }}</p>
          <div class="focus-links"><a v-for="slug in entry.links" :key="slug" :href="href(entry.route + slug + '/')">{{ model.nodes[entry.route + slug + '/'].title }}</a></div>
        </article>
      </div>
    </section>
    <section class="home-section" aria-labelledby="map-title"><h2 id="map-title">完整知识树</h2>
      <p class="start-intro">归属树确定每个知识的唯一位置；文章之间的依赖链接构成可递归学习路径。</p>
      <nav aria-label="完整知识树"><KnowledgeTree :routes="model.roots" scope="map" /></nav>
    </section>
    <section v-if="model.recent.length" class="home-section" aria-labelledby="recent-title"><h2 id="recent-title">Recently Updated</h2>
      <ul class="recent-list"><li v-for="r in model.recent" :key="r"><a :href="href(r)">{{ model.nodes[r].title }}</a><time :datetime="model.nodes[r].updated">{{ formatUpdated(model.nodes[r].updated) }}</time></li></ul>
    </section>
  </div>
</template>
