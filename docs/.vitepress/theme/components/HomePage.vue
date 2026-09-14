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
    <section class="home-section" aria-labelledby="map-title"><h2 id="map-title">Knowledge Map</h2>
      <nav aria-label="Knowledge Map"><KnowledgeTree :routes="model.roots" scope="map" /></nav>
    </section>
    <section class="home-section current-focus" aria-labelledby="focus-title"><h2 id="focus-title">Currently Studying</h2>
      <h3><a :href="href(model.focus)">{{ model.nodes[model.focus].title }}</a></h3>
      <p>Action Chunking with Transformers</p>
      <div class="focus-links"><a v-for="slug in ['architecture', 'cvae-in-act', 'training', 'inference']" :key="slug" :href="href(model.focus + slug + '/')">{{ model.nodes[model.focus + slug + '/'].title }}</a></div>
    </section>
    <section v-if="model.recent.length" class="home-section" aria-labelledby="recent-title"><h2 id="recent-title">Recently Updated</h2>
      <ul class="recent-list"><li v-for="r in model.recent" :key="r"><a :href="href(r)">{{ model.nodes[r].title }}</a><time :datetime="model.nodes[r].updated">{{ formatUpdated(model.nodes[r].updated) }}</time></li></ul>
    </section>
  </div>
</template>
