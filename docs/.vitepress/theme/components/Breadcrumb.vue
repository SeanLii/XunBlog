<script setup lang="ts">
import { useKnowledge } from '../composables/knowledge'
const { model, current, href } = useKnowledge()
</script>
<template>
  <nav v-if="current" class="breadcrumb" aria-label="Breadcrumb">
    <ol><li><a :href="href('/')">XunBlog</a></li><li v-for="r in current.breadcrumbs" :key="r"><a :href="href(r)" :aria-current="r === current.route ? 'page' : undefined">{{ model.nodes[r].title }}</a></li></ol>
  </nav>
</template>
