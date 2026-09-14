<script setup lang="ts">
import { formatUpdated, useKnowledge } from '../composables/knowledge'
const { current } = useKnowledge()
</script>
<template>
  <div v-if="current?.kind === 'canonical'" class="note-meta">
    <span v-if="current.updated">Updated <time :datetime="current.updated">{{ formatUpdated(current.updated) }}</time></span>
    <span>{{ current.minutes }} min read</span>
  </div>
</template>
