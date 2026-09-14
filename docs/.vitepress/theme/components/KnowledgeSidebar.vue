<script setup lang="ts">
import KnowledgeTree from './KnowledgeTree.vue'
import { useKnowledge } from '../composables/knowledge'
const { model } = useKnowledge()
</script>
<template><div class="knowledge-sidebar"><p>Knowledge Tree</p><KnowledgeTree :routes="model.roots" scope="sidebar" /></div></template>
