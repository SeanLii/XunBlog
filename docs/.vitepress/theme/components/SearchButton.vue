<script setup lang="ts">
const open = () => window.dispatchEvent(new Event('xunblog:search'))
</script>
<template><button class="header-search" aria-label="Search knowledge (⌘K / Ctrl K)" @click="open"><span>Search</span><kbd>⌘K</kbd></button></template>
