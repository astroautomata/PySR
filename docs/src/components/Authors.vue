<template>
    <div class="authors-container">
      <AuthorBadge
        v-for="author in pageAuthors"
        :key="author.name"
        :author="author.name"
        :avatar="author.avatar"
        :platform="author.platform"
        :link="author.link"
      />
    </div>
  </template>

  <script setup>
  import { useData } from 'vitepress'

  const { frontmatter } = useData()
  const pageAuthors = frontmatter.value.authors || []
  </script>

  <style scoped>
  .authors-container {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin: 1rem 0;
  }
  </style>
