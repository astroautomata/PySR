<template>
  <a
    v-if="link"
    :href="link"
    class="badge-link"
    target="_blank"
    rel="noopener noreferrer"
  >
    <span class="badge-container">
      <span class="badge-label">Author</span>
      <span class="author-badge" :style="{ backgroundColor: getColor(author) }">
        <img :src="getAvatarUrl" :alt="author" :class="{ 'platform-avatar': !props.avatar }" class="author-avatar">
        {{ author }}
      </span>
    </span>
  </a>
  <span v-else class="badge-container">
    <span class="badge-label">Author</span>
    <span class="author-badge" :style="{ backgroundColor: getColor(author) }">
      <img :src="getAvatarUrl"  :alt="author" :class="{ 'platform-avatar': !props.avatar }" class="author-avatar">
      {{ author }}
    </span>
  </span>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  author: {
    type: String,
    required: true
  },
  avatar: {
    type: String,
    default: ''
  },
  platform: {
    type: String,
    default: 'user'
  },
  link: {
    type: String,
    default: ''
  }
})

// Platform avatars mapping
const platformAvatars = {
  github: 'https://img.icons8.com/ios-filled/50/github.png',
  gitlab: 'https://img.icons8.com/ios-filled/50/gitlab.png',
  x: 'https://img.icons8.com/ios/50/twitterx--v2.png',
  linkedin: 'https://img.icons8.com/ios-filled/50/linkedin.png',
  bluesky: 'https://img.icons8.com/material-sharp/48/bluesky.png',
  mastodon: 'https://img.icons8.com/windows/64/mastodon.png',
  user: 'https://img.icons8.com/windows/64/user.png'
}

const getAvatarUrl = computed(() => {
  // If custom avatar is provided, use it
  if (props.avatar) {
    return props.avatar
  }
  // If platform is specified, use platform avatar
  if (props.platform && platformAvatars[props.platform.toLowerCase()]) {
    return platformAvatars[props.platform.toLowerCase()]
  }
  // Default to user avatar
  return platformAvatars.user
})

// Color function remains the same
const getColor = (name) => {
  const colors = [
    '#3eaf7c', // green
    '#476582', // blue
    '#c53e3e', // red
    '#986801', // orange
    '#8957e5'  // purple
  ]
  const hash = name.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0)
  return colors[hash % colors.length]
}
</script>

<style scoped>
.badge-container {
  display: inline-flex;
  height: 20px;
  line-height: 24px;
  font-size: 12px;
  font-weight: 500;
  border-radius: 5px;
  margin-right: 0.5rem;
  margin-bottom: 0.5rem;
  overflow: hidden;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
  vertical-align: middle;
}

.badge-label {
  padding: 0 8px;
  background-color: #555;
  color: white;
  display: flex;
  align-items: center;
}

.author-badge {
  display: inline-flex;
  align-items: center;
  padding: 0 8px;
  color: white;
}

.author-avatar {
  width: 15px;
  height: 15px;
  border-radius: 50%;
  margin-right: 0.25rem;
  margin-left: -0.25rem;
}
.platform-avatar {
  filter: brightness(0) invert(1);
}
.badge-link {
  text-decoration: none;
  color: inherit;
}

.badge-link:hover .author-badge {
  opacity: 0.9;
}

.badge-link:hover .badge-container {
  box-shadow: 0 0 0 1.25px rgba(248, 248, 247, 0.4);
  transition: all 0.2s ease;
}
</style>
