<!-- Adapted from https://github.com/MakieOrg/Makie.jl/blob/master/docs/src/.vitepress/theme/VersionPicker.vue -->

<script setup lang="ts">
import { ref, onMounted, computed} from 'vue'
import { useData } from 'vitepress'
import VPNavBarMenuGroup from 'vitepress/dist/client/theme-default/components/VPNavBarMenuGroup.vue'
import VPNavScreenMenuGroup from 'vitepress/dist/client/theme-default/components/VPNavScreenMenuGroup.vue'

declare global {
  interface Window {
    DOC_VERSIONS?: string[];
    DOCUMENTER_CURRENT_VERSION?: string;
  }
}

// from vitepress, MIT
function joinPath(base: string, path: string) {
  return `${base}${path}`.replace(/\/+/g, '/')
}

const absoluteRoot = __DEPLOY_ABSPATH__;

function absoluteUrl(relative) {
  // Avoid constructing URLs with origin during SSR
  if (typeof window === 'undefined') {
    return joinPath(absoluteRoot, relative);
  }

  const siteOrigin = window.location.origin;
  const withRoot = joinPath(absoluteRoot, relative);
  return siteOrigin + withRoot; // don't call `joinPath` on https:// part to not remove double slashes there
}

const props = defineProps<{ screenMenu?: boolean }>();
const versions = ref<Array<{ text: string, link: string, class?: string }>>([]);
const currentVersion = ref('Versions');
// Releases are shown as vX.Y; links still point at the full vX.Y.Z folder.
const minorLabel = (v: string) => /^(v\d+\.\d+)\.\d+$/.exec(v)?.[1] ?? v;
const isClient = ref(false);
const { site } = useData();

const isLocalBuild = () => {
  return typeof window !== 'undefined' && (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1');
};

const waitForScriptsToLoad = () => {
  return new Promise<boolean>((resolve) => {
    if (isLocalBuild() || typeof window === 'undefined') {
      resolve(false);
      return;
    }
    const checkInterval = setInterval(() => {
      if (window.DOC_VERSIONS && window.DOCUMENTER_CURRENT_VERSION) {
        clearInterval(checkInterval);
        resolve(true);
      }
    }, 100);
    setTimeout(() => {
      clearInterval(checkInterval);
      resolve(false);
    }, 5000);
  });
};

const loadVersions = async () => {
  if (typeof window === 'undefined') return;

  try {
    if (isLocalBuild()) {
      versions.value = [{ text: 'dev', link: '/' }];
      currentVersion.value = 'dev';
    } else {
      const scriptsLoaded = await waitForScriptsToLoad();

      if (scriptsLoaded && window.DOC_VERSIONS && window.DOCUMENTER_CURRENT_VERSION) {
        // DOC_VERSIONS lists releases newest first, so the first vX.Y.Z of each minor is its latest patch.
        // Aliases (v1, v1.5) and prereleases stay reachable by URL, out of the picker.
        const seenMinors = new Set();
        versions.value = window.DOC_VERSIONS
          .filter(v => {
            if (v === 'dev' || v === 'stable') return true;
            const minor = /^(v\d+\.\d+)\.\d+$/.exec(v)?.[1];
            if (!minor || seenMinors.has(minor)) return false;
            seenMinors.add(minor);
            return true;
          })
          .map(v => ({
            text: minorLabel(v),
            link: absoluteUrl(`/${v}/`),
          }));
        currentVersion.value = window.DOCUMENTER_CURRENT_VERSION;
      } else {
        versions.value = [{ text: 'dev', link: absoluteUrl('/dev/') }];
        currentVersion.value = 'dev';
      }
    }
  } catch (error) {
    console.warn('Error loading versions:', error);
    versions.value = [{ text: 'dev', link: absoluteUrl('/dev/') }];
    currentVersion.value = 'dev';
  }
  isClient.value = true;
};

const versionItems = computed(() => versions.value.map((v) => ({
  text: v.text,
  link: v.link,
  target: '_self',
  noIcon: true
})));

onMounted(() => {
  if (typeof window !== 'undefined') {
    currentVersion.value = window.DOCUMENTER_CURRENT_VERSION ?? 'Versions';
    loadVersions();
  }
});
</script>

<template>
  <template v-if="isClient">
    <VPNavBarMenuGroup
      v-if="!screenMenu && versions.length > 0"
      :item="{ text: minorLabel(currentVersion), items: versionItems }"
      class="VPVersionPicker"
    />
    <VPNavScreenMenuGroup
      v-else-if="screenMenu && versions.length > 0"
      :text="minorLabel(currentVersion)"
      :items="versionItems"
      class="VPVersionPicker"
    />
  </template>
</template>

<style scoped>
.VPVersionPicker :deep(button .text) {
  color: var(--vp-c-text-1) !important;
}
.VPVersionPicker:hover :deep(button .text) {
  color: var(--vp-c-text-2) !important;
}
</style>
