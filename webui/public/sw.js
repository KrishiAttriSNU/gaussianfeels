const CACHE_NAME = 'gf-cache-v1'
const CORE = ['/', '/index.html']

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(CORE)).then(() => self.skipWaiting())
  )
})

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) => Promise.all(keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k))))
      .then(() => self.clients.claim())
  )
})

self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url)
  if (url.origin === self.location.origin) {
    if (url.pathname === '/' || url.pathname.startsWith('/assets') || url.pathname.endsWith('.js') || url.pathname.endsWith('.css')) {
      event.respondWith(
        caches.match(event.request).then((cached) => cached || fetch(event.request).then((res) => {
          const respClone = res.clone()
          caches.open(CACHE_NAME).then((cache) => cache.put(event.request, respClone)).catch(()=>{})
          return res
        })).catch(() => caches.match('/index.html'))
      )
    }
  }
})

