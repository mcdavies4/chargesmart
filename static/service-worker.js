const CACHE_NAME = 'chargesmart-v1';
const STATIC_ASSETS = [
    '/',
    '/static/manifest.json',
    '/static/icon-192.png',
    '/static/icon-512.png',
    '/static/favicon.png',
];

// Install — cache static assets
self.addEventListener('install', event => {
    event.waitUntil(
        caches.open(CACHE_NAME).then(cache => {
            return cache.addAll(STATIC_ASSETS);
        })
    );
    self.skipWaiting();
});

// Activate — clean old caches
self.addEventListener('activate', event => {
    event.waitUntil(
        caches.keys().then(keys =>
            Promise.all(keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k)))
        )
    );
    self.clients.claim();
});

// Fetch — serve from cache, fallback to network
self.addEventListener('fetch', event => {
    const url = new URL(event.request.url);

    // Always fetch API calls from network
    if (url.pathname.startsWith('/nearby') || url.pathname.startsWith('/predict')) {
        event.respondWith(
            fetch(event.request).catch(() => {
                return new Response(JSON.stringify({
                    error: 'You are offline. Please connect to the internet to find chargers.'
                }), { headers: { 'Content-Type': 'application/json' } });
            })
        );
        return;
    }

    // For everything else — cache first, network fallback
    event.respondWith(
        caches.match(event.request).then(cached => {
            return cached || fetch(event.request).then(response => {
                // Cache successful responses
                if (response.status === 200) {
                    const clone = response.clone();
                    caches.open(CACHE_NAME).then(cache => cache.put(event.request, clone));
                }
                return response;
            });
        }).catch(() => {
            // Offline fallback for navigation
            if (event.request.mode === 'navigate') {
                return caches.match('/');
            }
        })
    );
});
