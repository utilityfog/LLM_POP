/* eslint-disable no-restricted-globals */

// Intercept fetch requests
self.addEventListener('fetch', (event) => {
  const proxyOrigin = 'http://localhost:3001';  // Proxy server origin
  
  // Parse the original request URL
  const originalUrl = new URL(event.request.url);
  
  // Reconstruct the proxied URL using the path and query parameters from the original request
  const proxiedUrl = new URL(originalUrl.pathname + originalUrl.search, proxyOrigin);
  console.log("proxied url: ", proxiedUrl);

  // Modify the request headers to include the original origin
  let modifiedHeaders = new Headers(event.request.headers);
  modifiedHeaders.append('X-Original-Origin', originalUrl.origin);  // Custom header with the original origin

  // Modify the request to redirect through the proxy server with the new URL
  const newRequest = new Request(proxiedUrl.toString(), {
    method: event.request.method,
    headers: modifiedHeaders,
    body: event.request.body,
    mode: 'cors',
    credentials: 'same-origin',
  });

  event.respondWith(fetch(newRequest));
});