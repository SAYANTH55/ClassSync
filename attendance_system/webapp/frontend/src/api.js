// Single API client — every fetch in the app goes through here.
// Same-origin requests include the session cookie automatically; a 401 means
// the session was lost, so we send the user back to the login page.

async function get(path) {
  const res = await fetch(path, { credentials: "include" });
  if (res.status === 401) {
    // full navigation resets the in-memory auth flag and shows the login page
    if (location.pathname !== "/login") location.assign("/login");
    throw new Error(`${path} -> 401`);
  }
  if (!res.ok) throw new Error(`${path} -> ${res.status}`);
  return res.json();
}

export const api = {
  dashboard: () => get("/api/dashboard"),
  health: () => get("/api/health"),
  settings: () => get("/api/settings"),
};

// WebSocket URL helper (works in dev via Vite proxy and in production). The
// session cookie is sent automatically on the same-origin WS handshake.
export function wsUrl(path) {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${location.host}${path}`;
}
