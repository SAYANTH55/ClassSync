// -----------------------------------------------------------------------------
// Authentication client — talks to the FastAPI backend.
//
// The backend verifies credentials and issues a signed HttpOnly session cookie
// (see webapp/backend/routers/auth.py). This file NO LONGER contains any
// credentials — verification is entirely server-side.
//
// `authed` is an in-memory flag used only by the React route guard for the
// "login required on every reload" demo UX. Real access control is enforced by
// the backend cookie on every API request and on the camera WebSocket.
// -----------------------------------------------------------------------------

let authed = false; // in-memory only; reset on reload / close / reopen

export async function login(username, password) {
  try {
    const res = await fetch("/api/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "include",
      body: JSON.stringify({ username, password }),
    });
    authed = res.ok;
    return res.ok;
  } catch {
    authed = false;
    return false;
  }
}

export function isAuthed() {
  return authed;
}

export async function logout() {
  authed = false;
  try {
    await fetch("/api/auth/logout", { method: "POST", credentials: "include" });
  } catch {
    /* ignore network errors on logout */
  }
}
