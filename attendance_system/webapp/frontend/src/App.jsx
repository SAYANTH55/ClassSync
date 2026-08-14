import { Routes, Route, Navigate } from "react-router-dom";
import Sidebar from "./components/Sidebar.jsx";
import Topbar from "./components/Topbar.jsx";
import Dashboard from "./pages/Dashboard.jsx";
import Camera from "./pages/Camera.jsx";
import Reports from "./pages/Reports.jsx";
import Students from "./pages/Students.jsx";
import Settings from "./pages/Settings.jsx";
import Login from "./pages/Login.jsx";
import { isAuthed } from "./auth.js";

// The main application shell (sidebar + topbar + routed pages). Rendered only
// when the demo session is authenticated; otherwise we redirect to /login.
function AppShell() {
  if (!isAuthed()) return <Navigate to="/login" replace />;
  return (
    <div className="flex min-h-screen">
      <Sidebar />
      <div className="flex min-w-0 flex-1 flex-col">
        <Topbar />
        <main className="mx-auto w-full max-w-6xl flex-1 px-6 py-6">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/camera" element={<Camera />} />
            <Route path="/reports" element={<Reports />} />
            <Route path="/students" element={<Students />} />
            <Route path="/settings" element={<Settings />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </main>
      </div>
    </div>
  );
}

export default function App() {
  return (
    <Routes>
      <Route path="/login" element={<Login />} />
      <Route path="/*" element={<AppShell />} />
    </Routes>
  );
}
