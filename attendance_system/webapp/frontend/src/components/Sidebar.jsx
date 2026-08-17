import { NavLink, useNavigate } from "react-router-dom";
import { useEffect, useRef, useState } from "react";
import { gsap } from "gsap";
import { useGSAP } from "@gsap/react";
import {
  LayoutDashboard, Camera, BarChart3, Users, Settings, LogOut,
} from "lucide-react";
import { api } from "../api.js";
import { logout } from "../auth.js";

const items = [
  { to: "/", label: "Dashboard", icon: LayoutDashboard },
  { to: "/camera", label: "Camera", icon: Camera },
  { to: "/reports", label: "Reports", icon: BarChart3 },
  { to: "/students", label: "Students", icon: Users },
  { to: "/settings", label: "Settings", icon: Settings },
];

export default function Sidebar() {
  const navigate = useNavigate();
  const ref = useRef(null);
  const [engineUp, setEngineUp] = useState(null);

  useEffect(() => {
    api.health().then((h) => setEngineUp(h.status === "ok")).catch(() => setEngineUp(false));
  }, []);

  useGSAP(() => {
    gsap.from(ref.current, { x: -28, opacity: 0, duration: 0.6, ease: "power3.out" });
    gsap.from(".nav-item", { x: -14, opacity: 0, stagger: 0.06, duration: 0.4, delay: 0.15, ease: "power2.out" });
  }, { scope: ref });

  return (
    <aside ref={ref} className="sticky top-0 hidden h-screen w-64 flex-shrink-0 flex-col border-r border-(--color-line) bg-(--color-card) px-3.5 py-5 md:flex">
      {/* brand */}
      <div className="mb-8 flex items-center gap-3 px-2">
        <div className="flex h-9 w-9 items-center justify-center rounded-xl"
          style={{ background: "linear-gradient(135deg,#6366f1,#8b5cf6)", boxShadow: "0 8px 22px -6px rgba(124,108,255,0.7)" }}>
          <svg width="19" height="19" viewBox="0 0 100 100" aria-hidden="true">
            <path d="M72 34 A26 26 0 1 0 72 66" fill="none" stroke="#fff" strokeWidth="10" strokeLinecap="round" />
            <circle cx="57" cy="50" r="6" fill="#fff" />
          </svg>
        </div>
        <div className="leading-tight">
          <div className="text-[15.5px] font-semibold tracking-tight">
            Class<span className="text-(--color-accent)">Sync</span>
          </div>
          <div className="text-[10px] uppercase tracking-[0.2em] text-(--color-ink-3)">AI attendance</div>
        </div>
      </div>

      <nav className="flex flex-col gap-1">
        {items.map(({ to, label, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            end={to === "/"}
            className={({ isActive }) =>
              `nav-item group relative flex items-center gap-3 rounded-xl px-3 py-2.5 text-[13.5px] transition-all ${
                isActive
                  ? "bg-white/[0.05] font-medium text-(--color-ink)"
                  : "text-(--color-ink-2) hover:bg-white/[0.03] hover:text-(--color-ink)"
              }`
            }
          >
            {({ isActive }) => (
              <>
                <span className={`absolute left-0 top-1/2 h-5 w-[3px] -translate-y-1/2 rounded-r-full bg-(--color-accent) transition-all ${isActive ? "opacity-100" : "opacity-0"}`}
                  style={isActive ? { boxShadow: "0 0 10px rgba(139,123,255,0.9)" } : undefined} />
                <Icon size={17.5} strokeWidth={2}
                  className={`transition-all group-hover:scale-110 ${isActive ? "text-(--color-accent)" : ""}`} />
                {label}
              </>
            )}
          </NavLink>
        ))}
      </nav>

      <div className="mt-auto border-t border-(--color-line) pt-3">
        <button
          onClick={() => { logout(); navigate("/login", { replace: true }); }}
          className="mb-1 flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-[13.5px] text-(--color-ink-2) transition-colors hover:bg-(--color-bad-soft) hover:text-(--color-bad)"
        >
          <LogOut size={17.5} strokeWidth={2} /> Sign out
        </button>
        <div className="flex items-center gap-2 px-3 py-2 text-xs text-(--color-ink-3)">
          <span
            className={`inline-block h-2 w-2 rounded-full ${
              engineUp === null ? "bg-(--color-ink-3)" : engineUp ? "bg-(--color-ok) softpulse" : "bg-(--color-bad)"
            }`}
            style={engineUp ? { boxShadow: "0 0 8px rgba(52,211,153,0.9)" } : undefined}
          />
          {engineUp === null ? "Connecting…" : engineUp ? "Engine online" : "Engine offline"}
        </div>
      </div>
    </aside>
  );
}
