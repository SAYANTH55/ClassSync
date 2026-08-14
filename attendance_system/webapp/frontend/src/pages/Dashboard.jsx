import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { gsap } from "gsap";
import { useGSAP } from "@gsap/react";
import {
  Camera, Users, UserCheck, UserX, ArrowRight, BarChart3, UserPlus, Sparkles,
} from "lucide-react";
import { api } from "../api.js";
import { useCountUp } from "../hooks/useCountUp.js";
import Ring from "../components/Ring.jsx";
import Avatar from "../components/Avatar.jsx";

// glass card that lets a soft highlight follow the cursor
function GlassCard({ className = "", children, ...rest }) {
  function onMove(e) {
    const r = e.currentTarget.getBoundingClientRect();
    e.currentTarget.style.setProperty("--mx", `${e.clientX - r.left}px`);
    e.currentTarget.style.setProperty("--my", `${e.clientY - r.top}px`);
  }
  return (
    <div onMouseMove={onMove}
      className={`glass-card rounded-2xl border border-(--color-line) bg-(--color-card) ${className}`}
      {...rest}>
      {children}
    </div>
  );
}

function StatCard({ icon: Icon, label, value, tone }) {
  const n = useCountUp(value);
  const tones = {
    ok: { text: "text-(--color-ok)", chip: "bg-(--color-ok-soft) text-(--color-ok)", glow: "rgba(52,211,153,0.5)" },
    bad: { text: "text-(--color-bad)", chip: "bg-(--color-bad-soft) text-(--color-bad)", glow: "rgba(248,113,113,0.5)" },
    accent: { text: "text-(--color-ink)", chip: "bg-(--color-accent-soft) text-(--color-accent)", glow: "rgba(139,123,255,0.55)" },
  };
  const t = tones[tone ?? "accent"];
  return (
    <GlassCard className="dash-stat p-4">
      <div className="mb-3 flex items-center justify-between">
        <span className="text-[12.5px] text-(--color-ink-2)">{label}</span>
        <span className={`flex h-8 w-8 items-center justify-center rounded-xl ${t.chip}`}
          style={{ boxShadow: `0 0 18px -4px ${t.glow}` }}>
          <Icon size={15} />
        </span>
      </div>
      <div className={`text-[28px] font-semibold tracking-tight tabular-nums ${t.text}`}>
        {Math.round(n)}
      </div>
    </GlassCard>
  );
}

function QuickAction({ to, icon: Icon, title, subtitle }) {
  return (
    <Link to={to} className="dash-action">
      <GlassCard className="group flex items-center gap-3 p-4">
        <span className="flex h-11 w-11 items-center justify-center rounded-xl bg-(--color-accent-soft) text-(--color-accent)"
          style={{ boxShadow: "0 0 18px -4px rgba(139,123,255,0.5)" }}>
          <Icon size={19} />
        </span>
        <div className="min-w-0 flex-1">
          <p className="text-[14px] font-medium">{title}</p>
          <p className="truncate text-[12px] text-(--color-ink-3)">{subtitle}</p>
        </div>
        <ArrowRight size={16} className="text-(--color-ink-3) transition-all group-hover:translate-x-1 group-hover:text-(--color-accent)" />
      </GlassCard>
    </Link>
  );
}

function Sparkline({ counts, labels }) {
  const [mounted, setMounted] = useState(false);
  useEffect(() => { const t = setTimeout(() => setMounted(true), 140); return () => clearTimeout(t); }, []);
  const max = Math.max(...counts, 1);
  return (
    <div className="flex h-28 items-end gap-1.5">
      {counts.map((c, i) => (
        <div key={i} className="group relative flex-1">
          <div
            className="w-full rounded-t-md transition-all duration-700"
            style={{
              height: mounted ? `${8 + (c / max) * 92}px` : "6px",
              transitionDelay: `${i * 35}ms`,
              background: c > 0
                ? "linear-gradient(180deg,#a78bfa,#6366f1)"
                : "var(--color-accent-soft)",
              boxShadow: c > 0 ? "0 0 14px -4px rgba(124,108,255,0.6)" : "none",
            }}
          />
          <span className="pointer-events-none absolute -top-7 left-1/2 -translate-x-1/2 whitespace-nowrap rounded-md bg-black/80 px-1.5 py-0.5 text-[10px] text-white opacity-0 transition-opacity group-hover:opacity-100">
            {labels[i]} · {c}
          </span>
        </div>
      ))}
    </div>
  );
}

export default function Dashboard() {
  const root = useRef(null);
  const orb = useRef(null);
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    api.dashboard().then(setData).catch((e) => setError(String(e)));
    const t = setInterval(() => api.dashboard().then(setData).catch(() => {}), 15000);
    return () => clearInterval(t);
  }, []);

  const pct = useCountUp(data?.percent ?? 0);

  useGSAP(() => {
    if (!data) return;
    const tl = gsap.timeline({ defaults: { ease: "power3.out" } });
    tl.from(".dash-hero", { opacity: 0, y: 26, scale: 0.98, duration: 0.65 })
      .from(".dash-stat", { opacity: 0, y: 22, stagger: 0.09, duration: 0.5 }, "-=0.3")
      .from(".dash-action", { opacity: 0, y: 16, stagger: 0.09, duration: 0.45 }, "-=0.25")
      .from(".dash-panel", { opacity: 0, y: 20, scale: 0.985, stagger: 0.12, duration: 0.55 }, "-=0.2");

    // very subtle parallax on the hero glow orb (max ~10px, feels near-invisible)
    const xTo = gsap.quickTo(orb.current, "x", { duration: 0.8, ease: "power2.out" });
    const yTo = gsap.quickTo(orb.current, "y", { duration: 0.8, ease: "power2.out" });
    const onMove = (e) => {
      const w = window.innerWidth, h = window.innerHeight;
      xTo((e.clientX / w - 0.5) * 20);
      yTo((e.clientY / h - 0.5) * 20);
    };
    window.addEventListener("pointermove", onMove);
    return () => window.removeEventListener("pointermove", onMove);
  }, { scope: root, dependencies: [!!data] });

  if (error)
    return <p className="text-sm text-(--color-bad)">Couldn’t reach the server. {error}</p>;
  if (!data)
    return <p className="text-sm text-(--color-ink-3)">Loading…</p>;

  const hour = new Date().getHours();
  const greeting = hour < 12 ? "Good morning" : hour < 17 ? "Good afternoon" : "Good evening";

  return (
    <div ref={root} className="flex flex-col gap-4">
      {/* ---- hero ---- */}
      <GlassCard className="dash-hero relative overflow-hidden !rounded-3xl p-6">
        <div ref={orb} className="pointer-events-none absolute -right-16 -top-24 h-64 w-64 rounded-full"
          style={{ background: "radial-gradient(circle, rgba(124,108,255,0.35), transparent 65%)", filter: "blur(20px)" }} />
        <div className="relative flex flex-col gap-6 lg:flex-row lg:items-center lg:justify-between">
          <div className="flex items-center gap-6">
            <Ring value={data.percent} size={136} stroke={11}>
              <span className="text-[27px] font-semibold tabular-nums">{pct.toFixed(0)}%</span>
              <span className="text-[11px] text-(--color-ink-3)">present</span>
            </Ring>
            <div>
              <h2 className="text-[25px] font-semibold tracking-tight">{greeting}</h2>
              <p className="mt-1 text-[13.5px] text-(--color-ink-2)">
                {data.present > 0
                  ? `${data.present} of ${data.total_students} students marked present today`
                  : "No attendance yet today — start a session to begin"}
              </p>
              <div className="mt-3 flex gap-4 text-[12.5px]">
                <span className="flex items-center gap-1.5"><span className="h-2 w-2 rounded-full bg-(--color-ok)" style={{ boxShadow: "0 0 8px rgba(52,211,153,0.8)" }} />{data.present} present</span>
                <span className="flex items-center gap-1.5"><span className="h-2 w-2 rounded-full bg-(--color-bad)" />{data.absent} absent</span>
              </div>
            </div>
          </div>
          <Link
            to="/camera"
            className="flex items-center justify-center gap-2 rounded-xl px-5 py-3 text-[14px] font-semibold text-white transition-all hover:-translate-y-0.5 hover:brightness-110 active:scale-[0.98]"
            style={{ background: "linear-gradient(135deg,#6366f1,#8b5cf6)", boxShadow: "0 14px 36px -10px rgba(124,108,255,0.6)" }}
          >
            <Camera size={17} /> Start attendance
          </Link>
        </div>
      </GlassCard>

      {/* ---- stats ---- */}
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <StatCard icon={Users} label="Total students" value={data.total_students} tone="accent" />
        <StatCard icon={UserCheck} label="Present today" value={data.present} tone="ok" />
        <StatCard icon={UserX} label="Absent" value={data.absent} tone="bad" />
        <StatCard icon={Camera} label="Sessions today" value={data.present > 0 ? 1 : 0} tone="accent" />
      </div>

      {/* ---- quick actions ---- */}
      <div className="grid gap-3 lg:grid-cols-3">
        <QuickAction to="/camera" icon={Camera} title="Take attendance" subtitle="Open the live classroom camera" />
        <QuickAction to="/reports" icon={BarChart3} title="View reports" subtitle="Daily and summary attendance" />
        <QuickAction to="/students" icon={UserPlus} title="Manage students" subtitle={`${data.total_students} enrolled`} />
      </div>

      {/* ---- chart + recent ---- */}
      <div className="grid gap-3 lg:grid-cols-5">
        <GlassCard className="dash-panel p-5 lg:col-span-3">
          <h3 className="mb-4 text-[13.5px] font-medium">Last 14 days</h3>
          <Sparkline counts={data.day_counts} labels={data.day_labels} />
        </GlassCard>

        <GlassCard className="dash-panel p-5 lg:col-span-2">
          <h3 className="mb-2 text-[13.5px] font-medium">Recent marks</h3>
          {data.recent.length === 0 ? (
            <div className="flex flex-col items-center py-8 text-center">
              <span className="floaty mb-3 flex h-12 w-12 items-center justify-center rounded-full bg-(--color-accent-soft) text-(--color-accent)"
                style={{ boxShadow: "0 0 24px -4px rgba(139,123,255,0.6)" }}>
                <Sparkles size={20} />
              </span>
              <p className="text-[13px] text-(--color-ink-2)">No attendance yet today</p>
              <p className="text-[12px] text-(--color-ink-3)">Start a session to see live marks here</p>
            </div>
          ) : (
            <ul className="divide-y divide-(--color-line)">
              {data.recent.map((r) => (
                <li key={r.student_id} className="flex items-center gap-3 py-2">
                  <Avatar id={r.student_id} name={r.name} size={30} tone="ok" />
                  <span className="flex-1 truncate text-[13px]">
                    {r.name} <span className="text-(--color-ink-3)">· {r.student_id}</span>
                  </span>
                  <span className="text-xs text-(--color-ink-3)">{r.time.slice(0, 5)}</span>
                  <span className="rounded-full bg-(--color-ok-soft) px-2 py-0.5 text-[11px] text-(--color-ok)">
                    {r.score.toFixed(2)}
                  </span>
                </li>
              ))}
            </ul>
          )}
        </GlassCard>
      </div>
    </div>
  );
}
