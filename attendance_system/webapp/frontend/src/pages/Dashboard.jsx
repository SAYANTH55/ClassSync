import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { gsap } from "gsap";
import { useGSAP } from "@gsap/react";
import {
  Camera, ArrowRight, BarChart3, UserPlus, Sparkles,
} from "lucide-react";
import { api } from "../api.js";
import { useCountUp } from "../hooks/useCountUp.js";
import Ring from "../components/Ring.jsx";
import Avatar from "../components/Avatar.jsx";

const REDUCE = typeof window !== "undefined" &&
  window.matchMedia("(prefers-reduced-motion: reduce)").matches;

function StatNum({ value, className }) {
  const n = useCountUp(value);
  return <span className={className}>{Math.round(n)}</span>;
}

function QuickAction({ to, icon: Icon, title, subtitle }) {
  return (
    <Link
      to={to}
      className="dash-in group relative flex items-center gap-3.5 rounded-2xl px-4 py-3.5 transition-all duration-200 hover:-translate-y-0.5 hover:bg-(--color-card) hover:shadow-[0_18px_40px_-24px_rgba(124,108,255,0.55)]"
    >
      <span className="flex h-10 w-10 items-center justify-center rounded-xl bg-white/[0.04] text-(--color-ink-2) transition-all duration-200 group-hover:bg-(--color-accent-soft) group-hover:text-(--color-accent)">
        <Icon size={18} />
      </span>
      <div className="min-w-0 flex-1">
        <p className="text-[14px] font-medium">{title}</p>
        <p className="truncate text-[12px] text-(--color-ink-3)">{subtitle}</p>
      </div>
      <ArrowRight size={16} className="text-(--color-ink-3) transition-all group-hover:translate-x-0.5 group-hover:text-(--color-accent)" />
    </Link>
  );
}

function TrendChart({ counts, labels }) {
  const [mounted, setMounted] = useState(REDUCE);
  useEffect(() => {
    if (REDUCE) return;
    const t = setTimeout(() => setMounted(true), 120);
    return () => clearTimeout(t);
  }, []);
  const max = Math.max(...counts, 1);
  return (
    <div className="relative">
      {/* subtle baseline grid */}
      <div className="pointer-events-none absolute inset-x-0 top-0 bottom-6 flex flex-col justify-between">
        {[0, 1, 2, 3].map((i) => <div key={i} className="h-px w-full bg-(--color-line)/60" />)}
      </div>
      <div className="relative flex h-36 items-end gap-1.5">
        {counts.map((c, i) => (
          <div key={i} className="group/bar relative flex flex-1 flex-col items-center">
            <div className="flex h-32 w-full items-end">
              <div
                className="w-full rounded-t-[5px] transition-[height] duration-700 ease-out"
                style={{
                  height: mounted ? `${6 + (c / max) * 116}px` : "4px",
                  transitionDelay: `${i * 28}ms`,
                  background: c > 0 ? "linear-gradient(180deg,#a78bfa,#6366f1)" : "var(--color-accent-soft)",
                  boxShadow: c > 0 ? "0 0 12px -5px rgba(124,108,255,0.7)" : "none",
                }}
              />
            </div>
            <span className="mt-1.5 text-[9.5px] tabular-nums text-(--color-ink-3)">{labels[i]?.slice(-2)}</span>
            <span className="pointer-events-none absolute -top-7 left-1/2 -translate-x-1/2 whitespace-nowrap rounded-md bg-black/85 px-1.5 py-0.5 text-[10px] text-white opacity-0 transition-opacity group-hover/bar:opacity-100">
              {labels[i]} · {c}
            </span>
          </div>
        ))}
      </div>
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
    if (!data || REDUCE) return;
    // no opacity here — a from-tween killed mid-flight by the 15s data poll
    // must never leave dashboard sections stuck invisible.
    gsap.from(".dash-in", { y: 18, stagger: 0.07, duration: 0.5, ease: "power3.out" });

    const xTo = gsap.quickTo(orb.current, "x", { duration: 0.9, ease: "power2.out" });
    const yTo = gsap.quickTo(orb.current, "y", { duration: 0.9, ease: "power2.out" });
    const onMove = (e) => {
      xTo((e.clientX / window.innerWidth - 0.5) * 18);
      yTo((e.clientY / window.innerHeight - 0.5) * 18);
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
  const sessions = data.present > 0 ? 1 : 0;

  return (
    <div ref={root} className="flex flex-col gap-7">
      {/* ---- header ---- */}
      <div className="dash-in flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-[27px] font-semibold tracking-tight">{greeting}</h1>
          <p className="mt-1 text-[14px] text-(--color-ink-2)">Here’s today’s attendance overview.</p>
        </div>
        <Link
          to="/camera"
          className="flex items-center justify-center gap-2 self-start rounded-xl px-5 py-3 text-[14px] font-semibold text-white transition-all hover:-translate-y-0.5 hover:brightness-110 active:scale-[0.98] sm:self-auto"
          style={{ background: "linear-gradient(135deg,#6366f1,#8b5cf6)", boxShadow: "0 16px 40px -12px rgba(124,108,255,0.7)" }}
        >
          <Camera size={17} /> Start attendance
        </Link>
      </div>

      {/* ---- attendance overview ---- */}
      <section className="dash-in relative overflow-hidden rounded-3xl border border-(--color-line) bg-(--color-card) p-6 lg:p-8">
        <div ref={orb} className="pointer-events-none absolute -right-20 -top-24 h-64 w-64 rounded-full"
          style={{ background: "radial-gradient(circle, rgba(124,108,255,0.28), transparent 65%)", filter: "blur(22px)" }} />
        <div className="relative mb-6 flex items-center justify-between">
          <p className="text-[11px] font-medium uppercase tracking-[0.22em] text-(--color-ink-3)">Today’s attendance</p>
          <span className="text-[11.5px] text-(--color-ink-3)">
            {sessions} session{sessions === 1 ? "" : "s"} today
          </span>
        </div>

        <div className="relative flex flex-col items-center gap-8 sm:flex-row sm:gap-10">
          <Ring value={data.percent} size={140} stroke={12}>
            <span className="text-[28px] font-semibold tabular-nums">{pct.toFixed(0)}%</span>
            <span className="text-[11px] text-(--color-ink-3)">present</span>
          </Ring>

          <div className="hidden h-28 w-px bg-(--color-line) sm:block" />

          <div className="w-full flex-1">
            <p className="mb-3 text-[14px] text-(--color-ink-2)">
              {data.present > 0
                ? <><span className="font-semibold text-(--color-ink)">{data.present}</span> of {data.total_students} students marked present</>
                : "No attendance yet today — start a session to begin"}
            </p>
            {/* progress */}
            <div className="mb-6 h-2.5 w-full overflow-hidden rounded-full bg-white/[0.05]">
              <div className="h-full rounded-full transition-[width] duration-1000 ease-out"
                style={{ width: `${data.percent}%`, background: "linear-gradient(90deg,#6366f1,#a78bfa)", boxShadow: "0 0 14px -2px rgba(124,108,255,0.7)" }} />
            </div>
            {/* connected stats */}
            <div className="grid grid-cols-3 divide-x divide-(--color-line)">
              <div className="pr-4">
                <div className="flex items-center gap-1.5 text-[12px] text-(--color-ink-2)">
                  <span className="h-2 w-2 rounded-full bg-(--color-ok)" style={{ boxShadow: "0 0 8px rgba(52,211,153,0.8)" }} /> Present
                </div>
                <StatNum value={data.present} className="mt-1 block text-[26px] font-semibold tabular-nums text-(--color-ok)" />
              </div>
              <div className="px-4">
                <div className="flex items-center gap-1.5 text-[12px] text-(--color-ink-2)">
                  <span className="h-2 w-2 rounded-full bg-(--color-bad)" /> Absent
                </div>
                <StatNum value={data.absent} className="mt-1 block text-[26px] font-semibold tabular-nums text-(--color-bad)" />
              </div>
              <div className="pl-4">
                <div className="flex items-center gap-1.5 text-[12px] text-(--color-ink-2)">
                  <span className="h-2 w-2 rounded-full bg-(--color-accent)" style={{ boxShadow: "0 0 8px rgba(139,123,255,0.8)" }} /> Total students
                </div>
                <StatNum value={data.total_students} className="mt-1 block text-[26px] font-semibold tabular-nums" />
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ---- quick actions ---- */}
      <div>
        <p className="dash-in mb-2 px-1 text-[11px] font-medium uppercase tracking-[0.22em] text-(--color-ink-3)">Quick actions</p>
        <div className="grid gap-1 lg:grid-cols-3">
          <QuickAction to="/camera" icon={Camera} title="Take attendance" subtitle="Start a live recognition session" />
          <QuickAction to="/reports" icon={BarChart3} title="View reports" subtitle="Review attendance history" />
          <QuickAction to="/students" icon={UserPlus} title="Manage students" subtitle={`${data.total_students} students enrolled`} />
        </div>
      </div>

      {/* ---- trend + recent ---- */}
      <div className="grid gap-5 lg:grid-cols-5">
        <section className="dash-in rounded-3xl border border-(--color-line) bg-(--color-card) p-6 lg:col-span-3">
          <div className="mb-5 flex items-baseline justify-between">
            <h3 className="text-[14px] font-medium">Attendance trend</h3>
            <span className="text-[11.5px] text-(--color-ink-3)">Last 14 days</span>
          </div>
          <TrendChart counts={data.day_counts} labels={data.day_labels} />
        </section>

        <section className="dash-in rounded-3xl border border-(--color-line) bg-(--color-card) p-6 lg:col-span-2">
          <h3 className="mb-4 text-[14px] font-medium">Recent attendance</h3>
          {data.recent.length === 0 ? (
            <div className="flex flex-col items-center py-10 text-center">
              <span className="floaty mb-3 flex h-12 w-12 items-center justify-center rounded-2xl bg-(--color-accent-soft) text-(--color-accent)"
                style={{ boxShadow: "0 0 26px -4px rgba(139,123,255,0.6)" }}>
                <Sparkles size={20} />
              </span>
              <p className="text-[13.5px] text-(--color-ink-2)">No attendance recorded yet</p>
              <p className="mt-0.5 max-w-[220px] text-[12px] text-(--color-ink-3)">
                Start an attendance session to see live recognition results here.
              </p>
            </div>
          ) : (
            <ul className="flex flex-col">
              {data.recent.map((r, i) => (
                <li key={r.student_id}
                  className={`flex items-center gap-3 py-2.5 ${i > 0 ? "border-t border-(--color-line)" : ""}`}>
                  <Avatar id={r.student_id} name={r.name} size={34} tone="ok" />
                  <div className="min-w-0 flex-1">
                    <p className="truncate text-[13.5px] font-medium">{r.name}</p>
                    <p className="text-[11.5px] text-(--color-ink-3)">{r.time.slice(0, 5)} · Present</p>
                  </div>
                  <span className="rounded-full bg-(--color-ok-soft) px-2.5 py-0.5 text-[11.5px] font-medium tabular-nums text-(--color-ok)">
                    {(r.score * 100).toFixed(1)}%
                  </span>
                </li>
              ))}
            </ul>
          )}
        </section>
      </div>
    </div>
  );
}
