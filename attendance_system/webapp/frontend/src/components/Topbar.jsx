import { useLocation } from "react-router-dom";
import { Moon, Sun } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { gsap } from "gsap";
import { useGSAP } from "@gsap/react";

const titles = {
  "/": "Dashboard",
  "/camera": "Attendance session",
  "/reports": "Reports",
  "/students": "Students",
  "/settings": "Settings",
};

export default function Topbar() {
  const { pathname } = useLocation();
  const ref = useRef(null);
  const [dark, setDark] = useState(
    () => document.documentElement.dataset.theme !== "light"
  );
  const [now, setNow] = useState(new Date());

  useEffect(() => {
    const t = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(t);
  }, []);

  useGSAP(() => {
    gsap.from(ref.current, { y: -16, opacity: 0, duration: 0.5, ease: "power3.out" });
  }, { scope: ref });

  function toggleTheme() {
    const next = !dark;
    setDark(next);
    document.documentElement.dataset.theme = next ? "dark" : "light";
    localStorage.setItem("classsync-theme", next ? "dark" : "light");
  }

  const date = now.toLocaleDateString(undefined, {
    weekday: "short", day: "numeric", month: "short", year: "numeric",
  });
  const time = now.toLocaleTimeString(undefined, { hour12: false });

  return (
    <header ref={ref} className="sticky top-0 z-10 flex h-16 items-center justify-between border-b border-(--color-line) bg-(--color-card)/55 px-6 backdrop-blur-xl">
      <div>
        <p className="text-[10.5px] font-medium uppercase tracking-[0.2em] text-(--color-accent)">ClassSync</p>
        <h1 className="text-[16px] font-semibold tracking-tight">
          {titles[pathname] ?? "ClassSync"}
        </h1>
      </div>
      <div className="flex items-center gap-3">
        <div className="hidden text-right sm:block">
          <div className="text-[13px] font-medium tabular-nums">{time}</div>
          <div className="text-[11px] text-(--color-ink-3)">{date}</div>
        </div>
        <button
          onClick={toggleTheme}
          aria-label="Toggle theme"
          className="flex h-9 w-9 items-center justify-center rounded-xl border border-(--color-line) text-(--color-ink-2) transition-all duration-300 hover:rotate-45 hover:border-(--color-accent)/50 hover:text-(--color-accent) hover:shadow-[0_0_18px_-4px_rgba(139,123,255,0.6)]"
        >
          {dark ? <Sun size={16} /> : <Moon size={16} />}
        </button>
      </div>
    </header>
  );
}
