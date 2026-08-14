import { useRef, useState } from "react";
import { useNavigate, Navigate } from "react-router-dom";
import { gsap } from "gsap";
import { useGSAP } from "@gsap/react";
import {
  Eye, EyeOff, ArrowRight, ScanFace, ShieldCheck, Clock, Loader2, Lock, User,
} from "lucide-react";
import { login, isAuthed } from "../auth.js";

const ACCENT = "#8b7bff";

function FaceMesh() {
  // abstract face-recognition motif: landmark dots + a detection frame, subtle
  return (
    <svg viewBox="0 0 300 340" className="h-full w-full" aria-hidden="true">
      <g stroke={ACCENT} strokeWidth="0.6" opacity="0.28" fill="none">
        <path d="M150 40 C90 40 70 110 70 170 C70 250 110 300 150 300 C190 300 230 250 230 170 C230 110 210 40 150 40Z" />
        <path d="M95 150 C120 135 180 135 205 150" />
        <path d="M100 210 C125 235 175 235 200 210" />
        <path d="M150 150 L150 205" />
      </g>
      <g fill={ACCENT} opacity="0.7">
        {[[112,150],[188,150],[150,190],[128,225],[172,225],[150,90],[95,170],[205,170],[150,258]].map(
          ([x, y], i) => <circle key={i} cx={x} cy={y} r="2.4" />)}
      </g>
      <g stroke={ACCENT} strokeWidth="0.5" opacity="0.4">
        <line x1="112" y1="150" x2="150" y2="90" />
        <line x1="188" y1="150" x2="150" y2="90" />
        <line x1="112" y1="150" x2="150" y2="190" />
        <line x1="188" y1="150" x2="150" y2="190" />
        <line x1="128" y1="225" x2="150" y2="190" />
        <line x1="172" y1="225" x2="150" y2="190" />
        <line x1="128" y1="225" x2="150" y2="258" />
        <line x1="172" y1="225" x2="150" y2="258" />
      </g>
      <g stroke="#4fd1e6" strokeWidth="1.5" opacity="0.5" fill="none">
        <path d="M60 60 L60 45 L75 45" /><path d="M240 60 L240 45 L225 45" />
        <path d="M60 280 L60 295 L75 295" /><path d="M240 280 L240 295 L225 295" />
      </g>
    </svg>
  );
}

export default function Login() {
  const navigate = useNavigate();
  const root = useRef(null);
  const cardRef = useRef(null);
  const scanRef = useRef(null);
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [showPw, setShowPw] = useState(false);
  const [error, setError] = useState(false);
  const [busy, setBusy] = useState(false);

  useGSAP(() => {
    const tl = gsap.timeline({ defaults: { ease: "power3.out" } });
    tl.from(".lg-bg", { opacity: 0, duration: 0.9 })
      .from(".lg-brand", { opacity: 0, y: 18, duration: 0.6 }, "-=0.35")
      .from(".lg-head", { opacity: 0, y: 30, duration: 0.7 }, "-=0.25")
      .from(".lg-sub", { opacity: 0, y: 18, duration: 0.6 }, "-=0.4")
      .from(".lg-feat", { opacity: 0, y: 14, stagger: 0.12, duration: 0.5 }, "-=0.3")
      .from(cardRef.current, { opacity: 0, y: 26, scale: 0.96, duration: 0.75 }, "-=0.55")
      .from(".lg-field", { opacity: 0, y: 16, stagger: 0.1, duration: 0.5 }, "-=0.35")
      .from(".lg-btn", { opacity: 0, y: 12, duration: 0.5 }, "-=0.2");

    gsap.to(scanRef.current, {
      y: 300, duration: 3.2, repeat: -1, yoyo: true, ease: "sine.inOut",
      keyframes: { opacity: [0, 0.7, 0] },
    });
  }, { scope: root });

  if (isAuthed()) return <Navigate to="/" replace />;

  async function submit(e) {
    e.preventDefault();
    if (busy) return;
    setBusy(true);
    const ok = await login(username, password);
    if (ok) {
      setError(false);
      gsap.timeline()
        .to(cardRef.current, { scale: 1.02, duration: 0.22, yoyo: true, repeat: 1, ease: "power2.inOut" })
        .to(root.current, { opacity: 0, duration: 0.5, ease: "power2.in",
          onComplete: () => navigate("/", { replace: true }) }, "+=0.15");
    } else {
      setBusy(false);
      setError(true);
      gsap.to(cardRef.current, {
        keyframes: { x: [-10, 10, -7, 7, -3, 0] }, duration: 0.5, ease: "power2.out",
      });
    }
  }

  const inputCls =
    "w-full rounded-xl border border-white/10 bg-white/[0.04] px-11 py-3 text-[14px] " +
    "text-[#f3f4fb] placeholder:text-[#6a6d82] outline-none " +
    "transition-all focus:border-[#8b7bff]/60 focus:bg-white/[0.07] " +
    "focus:shadow-[0_0_0_3px_rgba(139,123,255,0.18)]";

  return (
    <div
      ref={root}
      className="relative min-h-screen w-full overflow-hidden text-[#f3f4fb]"
      style={{ fontFamily: "'Inter','Segoe UI',sans-serif" }}
    >
      {/* ---- background layers ---- */}
      <div
        className="lg-bg absolute inset-0 -z-20"
        style={{
          background:
            "radial-gradient(1200px 820px at 100% -5%, rgba(124,108,255,0.30), transparent 55%)," +
            "radial-gradient(900px 700px at -10% 110%, rgba(79,209,230,0.14), transparent 52%)," +
            "radial-gradient(700px 600px at 100% 100%, rgba(139,92,246,0.16), transparent 55%)," +
            "linear-gradient(160deg,#0d0d1a 0%,#0a0a12 45%,#07070f 100%)",
        }}
      />
      <div className="absolute inset-0 -z-10 opacity-[0.6]" style={{
        background:
          "repeating-linear-gradient(115deg, transparent 0 90px, rgba(160,150,255,0.03) 90px 92px)",
      }} />
      <div className="absolute inset-0 -z-10 opacity-[0.6]" style={{
        maskImage: "radial-gradient(70% 60% at 50% 40%, black, transparent)",
        WebkitMaskImage: "radial-gradient(70% 60% at 50% 40%, black, transparent)",
        background:
          "linear-gradient(rgba(139,123,255,0.05) 1px, transparent 1px) 0 0/46px 46px," +
          "linear-gradient(90deg, rgba(139,123,255,0.05) 1px, transparent 1px) 0 0/46px 46px",
      }} />

      {/* ---- content ---- */}
      <div className="relative z-10 mx-auto flex min-h-screen max-w-6xl flex-col justify-center gap-10 px-6 py-10 lg:grid lg:grid-cols-2 lg:items-center">

        {/* LEFT — branding */}
        <div className="relative">
          <div className="pointer-events-none absolute -left-6 -top-10 h-72 w-64 opacity-70 lg:-top-4 lg:left-auto lg:right-8">
            <FaceMesh />
            <div ref={scanRef} className="absolute left-0 top-0 h-[2px] w-full"
              style={{ background: "linear-gradient(90deg, transparent, rgba(139,123,255,0.95), transparent)" }} />
          </div>

          <div className="lg-brand relative mb-8 flex items-center gap-2.5">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl"
              style={{ background: "linear-gradient(135deg,#6366f1,#8b5cf6)", boxShadow: "0 8px 24px -6px rgba(124,108,255,0.7)" }}>
              <svg width="20" height="20" viewBox="0 0 100 100" aria-hidden="true">
                <path d="M72 34 A26 26 0 1 0 72 66" fill="none" stroke="#fff" strokeWidth="10" strokeLinecap="round" />
                <circle cx="57" cy="50" r="6" fill="#fff" />
              </svg>
            </div>
            <span className="text-[17px] font-semibold tracking-tight">
              Class<span style={{ color: ACCENT }}>Sync</span>
            </span>
          </div>

          <p className="lg-brand mb-4 text-[12px] font-medium uppercase tracking-[0.25em]" style={{ color: ACCENT }}>
            AI-powered attendance
          </p>
          <h1 className="lg-head max-w-md text-[40px] font-semibold leading-[1.08] tracking-tight sm:text-[46px]">
            Smarter Attendance.
            <br />
            <span style={{ color: "#a78bfa" }}>Secure Recognition.</span>
          </h1>
          <p className="lg-sub mt-5 max-w-md text-[15px] leading-relaxed text-[#a6a9bd]">
            Real-time face recognition with intelligent liveness detection for
            modern classrooms — every student verified, every photo attack blocked.
          </p>

          <div className="mt-8 flex flex-wrap gap-3">
            {[
              { icon: ScanFace, label: "Face Recognition" },
              { icon: ShieldCheck, label: "Anti-Spoofing" },
              { icon: Clock, label: "Real-Time Attendance" },
            ].map(({ icon: Icon, label }) => (
              <span key={label}
                className="lg-feat flex items-center gap-2 rounded-full border border-white/10 bg-white/[0.05] px-3.5 py-2 text-[12.5px] text-[#d6d8e6] backdrop-blur-sm">
                <Icon size={15} style={{ color: ACCENT }} /> {label}
              </span>
            ))}
          </div>
        </div>

        {/* RIGHT — glass login card */}
        <div className="flex justify-center lg:justify-end">
          <form
            ref={cardRef}
            onSubmit={submit}
            className="w-full max-w-[400px] rounded-3xl border border-white/12 bg-white/[0.06] p-7 shadow-[0_30px_80px_-20px_rgba(0,0,0,0.75)] backdrop-blur-2xl sm:p-8"
          >
            <div className="mb-6">
              <h2 className="text-[22px] font-semibold tracking-tight">Welcome back</h2>
              <p className="mt-1 text-[13.5px] text-[#a6a9bd]">Sign in to your classroom</p>
            </div>

            {/* username */}
            <div className="lg-field mb-4">
              <label htmlFor="lg-user" className="mb-1.5 block text-[12.5px] font-medium text-[#a6a9bd]">
                Username
              </label>
              <div className="relative">
                <User size={16} className="pointer-events-none absolute left-3.5 top-1/2 -translate-y-1/2 text-[#6a6d82]" aria-hidden="true" />
                <input
                  id="lg-user" name="username" type="text" autoComplete="username"
                  value={username} onChange={(e) => { setUsername(e.target.value); setError(false); }}
                  placeholder="Enter your username" className={inputCls} required
                />
              </div>
            </div>

            {/* password */}
            <div className="lg-field mb-2">
              <label htmlFor="lg-pass" className="mb-1.5 block text-[12.5px] font-medium text-[#a6a9bd]">
                Password
              </label>
              <div className="relative">
                <Lock size={16} className="pointer-events-none absolute left-3.5 top-1/2 -translate-y-1/2 text-[#6a6d82]" aria-hidden="true" />
                <input
                  id="lg-pass" name="password" type={showPw ? "text" : "password"}
                  autoComplete="current-password"
                  value={password} onChange={(e) => { setPassword(e.target.value); setError(false); }}
                  placeholder="Enter your password" className={inputCls + " pr-11"} required
                />
                <button
                  type="button" onClick={() => setShowPw((v) => !v)}
                  aria-label={showPw ? "Hide password" : "Show password"}
                  className="absolute right-2.5 top-1/2 -translate-y-1/2 rounded-lg p-1.5 text-[#6a6d82] transition-colors hover:text-[#8b7bff]"
                >
                  {showPw ? <EyeOff size={17} /> : <Eye size={17} />}
                </button>
              </div>
            </div>

            {/* error */}
            <div className="min-h-[22px]" aria-live="polite">
              {error && (
                <p className="text-[12.5px] text-[#f87171]">Invalid credentials. Please try again.</p>
              )}
            </div>

            {/* submit */}
            <button
              type="submit" disabled={busy}
              className="lg-btn group mt-2 flex w-full items-center justify-center gap-2 rounded-xl px-4 py-3 text-[14.5px] font-semibold text-white transition-all hover:-translate-y-0.5 hover:brightness-110 active:scale-[0.98] disabled:opacity-70"
              style={{ background: "linear-gradient(135deg,#6366f1,#8b5cf6)", boxShadow: "0 14px 34px -8px rgba(124,108,255,0.6)" }}
            >
              {busy ? (
                <><Loader2 size={17} className="animate-spin" /> Verifying…</>
              ) : (
                <>Sign In <ArrowRight size={17} className="transition-transform group-hover:translate-x-1" /></>
              )}
            </button>

            <div className="mt-5 flex items-center justify-center gap-1.5 text-[11.5px] text-[#6a6d82]">
              <ShieldCheck size={13} /> Secure AI Attendance System
            </div>
          </form>
        </div>
      </div>
    </div>
  );
}
