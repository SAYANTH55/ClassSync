import { useEffect, useState } from "react";
import { Cpu, Database, Gauge, ShieldCheck } from "lucide-react";
import { api } from "../api.js";

function Card({ icon: Icon, title, children }) {
  return (
    <div className="rise rounded-2xl border border-(--color-line) bg-(--color-card) p-5">
      <div className="mb-3 flex items-center gap-2 text-[13.5px] font-medium">
        <Icon size={15} className="text-(--color-accent)" /> {title}
      </div>
      {children}
    </div>
  );
}

function Row({ k, v }) {
  return (
    <div className="flex items-baseline justify-between gap-4 border-b border-(--color-line) py-2 text-[13px] last:border-0">
      <span className="text-(--color-ink-2)">{k}</span>
      <span className="text-right">{v}</span>
    </div>
  );
}

export default function Settings() {
  const [s, setS] = useState(null);
  useEffect(() => { api.settings().then(setS); }, []);
  if (!s) return <p className="text-sm text-(--color-ink-3)">Loading…</p>;

  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <Card icon={Gauge} title="Recognition threshold">
        <div className="mb-3">
          <div className="relative h-2 rounded-full bg-(--color-line)">
            <div className="absolute inset-y-0 left-0 rounded-l-full bg-(--color-bad-soft)" style={{ width: "28.6%" }} />
            <div className="absolute inset-y-0 rounded-r-full bg-(--color-ok-soft)" style={{ left: "60.7%", right: 0 }} />
            <div className="absolute -top-1 h-4 w-1 rounded bg-(--color-accent)" style={{ left: `${s.threshold * 100}%` }} />
          </div>
          <div className="mt-1.5 flex justify-between text-[11px] text-(--color-ink-3)">
            <span>impostors ≤ 0.286</span>
            <span className="font-medium text-(--color-accent)">τ = {s.threshold}</span>
            <span>genuine ≥ 0.607</span>
          </div>
        </div>
        <p className="text-[12.5px] leading-relaxed text-(--color-ink-2)">
          A face is accepted only if its similarity to an enrolled student exceeds{" "}
          {s.threshold}. The value is calibrated from a cross-device evaluation —
          it sits in the middle of the empty band between the highest impostor
          score and the lowest genuine score, so both false accepts and false
          rejects were zero on the evaluation set.
        </p>
      </Card>

      <Card icon={Cpu} title="Recognition engine">
        <Row k="Model pack" v={s.model_pack} />
        <Row k="Face detector" v={s.detector} />
        <Row k="Face embedder" v={s.embedder} />
        <Row k="Compute" v="CPU (ONNX Runtime)" />
        <Row k="Confirmation" v={`${s.confirm_frames} consecutive frames`} />
      </Card>

      <Card icon={Database} title="Gallery">
        <Row k="Enrolled students" v={s.gallery_students} />
        <Row k="Face templates" v={s.gallery_templates} />
        <Row k="Data directory" v={<code className="text-[11.5px]">{s.data_dir}</code>} />
        <p className="mt-3 text-[12.5px] text-(--color-ink-2)">
          The gallery rebuilds automatically whenever a student is added or removed.
        </p>
      </Card>

      <Card icon={ShieldCheck} title="Privacy">
        <p className="text-[12.5px] leading-relaxed text-(--color-ink-2)">
          Camera frames are processed in memory and never stored. Attendance
          logs reference students by stable pseudonymous IDs. Enrollment
          images stay on this machine and removing a student preserves their
          images outside the recognition gallery.
        </p>
      </Card>
    </div>
  );
}
