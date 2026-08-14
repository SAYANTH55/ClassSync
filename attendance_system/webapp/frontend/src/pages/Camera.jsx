import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Check, Loader2, ScanFace, ShieldAlert, Users, X } from "lucide-react";
import { wsUrl } from "../api.js";
import Avatar from "../components/Avatar.jsx";

const FRAME_INTERVAL_MS = 1200; // classroom sweep cadence (seated students)

const BOX_TONE = {
  marked: "border-[#16a34a]",
  confirming: "border-[#d97706]",
  unknown: "border-[#dc2626]",
  spoof: "border-[#dc2626]",
};
const LABEL_TONE = {
  marked: "bg-[#16a34a]",
  confirming: "bg-[#d97706]",
  unknown: "bg-[#6b7280]",
  spoof: "bg-[#dc2626]",
};

export default function Camera() {
  const videoRef = useRef(null);
  const wsRef = useRef(null);
  const timerRef = useRef(null);
  const navigate = useNavigate();

  const [phase, setPhase] = useState("starting"); // starting|warming|live|error
  const [faces, setFaces] = useState([]);
  const [counts, setCounts] = useState({ detected: 0, recognized: 0, marked_session: 0 });
  const [marked, setMarked] = useState([]); // [{student_id, name}]
  const [clock, setClock] = useState(new Date());

  useEffect(() => {
    const t = setInterval(() => setClock(new Date()), 1000);
    return () => clearInterval(t);
  }, []);

  useEffect(() => {
    let stream;
    let cancelled = false;

    async function start() {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 1280, height: 720, facingMode: "user" },
        });
      } catch {
        setPhase("error");
        return;
      }
      if (cancelled) return;
      videoRef.current.srcObject = stream;
      await videoRef.current.play();

      const ws = new WebSocket(wsUrl("/ws/camera"));
      wsRef.current = ws;
      ws.onmessage = (ev) => {
        const msg = JSON.parse(ev.data);
        if (msg.state === "loading_engine") setPhase("warming");
        else if (msg.state === "ready") {
          setPhase("live");
          scheduleFrame();
        } else if (msg.state === "frame") {
          setFaces(msg.faces);
          setCounts(msg.counts);
          setMarked(msg.marked);
          scheduleFrame();
        }
      };
      ws.onerror = () => setPhase("error");
    }

    function scheduleFrame() {
      timerRef.current = setTimeout(sendFrame, FRAME_INTERVAL_MS);
    }

    function sendFrame() {
      const v = videoRef.current;
      const ws = wsRef.current;
      if (!v || !ws || ws.readyState !== WebSocket.OPEN) return;
      const canvas = document.createElement("canvas");
      canvas.width = 960; // higher than single-face so back-row faces survive
      canvas.height = Math.round((960 * v.videoHeight) / v.videoWidth) || 540;
      canvas.getContext("2d").drawImage(v, 0, 0, canvas.width, canvas.height);
      canvas.toBlob(
        (blob) => blob && blob.arrayBuffer().then((b) => ws.send(b)),
        "image/jpeg",
        0.8
      );
    }

    start();
    return () => {
      cancelled = true;
      clearTimeout(timerRef.current);
      wsRef.current?.close();
      stream?.getTracks().forEach((t) => t.stop());
    };
  }, []);

  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 text-[13px] text-(--color-ink-2)">
          <span className={`h-2 w-2 rounded-full ${phase === "live" ? "animate-pulse bg-(--color-bad)" : "bg-(--color-ink-3)"}`} />
          {phase === "live" ? "Classroom session live" : phase === "warming" ? "Warming up the engine…" : phase === "starting" ? "Starting camera…" : "Camera unavailable"}
          <span className="ml-3 tabular-nums">{clock.toLocaleTimeString()}</span>
        </div>
        <button
          onClick={() => navigate("/")}
          className="flex h-8 w-8 items-center justify-center rounded-lg border border-(--color-line) bg-(--color-card) text-(--color-ink-2) hover:text-(--color-ink)"
          aria-label="End session"
        >
          <X size={15} />
        </button>
      </div>

      <div className="grid gap-3 lg:grid-cols-[1fr_260px]">
        <div className="relative overflow-hidden rounded-2xl bg-black" style={{ aspectRatio: "16/9" }}>
          <video ref={videoRef} muted playsInline className="h-full w-full -scale-x-100 object-cover" />

          {faces.map((f, i) => (
            <div
              key={i}
              className={`absolute rounded-lg border-2 transition-all duration-200 ${BOX_TONE[f.state]}`}
              style={{
                right: `${f.box[0] * 100}%`, // video is mirrored -> mirror x
                top: `${f.box[1] * 100}%`,
                width: `${f.box[2] * 100}%`,
                height: `${f.box[3] * 100}%`,
              }}
            >
              <span className={`absolute -top-[22px] left-0 flex items-center gap-1 whitespace-nowrap rounded px-1.5 py-0.5 text-[11px] font-medium text-white ${LABEL_TONE[f.state]}`}>
                {f.state === "unknown" ? "unknown"
                  : f.state === "spoof" ? (
                    <><ShieldAlert size={11} /> spoof — photo/screen</>
                  ) : (
                  <>
                    {f.state === "marked" && <Check size={11} />}
                    {f.name}
                    {f.state === "confirming" ? ` ${f.streak}/${f.needed}` : ""}
                  </>
                )}
              </span>
            </div>
          ))}

          <div className="absolute left-3 top-3 flex gap-2 text-[12px] text-white">
            <span className="rounded-md bg-black/55 px-2 py-1 backdrop-blur">{counts.detected} detected</span>
            <span className="rounded-md bg-black/55 px-2 py-1 backdrop-blur">{counts.recognized} recognized</span>
            <span className="rounded-md bg-[#16a34a]/80 px-2 py-1 backdrop-blur">{counts.marked_session} marked</span>
            {faces.some((f) => f.state === "spoof") && (
              <span className="rounded-md bg-[#dc2626]/85 px-2 py-1 backdrop-blur">
                {faces.filter((f) => f.state === "spoof").length} spoof blocked
              </span>
            )}
          </div>

          {phase !== "live" && (
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="flex items-center gap-2.5 rounded-xl bg-black/60 px-4 py-3 text-[13.5px] text-white backdrop-blur">
                {phase === "warming" ? <Loader2 size={18} className="animate-spin" /> : phase === "error" ? <X size={18} /> : <ScanFace size={18} className="animate-pulse" />}
                {phase === "warming" ? "Loading models (~10s)…" : phase === "error" ? "Camera unavailable — allow access and reload" : "Starting camera…"}
              </div>
            </div>
          )}
          {phase === "live" && counts.detected === 0 && (
            <div className="absolute bottom-3 left-1/2 -translate-x-1/2 rounded-lg bg-black/55 px-3 py-1.5 text-[12.5px] text-white backdrop-blur">
              Waiting for students to appear…
            </div>
          )}
        </div>

        <div className="rounded-2xl border border-(--color-line) bg-(--color-card) p-4">
          <div className="mb-2 flex items-center gap-2 text-[13.5px] font-medium">
            <Users size={15} /> Marked present
            <span className="ml-auto rounded-full bg-(--color-ok-soft) px-2 py-0.5 text-[12px] text-(--color-ok)">
              {marked.length}
            </span>
          </div>
          {marked.length === 0 ? (
            <p className="py-6 text-center text-[12.5px] text-(--color-ink-3)">
              No students marked yet. Recognized faces are marked automatically
              after a few seconds in view.
            </p>
          ) : (
            <ul className="flex max-h-[430px] flex-col gap-1 overflow-y-auto">
              {marked.map((m) => (
                <li key={m.student_id} className="flex items-center gap-2.5 rounded-lg px-1 py-1.5">
                  <Avatar id={m.student_id} name={m.name} size={28} tone="ok" />
                  <span className="flex-1 truncate text-[13px]">{m.name}</span>
                  <span className="text-[11.5px] text-(--color-ink-3)">{m.student_id}</span>
                  <Check size={14} className="text-(--color-ok)" />
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </div>
  );
}
