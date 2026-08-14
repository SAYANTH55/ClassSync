import { useEffect, useRef, useState } from "react";
import { Loader2, Plus, Search, Trash2, Upload, X } from "lucide-react";
import Avatar from "../components/Avatar.jsx";

export default function Students() {
  const [data, setData] = useState(null);
  const [q, setQ] = useState("");
  const [sheetOpen, setSheetOpen] = useState(false);
  const [toast, setToast] = useState(null);
  const [confirmRemove, setConfirmRemove] = useState(null);
  const [removing, setRemoving] = useState(false);

  const load = () => fetch("/api/students").then((r) => r.json()).then(setData);
  useEffect(() => { load(); }, []);

  function showToast(ok, message) {
    setToast({ ok, message });
    setTimeout(() => setToast(null), 4000);
  }

  async function doRemove(sid) {
    setRemoving(true);
    const res = await fetch(`/api/students/${sid}`, { method: "DELETE" });
    const out = await res.json();
    setRemoving(false);
    setConfirmRemove(null);
    showToast(out.ok, out.message);
    load();
  }

  const rows = data?.students.filter((s) =>
    s.name.toLowerCase().includes(q.toLowerCase())) ?? [];

  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-center gap-3">
        <div className="relative">
          <Search size={14} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-(--color-ink-3)" />
          <input
            placeholder="Search students"
            value={q}
            onChange={(e) => setQ(e.target.value)}
            className="w-56 rounded-lg border border-(--color-line) bg-(--color-card) py-1.5 pl-8 pr-3 text-[13px] placeholder:text-(--color-ink-3)"
          />
        </div>
        <span className="text-[13px] text-(--color-ink-2)">
          {data ? `${data.enrolled} enrolled` : "…"}
        </span>
        <button
          onClick={() => setSheetOpen(true)}
          className="ml-auto flex items-center gap-1.5 rounded-xl bg-(--color-accent) px-3.5 py-2 text-[13px] font-medium text-white transition-transform hover:scale-[1.02] active:scale-[0.98]"
        >
          <Plus size={15} /> Add student
        </button>
      </div>

      <div className="rise overflow-hidden rounded-2xl border border-(--color-line) bg-(--color-card)">
        <table className="w-full text-[13px]">
          <thead>
            <tr className="text-left text-[11.5px] uppercase tracking-wide text-(--color-ink-3)">
              <th className="px-4 py-2.5 font-medium">ID</th>
              <th className="px-4 py-2.5 font-medium">Name</th>
              <th className="px-4 py-2.5 font-medium">Status</th>
              <th className="px-4 py-2.5 font-medium">Templates</th>
              <th className="px-4 py-2.5" />
            </tr>
          </thead>
          <tbody className="divide-y divide-(--color-line)">
            {rows.map((s) => (
              <tr key={s.student_id} className="hover:bg-(--color-page)">
                <td className="px-4 py-2 text-(--color-ink-2)">{s.student_id}</td>
                <td className="px-4 py-2">
                  <div className="flex items-center gap-2.5">
                    <Avatar id={s.student_id} name={s.name} size={28} />
                    {s.name}
                  </div>
                </td>
                <td className="px-4 py-2">
                  {s.enrolled ? (
                    <span className="rounded-full bg-(--color-ok-soft) px-2 py-0.5 text-[11.5px] text-(--color-ok)">enrolled</span>
                  ) : (
                    <span className="rounded-full bg-(--color-line) px-2 py-0.5 text-[11.5px] text-(--color-ink-2)">inactive</span>
                  )}
                </td>
                <td className="px-4 py-2 tabular-nums text-(--color-ink-2)">{s.templates}</td>
                <td className="px-4 py-2 text-right">
                  {s.enrolled && (
                    <button
                      onClick={() => setConfirmRemove(s)}
                      aria-label={`Remove ${s.name}`}
                      className="rounded-lg p-1.5 text-(--color-ink-3) transition-colors hover:bg-(--color-bad-soft) hover:text-(--color-bad)"
                    >
                      <Trash2 size={14} />
                    </button>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {sheetOpen && (
        <AddSheet
          onClose={() => setSheetOpen(false)}
          onDone={(ok, message) => { setSheetOpen(false); showToast(ok, message); load(); }}
        />
      )}

      {confirmRemove && (
        <div className="fixed inset-0 z-40 flex items-center justify-center bg-black/40 p-4">
          <div className="rise w-full max-w-sm rounded-2xl border border-(--color-line) bg-(--color-card) p-5">
            <h3 className="text-[15px] font-semibold">Remove {confirmRemove.name}?</h3>
            <p className="mt-1.5 text-[13px] text-(--color-ink-2)">
              They will no longer be recognized. Their images are kept on disk
              and their ID stays reserved, so past attendance records remain valid.
            </p>
            <div className="mt-4 flex justify-end gap-2">
              <button onClick={() => setConfirmRemove(null)}
                className="rounded-lg border border-(--color-line) px-3.5 py-1.5 text-[13px]">
                Cancel
              </button>
              <button onClick={() => doRemove(confirmRemove.student_id)} disabled={removing}
                className="flex items-center gap-1.5 rounded-lg bg-(--color-bad) px-3.5 py-1.5 text-[13px] font-medium text-white">
                {removing && <Loader2 size={13} className="animate-spin" />} Remove
              </button>
            </div>
          </div>
        </div>
      )}

      {toast && (
        <div className={`rise fixed bottom-5 right-5 z-50 max-w-md rounded-xl border px-4 py-3 text-[13px] ${
          toast.ok
            ? "border-(--color-ok)/30 bg-(--color-ok-soft) text-(--color-ok)"
            : "border-(--color-bad)/30 bg-(--color-bad-soft) text-(--color-bad)"
        }`}>
          {toast.message}
        </div>
      )}
    </div>
  );
}

function AddSheet({ onClose, onDone }) {
  const [name, setName] = useState("");
  const [files, setFiles] = useState([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const inputRef = useRef(null);

  async function submit() {
    if (!name.trim() || files.length === 0) {
      setError("Enter a name and add at least one photo");
      return;
    }
    setBusy(true);
    setError(null);
    const fd = new FormData();
    fd.append("name", name.trim());
    files.forEach((f) => fd.append("images", f));
    try {
      const res = await fetch("/api/students", { method: "POST", body: fd });
      const out = await res.json();
      if (!res.ok) throw new Error(out.detail ?? "Upload failed");
      if (!out.ok) { setError(out.message); setBusy(false); return; }
      onDone(true, out.message);
    } catch (e) {
      setError(String(e.message ?? e));
      setBusy(false);
    }
  }

  return (
    <div className="fixed inset-0 z-40 flex justify-end bg-black/40">
      <div className="rise h-full w-full max-w-md overflow-y-auto border-l border-(--color-line) bg-(--color-card) p-6">
        <div className="mb-5 flex items-center justify-between">
          <h3 className="text-[16px] font-semibold">Add student</h3>
          <button onClick={onClose} aria-label="Close"
            className="rounded-lg p-1.5 text-(--color-ink-3) hover:text-(--color-ink)">
            <X size={16} />
          </button>
        </div>

        <label className="mb-1.5 block text-[12.5px] font-medium text-(--color-ink-2)">
          Full name
        </label>
        <input
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="Aisha Verma"
          className="mb-4 w-full rounded-lg border border-(--color-line) bg-(--color-page) px-3 py-2 text-[13.5px] placeholder:text-(--color-ink-3)"
        />

        <label className="mb-1.5 block text-[12.5px] font-medium text-(--color-ink-2)">
          Photos (3–6 recommended — different angles, clear face)
        </label>
        <button
          onClick={() => inputRef.current.click()}
          className="mb-2 flex w-full flex-col items-center gap-1.5 rounded-xl border-2 border-dashed border-(--color-line) py-7 text-(--color-ink-3) transition-colors hover:border-(--color-accent) hover:text-(--color-accent)"
        >
          <Upload size={20} />
          <span className="text-[13px]">Click to choose images</span>
          <span className="text-[11.5px]">JPG, PNG or HEIC</span>
        </button>
        <input
          ref={inputRef}
          type="file"
          accept=".jpg,.jpeg,.png,.heic"
          multiple
          hidden
          onChange={(e) => setFiles([...e.target.files])}
        />

        {files.length > 0 && (
          <ul className="mb-3 flex flex-col gap-1">
            {files.map((f, i) => (
              <li key={i} className="flex items-center gap-2 rounded-lg bg-(--color-page) px-3 py-1.5 text-[12.5px]">
                <span className="flex-1 truncate">{f.name}</span>
                <span className="text-(--color-ink-3)">{(f.size / 1024 / 1024).toFixed(1)} MB</span>
                <button onClick={() => setFiles(files.filter((_, j) => j !== i))}
                  aria-label={`Remove ${f.name}`}
                  className="text-(--color-ink-3) hover:text-(--color-bad)">
                  <X size={13} />
                </button>
              </li>
            ))}
          </ul>
        )}

        {error && (
          <p className="mb-3 rounded-lg bg-(--color-bad-soft) px-3 py-2 text-[12.5px] text-(--color-bad)">
            {error}
          </p>
        )}

        <button
          onClick={submit}
          disabled={busy}
          className="flex w-full items-center justify-center gap-2 rounded-xl bg-(--color-accent) py-2.5 text-[13.5px] font-medium text-white disabled:opacity-60"
        >
          {busy && <Loader2 size={15} className="animate-spin" />}
          {busy ? "Validating faces and rebuilding gallery…" : "Enroll student"}
        </button>
        {busy && (
          <p className="mt-2 text-center text-[12px] text-(--color-ink-3)">
            Each photo is checked for exactly one face and identity consistency.
          </p>
        )}
      </div>
    </div>
  );
}
