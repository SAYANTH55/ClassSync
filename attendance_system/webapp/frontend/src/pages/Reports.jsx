import { useEffect, useMemo, useState } from "react";
import { Download, Loader2, Search, Trash2 } from "lucide-react";
import Avatar from "../components/Avatar.jsx";

function today() {
  return new Date().toISOString().slice(0, 10);
}

export default function Reports() {
  const [tab, setTab] = useState("daily");
  return (
    <div className="flex flex-col gap-4">
      <div className="flex w-fit rounded-xl border border-(--color-line) bg-(--color-card) p-1">
        {["daily", "summary"].map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`rounded-lg px-4 py-1.5 text-[13px] capitalize transition-colors ${
              tab === t
                ? "bg-(--color-accent-soft) font-medium text-(--color-accent)"
                : "text-(--color-ink-2) hover:text-(--color-ink)"
            }`}
          >
            {t === "daily" ? "Daily report" : "Summary"}
          </button>
        ))}
      </div>
      {tab === "daily" ? <Daily /> : <Summary />}
    </div>
  );
}

function Daily() {
  const [date, setDate] = useState(today());
  const [data, setData] = useState(null);
  const [q, setQ] = useState("");
  const [confirmClear, setConfirmClear] = useState(false);
  const [clearing, setClearing] = useState(false);
  const [toast, setToast] = useState(null);

  const load = () =>
    fetch(`/api/attendance/${date}`).then((r) => r.json()).then(setData);
  useEffect(() => { load(); }, [date]);

  async function clearDay() {
    setClearing(true);
    const res = await fetch(`/api/attendance/${date}`, { method: "DELETE" });
    const out = await res.json();
    setClearing(false);
    setConfirmClear(false);
    setToast(out.message);
    setTimeout(() => setToast(null), 4000);
    load();
  }

  const present = useMemo(
    () => data?.present.filter((r) => r.name.toLowerCase().includes(q.toLowerCase())) ?? [],
    [data, q]
  );
  const absent = useMemo(
    () => data?.absent.filter((r) => r.name.toLowerCase().includes(q.toLowerCase())) ?? [],
    [data, q]
  );

  return (
    <>
      <div className="flex flex-wrap items-center gap-3">
        <input
          type="date"
          value={date}
          max={today()}
          onChange={(e) => setDate(e.target.value)}
          className="rounded-lg border border-(--color-line) bg-(--color-card) px-3 py-1.5 text-[13px]"
        />
        <div className="relative">
          <Search size={14} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-(--color-ink-3)" />
          <input
            placeholder="Search students"
            value={q}
            onChange={(e) => setQ(e.target.value)}
            className="w-52 rounded-lg border border-(--color-line) bg-(--color-card) py-1.5 pl-8 pr-3 text-[13px] placeholder:text-(--color-ink-3)"
          />
        </div>
        <span className="text-[13px] text-(--color-ink-2)">
          {data ? `${data.present.length} present · ${data.absent.length} absent` : "…"}
        </span>
        <a
          href={`/api/attendance/${date}/export`}
          className="ml-auto flex items-center gap-1.5 rounded-lg border border-(--color-line) bg-(--color-card) px-3 py-1.5 text-[13px] text-(--color-ink-2) transition-colors hover:text-(--color-ink)"
        >
          <Download size={14} /> Export CSV
        </a>
        <button
          onClick={() => setConfirmClear(true)}
          disabled={!data || data.present.length === 0}
          className="flex items-center gap-1.5 rounded-lg border border-(--color-line) bg-(--color-card) px-3 py-1.5 text-[13px] text-(--color-ink-2) transition-colors enabled:hover:border-(--color-bad)/40 enabled:hover:text-(--color-bad) disabled:opacity-40"
        >
          <Trash2 size={14} /> Clear day
        </button>
      </div>

      <div className="grid gap-3 lg:grid-cols-3">
        <div className="rise overflow-hidden rounded-2xl border border-(--color-line) bg-(--color-card) lg:col-span-2">
          <div className="border-b border-(--color-line) px-4 py-3 text-[13.5px] font-medium">
            Present
          </div>
          {present.length === 0 ? (
            <p className="px-4 py-8 text-center text-[13px] text-(--color-ink-3)">
              No attendance records for this day.
            </p>
          ) : (
            <table className="w-full text-[13px]">
              <thead>
                <tr className="text-left text-[11.5px] uppercase tracking-wide text-(--color-ink-3)">
                  <th className="px-4 py-2 font-medium">Time</th>
                  <th className="px-4 py-2 font-medium">ID</th>
                  <th className="px-4 py-2 font-medium">Name</th>
                  <th className="px-4 py-2 text-right font-medium">Score</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-(--color-line)">
                {present.map((r) => (
                  <tr key={r.student_id} className="hover:bg-(--color-page)">
                    <td className="px-4 py-2 tabular-nums text-(--color-ink-2)">{r.time}</td>
                    <td className="px-4 py-2 text-(--color-ink-2)">{r.student_id}</td>
                    <td className="px-4 py-2">{r.name}</td>
                    <td className="px-4 py-2 text-right">
                      <span className="rounded-full bg-(--color-ok-soft) px-2 py-0.5 text-[11.5px] text-(--color-ok)">
                        {r.score.toFixed(2)}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>

        <div className="rise overflow-hidden rounded-2xl border border-(--color-line) bg-(--color-card)">
          <div className="border-b border-(--color-line) px-4 py-3 text-[13.5px] font-medium">
            Absent
          </div>
          {absent.length === 0 ? (
            <p className="px-4 py-8 text-center text-[13px] text-(--color-ink-3)">
              {data?.present.length ? "Everyone is present 🎉" : "No data for this day."}
            </p>
          ) : (
            <ul className="max-h-96 divide-y divide-(--color-line) overflow-y-auto">
              {absent.map((r) => (
                <li key={r.student_id} className="flex items-center gap-2.5 px-4 py-2">
                  <Avatar id={r.student_id} name={r.name} size={26} tone="bad" />
                  <span className="text-[13px]">{r.name}</span>
                  <span className="ml-auto text-[12px] text-(--color-ink-3)">{r.student_id}</span>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>

      {confirmClear && (
        <div className="fixed inset-0 z-40 flex items-center justify-center bg-black/40 p-4">
          <div className="rise w-full max-w-sm rounded-2xl border border-(--color-line) bg-(--color-card) p-5">
            <h3 className="text-[15px] font-semibold">Clear attendance for {date}?</h3>
            <p className="mt-1.5 text-[13px] text-(--color-ink-2)">
              This removes all {data?.present.length} mark(s) for this day so
              you can record attendance again. The records are archived and can
              be recovered if needed.
            </p>
            <div className="mt-4 flex justify-end gap-2">
              <button onClick={() => setConfirmClear(false)}
                className="rounded-lg border border-(--color-line) px-3.5 py-1.5 text-[13px]">
                Cancel
              </button>
              <button onClick={clearDay} disabled={clearing}
                className="flex items-center gap-1.5 rounded-lg bg-(--color-bad) px-3.5 py-1.5 text-[13px] font-medium text-white">
                {clearing && <Loader2 size={13} className="animate-spin" />} Clear day
              </button>
            </div>
          </div>
        </div>
      )}

      {toast && (
        <div className="rise fixed bottom-5 right-5 z-50 max-w-md rounded-xl border border-(--color-ok)/30 bg-(--color-ok-soft) px-4 py-3 text-[13px] text-(--color-ok)">
          {toast}
        </div>
      )}
    </>
  );
}

function Summary() {
  const [data, setData] = useState(null);
  const [sortBy, setSortBy] = useState("name");

  useEffect(() => {
    fetch("/api/attendance/summary").then((r) => r.json()).then(setData);
  }, []);

  const rows = useMemo(() => {
    if (!data) return [];
    const r = [...data.rows];
    if (sortBy === "percent") r.sort((a, b) => a.percent - b.percent);
    return r;
  }, [data, sortBy]);

  if (!data) return <p className="text-sm text-(--color-ink-3)">Loading…</p>;
  if (data.days_total === 0)
    return (
      <p className="rounded-2xl border border-(--color-line) bg-(--color-card) px-4 py-10 text-center text-[13px] text-(--color-ink-3)">
        No attendance days recorded yet. Run a camera session first.
      </p>
    );

  const low = data.rows.filter((r) => r.percent < 75).length;

  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center gap-3 text-[13px] text-(--color-ink-2)">
        <span>
          {data.days_total} day{data.days_total > 1 ? "s" : ""} · {data.first_day} → {data.last_day}
        </span>
        {low > 0 && (
          <span className="rounded-full bg-(--color-bad-soft) px-2.5 py-0.5 text-[12px] text-(--color-bad)">
            {low} below 75%
          </span>
        )}
        <select
          value={sortBy}
          onChange={(e) => setSortBy(e.target.value)}
          className="ml-auto rounded-lg border border-(--color-line) bg-(--color-card) px-2 py-1.5 text-[13px]"
        >
          <option value="name">Sort by name</option>
          <option value="percent">Sort by attendance</option>
        </select>
      </div>

      <div className="rise overflow-hidden rounded-2xl border border-(--color-line) bg-(--color-card)">
        <table className="w-full text-[13px]">
          <thead>
            <tr className="text-left text-[11.5px] uppercase tracking-wide text-(--color-ink-3)">
              <th className="px-4 py-2.5 font-medium">ID</th>
              <th className="px-4 py-2.5 font-medium">Name</th>
              <th className="px-4 py-2.5 font-medium">Days present</th>
              <th className="w-1/3 px-4 py-2.5 font-medium">Attendance</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-(--color-line)">
            {rows.map((r) => (
              <tr key={r.student_id} className="hover:bg-(--color-page)">
                <td className="px-4 py-2 text-(--color-ink-2)">{r.student_id}</td>
                <td className="px-4 py-2">{r.name}</td>
                <td className="px-4 py-2 tabular-nums text-(--color-ink-2)">
                  {r.days_present}/{data.days_total}
                </td>
                <td className="px-4 py-2">
                  <div className="flex items-center gap-2">
                    <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-(--color-line)">
                      <div
                        className={`h-full rounded-full ${r.percent < 75 ? "bg-(--color-bad)" : "bg-(--color-ok)"}`}
                        style={{ width: `${r.percent}%` }}
                      />
                    </div>
                    <span className={`w-12 text-right tabular-nums text-[12px] ${r.percent < 75 ? "text-(--color-bad)" : "text-(--color-ink-2)"}`}>
                      {r.percent}%
                    </span>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
