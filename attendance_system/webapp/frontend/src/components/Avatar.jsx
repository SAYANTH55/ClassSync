import { useState } from "react";

// Shows a student's face thumbnail, falling back to initials in a coloured
// circle if the photo is missing or fails to load.
export default function Avatar({ id, name, size = 28, tone = "accent" }) {
  const [failed, setFailed] = useState(false);
  const initials = (name || "?").slice(0, 2).toUpperCase();
  const box = { width: size, height: size };
  const tones = {
    accent: "bg-(--color-accent-soft) text-(--color-accent)",
    ok: "bg-(--color-ok-soft) text-(--color-ok)",
    bad: "bg-(--color-bad-soft) text-(--color-bad)",
  };

  if (id && !failed) {
    return (
      <img
        src={`/api/students/${id}/photo`}
        alt={name}
        onError={() => setFailed(true)}
        className="flex-shrink-0 rounded-full object-cover"
        style={box}
      />
    );
  }
  return (
    <span
      className={`flex flex-shrink-0 items-center justify-center rounded-full font-medium ${tones[tone] ?? tones.accent}`}
      style={{ ...box, fontSize: Math.round(size * 0.38) }}
    >
      {initials}
    </span>
  );
}
