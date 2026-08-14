// ClassSync brand mark — a "C" rendered as a face-scan arc with an eye dot.
// Fixed brand indigo so it matches the favicon exactly across themes.
export default function Logo({ size = 32 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 100 100"
         role="img" aria-label="ClassSync">
      <rect width="100" height="100" rx="26" fill="#4F46E5" />
      <path d="M72 34 A26 26 0 1 0 72 66" fill="none" stroke="#fff"
            strokeWidth="9" strokeLinecap="round" />
      <circle cx="57" cy="50" r="6" fill="#C7D2FE" />
    </svg>
  );
}
