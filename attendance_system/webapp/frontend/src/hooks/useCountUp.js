import { useEffect, useRef, useState } from "react";

// Animate a number from 0 to `target` on mount / when target changes.
// easeOutCubic for a lively-then-settling feel. Caller formats the value.
export function useCountUp(target, duration = 900) {
  const [val, setVal] = useState(0);
  const raf = useRef();

  useEffect(() => {
    let start = null;
    const tick = (t) => {
      if (start == null) start = t;
      const p = Math.min(1, (t - start) / duration);
      const eased = 1 - Math.pow(1 - p, 3);
      setVal(target * eased);
      if (p < 1) raf.current = requestAnimationFrame(tick);
      else setVal(target);
    };
    raf.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf.current);
  }, [target, duration]);

  return val;
}
