'use client';

import { useEffect, useState, useRef } from 'react';

export default function ValueFlicker({
  value,
  format = 'number',
  prefix = '',
  suffix = '',
  className = '',
}: {
  value: number | string;
  format?: 'number' | 'score' | 'step' | 'raw';
  prefix?: string;
  suffix?: string;
  className?: string;
}) {
  const [displayValue, setDisplayValue] = useState(formatVal(value, format));
  const [flickering, setFlickering] = useState(false);
  const prevValue = useRef(value);

  useEffect(() => {
    if (prevValue.current === value) return;
    prevValue.current = value;

    setFlickering(true);
    const chars = '0123456789ABCDEF+-.$'.split('');
    const target = formatVal(value, format);
    let frame = 0;
    const maxFrames = 6;

    const interval = setInterval(() => {
      frame++;
      if (frame >= maxFrames) {
        clearInterval(interval);
        setDisplayValue(target);
        setFlickering(false);
        return;
      }
      // Random characters for flicker effect
      const flickered = target.split('').map((ch, i) => {
        if (frame > maxFrames - 2 || Math.random() > 0.5) return ch;
        return /\d/.test(ch) ? chars[Math.floor(Math.random() * 10)] : ch;
      }).join('');
      setDisplayValue(flickered);
    }, 50);

    return () => clearInterval(interval);
  }, [value, format]);

  return (
    <span className={`nx-tabular ${flickering ? 'nx-flicker' : ''} ${className}`}>
      {prefix}{displayValue}{suffix}
    </span>
  );
}

function formatVal(v: number | string, format: string): string {
  if (typeof v === 'string') return v;
  switch (format) {
    case 'score': return v >= 0 ? `+${v.toFixed(4)}` : v.toFixed(4);
    case 'step': return String(Math.floor(v)).padStart(2, '0');
    case 'number': return v.toFixed(2);
    default: return String(v);
  }
}
