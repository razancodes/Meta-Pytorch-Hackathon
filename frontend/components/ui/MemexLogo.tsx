'use client';

export default function MemexLogo({ size = 'md' }: { size?: 'sm' | 'md' | 'lg' }) {
  const scale = size === 'sm' ? 0.6 : size === 'lg' ? 1.4 : 1;
  const w = 120 * scale;
  const h = 24 * scale;

  return (
    <svg width={w} height={h} viewBox="0 0 120 24" fill="none" xmlns="http://www.w3.org/2000/svg" className="nx-logo">
      {/* M */}
      <path d="M2 20V4H6L10 13L14 4H18V20H15V9L11 18H9L5 9V20H2Z" fill="#EA580C" />
      {/* E */}
      <path d="M22 4H34V7H25V10.5H32V13.5H25V17H34V20H22V4Z" fill="#EA580C" />
      {/* M */}
      <path d="M38 20V4H42L46 13L50 4H54V20H51V9L47 18H45L41 9V20H38Z" fill="#EA580C" />
      {/* E */}
      <path d="M58 4H70V7H61V10.5H68V13.5H61V17H70V20H58V4Z" fill="#EA580C" />
      {/* X */}
      <path d="M73 4H76.5L81 10.5L85.5 4H89L82.5 12L89.5 20H86L81 13.5L76 20H72.5L79.5 12L73 4Z" fill="#EA580C" />
      {/* Accent bar */}
      <rect x="93" y="19" width="27" height="1" fill="#EA580C" opacity="0.4" />
      {/* Version dot */}
      <rect x="115" y="4" width="5" height="5" fill="#D4334A" opacity="0.6" />
    </svg>
  );
}
