'use client';

export default function NexusLogo({ size = 'md' }: { size?: 'sm' | 'md' | 'lg' }) {
  const scale = size === 'sm' ? 0.6 : size === 'lg' ? 1.4 : 1;
  const w = 120 * scale;
  const h = 24 * scale;

  return (
    <svg width={w} height={h} viewBox="0 0 120 24" fill="none" xmlns="http://www.w3.org/2000/svg" className="nx-logo">
      {/* N */}
      <path d="M2 20V4H5L12 14V4H15V20H12L5 10V20H2Z" fill="#EA580C" />
      {/* E */}
      <path d="M19 4H31V7H22V10.5H29V13.5H22V17H31V20H19V4Z" fill="#EA580C" />
      {/* X */}
      <path d="M34 4H37.5L42 10.5L46.5 4H50L43.5 12L50.5 20H47L42 13.5L37 20H33.5L40.5 12L34 4Z" fill="#EA580C" />
      {/* U */}
      <path d="M53 4H56V15C56 16.5 57 17.5 59.5 17.5C62 17.5 63 16.5 63 15V4H66V15C66 18.5 63.5 20.5 59.5 20.5C55.5 20.5 53 18.5 53 15V4Z" fill="#EA580C" />
      {/* S */}
      <path d="M69 15.5L72 14.5C72.5 16.5 74 17.5 76 17.5C78 17.5 79 16.5 79 15.5C79 14.5 78 13.5 75.5 13C72.5 12.5 70 11 70 8.5C70 5.5 72.5 3.5 76 3.5C79 3.5 81 5 82 7.5L79 8.5C78.5 7 77.5 6 76 6C74.5 6 73.5 7 73.5 8C73.5 9 74.5 9.5 77 10C80 10.5 82.5 12 82.5 15C82.5 18 80 20.5 76 20.5C73 20.5 70 18.5 69 15.5Z" fill="#EA580C" />
      {/* Accent bar */}
      <rect x="86" y="19" width="34" height="1" fill="#EA580C" opacity="0.4" />
      {/* Version dot */}
      <rect x="115" y="4" width="5" height="5" fill="#E11D48" opacity="0.6" />
    </svg>
  );
}
