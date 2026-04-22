'use client';

export default function Sparkline({
  data,
  width = 120,
  height = 24,
  color = '#EA580C',
  negativeColor = '#E11D48',
  showZeroLine = false,
}: {
  data: number[];
  width?: number;
  height?: number;
  color?: string;
  negativeColor?: string;
  showZeroLine?: boolean;
}) {
  if (!data || data.length < 2) {
    return <svg width={width} height={height} />;
  }

  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  const padY = 2;

  const points = data.map((v, i) => {
    const x = (i / (data.length - 1)) * width;
    const y = padY + ((max - v) / range) * (height - padY * 2);
    return `${x},${y}`;
  });

  const pathD = `M${points.join(' L')}`;

  // Gradient from positive to negative color
  const lastVal = data[data.length - 1];
  const strokeColor = lastVal >= 0 ? color : negativeColor;

  const zeroY = padY + ((max - 0) / range) * (height - padY * 2);

  return (
    <svg width={width} height={height} style={{ display: 'block' }}>
      {showZeroLine && min < 0 && max > 0 && (
        <line x1={0} y1={zeroY} x2={width} y2={zeroY} stroke="#404040" strokeWidth={0.5} strokeDasharray="2,2" />
      )}
      <path d={pathD} fill="none" stroke={strokeColor} strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round" />
      {/* End dot */}
      <circle
        cx={(data.length - 1) / (data.length - 1) * width}
        cy={padY + ((max - lastVal) / range) * (height - padY * 2)}
        r={2}
        fill={strokeColor}
      />
    </svg>
  );
}
