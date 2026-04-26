'use client';

import styles from './ui.module.css';

type Status = 'connected' | 'replay' | 'disconnected';

export default function StatusIndicator({ status, label }: { status: Status; label?: string }) {
  const colorClass = status === 'connected' ? styles.statusGreen : status === 'replay' ? styles.statusAmber : styles.statusRed;
  const pulseClass = status !== 'disconnected' ? styles.statusPulse : '';

  return (
    <div className={styles.statusWrap}>
      <span className={`${styles.statusDot} ${colorClass} ${pulseClass}`} />
      {label && <span className={styles.statusLabel}>{label}</span>}
      <span style={{
        fontSize: '8px',
        fontWeight: 700,
        letterSpacing: '0.1em',
        color: '#737373',
        border: '1px solid #333',
        borderRadius: '3px',
        padding: '1px 4px',
        marginLeft: '6px',
      }}>AGUI</span>
    </div>
  );
}
