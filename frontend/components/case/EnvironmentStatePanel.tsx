'use client';

import { useState } from 'react';
import type { AGUIState } from '@/lib/types';
import Sparkline from '@/components/ui/Sparkline';
import ValueFlicker from '@/components/ui/ValueFlicker';
import styles from './case.module.css';

interface Props {
  aguiState: AGUIState | null;
  rewards: number[];
  currentStep: number;
  totalSteps: number;
}

export default function EnvironmentStatePanel({ aguiState, rewards, currentStep, totalSteps }: Props) {
  const [collapsed, setCollapsed] = useState({
    ram: false,
    disk: false,
    async: false,
    kernel: false,
    reward: false,
    episode: false,
  });

  const toggle = (key: keyof typeof collapsed) => {
    setCollapsed(prev => ({ ...prev, [key]: !prev[key] }));
  };

  if (!aguiState) {
    return (
      <div className={styles.envPanel}>
        <div className={styles.envEmpty}>
          <span className={styles.envEmptyIcon}>⟐</span>
          <span>&gt; AWAITING ENVIRONMENT STATE...</span>
        </div>
      </div>
    );
  }

  const ramParts = aguiState.ram_usage.capacity.split('/');
  const ramUsed = parseInt(ramParts[0]) || 0;
  const ramTotal = parseInt(ramParts[1]) || 2;
  const ramPct = (ramUsed / ramTotal) * 100;
  const ramFull = ramUsed >= ramTotal;

  const accReward = rewards.reduce((a, b) => a + b, 0);
  const lastReward = rewards.length > 0 ? rewards[rewards.length - 1] : 0;

  const stepPct = (currentStep / totalSteps) * 100;

  return (
    <div className={styles.envPanel}>
      {/* ── RAM Monitor ──────────────────── */}
      <div className={`nx-panel ${ramFull ? styles.envRamFull : ''}`}>
        <div className="nx-panel-header" style={{ cursor: 'pointer' }} onClick={() => toggle('ram')}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span style={{ fontSize: '10px', color: '#737373', transition: 'transform 0.2s', transform: collapsed.ram ? 'rotate(-90deg)' : 'none' }}>▼</span>
            <span className="nx-panel-title">RAM MONITOR</span>
          </div>
          <span className={`nx-panel-badge ${ramFull ? 'nx-tag--red' : 'nx-tag--orange'}`}>
            {aguiState.ram_usage.capacity}
          </span>
        </div>
        {!collapsed.ram && (
          <div className="nx-panel-body">
            <div className={styles.ramBar}>
              <div className={styles.ramFill} style={{ width: `${ramPct}%`, background: ramFull ? '#E11D48' : '#EA580C' }} />
            </div>
            <div className={styles.ramSlots}>
              {aguiState.ram_usage.active_context.map((ctx, i) => (
                <div key={i} className={styles.ramSlot}>
                  <span className={styles.ramSlotLabel}>SLOT {i + 1}</span>
                  <span className={styles.ramSlotText}>{ctx.length > 80 ? ctx.slice(0, 80) + '…' : ctx}</span>
                </div>
              ))}
              {ramUsed < ramTotal && (
                <div className={`${styles.ramSlot} ${styles.ramSlotEmpty}`}>
                  <span className={styles.ramSlotLabel}>SLOT {ramUsed + 1}</span>
                  <span className={styles.ramSlotText}>[ EMPTY ]</span>
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* ── Disk Storage ─────────────────── */}
      <div className="nx-panel">
        <div className="nx-panel-header" style={{ cursor: 'pointer' }} onClick={() => toggle('disk')}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span style={{ fontSize: '10px', color: '#737373', transition: 'transform 0.2s', transform: collapsed.disk ? 'rotate(-90deg)' : 'none' }}>▼</span>
            <span className="nx-panel-title">DISK STORAGE</span>
          </div>
          <span className="nx-panel-badge nx-tag--green">{aguiState.disk_storage.length} ENTRIES</span>
        </div>
        {!collapsed.disk && (
          <div className={`nx-panel-body ${styles.diskBody}`}>
            {aguiState.disk_storage.length === 0 ? (
              <div className={styles.diskEmpty}>&gt; No data paged to disk</div>
            ) : (
              aguiState.disk_storage.map((entry, i) => (
                <div key={i} className={styles.diskEntry}>
                  <span className={styles.diskIndex}>{String(i + 1).padStart(2, '0')}</span>
                  <span className={styles.diskText}>{entry}</span>
                </div>
              ))
            )}
          </div>
        )}
      </div>

      {/* ── Active Processes ──────────────── */}
      <div className="nx-panel">
        <div className="nx-panel-header" style={{ cursor: 'pointer' }} onClick={() => toggle('async')}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span style={{ fontSize: '10px', color: '#737373', transition: 'transform 0.2s', transform: collapsed.async ? 'rotate(-90deg)' : 'none' }}>▼</span>
            <span className="nx-panel-title">ASYNC QUEUE</span>
          </div>
          <span className="nx-panel-badge nx-tag">{aguiState.async_jobs.length} JOBS</span>
        </div>
        {!collapsed.async && (
          <div className="nx-panel-body">
            {aguiState.async_jobs.length === 0 ? (
              <div className={styles.diskEmpty}>&gt; No active background tasks</div>
            ) : (
              aguiState.async_jobs.map((job) => (
                <div key={job.id} className={styles.asyncJob}>
                  <span className={styles.asyncId}>{job.id}</span>
                  <span className={styles.asyncTool}>{job.tool}</span>
                  <span className={`nx-tag ${
                    job.status === 'ready' ? 'nx-tag--green' :
                    job.status === 'pending' ? 'nx-tag--orange' : ''
                  }`}>
                    {job.status === 'pending' ? `ETA: ${job.eta_steps}` : job.status.toUpperCase()}
                  </span>
                </div>
              ))
            )}
          </div>
        )}
      </div>

      {/* ── Kernel Directives ─────────────── */}
      <div className="nx-panel">
        <div className="nx-panel-header" style={{ cursor: 'pointer' }} onClick={() => toggle('kernel')}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span style={{ fontSize: '10px', color: '#737373', transition: 'transform 0.2s', transform: collapsed.kernel ? 'rotate(-90deg)' : 'none' }}>▼</span>
            <span className="nx-panel-title">KERNEL DIRECTIVES</span>
          </div>
          <span className="nx-panel-badge nx-tag">{aguiState.kernel_directives.length}</span>
        </div>
        {!collapsed.kernel && (
          <div className={`nx-panel-body ${styles.kernelBody}`}>
            {aguiState.kernel_directives.map((dir, i) => (
              <div key={i} className={`${styles.kernelEntry} ${i > 0 ? 'nx-highlight-new' : ''}`}>
                <span className={`nx-tag ${i === 0 ? '' : 'nx-tag--orange'}`}>
                  {i === 0 ? 'BASE' : 'INJECTED'}
                </span>
                <span className={styles.kernelText}>{dir}</span>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* ── Reward Signal ─────────────────── */}
      <div className="nx-panel">
        <div className="nx-panel-header" style={{ cursor: 'pointer' }} onClick={() => toggle('reward')}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span style={{ fontSize: '10px', color: '#737373', transition: 'transform 0.2s', transform: collapsed.reward ? 'rotate(-90deg)' : 'none' }}>▼</span>
            <span className="nx-panel-title">REWARD SIGNAL</span>
          </div>
        </div>
        {!collapsed.reward && (
          <div className="nx-panel-body">
            <div className={styles.rewardRow}>
              <div className={styles.rewardMetric}>
                <span className="nx-label">ACCUMULATED</span>
                <ValueFlicker
                  value={accReward}
                  format="score"
                  className={`${styles.rewardValue} ${accReward >= 0 ? 'nx-status-ok' : 'nx-status-danger'}`}
                />
              </div>
              <div className={styles.rewardMetric}>
                <span className="nx-label">LAST STEP</span>
                <ValueFlicker
                  value={lastReward}
                  format="score"
                  className={`${styles.rewardValueSm} ${lastReward >= 0 ? 'nx-status-ok' : 'nx-status-danger'}`}
                />
              </div>
            </div>
            {rewards.length > 1 && (
              <Sparkline data={rewards} width={200} height={28} showZeroLine />
            )}
          </div>
        )}
      </div>

      {/* ── Episode Progress ──────────────── */}
      <div className="nx-panel">
        <div className="nx-panel-header" style={{ cursor: 'pointer' }} onClick={() => toggle('episode')}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span style={{ fontSize: '10px', color: '#737373', transition: 'transform 0.2s', transform: collapsed.episode ? 'rotate(-90deg)' : 'none' }}>▼</span>
            <span className="nx-panel-title">EPISODE</span>
          </div>
          <span className="nx-panel-badge nx-tag">{currentStep}/{totalSteps}</span>
        </div>
        {!collapsed.episode && (
          <div className="nx-panel-body">
            <div className={styles.progressBar}>
              <div className={styles.progressFill} style={{ width: `${stepPct}%` }} />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
