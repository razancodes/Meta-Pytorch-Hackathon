'use client';

import { useState, useRef, useCallback, useEffect } from 'react';
import styles from './case.module.css';
import EnvironmentStatePanel from './EnvironmentStatePanel';
import EntityGraph from './EntityGraph';
import AgentTerminal from './AgentTerminal';
import type { AGUIState, ReplayStep } from '@/lib/types';

interface Props {
  steps: ReplayStep[];
  currentStepIndex: number;
  onBack: () => void;
  alertSummary?: string;
  caseId?: string;
}

export default function CaseTerminal({ steps, currentStepIndex, onBack, alertSummary, caseId }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [leftWidth, setLeftWidth] = useState(25);  // percentage
  const [rightWidth, setRightWidth] = useState(25);
  const dragging = useRef<'left' | 'right' | null>(null);

  const currentStep = steps[currentStepIndex] || null;
  const aguiState: AGUIState | null = currentStep?.agui_state || null;
  const visibleSteps = steps.slice(0, currentStepIndex + 1);
  const rewards = visibleSteps.map(s => s.observation.reward ?? 0);

  // Resizer drag handling
  const handleMouseDown = useCallback((side: 'left' | 'right') => {
    dragging.current = side;
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
  }, []);

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!dragging.current || !containerRef.current) return;
      const rect = containerRef.current.getBoundingClientRect();
      const pct = ((e.clientX - rect.left) / rect.width) * 100;

      if (dragging.current === 'left') {
        setLeftWidth(Math.max(15, Math.min(40, pct)));
      } else {
        setRightWidth(Math.max(15, Math.min(40, 100 - pct)));
      }
    };

    const handleMouseUp = () => {
      dragging.current = null;
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    };

    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, []);

  const centerWidth = 100 - leftWidth - rightWidth;

  return (
    <div className={styles.caseContainer}>
      {/* Top Bar */}
      <div className={styles.caseTopBar}>
        <button className="nx-btn" onClick={onBack}>
          ← MAP
        </button>
        <div className={styles.caseInfo}>
          <span className={styles.caseId}>{caseId || 'CASE-1MDB'}</span>
          <span className={styles.caseSep}>│</span>
          <span className={styles.caseSummary}>{alertSummary || 'Sovereign Wealth Fund Investigation'}</span>
        </div>
        <div className={styles.caseProgress}>
          <span className="nx-label">STEP</span>
          <span className={styles.caseStep}>
            {String(currentStepIndex + 1).padStart(2, '0')}/{String(steps.length).padStart(2, '0')}
          </span>
        </div>
      </div>

      {/* Three-Panel Layout */}
      <div className={styles.panelLayout} ref={containerRef}>
        <div className={styles.panel} style={{ width: `${leftWidth}%` }}>
          <EnvironmentStatePanel
            aguiState={aguiState}
            rewards={rewards}
            currentStep={currentStepIndex + 1}
            totalSteps={steps.length > 0 ? Math.max(steps.length, 25) : 25}
          />
        </div>

        <div
          className="nx-resizer"
          onMouseDown={() => handleMouseDown('left')}
        />

        <div className={styles.panel} style={{ width: `${centerWidth}%` }}>
          <EntityGraph currentStep={currentStep} />
        </div>

        <div
          className="nx-resizer"
          onMouseDown={() => handleMouseDown('right')}
        />

        <div className={styles.panel} style={{ width: `${rightWidth}%` }}>
          <AgentTerminal steps={visibleSteps} />
        </div>
      </div>
    </div>
  );
}
