'use client';

import { useState, useRef, useCallback, useEffect, useMemo } from 'react';
import styles from './case.module.css';
import EnvironmentStatePanel from './EnvironmentStatePanel';
import EntityGraph from './EntityGraph';
import AgentTerminal from './AgentTerminal';
import type { AGUIState, ReplayStep } from '@/lib/types';
import { scenarioToGraph } from '@/lib/dataTransform';

interface Props {
  steps: ReplayStep[];
  currentStepIndex: number;
  onBack: () => void;
  alertSummary?: string;
  caseId?: string;
}

type FullscreenPanel = 'graph' | 'env' | 'terminal' | null;

export default function CaseTerminal({ steps, currentStepIndex, onBack, alertSummary, caseId }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [leftWidth, setLeftWidth] = useState(60);  // Entity graph takes 60%
  const dragging = useRef<boolean>(false);
  const [fullscreen, setFullscreen] = useState<FullscreenPanel>(null);

  const currentStep = steps[currentStepIndex] || null;
  const aguiState: AGUIState | null = currentStep?.agui_state || null;
  const visibleSteps = steps.slice(0, currentStepIndex + 1);
  const rewards = visibleSteps.map(s => s.observation.reward ?? 0);

  // Compute graph data dynamically from visible steps
  const graphData = useMemo(() => {
    const scenarioData: any = {
      customer_profiles: {},
      network_graph: {},
      transactions: [],
      watchlist_results: {},
    };

    visibleSteps.forEach(step => {
      const tr = step.observation?.tool_result;
      if (!tr) return;

      if (step.action.tool === 'get_customer_profile' && tr.customer_profile) {
        const p = tr.customer_profile as any;
        const id = (step.action.parameters.customer_id as string) || p.customer_id;
        if (id) scenarioData.customer_profiles[id] = p;
      }
      else if (step.action.tool === 'trace_network' && tr.network) {
         const id = step.action.parameters.entity_id as string;
         if (id) {
           scenarioData.network_graph[id] = scenarioData.network_graph[id] || { connections: [] };
           scenarioData.network_graph[id].connections.push(...(tr.network as any).connections);
         }
      }
      else if (step.action.tool === 'query_transactions' && tr.transactions) {
         scenarioData.transactions.push(...(tr.transactions as any[]));
      }
      else if (step.action.tool === 'check_watchlist' && tr.watchlist_result) {
         const id = step.action.parameters.entity as string;
         if (id) scenarioData.watchlist_results[id] = tr.watchlist_result;
      }
    });

    // Seed the graph with the alert's main customer
    const alertMeta = steps[0]?.observation?.tool_result?.alert as any;
    if (alertMeta && alertMeta.customer_id) {
      if (!scenarioData.customer_profiles[alertMeta.customer_id]) {
        scenarioData.customer_profiles[alertMeta.customer_id] = {
           customer_id: alertMeta.customer_id,
           name: alertMeta.customer_id,
           type: 'individual',
           risk_rating: 'High'
        };
      }
    }

    return scenarioToGraph(scenarioData);
  }, [visibleSteps, steps]);

  // Single resizer drag handling
  const handleMouseDown = useCallback(() => {
    dragging.current = true;
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
  }, []);

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!dragging.current || !containerRef.current) return;
      const rect = containerRef.current.getBoundingClientRect();
      const pct = ((e.clientX - rect.left) / rect.width) * 100;
      setLeftWidth(Math.max(30, Math.min(75, pct)));
    };

    const handleMouseUp = () => {
      dragging.current = false;
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

  const toggleFullscreen = (panel: FullscreenPanel) => {
    setFullscreen(prev => (prev === panel ? null : panel));
  };

  // Escape key to exit fullscreen
  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && fullscreen) setFullscreen(null);
    };
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [fullscreen]);

  const rightWidth = 100 - leftWidth;

  // Fullscreen panel rendering
  const renderFullscreenBtn = (panel: FullscreenPanel) => (
    <button
      className="nx-fullscreen-btn"
      onClick={() => toggleFullscreen(panel)}
      title={fullscreen === panel ? 'Exit Fullscreen (Esc)' : 'Fullscreen'}
    >
      {fullscreen === panel ? '⛶' : '⛶'}
    </button>
  );

  // Graph panel content
  const graphPanel = (
    <div className={`${styles.panel} ${fullscreen === 'graph' ? styles.fullscreenOverlay : ''}`} style={fullscreen === 'graph' ? {} : { width: `${leftWidth}%` }}>
      <div className={styles.panelHeaderBar}>
        <span className="nx-panel-title">ENTITY GRAPH</span>
        <div className={styles.panelHeaderActions}>
          <span className="nx-panel-badge nx-tag">{graphData.nodes.length} NODES</span>
          {renderFullscreenBtn('graph')}
        </div>
      </div>
      <EntityGraph graphData={graphData} />
    </div>
  );

  // Environment panel content
  const envPanel = (
    <div className={`${styles.rightStackPanel} ${fullscreen === 'env' ? styles.fullscreenOverlay : ''}`}>
      <div className={styles.panelHeaderBar}>
        <span className="nx-panel-title">SYSTEM RESOURCES</span>
        <div className={styles.panelHeaderActions}>
          {renderFullscreenBtn('env')}
        </div>
      </div>
      <div className={styles.rightStackContent}>
        <EnvironmentStatePanel
          aguiState={aguiState}
          rewards={rewards}
          currentStep={currentStepIndex + 1}
          totalSteps={steps.length > 0 ? Math.max(steps.length, 25) : 25}
        />
      </div>
    </div>
  );

  // Terminal panel content
  const terminalPanel = (
    <div className={`${styles.rightStackPanel} ${fullscreen === 'terminal' ? styles.fullscreenOverlay : ''}`}>
      <div className={styles.panelHeaderBar}>
        <span className="nx-panel-title">AGENT TERMINAL</span>
        <div className={styles.panelHeaderActions}>
          <span className="nx-panel-badge nx-tag--orange">{visibleSteps.length} ACTIONS</span>
          {renderFullscreenBtn('terminal')}
        </div>
      </div>
      <div className={styles.rightStackContent}>
        <AgentTerminal steps={visibleSteps} />
      </div>
    </div>
  );

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

      {/* Two-Column Layout: [EntityGraph (left) | System+Terminal stacked (right)] */}
      <div className={styles.panelLayout} ref={containerRef}>
        {/* Left: Entity Graph (wide) */}
        {graphPanel}

        {/* Resizer */}
        <div
          className="nx-resizer"
          onMouseDown={handleMouseDown}
        />

        {/* Right: Stacked panels */}
        <div className={styles.panel} style={{ width: `${rightWidth}%` }}>
          <div className={styles.rightStack}>
            {envPanel}
            {terminalPanel}
          </div>
        </div>
      </div>
    </div>
  );
}
