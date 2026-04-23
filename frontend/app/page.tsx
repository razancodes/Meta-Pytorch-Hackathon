'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import dynamic from 'next/dynamic';
import styles from './page.module.css';
import MemexLogo from '@/components/ui/MemexLogo';
import StatusIndicator from '@/components/ui/StatusIndicator';
import ReplayControls from '@/components/ui/ReplayControls';
import CaseTerminal from '@/components/case/CaseTerminal';
import { ReplayEngine } from '@/lib/replay';
import { isBackendAvailable } from '@/lib/api';
import type { ViewState, ReplayStep } from '@/lib/types';

// Dynamic import for map (no SSR)
const GlobalThreatMap = dynamic(
  () => import('@/components/globe/GlobalThreatMap'),
  { ssr: false, loading: () => <div className={styles.mapLoading}>&gt; INITIALIZING MAP ENGINE...</div> }
);

export default function Home() {
  const [view, setView] = useState<ViewState>('globe');
  const [mode, setMode] = useState<'live' | 'replay'>('replay');
  const [backendStatus, setBackendStatus] = useState<'connected' | 'replay' | 'disconnected'>('replay');

  // Replay state
  const replayRef = useRef<ReplayEngine | null>(null);
  const [replaySteps, setReplaySteps] = useState<ReplayStep[]>([]);
  const [currentStepIndex, setCurrentStepIndex] = useState(-1);
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [replayLoaded, setReplayLoaded] = useState(false);
  const [allSteps, setAllSteps] = useState<ReplayStep[]>([]);

  // Transition state
  const [transitioning, setTransitioning] = useState(false);

  // Check backend availability
  useEffect(() => {
    isBackendAvailable().then(available => {
      setBackendStatus(available ? 'connected' : 'replay');
      setMode(available ? 'live' : 'replay');
    });
  }, []);

  // Initialize replay engine
  useEffect(() => {
    const engine = new ReplayEngine();
    replayRef.current = engine;

    engine.load().then(() => {
      setAllSteps(engine.getSteps());
      setReplayLoaded(true);
    });

    const unsub = engine.subscribe((step, index) => {
      setCurrentStepIndex(index);
      setReplaySteps(engine.getSteps().slice(0, index + 1));
    });

    return () => unsub();
  }, []);

  // Handle corridor/hub clicks
  const handleCorridorSelect = useCallback((corridorId: string) => {
    setTransitioning(true);
    setTimeout(() => {
      setView('case');
      setTransitioning(false);
      // Start replay
      if (replayRef.current && allSteps.length > 0) {
        replayRef.current.seekTo(0);
        setCurrentStepIndex(0);
        setReplaySteps(allSteps.slice(0, 1));
      }
    }, 800);
  }, [allSteps]);

  const handleHubSelect = useCallback((hubId: string) => {
    handleCorridorSelect(hubId);
  }, [handleCorridorSelect]);

  // Replay controls
  const handlePlay = useCallback(() => {
    replayRef.current?.play();
    setIsPlaying(true);
  }, []);

  const handlePause = useCallback(() => {
    replayRef.current?.pause();
    setIsPlaying(false);
  }, []);

  const handleStepForward = useCallback(() => {
    replayRef.current?.stepForward();
  }, []);

  const handleStepBack = useCallback(() => {
    replayRef.current?.stepBack();
  }, []);

  const handleSeek = useCallback((step: number) => {
    replayRef.current?.seekTo(step);
  }, []);

  const handleSpeedChange = useCallback((newSpeed: number) => {
    replayRef.current?.setSpeed(newSpeed);
    setSpeed(newSpeed);
  }, []);

  const handleBackToMap = useCallback(() => {
    replayRef.current?.pause();
    setIsPlaying(false);
    setTransitioning(true);
    setTimeout(() => {
      setView('globe');
      setTransitioning(false);
    }, 600);
  }, []);

  return (
    <div className={styles.app}>
      {/* Global Header */}
      <header className={styles.header}>
        <div className={styles.headerLeft}>
          <MemexLogo size="sm" />
          <span className={styles.headerSep}>│</span>
          <span className={styles.headerSubtitle}>FINANCIAL CRIME INTELLIGENCE</span>
        </div>
        <div className={styles.headerCenter}>
          {view === 'case' && replayLoaded && (
            <ReplayControls
              currentStep={Math.max(0, currentStepIndex)}
              totalSteps={allSteps.length}
              isPlaying={isPlaying}
              speed={speed}
              onPlay={handlePlay}
              onPause={handlePause}
              onStepForward={handleStepForward}
              onStepBack={handleStepBack}
              onSeek={handleSeek}
              onSpeedChange={handleSpeedChange}
            />
          )}
        </div>
        <div className={styles.headerRight}>
          <StatusIndicator
            status={backendStatus}
            label={backendStatus === 'connected' ? 'LIVE' : backendStatus === 'replay' ? 'REPLAY' : 'OFFLINE'}
          />
        </div>
      </header>

      {/* View Content */}
      <main className={`${styles.main} ${transitioning ? styles.transitioning : ''}`}>
        {view === 'globe' && (
          <div className={`${styles.viewContainer} ${transitioning ? styles.viewFadeOut : styles.viewFadeIn}`}>
            <GlobalThreatMap
              onCorridorSelect={handleCorridorSelect}
              onHubSelect={handleHubSelect}
            />
          </div>
        )}

        {view === 'case' && (
          <div className={`${styles.viewContainer} ${transitioning ? styles.viewFadeOut : styles.viewSlideIn}`}>
            <CaseTerminal
              steps={replaySteps}
              currentStepIndex={Math.max(0, currentStepIndex)}
              onBack={handleBackToMap}
              alertSummary={replayRef.current?.getMeta()?.alert?.summary}
              caseId={replayRef.current?.getMeta()?.alert?.alert_id}
            />
          </div>
        )}
      </main>
    </div>
  );
}
