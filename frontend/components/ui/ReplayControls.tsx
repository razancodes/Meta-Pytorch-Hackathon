'use client';

import styles from './ui.module.css';

export default function ReplayControls({
  currentStep,
  totalSteps,
  isPlaying,
  speed,
  onPlay,
  onPause,
  onStepForward,
  onStepBack,
  onSeek,
  onSpeedChange,
}: {
  currentStep: number;
  totalSteps: number;
  isPlaying: boolean;
  speed: number;
  onPlay: () => void;
  onPause: () => void;
  onStepForward: () => void;
  onStepBack: () => void;
  onSeek: (step: number) => void;
  onSpeedChange: (speed: number) => void;
}) {
  const speeds = [0.5, 1, 2, 4];

  return (
    <div className={styles.replayBar}>
      <div className={styles.replayGroup}>
        {speeds.map(s => (
          <button
            key={s}
            className={`${styles.replayBtn} ${speed === s ? styles.replayBtnActive : ''}`}
            onClick={() => onSpeedChange(s)}
          >
            {s}x
          </button>
        ))}
      </div>

      <div className={styles.replayGroupCenter}>
        <button className={styles.replayBtn} onClick={onStepBack} title="Step Back">◄</button>
        <button
          className={`${styles.replayBtn} ${isPlaying ? styles.replayBtnActive : ''} ${styles.replayPlayBtn}`}
          onClick={isPlaying ? onPause : onPlay}
          title={isPlaying ? 'Pause' : 'Play'}
        >
          {isPlaying ? '⏸' : '▶'}
        </button>
        <button className={styles.replayBtn} onClick={onStepForward} title="Step Forward">►</button>
      </div>

      <div className={styles.replayGroupRight}>
        <span className={styles.replayStep}>
          {String(currentStep + 1).padStart(2, '0')}/{String(totalSteps).padStart(2, '0')}
        </span>
        <input
          type="range"
          className={styles.replayScrubber}
          min={0}
          max={Math.max(totalSteps - 1, 0)}
          value={currentStep}
          onChange={e => onSeek(parseInt(e.target.value))}
        />
      </div>
    </div>
  );
}
