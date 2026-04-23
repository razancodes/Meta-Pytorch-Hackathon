'use client';

import styles from './case.module.css';
import type { CurriculumState } from '@/lib/types';

interface Props {
  curriculum: CurriculumState | null | undefined;
}

/**
 * 5th AGUI Panel — Curriculum Engine State.
 *
 * Displays PLR (Prioritized Level Replay) metrics:
 *   - Current scenario difficulty (color-coded)
 *   - Scenario regret (bar visualization)
 *   - Buffer mean regret, diversity, and size
 *
 * Only renders when curriculum.enabled === true.
 */
export default function CurriculumPanel({ curriculum }: Props) {
  if (!curriculum?.enabled) return null;

  const difficultyColors: Record<string, string> = {
    easy: '#22c55e',
    medium: '#f59e0b',
    hard: '#ef4444',
  };

  const diffColor = difficultyColors[curriculum.difficulty_label] ?? '#6b7280';

  // Regret bar width: 0-1 mapped to 0-100%
  const regretPct = Math.min(100, Math.max(0, curriculum.current_scenario_regret * 100));
  const meanRegretPct = Math.min(100, Math.max(0, curriculum.mean_regret * 100));

  // Difficulty gauge: 1.0 (easy) → 3.0 (hard), normalized to 0-100%
  const difficultyPct = Math.min(100, Math.max(0, ((curriculum.mean_difficulty - 1.0) / 2.0) * 100));

  // Coverage: fraction of 9 combos
  const coverageCount = Math.round(curriculum.buffer_diversity * 9);

  return (
    <div className={styles.curriculumPanel}>
      <div className={styles.curriculumHeader}>
        <span className={styles.curriculumTitle}>CURRICULUM ENGINE</span>
        <span className={styles.curriculumBadge}>PLR</span>
      </div>

      <div className={styles.curriculumGrid}>
        {/* Current Scenario Difficulty */}
        <div className={styles.curriculumMetric}>
          <span className={styles.curriculumLabel}>SCENARIO</span>
          <span
            className={styles.curriculumDifficulty}
            style={{ color: diffColor, borderColor: diffColor }}
          >
            {curriculum.difficulty_label.toUpperCase()}
          </span>
        </div>

        {/* Scenario Regret */}
        <div className={styles.curriculumMetric}>
          <span className={styles.curriculumLabel}>REGRET</span>
          <div className={styles.curriculumBar}>
            <div
              className={styles.curriculumBarFill}
              style={{
                width: `${regretPct}%`,
                backgroundColor: regretPct > 60 ? '#ef4444' : regretPct > 30 ? '#f59e0b' : '#22c55e',
              }}
            />
            <span className={styles.curriculumBarValue}>
              {curriculum.current_scenario_regret.toFixed(3)}
            </span>
          </div>
        </div>

        {/* Buffer Mean Regret */}
        <div className={styles.curriculumMetric}>
          <span className={styles.curriculumLabel}>MEAN REGRET</span>
          <div className={styles.curriculumBar}>
            <div
              className={styles.curriculumBarFill}
              style={{
                width: `${meanRegretPct}%`,
                backgroundColor: 'var(--accent-orange)',
              }}
            />
            <span className={styles.curriculumBarValue}>
              {curriculum.mean_regret.toFixed(3)}
            </span>
          </div>
        </div>

        {/* Mean Difficulty Gauge */}
        <div className={styles.curriculumMetric}>
          <span className={styles.curriculumLabel}>DIFFICULTY</span>
          <div className={styles.curriculumBar}>
            <div
              className={styles.curriculumBarFill}
              style={{
                width: `${difficultyPct}%`,
                background: 'linear-gradient(90deg, #22c55e, #f59e0b, #ef4444)',
              }}
            />
            <span className={styles.curriculumBarValue}>
              {curriculum.mean_difficulty.toFixed(1)} / 3.0
            </span>
          </div>
        </div>

        {/* Coverage */}
        <div className={styles.curriculumMetric}>
          <span className={styles.curriculumLabel}>COVERAGE</span>
          <span className={styles.curriculumValue}>{coverageCount}/9 scenarios</span>
        </div>

        {/* Buffer Size */}
        <div className={styles.curriculumMetric}>
          <span className={styles.curriculumLabel}>BUFFER</span>
          <span className={styles.curriculumValue}>{curriculum.buffer_size} entries</span>
        </div>
      </div>
    </div>
  );
}
