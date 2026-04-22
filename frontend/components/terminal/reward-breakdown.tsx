"use client";

import { motion } from "framer-motion";
import { TerminalPanel } from "@/components/ui/terminal-panel";
import { cn } from "@/lib/utils";

interface RewardCategory {
  label: string;
  weight: number;
  score: number;
  maxScore: number;
}

const REWARD_BREAKDOWN: RewardCategory[] = [
  { label: "Decision Accuracy", weight: 0.30, score: 0.30, maxScore: 0.30 },
  { label: "Typology Identified", weight: 0.15, score: 0.15, maxScore: 0.15 },
  { label: "Findings Coverage", weight: 0.25, score: 0.25, maxScore: 0.25 },
  { label: "Entity F1 Score", weight: 0.15, score: 0.15, maxScore: 0.15 },
  { label: "Efficiency", weight: 0.15, score: 0.15, maxScore: 0.15 },
];

export function RewardBreakdown() {
  const totalEarned = REWARD_BREAKDOWN.reduce((sum, r) => sum + r.score, 0);

  return (
    <TerminalPanel
      title="RL Environment Metrics"
      status="Composite Score"
      statusColor="success"
    >
      <div className="space-y-3">
        {REWARD_BREAKDOWN.map((item, i) => {
          const percentage = (item.score / item.maxScore) * 100;
          const isPerfect = item.score >= item.maxScore;

          return (
            <motion.div
              key={item.label}
              initial={{ opacity: 0, x: -12 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.3 + i * 0.1, duration: 0.4 }}
            >
              <div className="flex items-center justify-between mb-1">
                <span className="text-[10px] text-text">{item.label}</span>
                <div className="flex items-center gap-2">
                  <span className="text-[9px] text-text-dim">
                    w={item.weight.toFixed(2)}
                  </span>
                  <span className={cn(
                    "text-[10px] font-bold tabular-nums",
                    isPerfect ? "text-success" : "text-accent"
                  )}>
                    {item.score.toFixed(2)}/{item.maxScore.toFixed(2)}
                  </span>
                </div>
              </div>
              <div className="h-1.5 bg-bg rounded-[1px] overflow-hidden border border-muted/30">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${percentage}%` }}
                  transition={{ delay: 0.5 + i * 0.1, duration: 0.8, ease: "easeOut" }}
                  className={cn(
                    "h-full rounded-[1px]",
                    isPerfect ? "bg-success" : "bg-accent"
                  )}
                />
              </div>
            </motion.div>
          );
        })}

        {/* Total */}
        <div className="border-t border-muted pt-2 mt-3 flex items-center justify-between">
          <span className="text-[10px] text-text font-bold uppercase tracking-wider">
            Total Composite
          </span>
          <span className="text-sm font-bold text-success">
            {totalEarned.toFixed(2)} / 1.00
          </span>
        </div>
      </div>
    </TerminalPanel>
  );
}
