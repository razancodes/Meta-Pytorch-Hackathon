"use client";

import { useEffect, useRef } from "react";
import { motion } from "framer-motion";
import { useMemexStore } from "@/lib/store";
import { NetworkGraph } from "@/components/investigation/network-graph";
import { OSConsole } from "@/components/investigation/os-console";
import { EntityDrawer } from "@/components/entity/entity-drawer";
import { AgentScore } from "@/components/terminal/agent-score";
import { RewardBreakdown } from "@/components/terminal/reward-breakdown";
import { SARPayload } from "@/components/terminal/sar-payload";
import { ScanLine } from "@/components/ui/scan-line";
import { PulseDot } from "@/components/ui/pulse-dot";
import { StatusBadge } from "@/components/ui/status-badge";
import { cn, formatReward } from "@/lib/utils";
import { useRouter } from "next/navigation";

export default function InvestigationPage() {
  const router = useRouter();
  const currentFlow = useMemexStore((s) => s.currentFlow);
  const episodeMeta = useMemexStore((s) => s.episodeMeta);
  const steps = useMemexStore((s) => s.steps);
  const currentStepIndex = useMemexStore((s) => s.currentStepIndex);
  const isPlaying = useMemexStore((s) => s.isPlaying);
  const playbackSpeed = useMemexStore((s) => s.playbackSpeed);
  const advanceStep = useMemexStore((s) => s.advanceStep);
  const togglePlayback = useMemexStore((s) => s.togglePlayback);
  const setPlaybackSpeed = useMemexStore((s) => s.setPlaybackSpeed);
  const goToStep = useMemexStore((s) => s.goToStep);
  const loadEpisode = useMemexStore((s) => s.loadEpisode);

  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  // Load episode if not loaded
  useEffect(() => {
    if (!episodeMeta) {
      loadEpisode();
    }
  }, [episodeMeta, loadEpisode]);

  // Playback loop
  useEffect(() => {
    if (isPlaying) {
      intervalRef.current = setInterval(() => {
        advanceStep();
      }, playbackSpeed);
    } else {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    }
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [isPlaying, playbackSpeed, advanceStep]);

  const currentStep = steps[currentStepIndex];
  const isTerminal = currentFlow === "terminal";
  const cumulativeReward = steps
    .slice(0, currentStepIndex + 1)
    .reduce((sum, s) => sum + (s.observation.reward ?? 0), 0);

  return (
    <div className="h-screen flex flex-col bg-bg overflow-hidden">
      <ScanLine />
      <EntityDrawer />

      {/* Top Bar */}
      <header className="border-b border-muted px-4 py-2 shrink-0">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <button
              onClick={() => router.push("/")}
              className="text-accent font-bold text-sm tracking-wider hover:text-accent/80 transition-colors cursor-pointer"
            >
              MEMEX
            </button>
            <div className="w-px h-4 bg-muted" />
            <StatusBadge level="critical" label="1MDB INVESTIGATION" />
            <div className="text-[9px] text-text-dim">
              {episodeMeta?.alert.alert_id}
            </div>
          </div>

          <div className="flex items-center gap-4">
            {/* Step counter */}
            <div className="text-[10px] text-text-dim">
              Step <span className="text-accent font-bold">{currentStepIndex + 1}</span>
              <span className="text-text-dim">/{steps.length}</span>
            </div>

            {/* Current tool */}
            {currentStep && (
              <div className="text-[10px] text-text-dim">
                <span className="text-accent">{currentStep.action.tool}</span>
              </div>
            )}

            {/* Cumulative reward */}
            <div className="text-[10px]">
              <span className="text-text-dim">R: </span>
              <span className={cn(
                "font-bold tabular-nums",
                cumulativeReward >= 0 ? "text-success" : "text-threat"
              )}>
                {formatReward(cumulativeReward)}
              </span>
            </div>

            <div className="flex items-center gap-2">
              <PulseDot color={isPlaying ? "success" : "warning"} />
              <span className="text-[9px] uppercase tracking-wider text-text-dim">
                {isTerminal ? "COMPLETE" : isPlaying ? "RUNNING" : "PAUSED"}
              </span>
            </div>
          </div>
        </div>
      </header>

      {/* Playback Controls */}
      {!isTerminal && (
        <div className="border-b border-muted px-4 py-1.5 flex items-center gap-3 shrink-0 bg-surface">
          <button
            onClick={togglePlayback}
            className={cn(
              "px-3 py-1 rounded-[2px] border text-[10px] font-bold uppercase tracking-wider transition-all cursor-pointer",
              isPlaying
                ? "border-warning text-warning bg-warning/10 hover:bg-warning/20"
                : "border-success text-success bg-success/10 hover:bg-success/20"
            )}
          >
            {isPlaying ? "⏸ PAUSE" : "▶ PLAY"}
          </button>

          <button
            onClick={() => advanceStep()}
            disabled={isPlaying || currentStepIndex >= steps.length - 1}
            className="px-3 py-1 rounded-[2px] border border-muted text-[10px] text-text-dim hover:border-accent hover:text-accent transition-all disabled:opacity-30 cursor-pointer disabled:cursor-not-allowed"
          >
            STEP →
          </button>

          {/* Speed controls */}
          <div className="flex items-center gap-1 ml-2">
            <span className="text-[9px] text-text-dim uppercase tracking-wider">Speed:</span>
            {[
              { label: "0.5×", value: 4000 },
              { label: "1×", value: 2000 },
              { label: "2×", value: 1000 },
              { label: "4×", value: 500 },
            ].map((s) => (
              <button
                key={s.label}
                onClick={() => setPlaybackSpeed(s.value)}
                className={cn(
                  "px-1.5 py-0.5 rounded-[2px] text-[9px] transition-all cursor-pointer",
                  playbackSpeed === s.value
                    ? "bg-accent/20 text-accent border border-accent/30"
                    : "text-text-dim hover:text-text border border-transparent"
                )}
              >
                {s.label}
              </button>
            ))}
          </div>

          {/* Step timeline */}
          <div className="flex-1 mx-4">
            <div className="flex gap-0.5">
              {steps.map((step, i) => (
                <button
                  key={i}
                  onClick={() => goToStep(i)}
                  className={cn(
                    "flex-1 h-1.5 rounded-[1px] transition-all cursor-pointer",
                    i <= currentStepIndex
                      ? step.observation.done
                        ? "bg-success"
                        : "bg-accent"
                      : "bg-muted/30 hover:bg-muted/50"
                  )}
                  title={`Step ${step.step_number}: ${step.action.tool}`}
                />
              ))}
            </div>
          </div>

          {/* Reasoning */}
          {currentStep?.reasoning && (
            <div className="text-[9px] text-text-dim max-w-[300px] truncate">
              💭 {currentStep.reasoning}
            </div>
          )}
        </div>
      )}

      {/* Main Content */}
      {isTerminal ? (
        /* Terminal State — Flow D */
        <motion.main
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.6 }}
          className="flex-1 overflow-y-auto p-6"
        >
          <div className="max-w-3xl mx-auto space-y-6">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
              className="text-center mb-4"
            >
              <div className="text-[10px] uppercase tracking-[0.2em] text-success font-bold mb-1">
                ✓ Investigation Complete
              </div>
              <div className="text-xs text-text-dim">
                {episodeMeta?.scenario}
              </div>
            </motion.div>

            <AgentScore />

            <div className="grid grid-cols-2 gap-4">
              <SARPayload />
              <RewardBreakdown />
            </div>

            {/* Step History */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 1 }}
              className="rounded-[2px] border border-muted bg-surface"
            >
              <div className="px-3 py-2 border-b border-muted">
                <span className="text-[10px] uppercase tracking-[0.15em] font-bold text-accent">
                  Investigation Timeline
                </span>
              </div>
              <div className="p-3 space-y-1 max-h-[300px] overflow-y-auto">
                {steps.map((step) => (
                  <div
                    key={step.step_number}
                    className="grid grid-cols-[40px_1fr_80px] gap-2 text-[10px] py-1 border-b border-muted/20"
                  >
                    <span className="text-text-dim">#{String(step.step_number).padStart(2, "0")}</span>
                    <span className="text-text">{step.action.tool}</span>
                    <span className={cn(
                      "text-right font-bold tabular-nums",
                      (step.observation.reward ?? 0) >= 0 ? "text-success" : "text-threat"
                    )}>
                      {formatReward(step.observation.reward)}
                    </span>
                  </div>
                ))}
              </div>
            </motion.div>
          </div>
        </motion.main>
      ) : (
        /* Investigation Workspace — Flow B */
        <main className="flex-1 flex overflow-hidden">
          {/* Left: Network Graph (60%) */}
          <div className="w-[60%] h-full p-2">
            <NetworkGraph />
          </div>

          {/* Right: OS Console (40%) */}
          <div className="w-[40%] h-full border-l border-muted p-3 overflow-hidden">
            <OSConsole />
          </div>
        </main>
      )}
    </div>
  );
}
