"use client";

import { motion, AnimatePresence } from "framer-motion";
import { TerminalPanel } from "@/components/ui/terminal-panel";
import { StatusBadge } from "@/components/ui/status-badge";
import { PulseDot } from "@/components/ui/pulse-dot";
import { useMemexStore } from "@/lib/store";

export function AsyncProcesses() {
  const jobs = useMemexStore((s) => s.aguiState.async_jobs);

  return (
    <TerminalPanel
      title="Active Processes"
      status={jobs.length > 0 ? `${jobs.length} queued` : "idle"}
      statusColor={jobs.length > 0 ? "warning" : "muted"}
    >
      <div className="space-y-2 max-h-[200px] overflow-y-auto">
        <AnimatePresence>
          {jobs.length === 0 ? (
            <div className="text-[10px] text-text-dim py-2">
              No background processes running.
            </div>
          ) : (
            jobs.map((job) => (
              <motion.div
                key={job.id}
                initial={{ opacity: 0, y: -6 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: 6 }}
                transition={{ duration: 0.3 }}
                className="rounded-[2px] border border-muted/50 bg-bg p-2"
              >
                <div className="flex items-center justify-between mb-1">
                  <div className="flex items-center gap-2">
                    <span className="text-[10px] font-bold text-accent">{job.id}</span>
                    <span className="text-[9px] text-text-dim">{job.tool}</span>
                  </div>
                  <StatusBadge level={job.status as "pending" | "ready" | "retrieved"} />
                </div>
                <div className="flex items-center gap-2 mt-1">
                  {job.status === "pending" ? (
                    <>
                      <PulseDot color="warning" />
                      <span className="text-[9px] text-warning">
                        ETA: {job.eta_steps} step{job.eta_steps !== 1 ? "s" : ""}
                      </span>
                      {/* Progress bar */}
                      <div className="flex-1 h-1 bg-bg border border-muted/30 rounded-[1px] overflow-hidden">
                        <div
                          className="h-full bg-warning/60 transition-all duration-500"
                          style={{ width: `${Math.max(20, 100 - job.eta_steps * 30)}%` }}
                        />
                      </div>
                    </>
                  ) : (
                    <>
                      <PulseDot color="success" />
                      <span className="text-[9px] text-success font-bold">
                        Result available — retrieve now
                      </span>
                    </>
                  )}
                </div>
              </motion.div>
            ))
          )}
        </AnimatePresence>
      </div>
    </TerminalPanel>
  );
}
