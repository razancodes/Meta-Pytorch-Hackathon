"use client";

import { motion } from "framer-motion";
import { GlobalMetrics } from "@/components/triage/global-metrics";
import { ThreatHeatmap } from "@/components/triage/threat-heatmap";
import { CaseQueue } from "@/components/triage/case-queue";
import { GLOBAL_METRICS, CASE_QUEUE } from "@/lib/demo-data";
import { ScanLine } from "@/components/ui/scan-line";
import { PulseDot } from "@/components/ui/pulse-dot";

export default function TriagePage() {
  return (
    <div className="min-h-screen bg-bg">
      <ScanLine />

      {/* Top Bar */}
      <header className="border-b border-muted px-6 py-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <motion.div
              initial={{ opacity: 0, x: -12 }}
              animate={{ opacity: 1, x: 0 }}
              className="flex items-center gap-2"
            >
              <div className="text-accent font-bold text-sm tracking-wider">MEMEX</div>
              <div className="text-[9px] text-text-dim border border-muted rounded-[2px] px-1.5 py-0.5 uppercase tracking-wider">
                v0.2.0
              </div>
            </motion.div>
            <div className="w-px h-4 bg-muted" />
            <div className="text-[10px] text-text-dim uppercase tracking-[0.15em]">
              Anti-Money Laundering Intelligence Platform
            </div>
          </div>
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <PulseDot color="success" />
              <span className="text-[9px] text-success uppercase tracking-wider">System Online</span>
            </div>
            <div className="text-[9px] text-text-dim">
              {new Date().toISOString().split("T")[0]}
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="p-6 space-y-4 max-w-[1600px] mx-auto">
        {/* Section Title */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.1 }}
          className="flex items-center justify-between"
        >
          <div>
            <h1 className="text-sm font-bold text-text uppercase tracking-[0.1em]">
              Global Triage Dashboard
            </h1>
            <p className="text-[10px] text-text-dim mt-0.5">
              Macro threat landscape • Select a case to begin investigation
            </p>
          </div>
          <div className="flex items-center gap-2 text-[9px] text-text-dim">
            <span className="w-1.5 h-1.5 rounded-full bg-accent animate-pulse" />
            <span>LIVE</span>
          </div>
        </motion.div>

        {/* Metrics */}
        <GlobalMetrics metrics={GLOBAL_METRICS} />

        {/* Heatmap */}
        <ThreatHeatmap />

        {/* Case Queue */}
        <CaseQueue cases={CASE_QUEUE} />
      </main>

      {/* Footer */}
      <footer className="border-t border-muted px-6 py-2 mt-8">
        <div className="flex items-center justify-between text-[8px] text-text-dim uppercase tracking-wider">
          <span>Memex OS-Agent Benchmark // Meta × HuggingFace OpenEnv Hackathon</span>
          <span>Reinforcement Learning + OS Mechanics</span>
        </div>
      </footer>
    </div>
  );
}
