<<<<<<< HEAD
import Image from "next/image";

export default function Home() {
  return (
    <div className="flex flex-col flex-1 items-center justify-center bg-zinc-50 font-sans dark:bg-black">
      <main className="flex flex-1 w-full max-w-3xl flex-col items-center justify-between py-32 px-16 bg-white dark:bg-black sm:items-start">
        <Image
          className="dark:invert"
          src="/next.svg"
          alt="Next.js logo"
          width={100}
          height={20}
          priority
        />
        <div className="flex flex-col items-center gap-6 text-center sm:items-start sm:text-left">
          <h1 className="max-w-xs text-3xl font-semibold leading-10 tracking-tight text-black dark:text-zinc-50">
            To get started, edit the page.tsx file.
          </h1>
          <p className="max-w-md text-lg leading-8 text-zinc-600 dark:text-zinc-400">
            Looking for a starting point or more instructions? Head over to{" "}
            <a
              href="https://vercel.com/templates?framework=next.js&utm_source=create-next-app&utm_medium=appdir-template-tw&utm_campaign=create-next-app"
              className="font-medium text-zinc-950 dark:text-zinc-50"
            >
              Templates
            </a>{" "}
            or the{" "}
            <a
              href="https://nextjs.org/learn?utm_source=create-next-app&utm_medium=appdir-template-tw&utm_campaign=create-next-app"
              className="font-medium text-zinc-950 dark:text-zinc-50"
            >
              Learning
            </a>{" "}
            center.
          </p>
        </div>
        <div className="flex flex-col gap-4 text-base font-medium sm:flex-row">
          <a
            className="flex h-12 w-full items-center justify-center gap-2 rounded-full bg-foreground px-5 text-background transition-colors hover:bg-[#383838] dark:hover:bg-[#ccc] md:w-[158px]"
            href="https://vercel.com/new?utm_source=create-next-app&utm_medium=appdir-template-tw&utm_campaign=create-next-app"
            target="_blank"
            rel="noopener noreferrer"
          >
            <Image
              className="dark:invert"
              src="/vercel.svg"
              alt="Vercel logomark"
              width={16}
              height={16}
            />
            Deploy Now
          </a>
          <a
            className="flex h-12 w-full items-center justify-center rounded-full border border-solid border-black/[.08] px-5 transition-colors hover:border-transparent hover:bg-black/[.04] dark:border-white/[.145] dark:hover:bg-[#1a1a1a] md:w-[158px]"
            href="https://nextjs.org/docs?utm_source=create-next-app&utm_medium=appdir-template-tw&utm_campaign=create-next-app"
            target="_blank"
            rel="noopener noreferrer"
          >
            Documentation
          </a>
        </div>
      </main>
=======
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
>>>>>>> 85b18b358464e3203bde643fa27d6209bdda40c2
    </div>
  );
}
