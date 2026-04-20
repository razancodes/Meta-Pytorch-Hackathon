"use client";

import { motion, AnimatePresence } from "framer-motion";
import { TerminalPanel } from "@/components/ui/terminal-panel";
import { useMemexStore } from "@/lib/store";
import { cn } from "@/lib/utils";

export function KernelDirectives() {
  const directives = useMemexStore((s) => s.aguiState.kernel_directives);

  return (
    <TerminalPanel
      title="Kernel Directives"
      status={`${directives.length} rules`}
      statusColor="accent"
    >
      <div className="space-y-1.5 max-h-[200px] overflow-y-auto">
        <AnimatePresence>
          {directives.map((directive, i) => {
            const isBase = i === 0;
            const isInjected = directive.startsWith("Added (");
            const isHuman = directive.startsWith("[HUMAN");

            return (
              <motion.div
                key={`kernel-${i}`}
                initial={{ opacity: 0, x: -8 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.3, delay: i * 0.05 }}
                className={cn(
                  "rounded-[2px] border p-2 text-[10px] leading-relaxed",
                  isHuman
                    ? "border-success/50 bg-success/5"
                    : isInjected
                    ? "border-accent/30 bg-accent/5"
                    : "border-muted/30 bg-bg"
                )}
              >
                <span
                  className={cn(
                    "text-[8px] uppercase tracking-[0.12em] font-bold mr-2",
                    isHuman ? "text-success" : isInjected ? "text-accent" : "text-text-dim"
                  )}
                >
                  {isHuman ? "[OVERRIDE]" : isInjected ? "[INJECTED]" : "[BASE]"}
                </span>
                <span className="text-text">{directive}</span>
              </motion.div>
            );
          })}
        </AnimatePresence>
      </div>
    </TerminalPanel>
  );
}
