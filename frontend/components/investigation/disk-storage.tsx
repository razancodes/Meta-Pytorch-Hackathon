"use client";

import { motion, AnimatePresence } from "framer-motion";
import { TerminalPanel } from "@/components/ui/terminal-panel";
import { useMemexStore } from "@/lib/store";

export function DiskStorage() {
  const disk = useMemexStore((s) => s.aguiState.disk_storage);

  return (
    <TerminalPanel
      title="Disk Storage"
      status={`${disk.length} entries`}
      statusColor="success"
    >
      <div className="space-y-1.5 max-h-[200px] overflow-y-auto">
        <AnimatePresence>
          {disk.length === 0 ? (
            <div className="text-[10px] text-text-dim py-2">
              No persistent findings saved yet.
            </div>
          ) : (
            disk.map((entry, i) => (
              <motion.div
                key={`disk-${i}`}
                initial={{ opacity: 0, x: -8 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.3 }}
                className="flex gap-2 text-[10px] leading-relaxed rounded-[2px] border border-muted/30 bg-bg p-2"
              >
                <span className="text-success font-bold shrink-0">
                  {String(i + 1).padStart(2, "0")}
                </span>
                <span className="text-text">{entry}</span>
              </motion.div>
            ))
          )}
        </AnimatePresence>
      </div>
    </TerminalPanel>
  );
}
