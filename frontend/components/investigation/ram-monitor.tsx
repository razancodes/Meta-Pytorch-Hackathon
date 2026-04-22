"use client";

import { TerminalPanel } from "@/components/ui/terminal-panel";
import { useMemexStore } from "@/lib/store";
import { cn } from "@/lib/utils";

export function RAMMonitor() {
  const aguiState = useMemexStore((s) => s.aguiState);
  const pageFaultActive = useMemexStore((s) => s.pageFaultActive);
  const { capacity, active_context } = aguiState.ram_usage;

  const [used, total] = capacity.split("/").map((s) => parseInt(s));
  const isFull = used >= total;
  const percentage = total > 0 ? (used / total) * 100 : 0;

  return (
    <TerminalPanel
      title="RAM Monitor"
      status={isFull ? "FULL" : `${used}/${total}`}
      statusColor={isFull ? "threat" : "accent"}
      flash={pageFaultActive}
    >
      {/* Capacity Bar */}
      <div className="mb-3">
        <div className="h-1.5 bg-bg rounded-[1px] overflow-hidden border border-muted/50">
          <div
            className={cn(
              "h-full transition-all duration-500 rounded-[1px]",
              isFull ? "bg-threat" : "bg-accent"
            )}
            style={{ width: `${percentage}%` }}
          />
        </div>
        <div className="flex justify-between mt-1">
          <span className="text-[8px] text-text-dim">{capacity}</span>
          {pageFaultActive && (
            <span className="text-[8px] text-threat font-bold animate-flicker">
              ⚠ PAGE FAULT
            </span>
          )}
        </div>
      </div>

      {/* Context Slots */}
      <div className="space-y-2">
        {[0, 1].map((slot) => (
          <div
            key={slot}
            className={cn(
              "rounded-[2px] border p-2 text-[10px] leading-relaxed min-h-[40px]",
              active_context[slot]
                ? "border-muted bg-bg text-text"
                : "border-muted/30 bg-bg/50 text-text-dim"
            )}
          >
            {active_context[slot] ? (
              <div>
                <span className="text-[8px] text-accent uppercase tracking-wider">
                  Slot {slot + 1}
                </span>
                <p className="mt-0.5">{active_context[slot]}</p>
              </div>
            ) : (
              <span className="text-[8px] uppercase tracking-wider">
                Slot {slot + 1} — Empty
              </span>
            )}
          </div>
        ))}
      </div>
    </TerminalPanel>
  );
}
