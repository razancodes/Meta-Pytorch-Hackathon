"use client";

import { cn } from "@/lib/utils";
import type { RiskLevel } from "@/lib/types";

interface StatusBadgeProps {
  level: RiskLevel | "ready" | "pending" | "retrieved" | "open" | "in-progress" | "resolved";
  label?: string;
  className?: string;
}

export function StatusBadge({ level, label, className }: StatusBadgeProps) {
  const config: Record<string, { bg: string; text: string; defaultLabel: string }> = {
    critical: { bg: "bg-threat/20", text: "text-threat", defaultLabel: "CRITICAL" },
    high: { bg: "bg-accent/20", text: "text-accent", defaultLabel: "HIGH" },
    medium: { bg: "bg-warning/20", text: "text-warning", defaultLabel: "MEDIUM" },
    low: { bg: "bg-muted/30", text: "text-text-dim", defaultLabel: "LOW" },
    ready: { bg: "bg-success/20", text: "text-success", defaultLabel: "READY" },
    pending: { bg: "bg-accent/20", text: "text-accent", defaultLabel: "PENDING" },
    retrieved: { bg: "bg-muted/30", text: "text-text-dim", defaultLabel: "RETRIEVED" },
    open: { bg: "bg-accent/20", text: "text-accent", defaultLabel: "OPEN" },
    "in-progress": { bg: "bg-warning/20", text: "text-warning", defaultLabel: "IN PROGRESS" },
    resolved: { bg: "bg-success/20", text: "text-success", defaultLabel: "RESOLVED" },
  };

  const { bg, text, defaultLabel } = config[level] || config.low;

  return (
    <span
      className={cn(
        "inline-flex items-center px-1.5 py-0.5 text-[9px] font-bold uppercase tracking-[0.12em] rounded-[2px] border",
        bg, text,
        `border-current/20`,
        className
      )}
    >
      {label || defaultLabel}
    </span>
  );
}
