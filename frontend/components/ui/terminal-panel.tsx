"use client";

import { cn } from "@/lib/utils";

interface TerminalPanelProps {
  title: string;
  status?: string;
  statusColor?: "accent" | "threat" | "success" | "warning" | "muted";
  className?: string;
  headerRight?: React.ReactNode;
  children: React.ReactNode;
  flash?: boolean;
}

export function TerminalPanel({
  title,
  status,
  statusColor = "muted",
  className,
  headerRight,
  children,
  flash,
}: TerminalPanelProps) {
  const colorMap = {
    accent: "text-accent",
    threat: "text-threat",
    success: "text-success",
    warning: "text-warning",
    muted: "text-text-dim",
  };

  const borderColorMap = {
    accent: "border-accent",
    threat: "border-threat",
    success: "border-success",
    warning: "border-warning",
    muted: "border-muted",
  };

  return (
    <div
      className={cn(
        "rounded-[2px] border border-muted bg-surface overflow-hidden",
        flash && "page-fault-flash border-threat",
        className
      )}
    >
      {/* Header */}
      <div className={cn(
        "flex items-center justify-between px-3 py-2 border-b",
        flash ? "border-threat" : "border-muted"
      )}>
        <div className="flex items-center gap-2">
          <span className={cn("text-[10px] uppercase tracking-[0.15em] font-bold", colorMap[statusColor])}>
            {title}
          </span>
          {status && (
            <span className={cn("text-[9px] uppercase tracking-wider px-1.5 py-0.5 rounded-[2px] border", borderColorMap[statusColor], colorMap[statusColor])}>
              {status}
            </span>
          )}
        </div>
        {headerRight && <div className="flex items-center gap-2">{headerRight}</div>}
      </div>
      {/* Content */}
      <div className="p-3">
        {children}
      </div>
    </div>
  );
}
