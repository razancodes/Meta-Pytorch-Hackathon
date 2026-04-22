"use client";

import { motion } from "framer-motion";
import { useRouter } from "next/navigation";
import type { CaseItem } from "@/lib/types";
import { StatusBadge } from "@/components/ui/status-badge";
import { PulseDot } from "@/components/ui/pulse-dot";
import { useMemexStore } from "@/lib/store";
import { cn } from "@/lib/utils";

interface CaseQueueProps {
  cases: CaseItem[];
}

export function CaseQueue({ cases }: CaseQueueProps) {
  const router = useRouter();
  const loadEpisode = useMemexStore((s) => s.loadEpisode);

  const handleSelectCase = (caseItem: CaseItem) => {
    if (caseItem.id === "case-1mdb") {
      loadEpisode();
      router.push("/investigation");
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.6, duration: 0.5 }}
      className="rounded-[2px] border border-muted bg-surface"
    >
      <div className="flex items-center justify-between px-3 py-2 border-b border-muted">
        <span className="text-[10px] uppercase tracking-[0.15em] font-bold text-accent">
          Prioritized Case Queue
        </span>
        <span className="text-[9px] text-text-dim">
          {cases.length} cases
        </span>
      </div>

      {/* Table Header */}
      <div className="grid grid-cols-[2fr_1fr_0.8fr_1.2fr_0.6fr_0.8fr] gap-2 px-3 py-2 border-b border-muted text-[9px] uppercase tracking-[0.12em] text-text-dim">
        <span>Subject / Alert</span>
        <span>Typology</span>
        <span>Amount</span>
        <span>Jurisdictions</span>
        <span>Risk</span>
        <span>Status</span>
      </div>

      {/* Rows */}
      {cases.map((c, i) => (
        <motion.div
          key={c.id}
          initial={{ opacity: 0, x: -12 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.7 + i * 0.08, duration: 0.3 }}
          onClick={() => handleSelectCase(c)}
          className={cn(
            "grid grid-cols-[2fr_1fr_0.8fr_1.2fr_0.6fr_0.8fr] gap-2 px-3 py-3 border-b border-muted/50",
            "transition-all duration-200 cursor-pointer",
            c.id === "case-1mdb"
              ? "bg-threat/5 hover:bg-threat/10 border-l-2 border-l-threat"
              : "hover:bg-surface-raised"
          )}
        >
          {/* Subject */}
          <div>
            <div className="flex items-center gap-2">
              {c.risk_level === "critical" && <PulseDot color="threat" />}
              <span className={cn("text-xs font-bold", c.risk_level === "critical" ? "text-threat" : "text-text")}>
                {c.subject}
              </span>
            </div>
            <div className="text-[9px] text-text-dim mt-0.5 truncate">
              {c.alert_id} — {c.summary.slice(0, 60)}...
            </div>
          </div>

          {/* Typology */}
          <div className="text-[10px] text-text-dim self-center">
            {c.typology}
          </div>

          {/* Amount */}
          <div className={cn(
            "text-xs font-bold self-center",
            c.risk_level === "critical" ? "text-threat" : "text-accent"
          )}>
            {c.amount}
          </div>

          {/* Jurisdictions */}
          <div className="text-[9px] text-text-dim self-center">
            {c.jurisdictions.join(", ")}
          </div>

          {/* Risk */}
          <div className="self-center">
            <StatusBadge level={c.risk_level} label={`${c.risk_score}`} />
          </div>

          {/* Status */}
          <div className="self-center">
            <StatusBadge level={c.status} />
          </div>
        </motion.div>
      ))}
    </motion.div>
  );
}
