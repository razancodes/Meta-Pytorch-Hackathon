"use client";

import { motion } from "framer-motion";
import { TerminalPanel } from "@/components/ui/terminal-panel";
import { StatusBadge } from "@/components/ui/status-badge";
import { useMemexStore } from "@/lib/store";

export function SARPayload() {
  const episodeMeta = useMemexStore((s) => s.episodeMeta);
  const steps = useMemexStore((s) => s.steps);

  if (!episodeMeta) return null;

  const { ground_truth } = episodeMeta;
  const lastStep = steps[steps.length - 1];
  const sarAction = lastStep?.action;

  const findingLabels: Record<string, string> = {
    pep_connection: "Politically Exposed Person Connection",
    offshore_source: "Offshore Source of Funds",
    shared_registered_address: "Shared Registered Address (Shell Companies)",
    rapid_fan_out: "Rapid Fan-Out Pattern ($681M in 6 weeks)",
    no_source_documentation: "Missing Source Documentation",
    reversed_transaction: "Suspicious Reversal Transaction ($6M)",
  };

  return (
    <TerminalPanel
      title="Suspicious Activity Report"
      status="FILED"
      statusColor="success"
    >
      <div className="space-y-4">
        {/* Decision */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.2 }}
          className="flex items-center gap-3"
        >
          <div className="text-[10px] text-text-dim uppercase tracking-wider">Decision:</div>
          <StatusBadge level="critical" label="FILE SAR" />
        </motion.div>

        {/* Typology */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.3 }}
        >
          <div className="text-[10px] text-text-dim uppercase tracking-wider mb-1">Typology</div>
          <div className="text-xs font-bold text-accent uppercase">{ground_truth.typology}</div>
        </motion.div>

        {/* Entities */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.4 }}
        >
          <div className="text-[10px] text-text-dim uppercase tracking-wider mb-1">
            Entities Involved ({ground_truth.key_entities.length})
          </div>
          <div className="space-y-1">
            {ground_truth.key_entities.map((entity) => (
              <div key={entity} className="text-[10px] text-text bg-bg rounded-[2px] border border-muted/30 px-2 py-1">
                <span className="text-accent">▸</span> {entity}
              </div>
            ))}
          </div>
        </motion.div>

        {/* Findings */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5 }}
        >
          <div className="text-[10px] text-text-dim uppercase tracking-wider mb-1">
            Key Findings ({ground_truth.key_findings.length}/6)
          </div>
          <div className="space-y-1">
            {ground_truth.key_findings.map((finding, i) => (
              <motion.div
                key={finding}
                initial={{ opacity: 0, x: -8 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.6 + i * 0.08 }}
                className="flex items-start gap-2 text-[10px] text-text bg-bg rounded-[2px] border border-success/20 px-2 py-1.5"
              >
                <span className="text-success shrink-0">✓</span>
                <span>{findingLabels[finding] || finding}</span>
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* Investigation Summary */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.8 }}
          className="border-t border-muted pt-3"
        >
          <div className="grid grid-cols-3 gap-3 text-center">
            <div>
              <div className="text-[8px] text-text-dim uppercase tracking-wider">Steps</div>
              <div className="text-sm font-bold text-accent">{episodeMeta.total_steps}</div>
            </div>
            <div>
              <div className="text-[8px] text-text-dim uppercase tracking-wider">Entities</div>
              <div className="text-sm font-bold text-accent">{episodeMeta.entity_count}</div>
            </div>
            <div>
              <div className="text-[8px] text-text-dim uppercase tracking-wider">Transactions</div>
              <div className="text-sm font-bold text-accent">{episodeMeta.transaction_count}</div>
            </div>
          </div>
        </motion.div>
      </div>
    </TerminalPanel>
  );
}
