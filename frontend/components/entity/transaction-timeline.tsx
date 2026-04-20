"use client";

import { TerminalPanel } from "@/components/ui/terminal-panel";
import type { Transaction } from "@/lib/types";
import { formatCurrency, cn } from "@/lib/utils";

interface TransactionTimelineProps {
  transactions: Transaction[];
  entityId: string;
}

export function TransactionTimeline({ transactions, entityId }: TransactionTimelineProps) {
  const sorted = [...transactions].sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime());

  return (
    <TerminalPanel
      title="Transaction Timeline"
      status={`${transactions.length} txns`}
      statusColor="accent"
    >
      <div className="space-y-0 relative">
        {/* Vertical line */}
        <div className="absolute left-[7px] top-2 bottom-2 w-px bg-muted/50" />

        {sorted.length === 0 ? (
          <div className="text-[10px] text-text-dim py-2 pl-5">
            No transactions found for this entity.
          </div>
        ) : (
          sorted.map((txn, i) => {
            const isOutgoing = txn.from === entityId;
            const amountColor = txn.amount > 100_000_000 ? "text-threat" : txn.amount > 10_000_000 ? "text-accent" : "text-text";

            return (
              <div key={txn.transaction_id} className="flex gap-3 py-2 relative">
                {/* Dot */}
                <div className={cn(
                  "w-[15px] h-[15px] rounded-full border-2 shrink-0 mt-0.5 z-10 bg-surface",
                  isOutgoing ? "border-threat" : "border-success"
                )} />

                {/* Content */}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-[9px] text-text-dim">{txn.date}</span>
                    <span className={cn("text-[10px] font-bold", amountColor)}>
                      {isOutgoing ? "−" : "+"}{formatCurrency(txn.amount, txn.currency)}
                    </span>
                  </div>
                  <div className="text-[10px] text-text mt-0.5">
                    {isOutgoing ? `→ ${txn.to}` : `← ${txn.from}`}
                  </div>
                  <div className="text-[9px] text-text-dim mt-0.5">
                    {txn.type} — {txn.description}
                  </div>
                  {(txn.jurisdiction_from || txn.jurisdiction_to) && (
                    <div className="text-[8px] text-text-dim mt-0.5">
                      {txn.jurisdiction_from} → {txn.jurisdiction_to}
                    </div>
                  )}
                </div>
              </div>
            );
          })
        )}
      </div>
    </TerminalPanel>
  );
}
