"use client";

import { motion, AnimatePresence } from "framer-motion";
import { useMemexStore } from "@/lib/store";
import { KYCDetails } from "./kyc-details";
import { TransactionTimeline } from "./transaction-timeline";
import { WatchlistHits } from "./watchlist-hits";
import { cn } from "@/lib/utils";

export function EntityDrawer() {
  const selectedNodeId = useMemexStore((s) => s.selectedNodeId);
  const selectNode = useMemexStore((s) => s.selectNode);
  const entityProfiles = useMemexStore((s) => s.entityProfiles);
  const transactions = useMemexStore((s) => s.transactions);
  const watchlistResults = useMemexStore((s) => s.watchlistResults);

  const profile = selectedNodeId ? entityProfiles[selectedNodeId] : null;
  const entityTxns = selectedNodeId
    ? transactions.filter((t) => t.from === selectedNodeId || t.to === selectedNodeId)
    : [];
  const watchlist = selectedNodeId ? watchlistResults[selectedNodeId] : null;

  return (
    <AnimatePresence>
      {selectedNodeId && profile && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/60 z-40"
            onClick={() => selectNode(null)}
          />
          {/* Drawer */}
          <motion.div
            initial={{ x: "100%" }}
            animate={{ x: 0 }}
            exit={{ x: "100%" }}
            transition={{ type: "spring", damping: 30, stiffness: 300 }}
            className={cn(
              "fixed right-0 top-0 bottom-0 w-[420px] z-50",
              "bg-surface border-l border-muted overflow-y-auto"
            )}
          >
            {/* Header */}
            <div className="sticky top-0 bg-surface border-b border-muted px-4 py-3 z-10">
              <div className="flex items-center justify-between">
                <div>
                  <div className="text-[10px] uppercase tracking-[0.15em] text-accent font-bold">
                    Entity Deep Dive
                  </div>
                  <div className="text-sm font-bold text-text mt-0.5">
                    {profile.name}
                  </div>
                  <div className="text-[10px] text-text-dim">
                    {selectedNodeId}
                  </div>
                </div>
                <button
                  onClick={() => selectNode(null)}
                  className={cn(
                    "w-8 h-8 rounded-[2px] border border-muted",
                    "flex items-center justify-center text-text-dim",
                    "hover:border-accent hover:text-accent transition-colors cursor-pointer"
                  )}
                >
                  ✕
                </button>
              </div>
            </div>

            {/* Content */}
            <div className="p-4 space-y-4">
              <KYCDetails profile={profile} />
              {watchlist && <WatchlistHits watchlist={watchlist} entityId={selectedNodeId} />}
              <TransactionTimeline transactions={entityTxns} entityId={selectedNodeId} />
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}
