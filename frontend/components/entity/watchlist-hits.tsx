"use client";

import { TerminalPanel } from "@/components/ui/terminal-panel";
import { StatusBadge } from "@/components/ui/status-badge";
import type { WatchlistResult } from "@/lib/types";

interface WatchlistHitsProps {
  watchlist: WatchlistResult;
  entityId: string;
}

export function WatchlistHits({ watchlist, entityId }: WatchlistHitsProps) {
  return (
    <TerminalPanel
      title="Watchlist Screening"
      status={watchlist.match ? "HIT" : "CLEAR"}
      statusColor={watchlist.match ? "threat" : "success"}
    >
      {watchlist.match ? (
        <div className="space-y-2">
          <div className="flex gap-2 flex-wrap">
            {watchlist.lists.map((list) => (
              <StatusBadge key={list} level="critical" label={list} />
            ))}
          </div>
          <div className="text-[10px] text-threat bg-threat/10 rounded-[2px] border border-threat/20 p-2 leading-relaxed">
            {watchlist.details}
          </div>
        </div>
      ) : (
        <div className="text-[10px] text-success">
          ✓ No watchlist matches found for {entityId}
        </div>
      )}
    </TerminalPanel>
  );
}
