"use client";

import { TerminalPanel } from "@/components/ui/terminal-panel";
import { StatusBadge } from "@/components/ui/status-badge";
import type { EntityProfile } from "@/lib/types";
import { cn } from "@/lib/utils";

interface KYCDetailsProps {
  profile: EntityProfile;
}

export function KYCDetails({ profile }: KYCDetailsProps) {
  const riskMap: Record<string, "critical" | "high" | "medium" | "low"> = {
    "High": "high",
    "Medium": "medium",
    "Low": "low",
  };

  const fields = [
    { label: "Nationality", value: profile.nationality },
    { label: "Occupation", value: profile.occupation },
    { label: "Account Opened", value: profile.account_open_date || "N/A" },
    { label: "Annual Income", value: profile.annual_income || "N/A" },
    { label: "Address", value: profile.address || profile.registered_address || "N/A" },
    { label: "Registration", value: profile.registration || "N/A" },
  ].filter((f) => f.value !== "N/A");

  return (
    <TerminalPanel
      title="KYC Profile"
      status={profile.risk_rating}
      statusColor={riskMap[profile.risk_rating] === "high" ? "threat" : riskMap[profile.risk_rating] === "medium" ? "warning" : "muted"}
    >
      <div className="space-y-2">
        {/* Risk & PEP */}
        <div className="flex gap-2 flex-wrap">
          <StatusBadge level={riskMap[profile.risk_rating] || "medium"} label={`RISK: ${profile.risk_rating.toUpperCase()}`} />
          {profile.pep_status && (
            <StatusBadge level="critical" label="⚠ PEP" />
          )}
        </div>

        {profile.pep_details && (
          <div className="text-[10px] text-threat bg-threat/10 rounded-[2px] border border-threat/20 p-2">
            {profile.pep_details}
          </div>
        )}

        {/* Key-Value Grid */}
        <div className="grid grid-cols-2 gap-x-4 gap-y-1.5">
          {fields.map((f) => (
            <div key={f.label}>
              <div className="text-[8px] uppercase tracking-[0.12em] text-text-dim">{f.label}</div>
              <div className="text-[10px] text-text">{f.value}</div>
            </div>
          ))}
        </div>

        {/* Directors */}
        {profile.directors && profile.directors.length > 0 && (
          <div>
            <div className="text-[8px] uppercase tracking-[0.12em] text-text-dim mb-1">Directors</div>
            {profile.directors.map((d, i) => (
              <div key={i} className="text-[10px] text-text">• {d}</div>
            ))}
          </div>
        )}

        {/* Notes */}
        {profile.notes && (
          <div className={cn(
            "text-[10px] rounded-[2px] border border-muted/30 bg-bg p-2 text-text-dim leading-relaxed"
          )}>
            <span className="text-[8px] text-accent uppercase tracking-wider font-bold">Notes: </span>
            {profile.notes}
          </div>
        )}
      </div>
    </TerminalPanel>
  );
}
