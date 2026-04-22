import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";
import type { RiskLevel } from "./types";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatCurrency(amount: number, currency: string = "USD"): string {
  if (amount >= 1_000_000_000) {
    return `$${(amount / 1_000_000_000).toFixed(1)}B`;
  }
  if (amount >= 1_000_000) {
    return `$${(amount / 1_000_000).toFixed(0)}M`;
  }
  if (amount >= 1_000) {
    return `$${(amount / 1_000).toFixed(0)}K`;
  }
  return `${currency === "USD" ? "$" : currency + " "}${amount.toLocaleString()}`;
}

export function formatReward(reward: number | null): string {
  if (reward === null) return "—";
  return `${reward >= 0 ? "+" : ""}${reward.toFixed(4)}`;
}

export function riskColor(risk: RiskLevel): string {
  switch (risk) {
    case "critical": return "text-threat";
    case "high": return "text-accent";
    case "medium": return "text-warning";
    case "low": return "text-text-dim";
  }
}

export function riskBgColor(risk: RiskLevel): string {
  switch (risk) {
    case "critical": return "bg-threat/20 text-threat";
    case "high": return "bg-accent/20 text-accent";
    case "medium": return "bg-warning/20 text-warning";
    case "low": return "bg-muted/20 text-text-dim";
  }
}

export function truncate(str: string, maxLen: number): string {
  if (str.length <= maxLen) return str;
  return str.slice(0, maxLen - 3) + "...";
}

export function getNodeColor(risk: RiskLevel): string {
  switch (risk) {
    case "critical": return "#E11D48";
    case "high": return "#EA580C";
    case "medium": return "#F59E0B";
    case "low": return "#404040";
  }
}

export function getEdgeColor(strength: "strong" | "moderate" | "weak"): string {
  switch (strength) {
    case "strong": return "#EA580C";
    case "moderate": return "#F59E0B";
    case "weak": return "#404040";
  }
}
