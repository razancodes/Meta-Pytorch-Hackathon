"use client";

import { cn } from "@/lib/utils";

interface PulseDotProps {
  color?: "accent" | "threat" | "success" | "warning";
  size?: "sm" | "md";
  className?: string;
}

export function PulseDot({ color = "accent", size = "sm", className }: PulseDotProps) {
  const colorMap = {
    accent: "bg-accent",
    threat: "bg-threat",
    success: "bg-success",
    warning: "bg-warning",
  };

  const sizeMap = {
    sm: "w-1.5 h-1.5",
    md: "w-2.5 h-2.5",
  };

  return (
    <span className={cn("relative inline-flex", className)}>
      <span className={cn(
        "animate-ping absolute inline-flex rounded-full opacity-75",
        colorMap[color],
        sizeMap[size]
      )} />
      <span className={cn(
        "relative inline-flex rounded-full",
        colorMap[color],
        sizeMap[size]
      )} />
    </span>
  );
}
