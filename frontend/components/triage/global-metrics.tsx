"use client";

import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import type { GlobalMetric } from "@/lib/types";
import { cn } from "@/lib/utils";

interface GlobalMetricsProps {
  metrics: GlobalMetric[];
}

function AnimatedNumber({ value, unit }: { value: number; unit?: string }) {
  const [display, setDisplay] = useState(0);

  useEffect(() => {
    const duration = 1200;
    const steps = 40;
    const increment = value / steps;
    let current = 0;
    const timer = setInterval(() => {
      current += increment;
      if (current >= value) {
        setDisplay(value);
        clearInterval(timer);
      } else {
        setDisplay(Math.floor(current * 10) / 10);
      }
    }, duration / steps);
    return () => clearInterval(timer);
  }, [value]);

  return (
    <span>
      {unit === "%" ? display.toFixed(1) : Math.floor(display)}
      {unit || ""}
    </span>
  );
}

export function GlobalMetrics({ metrics }: GlobalMetricsProps) {
  const colorMap = {
    accent: "text-accent border-accent/30",
    threat: "text-threat border-threat/30",
    success: "text-success border-success/30",
    warning: "text-warning border-warning/30",
  };

  return (
    <div className="grid grid-cols-4 gap-3">
      {metrics.map((metric, i) => (
        <motion.div
          key={metric.label}
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: i * 0.1, duration: 0.4 }}
          className={cn(
            "rounded-[2px] border border-muted bg-surface p-4",
            "hover:border-accent/50 transition-colors duration-200"
          )}
        >
          <div className="text-[10px] uppercase tracking-[0.15em] text-text-dim mb-2">
            {metric.label}
          </div>
          <div className={cn("text-2xl font-bold", colorMap[metric.color]?.split(" ")[0])}>
            <AnimatedNumber value={metric.value} unit={metric.unit} />
          </div>
          {metric.change !== undefined && (
            <div className={cn(
              "text-[10px] mt-1",
              metric.change > 0 ? "text-threat" : "text-success"
            )}>
              {metric.change > 0 ? "▲" : "▼"} {Math.abs(metric.change)} since yesterday
            </div>
          )}
        </motion.div>
      ))}
    </div>
  );
}
