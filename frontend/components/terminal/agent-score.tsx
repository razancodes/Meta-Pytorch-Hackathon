"use client";

import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { useMemexStore } from "@/lib/store";
import { cn } from "@/lib/utils";

export function AgentScore() {
  const episodeMeta = useMemexStore((s) => s.episodeMeta);
  const [displayScore, setDisplayScore] = useState(0);

  const finalScore = episodeMeta?.final_score ?? 0;

  useEffect(() => {
    const duration = 2000;
    const fps = 60;
    const totalFrames = (duration / 1000) * fps;
    let frame = 0;

    const timer = setInterval(() => {
      frame++;
      const progress = frame / totalFrames;
      // Ease out cubic
      const eased = 1 - Math.pow(1 - progress, 3);
      setDisplayScore(eased * finalScore);

      if (frame >= totalFrames) {
        setDisplayScore(finalScore);
        clearInterval(timer);
      }
    }, 1000 / fps);

    return () => clearInterval(timer);
  }, [finalScore]);

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.8, ease: "easeOut" }}
      className="text-center py-8"
    >
      <div className="text-[10px] uppercase tracking-[0.2em] text-text-dim mb-2">
        Terminal Reward Score
      </div>
      <div
        className={cn(
          "text-7xl font-bold tabular-nums",
          finalScore > 0 ? "text-success text-glow-accent" : "text-threat"
        )}
        style={{
          textShadow: finalScore > 0
            ? "0 0 40px rgba(34, 197, 94, 0.4), 0 0 80px rgba(34, 197, 94, 0.2)"
            : "0 0 40px rgba(225, 29, 72, 0.4)",
        }}
      >
        {displayScore >= 0 ? "+" : ""}{displayScore.toFixed(2)}
      </div>
      <div className="text-[10px] text-text-dim mt-2 uppercase tracking-wider">
        Case Resolved Successfully
      </div>
    </motion.div>
  );
}
