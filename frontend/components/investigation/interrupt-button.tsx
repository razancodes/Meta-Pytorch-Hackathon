"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useMemexStore } from "@/lib/store";
import { cn } from "@/lib/utils";

export function InterruptButton() {
  const [isOpen, setIsOpen] = useState(false);
  const [input, setInput] = useState("");
  const injectContext = useMemexStore((s) => s.injectContext);

  const handleSubmit = () => {
    if (input.trim()) {
      injectContext(input.trim());
      setInput("");
      setIsOpen(false);
    }
  };

  return (
    <div className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className={cn(
          "w-full py-3 px-4 rounded-[2px] border border-accent bg-accent/10",
          "text-accent font-bold text-xs uppercase tracking-[0.15em]",
          "transition-all duration-200",
          "hover:bg-accent/20 hover:border-accent",
          "interrupt-glow",
          "cursor-pointer"
        )}
      >
        <div className="flex items-center justify-center gap-2">
          <span className="text-base">⚡</span>
          <span>INTERRUPT: Inject Context</span>
        </div>
      </button>

      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, y: -8, scaleY: 0.95 }}
            animate={{ opacity: 1, y: 0, scaleY: 1 }}
            exit={{ opacity: 0, y: -8, scaleY: 0.95 }}
            transition={{ duration: 0.2 }}
            className="absolute bottom-full left-0 right-0 mb-2 rounded-[2px] border border-accent bg-surface p-3 z-50"
          >
            <div className="text-[9px] uppercase tracking-[0.12em] text-accent font-bold mb-2">
              Human-in-the-Loop Context Injection
            </div>
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Enter compliance rule, evidence, or directive..."
              className={cn(
                "w-full h-20 bg-bg border border-muted rounded-[2px] p-2",
                "text-[11px] text-text placeholder:text-text-dim",
                "focus:outline-none focus:border-accent",
                "resize-none font-['JetBrains_Mono']"
              )}
            />
            <div className="flex gap-2 mt-2">
              <button
                onClick={handleSubmit}
                className={cn(
                  "flex-1 py-1.5 rounded-[2px] border border-accent bg-accent/20",
                  "text-accent text-[10px] font-bold uppercase tracking-wider",
                  "hover:bg-accent/30 transition-colors cursor-pointer"
                )}
              >
                Inject to Kernel
              </button>
              <button
                onClick={() => setIsOpen(false)}
                className={cn(
                  "px-3 py-1.5 rounded-[2px] border border-muted",
                  "text-text-dim text-[10px] uppercase tracking-wider",
                  "hover:border-text-dim transition-colors cursor-pointer"
                )}
              >
                Cancel
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
