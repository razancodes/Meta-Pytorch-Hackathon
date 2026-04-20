"use client";

import { useEffect, useRef } from "react";
import { motion } from "framer-motion";
import { THREAT_CLUSTERS } from "@/lib/demo-data";

export function ThreatHeatmap() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const w = rect.width;
    const h = rect.height;

    // Draw grid
    ctx.strokeStyle = "#1a1a1a";
    ctx.lineWidth = 0.5;
    const gridSize = 20;
    for (let x = 0; x < w; x += gridSize) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
      ctx.stroke();
    }
    for (let y = 0; y < h; y += gridSize) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }

    // Map lat/lng to canvas coords (simple Mercator-ish)
    const mapToCanvas = (lat: number, lng: number) => {
      const x = ((lng + 180) / 360) * w;
      const y = ((90 - lat) / 180) * h;
      return { x, y };
    };

    // Draw threat clusters
    let animFrame: number;
    let time = 0;

    const animate = () => {
      time += 0.02;

      // Clear only the cluster areas with slight fade
      ctx.fillStyle = "rgba(0, 0, 0, 0.08)";
      ctx.fillRect(0, 0, w, h);

      // Redraw grid
      ctx.strokeStyle = "#1a1a1a";
      ctx.lineWidth = 0.5;
      for (let x = 0; x < w; x += gridSize) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, h);
        ctx.stroke();
      }
      for (let y = 0; y < h; y += gridSize) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(w, y);
        ctx.stroke();
      }

      THREAT_CLUSTERS.forEach((cluster) => {
        const { x, y } = mapToCanvas(cluster.lat, cluster.lng);
        const pulseRadius = 6 + Math.sin(time * 2 + cluster.risk * 0.1) * 3;
        const alpha = cluster.risk / 100;

        // Outer glow
        const gradient = ctx.createRadialGradient(x, y, 0, x, y, pulseRadius * 3);
        if (cluster.risk > 80) {
          gradient.addColorStop(0, `rgba(225, 29, 72, ${alpha * 0.6})`);
          gradient.addColorStop(0.5, `rgba(225, 29, 72, ${alpha * 0.2})`);
          gradient.addColorStop(1, "rgba(225, 29, 72, 0)");
        } else if (cluster.risk > 60) {
          gradient.addColorStop(0, `rgba(234, 88, 12, ${alpha * 0.6})`);
          gradient.addColorStop(0.5, `rgba(234, 88, 12, ${alpha * 0.2})`);
          gradient.addColorStop(1, "rgba(234, 88, 12, 0)");
        } else {
          gradient.addColorStop(0, `rgba(245, 158, 11, ${alpha * 0.4})`);
          gradient.addColorStop(0.5, `rgba(245, 158, 11, ${alpha * 0.15})`);
          gradient.addColorStop(1, "rgba(245, 158, 11, 0)");
        }

        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(x, y, pulseRadius * 3, 0, Math.PI * 2);
        ctx.fill();

        // Core dot
        ctx.fillStyle = cluster.risk > 80 ? "#E11D48" : cluster.risk > 60 ? "#EA580C" : "#F59E0B";
        ctx.beginPath();
        ctx.arc(x, y, 3, 0, Math.PI * 2);
        ctx.fill();

        // Label
        ctx.fillStyle = "#D4D4D4";
        ctx.font = "8px 'JetBrains Mono'";
        ctx.textAlign = "left";
        ctx.fillText(`${cluster.id} [${cluster.risk}]`, x + 8, y + 3);
      });

      // Connection lines between high-risk clusters
      const highRisk = THREAT_CLUSTERS.filter(c => c.risk > 70);
      ctx.strokeStyle = "rgba(234, 88, 12, 0.12)";
      ctx.lineWidth = 0.5;
      ctx.setLineDash([3, 6]);
      for (let i = 0; i < highRisk.length; i++) {
        for (let j = i + 1; j < highRisk.length; j++) {
          const a = mapToCanvas(highRisk[i].lat, highRisk[i].lng);
          const b = mapToCanvas(highRisk[j].lat, highRisk[j].lng);
          ctx.beginPath();
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          ctx.stroke();
        }
      }
      ctx.setLineDash([]);

      animFrame = requestAnimationFrame(animate);
    };

    animate();

    return () => cancelAnimationFrame(animFrame);
  }, []);

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ delay: 0.4, duration: 0.6 }}
      className="rounded-[2px] border border-muted bg-surface overflow-hidden"
    >
      <div className="flex items-center justify-between px-3 py-2 border-b border-muted">
        <span className="text-[10px] uppercase tracking-[0.15em] font-bold text-accent">
          Threat Intelligence Map
        </span>
        <span className="text-[9px] text-text-dim">
          {THREAT_CLUSTERS.length} active clusters
        </span>
      </div>
      <div className="relative" style={{ height: "280px" }}>
        <canvas
          ref={canvasRef}
          className="w-full h-full"
          style={{ display: "block" }}
        />
        {/* Corner overlay labels */}
        <div className="absolute bottom-2 left-3 text-[8px] text-text-dim uppercase tracking-wider">
          Global Threat Distribution // Real-Time
        </div>
      </div>
    </motion.div>
  );
}
