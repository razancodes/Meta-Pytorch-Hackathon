"use client";

import { useEffect, useRef, useCallback } from "react";
import {
  forceSimulation,
  forceLink,
  forceManyBody,
  forceCenter,
  forceCollide,
  type SimulationNodeDatum,
  type SimulationLinkDatum,
} from "d3-force";
import { useMemexStore } from "@/lib/store";
import { getNodeColor, getEdgeColor } from "@/lib/utils";
import type { GraphNode, GraphEdge } from "@/lib/types";

interface SimNode extends SimulationNodeDatum {
  id: string;
  label: string;
  type: string;
  risk: string;
}

interface SimLink extends SimulationLinkDatum<SimNode> {
  relationship: string;
  amount?: number;
  strength: string;
}

export function NetworkGraph() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const simRef = useRef<ReturnType<typeof forceSimulation<SimNode>> | null>(null);
  const nodesRef = useRef<SimNode[]>([]);
  const linksRef = useRef<SimLink[]>([]);
  const hoveredRef = useRef<string | null>(null);
  const animRef = useRef<number>(0);
  const timeRef = useRef(0);

  const storeNodes = useMemexStore((s) => s.nodes);
  const storeEdges = useMemexStore((s) => s.edges);
  const selectNode = useMemexStore((s) => s.selectNode);
  const selectedNodeId = useMemexStore((s) => s.selectedNodeId);

  const draw = useCallback(() => {
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
    timeRef.current += 0.015;
    const t = timeRef.current;

    // Clear
    ctx.fillStyle = "#000000";
    ctx.fillRect(0, 0, w, h);

    // Grid
    ctx.strokeStyle = "#0d0d0d";
    ctx.lineWidth = 0.5;
    const gridSize = 30;
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

    // Draw edges
    linksRef.current.forEach((link) => {
      const source = link.source as SimNode;
      const target = link.target as SimNode;
      if (!source.x || !source.y || !target.x || !target.y) return;

      const color = getEdgeColor(link.strength as "strong" | "moderate" | "weak");
      ctx.strokeStyle = color;
      ctx.lineWidth = link.strength === "strong" ? 1.5 : 0.8;
      ctx.globalAlpha = 0.6;

      // Animated dash
      ctx.setLineDash([4, 4]);
      ctx.lineDashOffset = -t * 20;

      ctx.beginPath();
      ctx.moveTo(source.x, source.y);
      ctx.lineTo(target.x, target.y);
      ctx.stroke();

      ctx.setLineDash([]);
      ctx.globalAlpha = 1;

      // Edge label
      const mx = (source.x + target.x) / 2;
      const my = (source.y + target.y) / 2;
      ctx.fillStyle = "#404040";
      ctx.font = "7px 'JetBrains Mono'";
      ctx.textAlign = "center";
      ctx.fillText(link.relationship, mx, my - 4);
    });

    // Draw nodes
    nodesRef.current.forEach((node) => {
      if (!node.x || !node.y) return;

      const color = getNodeColor(node.risk as "critical" | "high" | "medium" | "low");
      const isHovered = hoveredRef.current === node.id;
      const isSelected = selectedNodeId === node.id;
      const radius = isHovered || isSelected ? 14 : 10;

      // Outer glow
      if (node.risk === "critical" || isSelected) {
        const pulse = Math.sin(t * 3) * 0.15 + 0.35;
        const glow = ctx.createRadialGradient(node.x, node.y, 0, node.x, node.y, radius * 2.5);
        glow.addColorStop(0, `${color}${Math.floor(pulse * 255).toString(16).padStart(2, "0")}`);
        glow.addColorStop(1, `${color}00`);
        ctx.fillStyle = glow;
        ctx.beginPath();
        ctx.arc(node.x, node.y, radius * 2.5, 0, Math.PI * 2);
        ctx.fill();
      }

      // Node shape
      if (node.type === "person") {
        // Circle for people
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(node.x, node.y, radius, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = isSelected ? "#D4D4D4" : color;
        ctx.lineWidth = isSelected ? 2 : 1;
        ctx.stroke();
      } else if (node.type === "fund") {
        // Diamond for funds
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.moveTo(node.x, node.y - radius);
        ctx.lineTo(node.x + radius, node.y);
        ctx.lineTo(node.x, node.y + radius);
        ctx.lineTo(node.x - radius, node.y);
        ctx.closePath();
        ctx.fill();
        ctx.strokeStyle = isSelected ? "#D4D4D4" : color;
        ctx.lineWidth = isSelected ? 2 : 1;
        ctx.stroke();
      } else if (node.type === "decoy") {
        // Small circle with X for decoys
        ctx.strokeStyle = color;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.arc(node.x, node.y, radius * 0.7, 0, Math.PI * 2);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(node.x - 4, node.y - 4);
        ctx.lineTo(node.x + 4, node.y + 4);
        ctx.moveTo(node.x + 4, node.y - 4);
        ctx.lineTo(node.x - 4, node.y + 4);
        ctx.stroke();
      } else {
        // Rounded rectangle for entities
        const s = radius;
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.moveTo(node.x - s, node.y - s + 2);
        ctx.lineTo(node.x - s + 2, node.y - s);
        ctx.lineTo(node.x + s - 2, node.y - s);
        ctx.lineTo(node.x + s, node.y - s + 2);
        ctx.lineTo(node.x + s, node.y + s - 2);
        ctx.lineTo(node.x + s - 2, node.y + s);
        ctx.lineTo(node.x - s + 2, node.y + s);
        ctx.lineTo(node.x - s, node.y + s - 2);
        ctx.closePath();
        ctx.fill();
        ctx.strokeStyle = isSelected ? "#D4D4D4" : color;
        ctx.lineWidth = isSelected ? 2 : 1;
        ctx.stroke();
      }

      // Label
      ctx.fillStyle = isHovered || isSelected ? "#FFFFFF" : "#D4D4D4";
      ctx.font = `${isHovered || isSelected ? "bold " : ""}9px 'JetBrains Mono'`;
      ctx.textAlign = "center";
      ctx.fillText(node.label, node.x, node.y + radius + 12);

      // ID tag
      ctx.fillStyle = "#404040";
      ctx.font = "7px 'JetBrains Mono'";
      ctx.fillText(node.id, node.x, node.y + radius + 21);
    });

    // HUD overlay
    ctx.fillStyle = "#404040";
    ctx.font = "8px 'JetBrains Mono'";
    ctx.textAlign = "left";
    ctx.fillText(`NODES: ${nodesRef.current.length}  EDGES: ${linksRef.current.length}`, 12, 16);
    ctx.textAlign = "right";
    ctx.fillText("FORCE-DIRECTED GRAPH // MEMEX", w - 12, 16);

    animRef.current = requestAnimationFrame(draw);
  }, [selectedNodeId]);

  // Initialize simulation
  useEffect(() => {
    const simNodes: SimNode[] = storeNodes.map((n) => ({ ...n }));
    const simLinks: SimLink[] = storeEdges.map((e) => ({
      source: typeof e.source === "string" ? e.source : (e.source as GraphNode).id,
      target: typeof e.target === "string" ? e.target : (e.target as GraphNode).id,
      relationship: e.relationship,
      amount: e.amount,
      strength: e.strength,
    }));

    nodesRef.current = simNodes;
    linksRef.current = simLinks as SimLink[];

    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();

    const sim = forceSimulation<SimNode>(simNodes)
      .force(
        "link",
        forceLink<SimNode, SimLink>(simLinks as SimLink[])
          .id((d) => d.id)
          .distance(120)
      )
      .force("charge", forceManyBody().strength(-350))
      .force("center", forceCenter(rect.width / 2, rect.height / 2))
      .force("collide", forceCollide(30))
      .alphaDecay(0.02);

    simRef.current = sim;

    sim.on("tick", () => {
      /* draw loop handles rendering */
    });

    animRef.current = requestAnimationFrame(draw);

    return () => {
      sim.stop();
      cancelAnimationFrame(animRef.current);
    };
  }, [storeNodes, storeEdges, draw]);

  // Click handler
  const handleClick = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;

      let clicked: string | null = null;
      for (const node of nodesRef.current) {
        if (!node.x || !node.y) continue;
        const dx = x - node.x;
        const dy = y - node.y;
        if (Math.sqrt(dx * dx + dy * dy) < 35) {
          clicked = node.id;
          break;
        }
      }
      selectNode(clicked);
    },
    [selectNode]
  );

  // Hover handler
  const handleMouseMove = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;

      let found: string | null = null;
      for (const node of nodesRef.current) {
        if (!node.x || !node.y) continue;
        const dx = x - node.x;
        const dy = y - node.y;
        if (Math.sqrt(dx * dx + dy * dy) < 35) {
          found = node.id;
          break;
        }
      }
      hoveredRef.current = found;
      canvas.style.cursor = found ? "pointer" : "default";
    },
    []
  );

  return (
    <div className="w-full h-full relative rounded-[2px] border border-muted overflow-hidden bg-bg">
      <canvas
        ref={canvasRef}
        className="w-full h-full"
        style={{ display: "block" }}
        onClick={handleClick}
        onMouseMove={handleMouseMove}
      />
      {/* Legend */}
      <div className="absolute bottom-3 left-3 flex gap-4 text-[8px] uppercase tracking-wider text-text-dim">
        <span className="flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-full bg-threat" /> Critical
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-full bg-accent" /> High
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-full bg-warning" /> Medium
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-full bg-muted" /> Low
        </span>
      </div>
    </div>
  );
}
