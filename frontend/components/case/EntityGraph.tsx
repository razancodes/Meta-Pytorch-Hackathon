'use client';

import { useEffect, useRef, useState, useCallback } from 'react';
import styles from './case.module.css';
import type { ReplayStep, GraphNode, GraphEdge } from '@/lib/types';
import { scenarioToGraph, riskToColor, nodeSize } from '@/lib/dataTransform';

// Mock scenario graph data from the 1MDB case
const MOCK_GRAPH_DATA = {
  nodes: [
    { id: 'CUST-1MDB-001', label: 'Taek Jho Lowe', type: 'person' as const, risk: 92, jurisdiction: 'Malaysia', flagged: true, pep: true },
    { id: 'ENT-GSTAR-001', label: 'Golden Star Holdings', type: 'shell' as const, risk: 88, jurisdiction: 'Seychelles', flagged: true },
    { id: 'ENT-ARBL-001', label: 'Arabella Investments', type: 'shell' as const, risk: 85, jurisdiction: 'BVI', flagged: false },
    { id: 'ENT-PETRA-001', label: 'PetraStar Energy Fund', type: 'company' as const, risk: 60, jurisdiction: 'Malaysia', flagged: false },
    { id: 'CUST-CHEN-002', label: 'Sarah Chen', type: 'person' as const, risk: 15, jurisdiction: 'Singapore', flagged: false },
    { id: 'real-estate-ny', label: 'Real Estate Holdings (NY)', type: 'asset' as const, risk: 40 },
    { id: 'art-dealer', label: 'Art Dealer (Geneva)', type: 'asset' as const, risk: 35 },
  ],
  edges: [
    { id: 'e1', source: 'ENT-PETRA-001', target: 'ENT-GSTAR-001', type: 'transaction' as const, label: '$681M', amount: 681000000, suspicious: true },
    { id: 'e2', source: 'ENT-GSTAR-001', target: 'ENT-ARBL-001', type: 'transaction' as const, label: '$260M', amount: 260000000, suspicious: true },
    { id: 'e3', source: 'ENT-ARBL-001', target: 'real-estate-ny', type: 'transaction' as const, label: '$250M', amount: 250000000, suspicious: true },
    { id: 'e4', source: 'ENT-ARBL-001', target: 'art-dealer', type: 'transaction' as const, label: '$135M', amount: 135000000, suspicious: true },
    { id: 'e5', source: 'ENT-GSTAR-001', target: 'CUST-1MDB-001', type: 'transaction' as const, label: '$30M', amount: 30000000, suspicious: true },
    { id: 'e6', source: 'ENT-ARBL-001', target: 'ENT-PETRA-001', type: 'transaction' as const, label: '$6M (reversal)', amount: 6000000, suspicious: false },
    { id: 'e7', source: 'CUST-1MDB-001', target: 'ENT-GSTAR-001', type: 'ownership' as const, label: 'Director' },
    { id: 'e8', source: 'ENT-GSTAR-001', target: 'ENT-ARBL-001', type: 'suspicious' as const, label: 'Shared Address' },
    { id: 'e9', source: 'CUST-1MDB-001', target: 'ENT-PETRA-001', type: 'association' as const, label: 'Consultant' },
  ],
};

const NODE_SHAPES: Record<string, string> = {
  person: 'ellipse',
  company: 'rectangle',
  account: 'diamond',
  transaction: 'triangle',
  shell: 'hexagon',
  jurisdiction: 'pentagon',
  asset: 'barrel',
};

const EDGE_COLORS: Record<string, string> = {
  ownership: '#8B5CF6',
  transaction: '#EA580C',
  association: '#404040',
  suspicious: '#E11D48',
  director: '#8B5CF6',
};

const getIconSvg = (type: string, color: string) => {
  let path = '';
  switch (type) {
    case 'person': path = 'M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z'; break;
    case 'company': path = 'M12 7V3H2v18h20V7H12zM6 19H4v-2h2v2zm0-4H4v-2h2v2zm0-4H4V9h2v2zm0-4H4V5h2v2zm4 12H8v-2h2v2zm0-4H8v-2h2v2zm0-4H8V9h2v2zm0-4H8V5h2v2zm10 12h-8v-2h2v-2h-2v-2h2v-2h-2V9h8v10zm-2-8h-2v2h2v-2zm0 4h-2v2h2v-2z'; break;
    case 'shell': path = 'M21 16.5c0 .38-.21.71-.53.88l-7.9 4.44c-.16.12-.36.18-.57.18-.21 0-.41-.06-.57-.18l-7.9-4.44A.991.991 0 0 1 3 16.5v-9c0-.38.21-.71.53-.88l7.9-4.44c.16-.12.36-.18.57-.18.21 0 .41.06.57.18l7.9 4.44c.32.17.53.5.53.88v9zM12 4.15L6.04 7.5 12 10.85l5.96-3.35L12 4.15zM5 15.91l6 3.38v-6.71L5 9.21v6.7zM19 15.91v-6.7l-6 3.37v6.71l6-3.38z'; break;
    case 'account': path = 'M4 10h3v7H4zM10.5 10h3v7h-3zM2 19h20v3H2zM17 10h3v7h-3zM12 1L2 6v2h20V6z'; break;
    case 'transaction': path = 'M12 4V1L8 5l4 4V6c3.31 0 6 2.69 6 6 0 1.01-.25 1.97-.7 2.8l1.46 1.46A7.93 7.93 0 0 0 20 12c0-4.42-3.58-8-8-8zm0 14c-3.31 0-6-2.69-6-6 0-1.01.25-1.97.7-2.8L5.24 7.74A7.93 7.93 0 0 0 4 12c0 4.42 3.58 8 8 8v3l4-4-4-4v3z'; break;
    case 'asset': path = 'M12 3L2 12h3v8h6v-6h2v6h6v-8h3L12 3zm5 15h-2v-6H9v6H7v-7.81l5-4.5 5 4.5V18z'; break;
    default: path = 'M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z'; break;
  }
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="${color}"><path d="${path}"/></svg>`;
  return `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`;
};

interface Props {
  currentStep: ReplayStep | null;
}

export default function EntityGraph({ currentStep }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const cyRef = useRef<any>(null);
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const [tooltipData, setTooltipData] = useState<{ node: GraphNode; x: number; y: number } | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;
    let mounted = true;

    const initCytoscape = async () => {
      const cytoscape = (await import('cytoscape')).default;
      const cola = (await import('cytoscape-cola')).default;

      if (!mounted || !containerRef.current) return;

      try {
        cytoscape.use(cola);
      } catch { /* already registered */ }

      const data = MOCK_GRAPH_DATA;

      const elements = [
        ...data.nodes.map(n => ({
          data: { id: n.id, label: n.label, type: n.type, risk: n.risk, jurisdiction: n.jurisdiction, flagged: n.flagged, pep: n.pep },
        })),
        ...data.edges.map(e => ({
          data: { id: e.id, source: e.source, target: e.target, type: e.type, label: e.label, amount: e.amount, suspicious: e.suspicious },
        })),
      ];

      const cy = cytoscape({
        container: containerRef.current,
        elements,
        style: [
          {
            selector: 'node',
            style: {
              'background-color': '#050505',
              'width': (ele: any) => nodeSize(ele.data('risk') || 30),
              'height': (ele: any) => nodeSize(ele.data('risk') || 30),
              'shape': (ele: any) => NODE_SHAPES[ele.data('type')] || 'ellipse',
              'label': 'data(label)',
              'color': '#D4D4D4',
              'font-size': '9px',
              'font-family': "'JetBrains Mono', monospace",
              'text-valign': 'bottom',
              'text-margin-y': 6,
              'text-outline-width': 4,
              'text-outline-color': '#000',
              'border-width': 2,
              'border-style': (ele: any) => ele.data('type') === 'shell' ? 'dashed' : 'solid',
              'border-color': (ele: any) => {
                if (ele.data('flagged')) return '#E11D48';
                return riskToColor(ele.data('risk') || 30);
              },
              'background-image': (ele: any) => getIconSvg(ele.data('type'), riskToColor(ele.data('risk') || 30)),
              'background-width': '50%',
              'background-height': '50%',
              'background-position-x': '50%',
              'background-position-y': '50%',
            } as any,
          },
          {
            selector: 'node:selected',
            style: {
              'border-width': 3,
              'border-color': '#EA580C',
              'overlay-color': '#EA580C',
              'overlay-opacity': 0.1,
            },
          },
          {
            selector: 'edge',
            style: {
              'width': (ele: any) => {
                const amt = ele.data('amount');
                if (amt) return Math.max(1, Math.min(4, Math.log10(amt / 100000)));
                return 1;
              },
              'line-color': (ele: any) => EDGE_COLORS[ele.data('type')] || '#404040',
              'target-arrow-color': (ele: any) => EDGE_COLORS[ele.data('type')] || '#404040',
              'target-arrow-shape': 'triangle',
              'curve-style': 'bezier',
              'label': 'data(label)',
              'font-size': '8px',
              'font-family': "'JetBrains Mono', monospace",
              'color': '#D4D4D4',
              'text-rotation': 'autorotate',
              'text-background-opacity': 1,
              'text-background-color': '#000',
              'text-background-padding': 4,
              'text-border-opacity': 1,
              'text-border-color': '#1A1A1A',
              'text-border-width': 1,
              'text-border-shape': 'roundrectangle',
              'control-point-step-size': 60,
            } as any,
          },
          {
            selector: 'edge[?suspicious]',
            style: {
              'line-color': '#E11D48',
              'target-arrow-color': '#E11D48',
              'line-style': 'solid',
              'width': 2.5,
            },
          },
          {
            selector: '.dimmed',
            style: { 'opacity': 0.12 },
          },
        ],
        layout: {
          name: 'circle',
          radius: 180,
          startAngle: 3 / 2 * Math.PI,
          animate: true,
          animationDuration: 800,
          fit: true,
          padding: 60,
        } as any,
      });

      cyRef.current = cy;

      cy.on('mouseover', 'node', (evt: any) => {
        const node = evt.target;
        cy.elements().addClass('dimmed');
        node.neighborhood().add(node).removeClass('dimmed');

        const pos = evt.renderedPosition;
        const data = node.data();
        
        let x = pos.x + 16;
        let y = pos.y - 10;
        
        if (containerRef.current) {
          const rect = containerRef.current.getBoundingClientRect();
          const tooltipWidth = 220; // approximate max width
          const tooltipHeight = 120; // approximate max height
          
          if (x + tooltipWidth > rect.width) {
            x = pos.x - tooltipWidth - 16;
          }
          if (y + 40 + tooltipHeight > rect.height) {
            y = rect.height - tooltipHeight - 50;
          }
        }

        setTooltipData({
          node: data as GraphNode,
          x,
          y,
        });
      });

      cy.on('mouseout', 'node', () => {
        cy.elements().removeClass('dimmed');
        setTooltipData(null);
      });

      cy.on('tap', 'node', (evt: any) => {
        const nodeData = evt.target.data() as GraphNode;
        setSelectedNode(prev => {
          if (prev?.id === nodeData.id) {
            evt.target.unselect(); // Explicitly remove focus ring
            return null; // Toggle off
          }
          return nodeData; // Select new
        });
      });

      // Click background to deselect
      cy.on('tap', (evt: any) => {
        if (evt.target === cy) {
          setSelectedNode(null);
          cy.elements().unselect(); // Explicitly remove all focus rings
        }
      });
    };

    initCytoscape();

    return () => {
      mounted = false;
      if (cyRef.current) {
        cyRef.current.destroy();
        cyRef.current = null;
      }
    };
  }, []);

  return (
    <div className={styles.graphContainer}>
      <div className="nx-panel-header">
        <span className="nx-panel-title">ENTITY GRAPH</span>
        <span className="nx-panel-badge nx-tag">{MOCK_GRAPH_DATA.nodes.length} NODES</span>
      </div>
      <div ref={containerRef} className={styles.graphCanvas} />

      {/* Floating tooltip */}
      {tooltipData && (
        <div className={styles.graphTooltip} style={{ left: tooltipData.x, top: tooltipData.y + 40 }}>
          <div className={styles.ttHeader}>
            <span className={styles.ttType}>{tooltipData.node.type?.toUpperCase()}</span>
            <span className={styles.ttRisk} style={{ color: riskToColor(tooltipData.node.risk || 0) }}>
              RISK {tooltipData.node.risk || 0}
            </span>
          </div>
          <div className={styles.ttName}>{tooltipData.node.label}</div>
          {tooltipData.node.jurisdiction && (
            <div className={styles.ttRow}><span>Jurisdiction</span><span>{tooltipData.node.jurisdiction}</span></div>
          )}
          {tooltipData.node.pep && (
            <div className={styles.ttFlag}>⚠ PEP — Politically Exposed Person</div>
          )}
          {tooltipData.node.flagged && !tooltipData.node.pep && (
            <div className={styles.ttFlag}>⚠ Flagged for review</div>
          )}
        </div>
      )}

      {/* Selected node detail tray */}
      {selectedNode && (
        <div className={styles.detailTray}>
          <div className={styles.detailHeader}>
            <span className={styles.detailTitle}>{selectedNode.label}</span>
            <button className="nx-btn" onClick={() => {
              setSelectedNode(null);
              cyRef.current?.elements().unselect();
            }}>✕</button>
          </div>
          <div className={styles.detailBody}>
            <div className={styles.detailRow}>
              <span className="nx-label">TYPE</span>
              <span>{selectedNode.type?.toUpperCase()}</span>
            </div>
            <div className={styles.detailRow}>
              <span className="nx-label">RISK SCORE</span>
              <span style={{ color: riskToColor(selectedNode.risk || 0) }}>{selectedNode.risk}</span>
            </div>
            {selectedNode.jurisdiction && (
              <div className={styles.detailRow}>
                <span className="nx-label">JURISDICTION</span>
                <span>{selectedNode.jurisdiction}</span>
              </div>
            )}
            {selectedNode.pep && (
              <div className={styles.detailFlag}>PEP — Politically Exposed Person</div>
            )}
            {selectedNode.flagged && (
              <div className={styles.detailFlag}>⚠ WATCHLIST HIT</div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
