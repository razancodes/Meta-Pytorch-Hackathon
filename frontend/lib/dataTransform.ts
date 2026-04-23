// ═══════════════════════════════════════════════════════════════
// MEMEX — Data Transform Utilities
// ═══════════════════════════════════════════════════════════════

import { GraphNode, GraphEdge, EntityNodeType, EdgeType } from './types';

// ── Scenario → Cytoscape Graph Elements ──────────────────────

interface ScenarioData {
  customer_profiles?: Record<string, Record<string, unknown>>;
  network_graph?: Record<string, { connections?: Array<Record<string, unknown>>; entity_name?: string }>;
  transactions?: Array<Record<string, unknown>>;
  watchlist_results?: Record<string, Record<string, unknown>>;
}

function inferNodeType(profile: Record<string, unknown>): EntityNodeType {
  const type = (profile.type as string || '').toLowerCase();
  const name = (profile.name as string || '').toLowerCase();
  const notes = (profile.notes as string || '').toLowerCase();

  if (type === 'individual' || profile.date_of_birth || profile.occupation || profile.nationality) return 'person';
  if (notes.includes('shell') || notes.includes('no employees') || notes.includes('no commercial')) return 'shell';
  if (type === 'business' || profile.business_type || profile.directors) return 'company';
  if (name.includes('account') || name.includes('acc-')) return 'account';
  return 'company';
}

function inferRisk(profile: Record<string, unknown>, watchlist?: Record<string, Record<string, unknown>>): number {
  let score = 30;
  const riskRating = (profile.risk_rating as string || '').toLowerCase();
  if (riskRating === 'high') score += 40;
  else if (riskRating === 'medium') score += 20;

  if (profile.pep_status) score += 25;
  
  const name = profile.name as string || '';
  if (watchlist && watchlist[name]?.hit) score += 20;

  const notes = (profile.notes as string || '').toLowerCase();
  if (notes.includes('shell') || notes.includes('no employees')) score += 15;
  if (notes.includes('shared') || notes.includes('nominee')) score += 10;

  return Math.min(score, 100);
}

export function scenarioToGraph(scenario: ScenarioData): { nodes: GraphNode[]; edges: GraphEdge[] } {
  const nodes: GraphNode[] = [];
  const edges: GraphEdge[] = [];
  const nodeIds = new Set<string>();

  const profiles = scenario.customer_profiles || {};
  const network = scenario.network_graph || {};
  const watchlist = scenario.watchlist_results || {};

  // Build nodes from profiles
  for (const [id, profile] of Object.entries(profiles)) {
    if (nodeIds.has(id)) continue;
    nodeIds.add(id);
    nodes.push({
      id,
      label: (profile.name as string) || id,
      type: inferNodeType(profile),
      risk: inferRisk(profile, watchlist),
      jurisdiction: (profile.nationality as string) || (profile.jurisdiction as string),
      flagged: !!(watchlist[(profile.name as string)]?.hit || watchlist[id]?.hit),
      pep: !!profile.pep_status,
      meta: profile,
    });
  }

  // Build edges from network graph
  const edgeIds = new Set<string>();
  for (const [entityId, node] of Object.entries(network)) {
    if (!node.connections) continue;

    // Ensure source node exists
    if (!nodeIds.has(entityId)) {
      nodeIds.add(entityId);
      nodes.push({
        id: entityId,
        label: entityId, // fallback
        type: entityId.startsWith('CUST') ? 'person' : 'company',
        risk: 40,
        flagged: false,
        pep: false,
      });
    }

    for (const conn of node.connections) {
      const targetId = (conn.entity_id as string) || (conn.entity as string) || '';
      if (!targetId) continue;

      const edgeId = `e-${entityId}-${targetId}`;
      const reverseId = `e-${targetId}-${entityId}`;
      if (edgeIds.has(edgeId) || edgeIds.has(reverseId)) continue;
      edgeIds.add(edgeId);

      // Ensure target node exists
      if (!nodeIds.has(targetId)) {
        nodeIds.add(targetId);
        const targetName = (conn.entity_name as string) || targetId;
        const isPep = !!(conn.pep || (conn.director as string || '').includes('PEP'));
        nodes.push({
          id: targetId,
          label: targetName,
          type: isPep ? 'person' : 'company',
          risk: isPep ? 80 : 40,
          flagged: isPep,
          pep: isPep,
        });
      }

      const rel = (conn.relationship as string || '').toLowerCase();
      let edgeType: EdgeType = 'association';
      if (rel.includes('wire') || rel.includes('transfer') || rel.includes('inbound') || rel.includes('outbound')) edgeType = 'transaction';
      else if (rel.includes('director') || rel.includes('owns') || rel.includes('owner')) edgeType = 'ownership';
      else if (rel.includes('shared') || rel.includes('address')) edgeType = 'suspicious';

      edges.push({
        id: edgeId,
        source: entityId,
        target: targetId,
        type: edgeType,
        label: conn.relationship as string,
        amount: conn.amount as number | undefined,
        suspicious: edgeType === 'suspicious' || !!(conn.note as string || '').includes('SHARED'),
      });
    }
  }

  // Build edges from transactions
  const txns = scenario.transactions || [];
  for (const txn of txns) {
    const from = (txn.from as string) || (txn.sender_id as string) || '';
    const to = (txn.to as string) || (txn.receiver_id as string) || '';
    if (!from || !to) continue;

    const edgeId = `txn-${txn.transaction_id}`;
    if (edgeIds.has(edgeId)) continue;
    edgeIds.add(edgeId);

    // Ensure nodes exist for external entities
    for (const [eid, ename] of [[from, from], [to, to]]) {
      if (!nodeIds.has(eid)) {
        nodeIds.add(eid);
        nodes.push({
          id: eid,
          label: ename,
          type: (ename as string).includes('ENT') ? 'company' : 'asset',
          risk: 20,
        });
      }
    }

    edges.push({
      id: edgeId,
      source: from,
      target: to,
      type: 'transaction',
      label: `$${formatCompact(txn.amount as number)}`,
      amount: txn.amount as number,
      suspicious: (txn.amount as number) > 100000,
    });
  }

  return { nodes, edges };
}

function formatCompact(n: number): string {
  if (!n) return '0';
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(0)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(0)}K`;
  return String(n);
}

// ── Risk Color ───────────────────────────────────────────────
export function riskToColor(score: number): string {
  if (score >= 80) return '#E11D48';
  if (score >= 60) return '#EA580C';
  if (score >= 40) return '#EAB308';
  return '#22C55E';
}

export function nodeSize(risk: number): number {
  return 24 + (risk / 100) * 24;
}
