// ═══════════════════════════════════════════════════════════════
// MEMEX — Type Definitions
// ═══════════════════════════════════════════════════════════════

// ── AGUI State (from backend models.py) ──────────────────────
export interface RAMUsage {
  capacity: string;
  active_context: string[];
}

export interface AGUIState {
  ram_usage: RAMUsage;
  disk_storage: string[];
  async_jobs: AsyncJob[];
  kernel_directives: string[];
}

export interface AsyncJob {
  id: string;
  tool: string;
  eta_steps: number;
  status: 'pending' | 'ready' | 'retrieved';
}

// ── Observation (from /step response) ────────────────────────
export interface Observation {
  tool_result: Record<string, unknown>;
  available_tools: string[];
  message: string;
  done: boolean;
  metadata: Record<string, unknown> & {
    agui_state?: AGUIState;
    step?: number;
    episode_id?: string;
    page_fault?: boolean;
    eviction_occurred?: boolean;
  };
}

export interface StepResponse {
  observation: Observation;
  reward: number | null;
  done: boolean;
}

// ── Replay Step ──────────────────────────────────────────────
export interface ReplayStep {
  step_number: number;
  timestamp: number;
  action: {
    tool: string;
    parameters: Record<string, unknown>;
  };
  reasoning: string;
  observation: {
    tool_result: Record<string, unknown>;
    message: string;
    done: boolean;
    reward: number | null;
  };
  agui_state: AGUIState;
}

export interface EpisodeMeta {
  scenario: string;
  model: string;
  total_steps: number;
  final_score: number;
  ground_truth: GroundTruth;
  alert: Alert;
  entity_count: number;
  transaction_count: number;
  steps_summary: {
    step: number;
    tool: string;
    reward: number | null;
    done: boolean;
  }[];
}

// ── Scenario Data ────────────────────────────────────────────
export interface Alert {
  alert_id: string;
  customer_id: string;
  summary: string;
  type?: string;
  risk_score?: number;
  date?: string;
}

export interface GroundTruth {
  correct_decision: string;
  typology: string;
  key_entities: string[];
  excluded_entities: string[];
  key_findings: string[];
}

export interface CustomerProfile {
  customer_id: string;
  name: string;
  nationality?: string;
  occupation?: string;
  risk_rating?: string;
  pep_status?: boolean;
  pep_details?: string;
  jurisdiction?: string;
  registered_address?: string;
  directors?: string[];
  notes?: string;
  [key: string]: unknown;
}

export interface Transaction {
  transaction_id: string;
  from?: string;
  to?: string;
  sender_id?: string;
  receiver_id?: string;
  customer_id?: string;
  amount: number;
  currency: string;
  date: string;
  type: string;
  description?: string;
  jurisdiction_from?: string;
  jurisdiction_to?: string;
}

export interface NetworkConnection {
  entity?: string;
  entity_id?: string;
  entity_name?: string;
  relationship: string;
  strength?: string;
  amount?: number;
  pep?: boolean;
  note?: string;
  registered_address?: string;
  director?: string;
}

export interface NetworkNode {
  entity_id: string;
  entity_name?: string;
  connections: NetworkConnection[];
  note?: string;
}

// ── Map Types ────────────────────────────────────────────────
export interface MLHub {
  id: string;
  name: string;
  lat: number;
  lng: number;
  riskLevel: 'low' | 'medium' | 'high' | 'critical';
  entityCount: number;
  region: string;
}

export interface MLCorridor {
  id: string;
  source: string; // hub ID
  target: string; // hub ID
  volume: number;
  riskScore: number;
  typology: string;
  caseId?: string;
  label?: string;
}

// ── Graph Types (Cytoscape) ──────────────────────────────────
export type EntityNodeType = 'person' | 'company' | 'account' | 'transaction' | 'shell' | 'jurisdiction' | 'asset';
export type EdgeType = 'ownership' | 'transaction' | 'association' | 'suspicious' | 'director';

export interface GraphNode {
  id: string;
  label: string;
  type: EntityNodeType;
  risk: number;
  jurisdiction?: string;
  flagged?: boolean;
  pep?: boolean;
  totalVolume?: number;
  meta?: Record<string, unknown>;
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  type: EdgeType;
  label?: string;
  amount?: number;
  suspicious?: boolean;
}

// ── Environment Mode ─────────────────────────────────────────
export type AppMode = 'live' | 'replay';
export type ViewState = 'globe' | 'transitioning' | 'case';

export interface CaseContext {
  caseId: string;
  hubId?: string;
  corridorId?: string;
  scenario?: Record<string, unknown>;
}
