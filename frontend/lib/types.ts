// ═══════════════════════════════════════════════════════════════
// Memex Frontend — TypeScript Interfaces
// Mirrors backend models.py Pydantic contracts
// ═══════════════════════════════════════════════════════════════

// --- AGUI State (from state_manager.py) ---

export interface RAMUsage {
  capacity: string;
  active_context: string[];
}

export interface AsyncJob {
  id: string;
  tool: string;
  eta_steps: number;
  status: "pending" | "ready" | "retrieved";
}

export interface AGUIState {
  ram_usage: RAMUsage;
  disk_storage: string[];
  async_jobs: AsyncJob[];
  kernel_directives: string[];
}

// --- Step Record (from demo_eval.py AGUIRecorder) ---

export interface StepRecord {
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

// --- Episode Meta (from demo_eval.py) ---

export interface EpisodeMeta {
  scenario: string;
  model: string;
  total_steps: number;
  final_score: number;
  ground_truth: {
    correct_decision: string;
    typology: string;
    key_entities: string[];
    excluded_entities?: string[];
    key_findings: string[];
  };
  alert: {
    alert_id: string;
    customer_id: string;
    summary: string;
    type: string;
    risk_score: number;
    date: string;
  };
  entity_count: number;
  transaction_count: number;
  steps_summary: {
    step: number;
    tool: string;
    reward: number | null;
    done: boolean;
  }[];
}

// --- Network Graph Types ---

export type NodeType = "person" | "entity" | "fund" | "decoy";
export type RiskLevel = "critical" | "high" | "medium" | "low";

export interface GraphNode {
  id: string;
  label: string;
  type: NodeType;
  risk: RiskLevel;
  x?: number;
  y?: number;
  fx?: number | null;
  fy?: number | null;
  vx?: number;
  vy?: number;
}

export interface GraphEdge {
  source: string | GraphNode;
  target: string | GraphNode;
  relationship: string;
  amount?: number;
  strength: "strong" | "moderate" | "weak";
}

// --- Entity Detail (for Flow C drawer) ---

export interface EntityProfile {
  customer_id: string;
  name: string;
  nationality: string;
  occupation: string;
  account_open_date?: string;
  risk_rating: string;
  pep_status?: boolean;
  pep_details?: string;
  address?: string;
  annual_income?: string;
  notes?: string;
  registration?: string;
  directors?: string[];
  registered_address?: string;
}

export interface Transaction {
  transaction_id: string;
  from: string;
  to: string;
  amount: number;
  currency: string;
  date: string;
  type: string;
  description: string;
  jurisdiction_from?: string;
  jurisdiction_to?: string;
}

export interface WatchlistResult {
  match: boolean;
  lists: string[];
  details: string;
}

// --- Triage Types ---

export interface CaseItem {
  id: string;
  alert_id: string;
  subject: string;
  summary: string;
  risk_score: number;
  risk_level: RiskLevel;
  typology: string;
  amount: string;
  jurisdictions: string[];
  date: string;
  status: "open" | "in-progress" | "resolved";
}

export interface GlobalMetric {
  label: string;
  value: number;
  change?: number;
  unit?: string;
  color: "accent" | "threat" | "success" | "warning";
}

// --- Zustand Store ---

export type AppFlow = "triage" | "investigation" | "entity-detail" | "terminal";

export interface MemexStore {
  // Current flow
  currentFlow: AppFlow;
  setFlow: (flow: AppFlow) => void;

  // Episode data
  episodeMeta: EpisodeMeta | null;
  steps: StepRecord[];
  currentStepIndex: number;

  // Live AGUI state
  aguiState: AGUIState;

  // Graph
  nodes: GraphNode[];
  edges: GraphEdge[];
  selectedNodeId: string | null;

  // Entity detail
  entityProfiles: Record<string, EntityProfile>;
  transactions: Transaction[];
  watchlistResults: Record<string, WatchlistResult>;

  // Playback
  isPlaying: boolean;
  playbackSpeed: number;

  // Page fault indicator
  pageFaultActive: boolean;

  // Actions
  loadEpisode: () => void;
  advanceStep: () => void;
  goToStep: (index: number) => void;
  selectNode: (id: string | null) => void;
  setPlaybackSpeed: (speed: number) => void;
  togglePlayback: () => void;
  triggerPageFault: () => void;
  injectContext: (data: string) => void;
}
