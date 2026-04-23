// ═══════════════════════════════════════════════════════════════
// MEMEX — Replay Engine & Mock Data
// ═══════════════════════════════════════════════════════════════

import { ReplayStep, EpisodeMeta, AGUIState } from './types';

// ── Mock Replay Data (1MDB scenario — used when no demo_output exists) ──

const MOCK_AGUI_STATES: AGUIState[] = [
  {
    ram_usage: { capacity: '1/2 observations', active_context: ['Alert ALERT-2024-1MDB-7701: Suspicious international wire transfers totaling $681M'] },
    disk_storage: [],
    async_jobs: [],
    kernel_directives: ['You are an AML compliance investigator. Gather evidence, assess risk, and decide: file_sar or close_alert.'],
  },
  {
    ram_usage: { capacity: '2/2 observations', active_context: ['Alert ALERT-2024-1MDB-7701: Suspicious international wire transfers totaling $681M', 'Step 1 [review_alert]: Alert details retrieved.'] },
    disk_storage: [],
    async_jobs: [],
    kernel_directives: ['You are an AML compliance investigator. Gather evidence, assess risk, and decide: file_sar or close_alert.'],
  },
  {
    ram_usage: { capacity: '2/2 observations', active_context: ['Step 1 [review_alert]: Alert details retrieved.', 'Step 2 [get_customer_profile]: Customer profile retrieved for CUST-1MDB-001.'] },
    disk_storage: [],
    async_jobs: [],
    kernel_directives: ['You are an AML compliance investigator. Gather evidence, assess risk, and decide: file_sar or close_alert.'],
  },
  {
    ram_usage: { capacity: '2/2 observations', active_context: ['Step 2 [get_customer_profile]: Customer profile retrieved.', 'Step 3 [write_to_case_file]: Data paged to case file.'] },
    disk_storage: ['Subject is PEP, connected to Minister of Finance. High risk.'],
    async_jobs: [],
    kernel_directives: ['You are an AML compliance investigator. Gather evidence, assess risk, and decide: file_sar or close_alert.'],
  },
  {
    ram_usage: { capacity: '2/2 observations', active_context: ['Step 3 [write_to_case_file]: Data paged to case file.', 'Step 4 [query_transactions]: Found 8 transaction(s) matching your query.'] },
    disk_storage: ['Subject is PEP, connected to Minister of Finance. High risk.'],
    async_jobs: [],
    kernel_directives: ['You are an AML compliance investigator. Gather evidence, assess risk, and decide: file_sar or close_alert.'],
  },
  {
    ram_usage: { capacity: '2/2 observations', active_context: ['Step 4 [query_transactions]: Found 8 transactions.', 'Step 5 [search_compliance_manual]: Found 3 compliance rule(s).'] },
    disk_storage: ['Subject is PEP, connected to Minister of Finance. High risk.'],
    async_jobs: [],
    kernel_directives: ['You are an AML compliance investigator. Gather evidence, assess risk, and decide: file_sar or close_alert.'],
  },
  {
    ram_usage: { capacity: '2/2 observations', active_context: ['Step 5 [search_compliance_manual]: Found 3 compliance rules.', 'Step 6 [update_system_prompt]: Kernel directive injected.'] },
    disk_storage: ['Subject is PEP, connected to Minister of Finance. High risk.'],
    async_jobs: [],
    kernel_directives: ['You are an AML compliance investigator. Gather evidence, assess risk, and decide: file_sar or close_alert.', 'Added (Step 6): Apply enhanced due diligence for PEP with offshore transfers'],
  },
  {
    ram_usage: { capacity: '2/2 observations', active_context: ['Step 6 [update_system_prompt]: Kernel directive injected.', 'Step 7 [trace_network]: Network trace for ENT-GSTAR-001 (depth 1): 2 connections found.'] },
    disk_storage: ['Subject is PEP, connected to Minister of Finance. High risk.'],
    async_jobs: [],
    kernel_directives: ['You are an AML compliance investigator. Gather evidence, assess risk, and decide: file_sar or close_alert.', 'Added (Step 6): Apply enhanced due diligence for PEP with offshore transfers'],
  },
  {
    ram_usage: { capacity: '2/2 observations', active_context: ['Step 7 [trace_network]: 2 connections found.', 'Step 8 [write_to_case_file]: Data paged to case file.'] },
    disk_storage: ['Subject is PEP, connected to Minister of Finance. High risk.', 'Golden Star and Arabella share registered address in Seychelles. Classic layering.'],
    async_jobs: [],
    kernel_directives: ['You are an AML compliance investigator. Gather evidence, assess risk, and decide: file_sar or close_alert.', 'Added (Step 6): Apply enhanced due diligence for PEP with offshore transfers'],
  },
  {
    ram_usage: { capacity: '2/2 observations', active_context: ['Step 8 [write_to_case_file]: Data paged.', 'Step 9 [check_watchlist]: Watchlist check — HIT.'] },
    disk_storage: ['Subject is PEP, connected to Minister of Finance. High risk.', 'Golden Star and Arabella share registered address in Seychelles. Classic layering.'],
    async_jobs: [],
    kernel_directives: ['You are an AML compliance investigator. Gather evidence, assess risk, and decide: file_sar or close_alert.', 'Added (Step 6): Apply enhanced due diligence for PEP with offshore transfers'],
  },
  {
    ram_usage: { capacity: '2/2 observations', active_context: ['Step 9 [check_watchlist]: HIT.', 'Step 10 [check_source_of_funds]: Source NOT verified.'] },
    disk_storage: ['Subject is PEP, connected to Minister of Finance. High risk.', 'Golden Star and Arabella share registered address in Seychelles. Classic layering.'],
    async_jobs: [],
    kernel_directives: ['You are an AML compliance investigator. Gather evidence, assess risk, and decide: file_sar or close_alert.', 'Added (Step 6): Apply enhanced due diligence for PEP with offshore transfers'],
  },
  {
    ram_usage: { capacity: '2/2 observations', active_context: ['Step 10 [check_source_of_funds]: NOT verified.', 'Step 11 [request_wire_trace]: Wire trace enqueued as REQ-001 (ETA: 3 steps).'] },
    disk_storage: ['Subject is PEP, connected to Minister of Finance. High risk.', 'Golden Star and Arabella share registered address in Seychelles. Classic layering.'],
    async_jobs: [{ id: 'REQ-001', tool: 'request_wire_trace', eta_steps: 3, status: 'pending' }],
    kernel_directives: ['You are an AML compliance investigator. Gather evidence, assess risk, and decide: file_sar or close_alert.', 'Added (Step 6): Apply enhanced due diligence for PEP with offshore transfers'],
  },
  {
    ram_usage: { capacity: '2/2 observations', active_context: ['Step 11 [request_wire_trace]: Enqueued REQ-001.', 'Step 12 [get_customer_profile]: Profile retrieved for ENT-ARBL-001.'] },
    disk_storage: ['Subject is PEP, connected to Minister of Finance. High risk.', 'Golden Star and Arabella share registered address in Seychelles. Classic layering.'],
    async_jobs: [{ id: 'REQ-001', tool: 'request_wire_trace', eta_steps: 2, status: 'pending' }],
    kernel_directives: ['You are an AML compliance investigator. Gather evidence, assess risk, and decide: file_sar or close_alert.', 'Added (Step 6): Apply enhanced due diligence for PEP with offshore transfers'],
  },
  {
    ram_usage: { capacity: '2/2 observations', active_context: ['Step 12 [get_customer_profile]: Arabella profile.', 'Step 13 [check_source_of_funds]: $30M consulting fee — backdated contract.'] },
    disk_storage: ['Subject is PEP, connected to Minister of Finance. High risk.', 'Golden Star and Arabella share registered address in Seychelles. Classic layering.'],
    async_jobs: [{ id: 'REQ-001', tool: 'request_wire_trace', eta_steps: 1, status: 'pending' }],
    kernel_directives: ['You are an AML compliance investigator. Gather evidence, assess risk, and decide: file_sar or close_alert.', 'Added (Step 6): Apply enhanced due diligence for PEP with offshore transfers'],
  },
  {
    ram_usage: { capacity: '2/2 observations', active_context: ['Step 13 [check_source_of_funds]: Backdated contract.', 'Step 14 [retrieve_async_result]: Wire trace completed.'] },
    disk_storage: ['Subject is PEP, connected to Minister of Finance. High risk.', 'Golden Star and Arabella share registered address in Seychelles. Classic layering.'],
    async_jobs: [{ id: 'REQ-001', tool: 'request_wire_trace', eta_steps: 0, status: 'retrieved' }],
    kernel_directives: ['You are an AML compliance investigator. Gather evidence, assess risk, and decide: file_sar or close_alert.', 'Added (Step 6): Apply enhanced due diligence for PEP with offshore transfers'],
  },
];

const MOCK_STEPS: ReplayStep[] = [
  { step_number: 1, timestamp: Date.now() / 1000, action: { tool: 'review_alert', parameters: {} }, reasoning: 'Begin by reviewing the alert details to understand the scope of suspicious activity.', observation: { tool_result: { alert: { alert_id: 'ALERT-2024-1MDB-7701' } }, message: 'Alert details retrieved. Review the alert summary and begin investigation.', done: false, reward: 0.01 }, agui_state: MOCK_AGUI_STATES[1] },
  { step_number: 2, timestamp: Date.now() / 1000 + 2, action: { tool: 'get_customer_profile', parameters: { customer_id: 'CUST-1MDB-001' } }, reasoning: 'Profile the primary subject — need KYC data and PEP status.', observation: { tool_result: { customer_profile: { name: 'Taek Jho Lowe', pep_status: true } }, message: 'Customer profile retrieved for CUST-1MDB-001.', done: false, reward: 0.01 }, agui_state: MOCK_AGUI_STATES[2] },
  { step_number: 3, timestamp: Date.now() / 1000 + 4, action: { tool: 'write_to_case_file', parameters: { note: 'Subject is PEP, connected to Minister of Finance. High risk.' } }, reasoning: 'PEP connection is critical — save to disk before RAM eviction.', observation: { tool_result: { written: true, disk_entry_count: 1 }, message: 'Data paged to case file. Total disk entries: 1.', done: false, reward: 0.08 }, agui_state: MOCK_AGUI_STATES[3] },
  { step_number: 4, timestamp: Date.now() / 1000 + 6, action: { tool: 'query_transactions', parameters: { customer_id: 'CUST-1MDB-001' } }, reasoning: 'Analyze transaction history — looking for layering patterns.', observation: { tool_result: { transactions: [], count: 8 }, message: 'Found 8 transaction(s) matching your query.', done: false, reward: 0.01 }, agui_state: MOCK_AGUI_STATES[4] },
  { step_number: 5, timestamp: Date.now() / 1000 + 8, action: { tool: 'search_compliance_manual', parameters: { query: 'layering wire transfer offshore' } }, reasoning: 'Look up compliance rules for offshore layering patterns.', observation: { tool_result: { results: [{ title: 'Wire Transfer Due Diligence' }], count: 3 }, message: 'Found 3 compliance rule(s).', done: false, reward: 0.01 }, agui_state: MOCK_AGUI_STATES[5] },
  { step_number: 6, timestamp: Date.now() / 1000 + 10, action: { tool: 'update_system_prompt', parameters: { directive: 'Apply enhanced due diligence for PEP with offshore transfers' } }, reasoning: 'Inject PEP due diligence rules into kernel.', observation: { tool_result: { injected: true, directive_count: 2 }, message: 'Kernel directive injected. Total directives: 2.', done: false, reward: 0.13 }, agui_state: MOCK_AGUI_STATES[6] },
  { step_number: 7, timestamp: Date.now() / 1000 + 12, action: { tool: 'trace_network', parameters: { entity_id: 'ENT-GSTAR-001' } }, reasoning: 'Trace Golden Star Holdings — need to find shell company connections.', observation: { tool_result: { network: { connections: [{ entity: 'ENT-ARBL-001', relationship: 'Shared registered address' }] } }, message: 'Network trace: 2 connections found.', done: false, reward: 0.01 }, agui_state: MOCK_AGUI_STATES[7] },
  { step_number: 8, timestamp: Date.now() / 1000 + 14, action: { tool: 'write_to_case_file', parameters: { note: 'Golden Star and Arabella share registered address in Seychelles. Classic layering.' } }, reasoning: 'Persist network findings to disk — shared address is key evidence.', observation: { tool_result: { written: true, disk_entry_count: 2 }, message: 'Data paged to case file. Total disk entries: 2.', done: false, reward: 0.08 }, agui_state: MOCK_AGUI_STATES[8] },
  { step_number: 9, timestamp: Date.now() / 1000 + 16, action: { tool: 'check_watchlist', parameters: { entity: 'CUST-1MDB-001' } }, reasoning: 'Check subject against sanctions lists for PEP confirmation.', observation: { tool_result: { watchlist_result: { hit: true, lists: ['PEP'] } }, message: 'Watchlist check — HIT: PEP.', done: false, reward: 0.01 }, agui_state: MOCK_AGUI_STATES[9] },
  { step_number: 10, timestamp: Date.now() / 1000 + 18, action: { tool: 'check_source_of_funds', parameters: { transaction_id: 'TXN-1MDB-001' } }, reasoning: 'Verify source of the $681M wire — expecting unverified.', observation: { tool_result: { source_of_funds: { verified: false } }, message: 'Source of funds: NOT verified.', done: false, reward: 0.01 }, agui_state: MOCK_AGUI_STATES[10] },
  { step_number: 11, timestamp: Date.now() / 1000 + 20, action: { tool: 'request_wire_trace', parameters: { transaction_id: 'TXN-1MDB-002' } }, reasoning: 'Request async trace on the $260M intercompany transfer.', observation: { tool_result: { job_id: 'REQ-001', status: 'pending', eta_steps: 3 }, message: 'Wire trace enqueued as REQ-001 (ETA: 3 steps).', done: false, reward: -0.02 }, agui_state: MOCK_AGUI_STATES[11] },
  { step_number: 12, timestamp: Date.now() / 1000 + 22, action: { tool: 'get_customer_profile', parameters: { customer_id: 'ENT-ARBL-001' } }, reasoning: 'Profile the second shell company while waiting for async result.', observation: { tool_result: { customer_profile: { name: 'Arabella Investments PJS Ltd', jurisdiction: 'BVI' } }, message: 'Customer profile retrieved for ENT-ARBL-001.', done: false, reward: 0.01 }, agui_state: MOCK_AGUI_STATES[12] },
  { step_number: 13, timestamp: Date.now() / 1000 + 24, action: { tool: 'check_source_of_funds', parameters: { transaction_id: 'TXN-1MDB-005' } }, reasoning: 'Check the $30M consulting fee — looking for backdated contracts.', observation: { tool_result: { source_of_funds: { source: 'Golden Star Holdings Ltd', documentation: 'Consulting agreement (backdated)' } }, message: 'Source of funds: backdated contract found.', done: false, reward: 0.01 }, agui_state: MOCK_AGUI_STATES[13] },
  { step_number: 14, timestamp: Date.now() / 1000 + 26, action: { tool: 'retrieve_async_result', parameters: { job_id: 'REQ-001' } }, reasoning: 'Retrieve completed wire trace — ETA should be 0 now.', observation: { tool_result: { wire_trace: { trace_target: 'TXN-1MDB-002', intermediary_count: 2 } }, message: 'Async job REQ-001 retrieved successfully.', done: false, reward: -0.02 }, agui_state: MOCK_AGUI_STATES[14] },
  { step_number: 15, timestamp: Date.now() / 1000 + 28, action: { tool: 'file_sar', parameters: { typology: 'layering', entities_involved: ['CUST-1MDB-001', 'ENT-GSTAR-001', 'ENT-ARBL-001'], findings: ['pep_connection', 'offshore_source', 'shared_registered_address', 'rapid_fan_out', 'no_source_documentation', 'reversed_transaction'] } }, reasoning: 'File SAR with complete evidence chain — all six findings identified.', observation: { tool_result: { sar_filed: true, final_score: 1.01 }, message: 'SAR filed successfully. Final episode score: +1.0100.', done: true, reward: 1.01 }, agui_state: MOCK_AGUI_STATES[14] },
];

const MOCK_META: EpisodeMeta = {
  scenario: '1MDB-Inspired Sovereign Wealth Fund Investigation',
  model: 'scripted',
  total_steps: 15,
  final_score: 1.01,
  ground_truth: {
    correct_decision: 'file_sar',
    typology: 'layering',
    key_entities: ['CUST-1MDB-001', 'ENT-GSTAR-001', 'ENT-ARBL-001'],
    excluded_entities: ['CUST-CHEN-002'],
    key_findings: ['pep_connection', 'offshore_source', 'shared_registered_address', 'rapid_fan_out', 'no_source_documentation', 'reversed_transaction'],
  },
  alert: {
    alert_id: 'ALERT-2024-1MDB-7701',
    customer_id: 'CUST-1MDB-001',
    summary: 'Suspicious international wire transfers totaling $681M from PetraStar Energy Fund through offshore shell entities.',
  },
  entity_count: 5,
  transaction_count: 8,
  steps_summary: MOCK_STEPS.map(s => ({ step: s.step_number, tool: s.action.tool, reward: s.observation.reward, done: s.observation.done })),
};

// ── Replay Engine ────────────────────────────────────────────

type ReplayListener = (step: ReplayStep, index: number) => void;

export class ReplayEngine {
  private steps: ReplayStep[] = [];
  private meta: EpisodeMeta | null = null;
  private currentIndex: number = -1;
  private playing: boolean = false;
  private speed: number = 1;
  private timer: ReturnType<typeof setTimeout> | null = null;
  private listeners: Set<ReplayListener> = new Set();
  private baseDelay: number = 2000; // ms per step

  async load(): Promise<void> {
    // Try loading from public/demo_output first
    try {
      const metaRes = await fetch('/demo_output/episode_meta.json');
      if (metaRes.ok) {
        this.meta = await metaRes.json();
        const totalSteps = this.meta!.total_steps;
        const stepPromises = [];
        for (let i = 1; i <= totalSteps; i++) {
          stepPromises.push(
            fetch(`/demo_output/step_${String(i).padStart(3, '0')}.json`)
              .then(r => r.ok ? r.json() : null)
          );
        }
        const results = await Promise.all(stepPromises);
        this.steps = results.filter(Boolean);
        if (this.steps.length > 0) return;
      }
    } catch { /* fall through to mock */ }

    // Fall back to mock data
    this.steps = MOCK_STEPS;
    this.meta = MOCK_META;
  }

  getMeta(): EpisodeMeta | null { return this.meta; }
  getSteps(): ReplayStep[] { return this.steps; }
  getCurrentIndex(): number { return this.currentIndex; }
  getCurrentStep(): ReplayStep | null { return this.currentIndex >= 0 ? this.steps[this.currentIndex] : null; }
  getTotalSteps(): number { return this.steps.length; }
  isPlaying(): boolean { return this.playing; }
  getSpeed(): number { return this.speed; }

  subscribe(listener: ReplayListener): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  private emit(step: ReplayStep, index: number) {
    this.listeners.forEach(fn => fn(step, index));
  }

  play() {
    if (this.currentIndex < 0) this.currentIndex = 0;
    this.playing = true;
    this.scheduleNext();
  }

  pause() {
    this.playing = false;
    if (this.timer) { clearTimeout(this.timer); this.timer = null; }
  }

  setSpeed(multiplier: number) {
    this.speed = multiplier;
    if (this.playing) {
      if (this.timer) clearTimeout(this.timer);
      this.scheduleNext();
    }
  }

  stepForward(): ReplayStep | null {
    if (this.currentIndex < this.steps.length - 1) {
      this.currentIndex++;
      const step = this.steps[this.currentIndex];
      this.emit(step, this.currentIndex);
      return step;
    }
    return null;
  }

  stepBack(): ReplayStep | null {
    if (this.currentIndex > 0) {
      this.currentIndex--;
      const step = this.steps[this.currentIndex];
      this.emit(step, this.currentIndex);
      return step;
    }
    return null;
  }

  seekTo(index: number): ReplayStep | null {
    if (index >= 0 && index < this.steps.length) {
      this.currentIndex = index;
      const step = this.steps[this.currentIndex];
      this.emit(step, this.currentIndex);
      return step;
    }
    return null;
  }

  reset() {
    this.pause();
    this.currentIndex = -1;
  }

  private scheduleNext() {
    if (!this.playing || this.currentIndex >= this.steps.length - 1) {
      this.playing = false;
      return;
    }
    const delay = this.baseDelay / this.speed;
    this.timer = setTimeout(() => {
      this.stepForward();
      this.scheduleNext();
    }, delay);
  }
}
