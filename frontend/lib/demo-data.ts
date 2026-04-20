// ═══════════════════════════════════════════════════════════════
// Memex Frontend — Static Demo Data
// Hardcoded data for the Triage page + 1MDB Investigation
// ═══════════════════════════════════════════════════════════════

import type {
  CaseItem,
  GlobalMetric,
  EntityProfile,
  Transaction,
  WatchlistResult,
  GraphNode,
  GraphEdge,
  StepRecord,
  EpisodeMeta,
  AGUIState,
} from "./types";

// --- Global Triage Metrics ---

export const GLOBAL_METRICS: GlobalMetric[] = [
  { label: "Active Cases", value: 147, change: 12, color: "accent" },
  { label: "Critical Alerts", value: 23, change: 3, color: "threat" },
  { label: "Jurisdictions Flagged", value: 9, color: "warning" },
  { label: "Model Accuracy", value: 94.7, unit: "%", color: "success" },
];

// --- Threat Heatmap Data ---

export const THREAT_CLUSTERS = [
  { id: "MY", name: "Malaysia", lat: 4.2, lng: 101.9, risk: 92, cases: 8 },
  { id: "SC", name: "Seychelles", lat: -4.6, lng: 55.4, risk: 88, cases: 5 },
  { id: "VG", name: "British Virgin Islands", lat: 18.4, lng: -64.6, risk: 85, cases: 7 },
  { id: "SG", name: "Singapore", lat: 1.3, lng: 103.8, risk: 72, cases: 4 },
  { id: "CH", name: "Switzerland", lat: 46.8, lng: 8.2, risk: 68, cases: 3 },
  { id: "PA", name: "Panama", lat: 8.9, lng: -79.5, risk: 65, cases: 6 },
  { id: "AE", name: "UAE", lat: 23.4, lng: 53.8, risk: 60, cases: 4 },
  { id: "HK", name: "Hong Kong", lat: 22.3, lng: 114.1, risk: 55, cases: 3 },
  { id: "CY", name: "Cyprus", lat: 35.1, lng: 33.4, risk: 50, cases: 2 },
];

// --- Case Queue ---

export const CASE_QUEUE: CaseItem[] = [
  {
    id: "case-1mdb",
    alert_id: "ALERT-2024-1MDB-7701",
    subject: "Taek Jho Lowe",
    summary: "Suspicious international wire transfers totaling $681M from PetraStar Energy Fund through offshore shell entities. Multiple jurisdictions flagged.",
    risk_score: 92,
    risk_level: "critical",
    typology: "Layering",
    amount: "$681M",
    jurisdictions: ["Malaysia", "Seychelles", "BVI", "Singapore"],
    date: "2024-03-15",
    status: "open",
  },
  {
    id: "case-002",
    alert_id: "ALERT-2024-0892",
    subject: "Marcus Fielding",
    summary: "Multiple sub-$10K deposits across 6 accounts within 48-hour window. Structuring pattern detected.",
    risk_score: 78,
    risk_level: "high",
    typology: "Structuring",
    amount: "$2.4M",
    jurisdictions: ["United States"],
    date: "2024-03-14",
    status: "open",
  },
  {
    id: "case-003",
    alert_id: "ALERT-2024-0755",
    subject: "Jade Commerce Ltd",
    summary: "Invoice values 340% above market price for commodity shipments. Trade-based ML suspected.",
    risk_score: 71,
    risk_level: "high",
    typology: "Trade-Based ML",
    amount: "$18.2M",
    jurisdictions: ["Hong Kong", "Panama"],
    date: "2024-03-12",
    status: "open",
  },
  {
    id: "case-004",
    alert_id: "ALERT-2024-0688",
    subject: "Omega Financial Group",
    summary: "Rapid movement of funds through 4 shell entities in 72 hours. All entities share nominee directors.",
    risk_score: 65,
    risk_level: "medium",
    typology: "Layering",
    amount: "$45M",
    jurisdictions: ["Cyprus", "UAE"],
    date: "2024-03-10",
    status: "in-progress",
  },
  {
    id: "case-005",
    alert_id: "ALERT-2024-0543",
    subject: "Chen Wei Trading",
    summary: "Routine salary deposits flagged by automated system. Likely false positive after review.",
    risk_score: 15,
    risk_level: "low",
    typology: "N/A",
    amount: "$95K",
    jurisdictions: ["Singapore"],
    date: "2024-03-08",
    status: "open",
  },
];

// --- Entity Profiles (1MDB Case) ---

export const ENTITY_PROFILES: Record<string, EntityProfile> = {
  "CUST-1MDB-001": {
    customer_id: "CUST-1MDB-001",
    name: "Taek Jho Lowe",
    nationality: "Malaysian",
    occupation: "Investment Consultant",
    account_open_date: "2023-01-10",
    risk_rating: "High",
    pep_status: true,
    pep_details: "Connected to Minister of Finance (MY)",
    address: "15 Jalan Ampang, Kuala Lumpur, Malaysia",
    annual_income: "$120,000",
    notes: "Multiple passports. Frequent travel to Seychelles, Switzerland.",
  },
  "ENT-GSTAR-001": {
    customer_id: "ENT-GSTAR-001",
    name: "Golden Star Holdings Ltd",
    nationality: "Seychelles",
    occupation: "Investment Holding Company",
    account_open_date: "2022-06-01",
    risk_rating: "High",
    registration: "Seychelles IBC, incorporated 2022-05-14",
    directors: ["Taek Jho Lowe", "Nominee Director Services (SC) Ltd"],
    registered_address: "Suite 4, Mahe Business Centre, Victoria, Seychelles",
    notes: "No employees. No commercial operations. Sole purpose: fund transfers.",
  },
  "ENT-ARBL-001": {
    customer_id: "ENT-ARBL-001",
    name: "Arabella Investments PJS Ltd",
    nationality: "British Virgin Islands",
    occupation: "Private Equity Vehicle",
    account_open_date: "2022-09-20",
    risk_rating: "High",
    registration: "BVI Business Company, incorporated 2022-08-30",
    directors: ["Nominee Director (BVI) Corp"],
    registered_address: "Suite 4, Mahe Business Centre, Victoria, Seychelles",
    notes: "SHARES REGISTERED ADDRESS with Golden Star Holdings. Beneficial owner undisclosed.",
  },
  "ENT-PETRA-001": {
    customer_id: "ENT-PETRA-001",
    name: "PetraStar Energy Fund",
    nationality: "Malaysia",
    occupation: "Sovereign Wealth Fund",
    risk_rating: "Medium",
    notes: "Government-linked investment fund. Board includes Minister of Finance.",
  },
  "CUST-CHEN-002": {
    customer_id: "CUST-CHEN-002",
    name: "Sarah Chen",
    nationality: "Singaporean",
    occupation: "Software Engineer",
    account_open_date: "2021-03-15",
    risk_rating: "Low",
    annual_income: "$95,000",
    notes: "Legitimate retail customer. Regular salary deposits.",
  },
};

// --- Transactions ---

export const TRANSACTIONS: Transaction[] = [
  { transaction_id: "TXN-1MDB-001", from: "ENT-PETRA-001", to: "ENT-GSTAR-001", amount: 681000000, currency: "USD", date: "2024-01-15", type: "International Wire", description: "Investment partnership contribution", jurisdiction_from: "Malaysia", jurisdiction_to: "Seychelles" },
  { transaction_id: "TXN-1MDB-002", from: "ENT-GSTAR-001", to: "ENT-ARBL-001", amount: 260000000, currency: "USD", date: "2024-01-22", type: "International Wire", description: "Intercompany transfer", jurisdiction_from: "Seychelles", jurisdiction_to: "BVI" },
  { transaction_id: "TXN-1MDB-003", from: "ENT-ARBL-001", to: "Real Estate Holdings (NY)", amount: 250000000, currency: "USD", date: "2024-02-05", type: "International Wire", description: "Real estate acquisition", jurisdiction_from: "BVI", jurisdiction_to: "United States" },
  { transaction_id: "TXN-1MDB-004", from: "ENT-ARBL-001", to: "Art dealer (Geneva)", amount: 135000000, currency: "USD", date: "2024-02-14", type: "International Wire", description: "Art collection purchase", jurisdiction_from: "BVI", jurisdiction_to: "Switzerland" },
  { transaction_id: "TXN-1MDB-005", from: "ENT-GSTAR-001", to: "CUST-1MDB-001", amount: 30000000, currency: "USD", date: "2024-02-28", type: "Wire Transfer", description: "Consulting fees", jurisdiction_from: "Seychelles", jurisdiction_to: "Malaysia" },
  { transaction_id: "TXN-1MDB-006", from: "ENT-ARBL-001", to: "ENT-PETRA-001", amount: 6000000, currency: "USD", date: "2024-03-01", type: "Wire Transfer", description: "Return of overpayment", jurisdiction_from: "BVI", jurisdiction_to: "Malaysia" },
  { transaction_id: "TXN-LEGIT-001", from: "Employer (TechCorp SG)", to: "CUST-CHEN-002", amount: 7900, currency: "SGD", date: "2024-01-31", type: "Salary", description: "Monthly salary" },
  { transaction_id: "TXN-LEGIT-002", from: "CUST-CHEN-002", to: "DBS Savings", amount: 3000, currency: "SGD", date: "2024-02-01", type: "Transfer", description: "Savings transfer" },
];

// --- Watchlist ---

export const WATCHLIST_RESULTS: Record<string, WatchlistResult> = {
  "CUST-1MDB-001": { match: true, lists: ["PEP"], details: "Politically Exposed Person — connected to Malaysian government officials" },
  "ENT-GSTAR-001": { match: true, lists: ["FATF-Monitored"], details: "Seychelles entity — FATF-monitored jurisdiction (grey list)" },
  "ENT-ARBL-001": { match: false, lists: [], details: "No direct watchlist match" },
  "CUST-CHEN-002": { match: false, lists: [], details: "No matches" },
};

// --- Network Graph ---

export const INITIAL_NODES: GraphNode[] = [
  { id: "CUST-1MDB-001", label: "Taek Jho Lowe", type: "person", risk: "critical" },
  { id: "ENT-GSTAR-001", label: "Golden Star Holdings", type: "entity", risk: "high" },
  { id: "ENT-ARBL-001", label: "Arabella Investments", type: "entity", risk: "high" },
  { id: "ENT-PETRA-001", label: "PetraStar Energy Fund", type: "fund", risk: "medium" },
  { id: "CUST-CHEN-002", label: "Sarah Chen", type: "decoy", risk: "low" },
];

export const INITIAL_EDGES: GraphEdge[] = [
  { source: "CUST-1MDB-001", target: "ENT-GSTAR-001", relationship: "Director", strength: "strong" },
  { source: "CUST-1MDB-001", target: "ENT-PETRA-001", relationship: "Consultant", strength: "moderate" },
  { source: "ENT-GSTAR-001", target: "ENT-ARBL-001", relationship: "Shared Address", strength: "strong" },
  { source: "ENT-PETRA-001", target: "ENT-GSTAR-001", relationship: "$681M Wire", amount: 681000000, strength: "strong" },
  { source: "ENT-GSTAR-001", target: "ENT-ARBL-001", relationship: "$260M Transfer", amount: 260000000, strength: "strong" },
  { source: "ENT-GSTAR-001", target: "CUST-1MDB-001", relationship: "$30M Consulting Fee", amount: 30000000, strength: "moderate" },
  { source: "ENT-ARBL-001", target: "ENT-PETRA-001", relationship: "$6M Return", amount: 6000000, strength: "weak" },
];

// --- Simulated AGUI Steps (mimicking demo_eval.py --dry-run) ---

const EMPTY_AGUI: AGUIState = {
  ram_usage: { capacity: "0/2 observations", active_context: [] },
  disk_storage: [],
  async_jobs: [],
  kernel_directives: ["You are an AML compliance investigator. Gather evidence, assess risk, and decide: file_sar or close_alert."],
};

export const DEMO_STEPS: StepRecord[] = [
  {
    step_number: 1,
    timestamp: Date.now(),
    action: { tool: "review_alert", parameters: {} },
    reasoning: "Begin by reviewing the alert details to understand the scope of the investigation.",
    observation: {
      tool_result: { alert_id: "ALERT-2024-1MDB-7701", risk_score: 92, summary: "Suspicious international wire transfers totaling $681M" },
      message: "Alert ALERT-2024-1MDB-7701 reviewed. Risk score: 92. $681M suspicious wire transfers.",
      done: false,
      reward: 0.01,
    },
    agui_state: {
      ram_usage: { capacity: "1/2 observations", active_context: ["Alert ALERT-2024-1MDB-7701: Suspicious wire transfers $681M. Customer: CUST-1MDB-001"] },
      disk_storage: [],
      async_jobs: [],
      kernel_directives: ["You are an AML compliance investigator. Gather evidence, assess risk, and decide: file_sar or close_alert."],
    },
  },
  {
    step_number: 2,
    timestamp: Date.now() + 1000,
    action: { tool: "get_customer_profile", parameters: { customer_id: "CUST-1MDB-001" } },
    reasoning: "Profile the primary subject — Taek Jho Lowe — to check PEP status and risk indicators.",
    observation: {
      tool_result: { name: "Taek Jho Lowe", pep_status: true, risk_rating: "High", nationality: "Malaysian" },
      message: "Profile retrieved for CUST-1MDB-001. PEP confirmed. High risk.",
      done: false,
      reward: 0.03,
    },
    agui_state: {
      ram_usage: { capacity: "2/2 observations", active_context: [
        "Alert ALERT-2024-1MDB-7701: Suspicious wire transfers $681M. Customer: CUST-1MDB-001",
        "Profile CUST-1MDB-001: Taek Jho Lowe, Malaysian, PEP (Minister of Finance connection), High Risk"
      ]},
      disk_storage: [],
      async_jobs: [],
      kernel_directives: ["You are an AML compliance investigator. Gather evidence, assess risk, and decide: file_sar or close_alert."],
    },
  },
  {
    step_number: 3,
    timestamp: Date.now() + 2000,
    action: { tool: "write_to_case_file", parameters: { note: "Subject is PEP, connected to Minister of Finance. High risk." } },
    reasoning: "RAM is full (2/2). Save critical PEP finding to disk before it gets evicted.",
    observation: {
      tool_result: { status: "saved", entry_number: 1 },
      message: "Written to case file: 'Subject is PEP, connected to Minister of Finance. High risk.'",
      done: false,
      reward: 0.10,
    },
    agui_state: {
      ram_usage: { capacity: "2/2 observations", active_context: [
        "Alert ALERT-2024-1MDB-7701: Suspicious wire transfers $681M. Customer: CUST-1MDB-001",
        "Profile CUST-1MDB-001: Taek Jho Lowe, Malaysian, PEP (Minister of Finance connection), High Risk"
      ]},
      disk_storage: ["Subject is PEP, connected to Minister of Finance. High risk."],
      async_jobs: [],
      kernel_directives: ["You are an AML compliance investigator. Gather evidence, assess risk, and decide: file_sar or close_alert."],
    },
  },
  {
    step_number: 4,
    timestamp: Date.now() + 3000,
    action: { tool: "query_transactions", parameters: { customer_id: "CUST-1MDB-001" } },
    reasoning: "Analyze transaction history to identify the money flow pattern.",
    observation: {
      tool_result: { transaction_count: 8, total_volume: "$681M+", suspicious_patterns: ["layering", "rapid_fan_out"] },
      message: "8 transactions found. $681M from PetraStar → Golden Star → Arabella. Classic layering pattern.",
      done: false,
      reward: 0.03,
    },
    agui_state: {
      ram_usage: { capacity: "2/2 observations", active_context: [
        "Profile CUST-1MDB-001: Taek Jho Lowe, Malaysian, PEP (Minister of Finance connection), High Risk",
        "Transactions for CUST-1MDB-001: 8 txns, $681M total. PetraStar→GoldenStar→Arabella. Layering pattern."
      ]},
      disk_storage: ["Subject is PEP, connected to Minister of Finance. High risk."],
      async_jobs: [],
      kernel_directives: ["You are an AML compliance investigator. Gather evidence, assess risk, and decide: file_sar or close_alert."],
    },
  },
  {
    step_number: 5,
    timestamp: Date.now() + 4000,
    action: { tool: "search_compliance_manual", parameters: { query: "layering wire transfer offshore" } },
    reasoning: "Look up compliance rules to strengthen the kernel for layering-specific analysis.",
    observation: {
      tool_result: { rules_found: 2, top_match: "CTR threshold: $10,000 (31 USC §5313)" },
      message: "Found 2 relevant compliance rules for layering/wire transfers.",
      done: false,
      reward: 0.03,
    },
    agui_state: {
      ram_usage: { capacity: "2/2 observations", active_context: [
        "Transactions for CUST-1MDB-001: 8 txns, $681M total. PetraStar→GoldenStar→Arabella. Layering pattern.",
        "Compliance search 'layering wire transfer offshore': CTR threshold $10K (31 USC §5313), FATF high-risk jurisdictions"
      ]},
      disk_storage: ["Subject is PEP, connected to Minister of Finance. High risk."],
      async_jobs: [],
      kernel_directives: ["You are an AML compliance investigator. Gather evidence, assess risk, and decide: file_sar or close_alert."],
    },
  },
  {
    step_number: 6,
    timestamp: Date.now() + 5000,
    action: { tool: "update_system_prompt", parameters: { directive: "Apply enhanced due diligence for PEP with offshore transfers" } },
    reasoning: "Inject a compliance rule into the kernel to guide further investigation steps.",
    observation: {
      tool_result: { status: "injected" },
      message: "Kernel directive injected: Enhanced due diligence for PEP + offshore.",
      done: false,
      reward: 0.15,
    },
    agui_state: {
      ram_usage: { capacity: "2/2 observations", active_context: [
        "Transactions for CUST-1MDB-001: 8 txns, $681M total. PetraStar→GoldenStar→Arabella. Layering pattern.",
        "Compliance search 'layering wire transfer offshore': CTR threshold $10K (31 USC §5313), FATF high-risk jurisdictions"
      ]},
      disk_storage: ["Subject is PEP, connected to Minister of Finance. High risk."],
      async_jobs: [],
      kernel_directives: [
        "You are an AML compliance investigator. Gather evidence, assess risk, and decide: file_sar or close_alert.",
        "Added (Step 6): Apply enhanced due diligence for PEP with offshore transfers"
      ],
    },
  },
  {
    step_number: 7,
    timestamp: Date.now() + 6000,
    action: { tool: "trace_network", parameters: { entity_id: "ENT-GSTAR-001" } },
    reasoning: "Trace Golden Star Holdings network to map the full shell company structure.",
    observation: {
      tool_result: { connections: [{ entity: "ENT-ARBL-001", relationship: "Shared registered address" }] },
      message: "Network traced for ENT-GSTAR-001. Connected to Arabella Investments via shared registered address.",
      done: false,
      reward: 0.03,
    },
    agui_state: {
      ram_usage: { capacity: "2/2 observations", active_context: [
        "Compliance search 'layering wire transfer offshore': CTR threshold $10K, FATF high-risk jurisdictions",
        "Network ENT-GSTAR-001: Connected to ENT-ARBL-001 (shared registered address, Seychelles). Director: Taek Jho Lowe."
      ]},
      disk_storage: ["Subject is PEP, connected to Minister of Finance. High risk."],
      async_jobs: [],
      kernel_directives: [
        "You are an AML compliance investigator. Gather evidence, assess risk, and decide: file_sar or close_alert.",
        "Added (Step 6): Apply enhanced due diligence for PEP with offshore transfers"
      ],
    },
  },
  {
    step_number: 8,
    timestamp: Date.now() + 7000,
    action: { tool: "write_to_case_file", parameters: { note: "Golden Star and Arabella share registered address in Seychelles. Classic layering." } },
    reasoning: "Persist the shared-address finding before RAM eviction.",
    observation: {
      tool_result: { status: "saved", entry_number: 2 },
      message: "Written to case file: Shared registered address finding.",
      done: false,
      reward: 0.10,
    },
    agui_state: {
      ram_usage: { capacity: "2/2 observations", active_context: [
        "Compliance search 'layering wire transfer offshore': CTR threshold $10K, FATF high-risk jurisdictions",
        "Network ENT-GSTAR-001: Connected to ENT-ARBL-001 (shared registered address, Seychelles). Director: Taek Jho Lowe."
      ]},
      disk_storage: [
        "Subject is PEP, connected to Minister of Finance. High risk.",
        "Golden Star and Arabella share registered address in Seychelles. Classic layering."
      ],
      async_jobs: [],
      kernel_directives: [
        "You are an AML compliance investigator. Gather evidence, assess risk, and decide: file_sar or close_alert.",
        "Added (Step 6): Apply enhanced due diligence for PEP with offshore transfers"
      ],
    },
  },
  {
    step_number: 9,
    timestamp: Date.now() + 8000,
    action: { tool: "check_watchlist", parameters: { entity: "CUST-1MDB-001" } },
    reasoning: "Check the primary subject against global sanctions and watchlists.",
    observation: {
      tool_result: { match: true, lists: ["PEP"], details: "Politically Exposed Person" },
      message: "Watchlist HIT for CUST-1MDB-001: PEP — connected to Malaysian government.",
      done: false,
      reward: 0.03,
    },
    agui_state: {
      ram_usage: { capacity: "2/2 observations", active_context: [
        "Network ENT-GSTAR-001: Connected to ENT-ARBL-001 (shared registered address). Director: Taek Jho Lowe.",
        "Watchlist CUST-1MDB-001: MATCH — PEP, connected to Malaysian government officials"
      ]},
      disk_storage: [
        "Subject is PEP, connected to Minister of Finance. High risk.",
        "Golden Star and Arabella share registered address in Seychelles. Classic layering."
      ],
      async_jobs: [],
      kernel_directives: [
        "You are an AML compliance investigator. Gather evidence, assess risk, and decide: file_sar or close_alert.",
        "Added (Step 6): Apply enhanced due diligence for PEP with offshore transfers"
      ],
    },
  },
  {
    step_number: 10,
    timestamp: Date.now() + 9000,
    action: { tool: "check_source_of_funds", parameters: { transaction_id: "TXN-1MDB-001" } },
    reasoning: "Verify the source of the $681M wire from PetraStar.",
    observation: {
      tool_result: { source: "PetraStar Energy Fund", risk_flags: ["No board resolution", "No independent audit"] },
      message: "Source of funds: PetraStar (unverified). Risk flags: no board resolution, no audit.",
      done: false,
      reward: 0.03,
    },
    agui_state: {
      ram_usage: { capacity: "2/2 observations", active_context: [
        "Watchlist CUST-1MDB-001: MATCH — PEP, connected to Malaysian government officials",
        "Source of funds TXN-1MDB-001: PetraStar Energy Fund. FLAGS: No board resolution, no independent audit."
      ]},
      disk_storage: [
        "Subject is PEP, connected to Minister of Finance. High risk.",
        "Golden Star and Arabella share registered address in Seychelles. Classic layering."
      ],
      async_jobs: [],
      kernel_directives: [
        "You are an AML compliance investigator. Gather evidence, assess risk, and decide: file_sar or close_alert.",
        "Added (Step 6): Apply enhanced due diligence for PEP with offshore transfers"
      ],
    },
  },
  {
    step_number: 11,
    timestamp: Date.now() + 10000,
    action: { tool: "request_wire_trace", parameters: { transaction_id: "TXN-1MDB-002" } },
    reasoning: "Request async trace on the $260M intercompany transfer. ETA: 2 steps.",
    observation: {
      tool_result: { job_id: "REQ-001", eta_steps: 2, status: "pending" },
      message: "Async wire trace queued: REQ-001. ETA: 2 steps.",
      done: false,
      reward: 0.03,
    },
    agui_state: {
      ram_usage: { capacity: "2/2 observations", active_context: [
        "Source of funds TXN-1MDB-001: PetraStar Energy Fund. FLAGS: No board resolution, no independent audit.",
        "Async request: wire_trace TXN-1MDB-002 → REQ-001 (ETA: 2 steps)"
      ]},
      disk_storage: [
        "Subject is PEP, connected to Minister of Finance. High risk.",
        "Golden Star and Arabella share registered address in Seychelles. Classic layering."
      ],
      async_jobs: [{ id: "REQ-001", tool: "wire_trace", eta_steps: 2, status: "pending" }],
      kernel_directives: [
        "You are an AML compliance investigator. Gather evidence, assess risk, and decide: file_sar or close_alert.",
        "Added (Step 6): Apply enhanced due diligence for PEP with offshore transfers"
      ],
    },
  },
  {
    step_number: 12,
    timestamp: Date.now() + 11000,
    action: { tool: "get_customer_profile", parameters: { customer_id: "ENT-ARBL-001" } },
    reasoning: "Profile the second shell company while waiting for async trace.",
    observation: {
      tool_result: { name: "Arabella Investments PJS Ltd", nationality: "BVI", risk_rating: "High" },
      message: "Profile ENT-ARBL-001: Arabella Investments, BVI, High Risk. Nominee directors.",
      done: false,
      reward: 0.03,
    },
    agui_state: {
      ram_usage: { capacity: "2/2 observations", active_context: [
        "Async request: wire_trace TXN-1MDB-002 → REQ-001 (ETA: 2 steps)",
        "Profile ENT-ARBL-001: Arabella Investments PJS Ltd, BVI, High Risk. Nominee directors only."
      ]},
      disk_storage: [
        "Subject is PEP, connected to Minister of Finance. High risk.",
        "Golden Star and Arabella share registered address in Seychelles. Classic layering."
      ],
      async_jobs: [{ id: "REQ-001", tool: "wire_trace", eta_steps: 1, status: "pending" }],
      kernel_directives: [
        "You are an AML compliance investigator. Gather evidence, assess risk, and decide: file_sar or close_alert.",
        "Added (Step 6): Apply enhanced due diligence for PEP with offshore transfers"
      ],
    },
  },
  {
    step_number: 13,
    timestamp: Date.now() + 12000,
    action: { tool: "check_source_of_funds", parameters: { transaction_id: "TXN-1MDB-005" } },
    reasoning: "Check the $30M consulting fee — likely no legitimate deliverables.",
    observation: {
      tool_result: { source: "Golden Star Holdings", risk_flags: ["Backdated contract", "No deliverables"] },
      message: "Source: Golden Star. FLAGS: Backdated consulting contract, no deliverables documented.",
      done: false,
      reward: 0.03,
    },
    agui_state: {
      ram_usage: { capacity: "2/2 observations", active_context: [
        "Profile ENT-ARBL-001: Arabella Investments PJS Ltd, BVI, High Risk. Nominee directors only.",
        "Source of funds TXN-1MDB-005: Golden Star Holdings. FLAGS: Backdated contract, no deliverables."
      ]},
      disk_storage: [
        "Subject is PEP, connected to Minister of Finance. High risk.",
        "Golden Star and Arabella share registered address in Seychelles. Classic layering."
      ],
      async_jobs: [{ id: "REQ-001", tool: "wire_trace", eta_steps: 0, status: "ready" }],
      kernel_directives: [
        "You are an AML compliance investigator. Gather evidence, assess risk, and decide: file_sar or close_alert.",
        "Added (Step 6): Apply enhanced due diligence for PEP with offshore transfers"
      ],
    },
  },
  {
    step_number: 14,
    timestamp: Date.now() + 13000,
    action: { tool: "retrieve_async_result", parameters: { job_id: "REQ-001" } },
    reasoning: "REQ-001 is READY. Retrieve the wire trace result.",
    observation: {
      tool_result: { trace: "TXN-1MDB-002: $260M routed Seychelles → BVI → US Real Estate + Swiss Art" },
      message: "Wire trace complete: $260M layered through BVI to US real estate and Swiss art purchases.",
      done: false,
      reward: 0.03,
    },
    agui_state: {
      ram_usage: { capacity: "2/2 observations", active_context: [
        "Source of funds TXN-1MDB-005: Golden Star Holdings. FLAGS: Backdated contract, no deliverables.",
        "Wire trace REQ-001: $260M routed Seychelles→BVI→US Real Estate ($250M) + Swiss Art ($135M)"
      ]},
      disk_storage: [
        "Subject is PEP, connected to Minister of Finance. High risk.",
        "Golden Star and Arabella share registered address in Seychelles. Classic layering."
      ],
      async_jobs: [],
      kernel_directives: [
        "You are an AML compliance investigator. Gather evidence, assess risk, and decide: file_sar or close_alert.",
        "Added (Step 6): Apply enhanced due diligence for PEP with offshore transfers"
      ],
    },
  },
  {
    step_number: 15,
    timestamp: Date.now() + 14000,
    action: { tool: "file_sar", parameters: {
      typology: "layering",
      entities_involved: ["CUST-1MDB-001", "ENT-GSTAR-001", "ENT-ARBL-001"],
      findings: ["pep_connection", "offshore_source", "shared_registered_address", "rapid_fan_out", "no_source_documentation", "reversed_transaction"],
    }},
    reasoning: "All evidence gathered. Filing SAR with complete evidence chain.",
    observation: {
      tool_result: { decision: "file_sar", typology: "layering", entities: 3, findings: 6, score: 1.01 },
      message: "SAR FILED. Decision: file_sar. Typology: layering. Terminal reward: +1.01",
      done: true,
      reward: 0.30,
    },
    agui_state: {
      ram_usage: { capacity: "2/2 observations", active_context: [
        "Wire trace REQ-001: $260M routed Seychelles→BVI→US Real Estate ($250M) + Swiss Art ($135M)",
        "SAR FILED: layering, 3 entities, 6 findings. Decision: file_sar. CASE CLOSED."
      ]},
      disk_storage: [
        "Subject is PEP, connected to Minister of Finance. High risk.",
        "Golden Star and Arabella share registered address in Seychelles. Classic layering."
      ],
      async_jobs: [],
      kernel_directives: [
        "You are an AML compliance investigator. Gather evidence, assess risk, and decide: file_sar or close_alert.",
        "Added (Step 6): Apply enhanced due diligence for PEP with offshore transfers"
      ],
    },
  },
];

// --- Episode Meta ---

export const DEMO_EPISODE_META: EpisodeMeta = {
  scenario: "1MDB-Inspired Sovereign Wealth Fund Investigation",
  model: "scripted",
  total_steps: 15,
  final_score: 1.01,
  ground_truth: {
    correct_decision: "file_sar",
    typology: "layering",
    key_entities: ["CUST-1MDB-001", "ENT-GSTAR-001", "ENT-ARBL-001"],
    excluded_entities: ["CUST-CHEN-002"],
    key_findings: ["pep_connection", "offshore_source", "shared_registered_address", "rapid_fan_out", "no_source_documentation", "reversed_transaction"],
  },
  alert: {
    alert_id: "ALERT-2024-1MDB-7701",
    customer_id: "CUST-1MDB-001",
    summary: "Suspicious international wire transfers totaling $681M from PetraStar Energy Fund through offshore shell entities.",
    type: "Suspicious Wire Transfer",
    risk_score: 92,
    date: "2024-03-15",
  },
  entity_count: 5,
  transaction_count: 8,
  steps_summary: DEMO_STEPS.map(s => ({ step: s.step_number, tool: s.action.tool, reward: s.observation.reward, done: s.observation.done })),
};
