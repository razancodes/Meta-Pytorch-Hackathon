//! AgentOS-Kernel — Async Tool Execution Runtime
//!
//! A Tokio-based concurrent tool executor exposed to Python via PyO3.
//! Supports two execution modes:
//!   - Mock: Built-in handlers with simulated latency (for testing)
//!   - HTTP: Dispatches to `http://<base_url>/tools/<name>` via reqwest

use std::time::Instant;

use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use tokio::time::{self, Duration};

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Result types
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Status of a single tool execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolStatus {
    Success,
    Timeout,
    Error,
}

/// Structured result returned for each tool call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub tool: String,
    pub status: ToolStatus,
    pub data: serde_json::Value,
    pub elapsed_ms: u64,
    pub error: Option<String>,
}

/// Batch response containing all tool results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchResult {
    pub results: Vec<ToolResult>,
    pub total_elapsed_ms: u64,
    pub tools_succeeded: usize,
    pub tools_timed_out: usize,
    pub tools_errored: usize,
}

/// Runtime execution mode.
#[derive(Debug, Clone)]
pub enum ExecutionMode {
    /// Built-in mock handlers with simulated latency.
    Mock,
    /// HTTP dispatch to a tool server.
    Http { base_url: String },
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Mock tool executors (testing / demo mode)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Mock tool handlers with realistic AML data and simulated I/O latency.
async fn execute_mock_tool(
    tool_name: &str,
    params_json: &str,
) -> Result<serde_json::Value, String> {
    let params: serde_json::Value = serde_json::from_str(params_json)
        .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));

    match tool_name {
        "trace_network" => {
            time::sleep(Duration::from_millis(2000)).await;
            let entity_id = params
                .get("entity_id")
                .and_then(|v| v.as_str())
                .unwrap_or("UNKNOWN");
            Ok(serde_json::json!({
                "entity_id": entity_id,
                "depth": params.get("depth").and_then(|v| v.as_u64()).unwrap_or(1),
                "connections": [
                    {"target": "ENT_SHELL_A", "relationship": "beneficial_owner", "weight": 0.85},
                    {"target": "ENT_SHELL_B", "relationship": "transfer_to", "weight": 0.72},
                    {"target": "ENT_BANK_C",  "relationship": "correspondent", "weight": 0.61}
                ],
                "source": "mock_graph_db"
            }))
        }

        "check_watchlist" => {
            time::sleep(Duration::from_millis(500)).await;
            let entity_name = params
                .get("entity_name")
                .and_then(|v| v.as_str())
                .unwrap_or("John Doe");
            Ok(serde_json::json!({
                "entity": entity_name,
                "hit": true,
                "lists": ["OFAC SDN", "PEP"],
                "match_score": 0.94,
                "details": "Politically Exposed Person — former minister of finance",
                "source": "mock_screening_api"
            }))
        }

        "query_transactions" => {
            time::sleep(Duration::from_millis(1000)).await;
            let customer_id = params
                .get("customer_id")
                .and_then(|v| v.as_str())
                .unwrap_or("CUST-000");
            Ok(serde_json::json!({
                "customer_id": customer_id,
                "count": 12,
                "transactions": [
                    {"id": "TXN-001", "amount": 9800.0, "currency": "USD", "direction": "outgoing"},
                    {"id": "TXN-002", "amount": 9750.0, "currency": "USD", "direction": "outgoing"},
                    {"id": "TXN-003", "amount": 9900.0, "currency": "USD", "direction": "outgoing"}
                ],
                "total_amount": 117000.0,
                "source": "mock_core_banking"
            }))
        }

        "request_wire_trace" => {
            time::sleep(Duration::from_millis(1800)).await;
            let target = params
                .get("entity_id")
                .or_else(|| params.get("transaction_id"))
                .and_then(|v| v.as_str())
                .unwrap_or("UNKNOWN");
            Ok(serde_json::json!({
                "trace_target": target,
                "correspondent_banks": ["Deutsche Bank AG", "HSBC Holdings", "Standard Chartered"],
                "intermediary_count": 3,
                "swift_messages": 7,
                "jurisdictions_touched": ["DE", "HK", "SG", "KY"],
                "source": "mock_swift_gpi"
            }))
        }

        "check_device_overlap" => {
            time::sleep(Duration::from_millis(300)).await;
            let entity_id = params
                .get("entity_id")
                .and_then(|v| v.as_str())
                .unwrap_or("UNKNOWN");
            Ok(serde_json::json!({
                "entity_id": entity_id,
                "shared_devices": 2,
                "shared_ips": 1,
                "overlapping_entities": ["ENT_MULE_1", "ENT_MULE_2"],
                "risk_indicator": "mule_ring_suspected",
                "source": "mock_device_graph"
            }))
        }

        "assess_risk" => {
            time::sleep(Duration::from_millis(200)).await;
            let customer_id = params
                .get("customer_id")
                .and_then(|v| v.as_str())
                .unwrap_or("CUST-000");
            Ok(serde_json::json!({
                "customer_id": customer_id,
                "risk_score": 78,
                "risk_level": "HIGH",
                "recommendation": "SAR filing recommended",
                "risk_factors": ["sub_threshold_structuring", "pep_connection", "offshore_transfers"],
                "source": "mock_risk_engine"
            }))
        }

        _ => {
            time::sleep(Duration::from_millis(100)).await;
            Ok(serde_json::json!({
                "tool": tool_name,
                "echo_params": params,
                "message": "No specific handler — params echoed",
                "source": "mock_default"
            }))
        }
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// HTTP tool dispatch (production mode)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Execute a tool by POSTing to an HTTP endpoint.
async fn execute_http_tool(
    base_url: &str,
    tool_name: &str,
    params_json: &str,
) -> Result<serde_json::Value, String> {
    let url = format!("{}/tools/{}", base_url.trim_end_matches('/'), tool_name);

    let body: serde_json::Value = serde_json::from_str(params_json)
        .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));

    let client = reqwest::Client::new();
    let resp = client
        .post(&url)
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("HTTP request failed: {}", e))?;

    let status = resp.status();
    let text = resp
        .text()
        .await
        .map_err(|e| format!("Failed to read response body: {}", e))?;

    if !status.is_success() {
        return Err(format!("HTTP {} from {}: {}", status, url, text));
    }

    serde_json::from_str(&text)
        .map_err(|e| format!("JSON parse error: {} — body: {}", e, &text[..text.len().min(200)]))
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Async execution logic
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Execute a single tool call with timeout and panic safety.
async fn run_single_tool(
    tool_name: String,
    params_json: String,
    timeout_ms: u64,
    mode: ExecutionMode,
) -> ToolResult {
    let start = Instant::now();
    let deadline = Duration::from_millis(timeout_ms);
    let name_for_result = tool_name.clone();

    let task = tokio::spawn(async move {
        match mode {
            ExecutionMode::Mock => execute_mock_tool(&tool_name, &params_json).await,
            ExecutionMode::Http { base_url } => {
                execute_http_tool(&base_url, &tool_name, &params_json).await
            }
        }
    });

    let outcome = time::timeout(deadline, task).await;

    match outcome {
        Ok(Ok(Ok(data))) => ToolResult {
            tool: name_for_result,
            status: ToolStatus::Success,
            data,
            elapsed_ms: start.elapsed().as_millis() as u64,
            error: None,
        },
        Ok(Ok(Err(tool_err))) => ToolResult {
            tool: name_for_result,
            status: ToolStatus::Error,
            data: serde_json::Value::Null,
            elapsed_ms: start.elapsed().as_millis() as u64,
            error: Some(tool_err),
        },
        Ok(Err(join_err)) => ToolResult {
            tool: name_for_result,
            status: ToolStatus::Error,
            data: serde_json::Value::Null,
            elapsed_ms: start.elapsed().as_millis() as u64,
            error: Some(format!("Task panicked: {}", join_err)),
        },
        Err(_) => ToolResult {
            tool: name_for_result,
            status: ToolStatus::Timeout,
            data: serde_json::Value::Null,
            elapsed_ms: start.elapsed().as_millis() as u64,
            error: Some(format!("Exceeded {}ms timeout", timeout_ms)),
        },
    }
}

/// Execute a batch of tool calls concurrently.
async fn run_batch(calls: Vec<(String, String)>, timeout_ms: u64, mode: ExecutionMode) -> BatchResult {
    let batch_start = Instant::now();
    let mut handles = Vec::with_capacity(calls.len());

    for (tool_name, params_json) in calls {
        let timeout = timeout_ms;
        let m = mode.clone();
        handles.push(tokio::spawn(async move {
            run_single_tool(tool_name, params_json, timeout, m).await
        }));
    }

    let mut results = Vec::with_capacity(handles.len());
    for handle in handles {
        match handle.await {
            Ok(result) => results.push(result),
            Err(join_err) => {
                results.push(ToolResult {
                    tool: "unknown".to_string(),
                    status: ToolStatus::Error,
                    data: serde_json::Value::Null,
                    elapsed_ms: 0,
                    error: Some(format!("Join error: {}", join_err)),
                });
            }
        }
    }

    let succeeded = results.iter().filter(|r| matches!(r.status, ToolStatus::Success)).count();
    let timed_out = results.iter().filter(|r| matches!(r.status, ToolStatus::Timeout)).count();
    let errored  = results.iter().filter(|r| matches!(r.status, ToolStatus::Error)).count();

    BatchResult {
        results,
        total_elapsed_ms: batch_start.elapsed().as_millis() as u64,
        tools_succeeded: succeeded,
        tools_timed_out: timed_out,
        tools_errored: errored,
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// PyO3 interface
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Tokio-powered async tool execution runtime.
///
/// Usage from Python:
/// ```python
/// from agent_os_core import ToolRuntime
///
/// # Mock mode (built-in handlers)
/// rt = ToolRuntime()
///
/// # HTTP mode (dispatch to real services)
/// rt = ToolRuntime()
/// rt.set_mode("http", "http://localhost:8080")
/// ```
#[pyclass]
pub struct ToolRuntime {
    worker_threads: usize,
    mode: ExecutionMode,
}

#[pymethods]
impl ToolRuntime {
    /// Create a new ToolRuntime in mock mode.
    #[new]
    #[pyo3(signature = (worker_threads=None))]
    fn new(worker_threads: Option<usize>) -> Self {
        let threads = worker_threads.unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4)
        });
        ToolRuntime {
            worker_threads: threads,
            mode: ExecutionMode::Mock,
        }
    }

    /// Set the execution mode.
    ///
    /// Args:
    ///     mode: "mock" or "http"
    ///     base_url: Required if mode == "http". E.g. "http://localhost:8080"
    #[pyo3(signature = (mode, base_url=None))]
    fn set_mode(&mut self, mode: &str, base_url: Option<String>) -> PyResult<()> {
        self.mode = match mode {
            "mock" => ExecutionMode::Mock,
            "http" => {
                let url = base_url.ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err("base_url required for http mode")
                })?;
                ExecutionMode::Http { base_url: url }
            }
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    format!("Unknown mode '{}'. Use 'mock' or 'http'.", mode),
                ));
            }
        };
        Ok(())
    }

    /// Get the current execution mode.
    fn get_mode(&self) -> String {
        match &self.mode {
            ExecutionMode::Mock => "mock".to_string(),
            ExecutionMode::Http { base_url } => format!("http:{}", base_url),
        }
    }

    /// Execute a single tool call with a timeout. Returns JSON string.
    #[pyo3(signature = (tool_name, params_json, timeout_ms=10000))]
    fn execute_one(
        &self,
        tool_name: String,
        params_json: String,
        timeout_ms: u64,
    ) -> PyResult<String> {
        let rt = self.build_runtime()?;
        let result = rt.block_on(run_single_tool(
            tool_name, params_json, timeout_ms, self.mode.clone(),
        ));
        serde_json::to_string_pretty(&result).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Serialization error: {}", e))
        })
    }

    /// Execute a batch of tool calls concurrently. Returns JSON string.
    #[pyo3(signature = (calls, timeout_ms=10000))]
    fn execute_batch(
        &self,
        calls: Vec<(String, String)>,
        timeout_ms: u64,
    ) -> PyResult<String> {
        let rt = self.build_runtime()?;
        let result = rt.block_on(run_batch(calls, timeout_ms, self.mode.clone()));
        serde_json::to_string_pretty(&result).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Serialization error: {}", e))
        })
    }
}

impl ToolRuntime {
    fn build_runtime(&self) -> PyResult<tokio::runtime::Runtime> {
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(self.worker_threads)
            .enable_all()
            .build()
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Failed to create Tokio runtime: {}",
                    e
                ))
            })
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Python module registration
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[pymodule]
fn agent_os_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ToolRuntime>()?;
    Ok(())
}
