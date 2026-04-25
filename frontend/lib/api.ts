// ═══════════════════════════════════════════════════════════════
// MEMEX — API Client (Live Mode)
// ═══════════════════════════════════════════════════════════════

import { StepResponse } from './types';

// When served from HF Spaces (same origin as FastAPI), use '' for same-origin.
// In local dev, NEXT_PUBLIC_API_URL can point to the backend.
const API_BASE = process.env.NEXT_PUBLIC_API_URL || '';

async function apiFetch<T>(path: string, options?: RequestInit): Promise<T> {
  const url = `${API_BASE}${path}`;
  const res = await fetch(url, {
    ...options,
    headers: { 'Content-Type': 'application/json', ...options?.headers },
  });
  if (!res.ok) {
    throw new Error(`API ${res.status}: ${res.statusText}`);
  }
  return res.json();
}

export async function healthCheck(): Promise<{ status: string }> {
  return apiFetch('/health');
}

export async function resetEnvironment(
  taskId: string = 'easy',
  seed?: number,
  episodeId?: string
): Promise<StepResponse> {
  const body: Record<string, unknown> = { task_id: taskId };
  if (seed !== undefined) body.seed = seed;
  if (episodeId) body.episode_id = episodeId;
  return apiFetch('/reset', { method: 'POST', body: JSON.stringify(body) });
}

export async function stepEnvironment(
  tool: string,
  parameters: Record<string, unknown> = {}
): Promise<StepResponse> {
  return apiFetch('/step', {
    method: 'POST',
    body: JSON.stringify({
      action: { tool, parameters, metadata: {} },
    }),
  });
}

export async function getState(): Promise<Record<string, unknown>> {
  return apiFetch('/state');
}

export async function isBackendAvailable(): Promise<boolean> {
  try {
    await healthCheck();
    return true;
  } catch {
    return false;
  }
}
