import type {
  ModelInfo,
  ExperimentConfig,
  ExperimentResult,
  ExperimentSummary,
  CaptureResult,
  TokenInfo,
} from './types';

const BASE = '/api';

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    const detail = body.detail;
    const message =
      typeof detail === 'string'
        ? detail
        : Array.isArray(detail)
          ? detail.map((d: { msg?: string }) => d.msg || JSON.stringify(d)).join('; ')
          : `Request failed: ${res.status}`;
    throw new Error(message);
  }
  return res.json();
}

// -- Model --

export async function getModelInfo(): Promise<ModelInfo | { loaded: false }> {
  return request('/model/info');
}

export async function loadModel(
  modelPath = 'LLM',
  device = 'cuda',
): Promise<ModelInfo> {
  return request('/model/load', {
    method: 'POST',
    body: JSON.stringify({ model_path: modelPath, device }),
  });
}

export async function unloadModel(): Promise<void> {
  await request('/model/unload', { method: 'POST' });
}

export async function getModuleTree(): Promise<Record<string, string>> {
  return request('/model/modules');
}

export async function tokenize(text: string): Promise<TokenInfo[]> {
  return request(`/model/tokenize?text=${encodeURIComponent(text)}`, {
    method: 'POST',
  });
}

// -- Experiments --

export async function runExperiment(
  config: ExperimentConfig,
): Promise<ExperimentResult> {
  return request('/experiments/run', {
    method: 'POST',
    body: JSON.stringify(config),
  });
}

export async function runSweep(
  config: ExperimentConfig,
  layers?: number[],
): Promise<ExperimentResult[]> {
  return request('/experiments/sweep', {
    method: 'POST',
    body: JSON.stringify({ config, layers: layers ?? null }),
  });
}

export async function listExperiments(): Promise<ExperimentSummary[]> {
  return request('/experiments/');
}

export async function getExperiment(id: string): Promise<ExperimentResult> {
  return request(`/experiments/${id}`);
}

export async function deleteExperiment(id: string): Promise<void> {
  await request(`/experiments/${id}`, { method: 'DELETE' });
}

// -- Activations --

export async function captureActivations(
  inputText: string,
  targets: Array<Record<string, unknown>>,
): Promise<CaptureResult[]> {
  return request('/activations/capture', {
    method: 'POST',
    body: JSON.stringify({ input_text: inputText, targets }),
  });
}
