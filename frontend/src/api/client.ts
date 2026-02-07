import type {
  ModelInfo,
  ExperimentConfig,
  ExperimentResult,
  ExperimentSummary,
  CaptureResult,
  TokenInfo,
  Insight,
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
): Promise<{ result: ExperimentResult; insights: Insight[] }> {
  return request('/experiments/run', {
    method: 'POST',
    body: JSON.stringify(config),
  });
}

export async function runSweep(
  config: ExperimentConfig,
  layers?: number[],
): Promise<{ results: ExperimentResult[]; insights: Insight[] }> {
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

// -- Reports (PDF) --

export async function downloadSweepReport(
  config: ExperimentConfig,
  layers?: number[],
): Promise<void> {
  const res = await fetch(`${BASE}/reports/sweep`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ config, layers: layers ?? null }),
  });
  if (!res.ok) throw new Error(`Report failed: ${res.status}`);
  const blob = await res.blob();
  _downloadBlob(blob, 'neuronscope_sweep_report.pdf');
}

export async function downloadExperimentReport(id: string): Promise<void> {
  const res = await fetch(`${BASE}/reports/experiment/${id}`);
  if (!res.ok) throw new Error(`Report failed: ${res.status}`);
  const blob = await res.blob();
  _downloadBlob(blob, `neuronscope_experiment_${id}.pdf`);
}

function _downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
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
