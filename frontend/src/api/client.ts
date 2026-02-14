import type {
  ModelInfo,
  ExperimentConfig,
  ExperimentResult,
  ExperimentSummary,
  SweepSummary,
  SweepDetail,
  CaptureResult,
  TokenInfo,
  Insight,
  AttributionResult,
  StreamMessage,
  StoredActivation,
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
): Promise<{ results: ExperimentResult[]; insights: Insight[]; sweep_id: string }> {
  return request('/experiments/sweep', {
    method: 'POST',
    body: JSON.stringify({ config, layers: layers ?? null }),
  });
}

export async function runHeadSweep(
  config: ExperimentConfig,
  layer: number,
  heads?: number[],
): Promise<{ results: ExperimentResult[]; insights: Insight[]; sweep_id: string }> {
  return request('/experiments/sweep-heads', {
    method: 'POST',
    body: JSON.stringify({ config, layer, heads: heads ?? null }),
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

// -- Sweeps --

export async function listSweeps(): Promise<SweepSummary[]> {
  return request('/experiments/sweeps');
}

export async function getSweep(id: string): Promise<SweepDetail> {
  return request(`/experiments/sweeps/${id}`);
}

export async function deleteSweep(id: string): Promise<void> {
  await request(`/experiments/sweeps/${id}`, { method: 'DELETE' });
}

export async function downloadSweepReportById(id: string): Promise<void> {
  const res = await fetch(`${BASE}/reports/sweep/${id}`);
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(typeof body.detail === 'string' ? body.detail : `Report failed: ${res.status}`);
  }
  const blob = await res.blob();
  _downloadBlob(blob, `neuronscope_sweep_${id}.pdf`);
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
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(typeof body.detail === 'string' ? body.detail : `Report failed: ${res.status}`);
  }
  const blob = await res.blob();
  _downloadBlob(blob, 'neuronscope_sweep_report.pdf');
}

export async function downloadExperimentReport(id: string): Promise<void> {
  const res = await fetch(`${BASE}/reports/experiment/${id}`);
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(typeof body.detail === 'string' ? body.detail : `Report failed: ${res.status}`);
  }
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
  save = false,
  experimentId?: string,
): Promise<CaptureResult[]> {
  return request('/activations/capture', {
    method: 'POST',
    body: JSON.stringify({
      input_text: inputText,
      targets,
      save,
      experiment_id: experimentId ?? null,
    }),
  });
}

// -- Stored Activations --

export async function listStoredActivations(): Promise<StoredActivation[]> {
  return request('/activations/stored');
}

export async function getStoredActivation(id: string): Promise<StoredActivation> {
  return request(`/activations/stored/${id}`);
}

export async function deleteStoredActivation(id: string): Promise<void> {
  await request(`/activations/stored/${id}`, { method: 'DELETE' });
}

// -- Attribution --

export async function runAttribution(
  baseInput: string,
  interventionType = 'zero',
  component = 'mlp_output',
  layers?: number[],
): Promise<AttributionResult> {
  return request('/analysis/attribution', {
    method: 'POST',
    body: JSON.stringify({
      base_input: baseInput,
      intervention_type: interventionType,
      component,
      layers: layers ?? null,
    }),
  });
}

// -- WebSocket Streaming --

export function streamActivations(
  inputText: string,
  targets: Array<Record<string, unknown>>,
  onMessage: (msg: StreamMessage) => void,
  onError?: (error: Event) => void,
): WebSocket {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const ws = new WebSocket(`${protocol}//${window.location.host}/api/ws/stream`);

  ws.onopen = () => {
    ws.send(JSON.stringify({ input_text: inputText, targets }));
  };

  ws.onmessage = (event) => {
    const msg: StreamMessage = JSON.parse(event.data);
    onMessage(msg);
  };

  ws.onerror = (event) => {
    onError?.(event);
  };

  return ws;
}
