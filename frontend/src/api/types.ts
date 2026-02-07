// Mirrors backend Pydantic models exactly

export interface ModelInfo {
  name: string;
  path: string;
  architecture: string;
  num_layers: number;
  hidden_size: number;
  intermediate_size: number;
  num_attention_heads: number | null;
  num_key_value_heads: number | null;
  vocab_size: number;
  dtype: string;
  device: string;
  sliding_window: number | null;
  has_vision: boolean;
  module_names: string[];
}

export interface InterventionSpec {
  target_layer: number;
  target_component: ComponentType;
  target_head: number | null;
  target_position: number | null;
  target_neuron: number | null;
  intervention_type: InterventionType;
  intervention_params: Record<string, unknown>;
}

export interface CaptureSpec {
  layer: number;
  component: ComponentType;
  head: number | null;
  token_position: number | null;
  neuron_index: number | null;
}

export interface ExperimentConfig {
  name: string;
  base_input: string;
  source_input: string | null;
  interventions: InterventionSpec[];
  seed: number;
  model_path: string;
  max_new_tokens: number;
  temperature: number;
  capture_targets: CaptureSpec[];
}

export interface TokenPrediction {
  token: string;
  token_id: number;
  logit: number;
  prob: number;
}

export interface ExperimentResult {
  id: string;
  config: ExperimentConfig;
  config_hash: string;
  clean_top_k: TokenPrediction[];
  clean_output_token: string;
  clean_output_prob: number;
  intervention_top_k: TokenPrediction[];
  intervention_output_token: string;
  intervention_output_prob: number;
  kl_divergence: number;
  logit_diff_change: number | null;
  top_token_changed: boolean;
  rank_changes: Record<string, RankChange>;
  effect_size: number | null;
  timestamp: string;
  duration_seconds: number;
  device: string;
}

export interface RankChange {
  token_id: number;
  clean_rank: number;
  intervention_rank: number;
  rank_delta: number;
}

export interface ExperimentSummary {
  id: string;
  name: string;
  config_hash: string;
  kl_divergence: number;
  top_token_changed: boolean;
  clean_token: string;
  intervention_token: string;
  timestamp: string;
}

export interface CaptureResult {
  target_key: string;
  shape: number[];
  stats: TensorStats;
  values: number[] | null;
}

export interface TensorStats {
  mean: number;
  std: number;
  min: number;
  max: number;
  norm: number;
  shape: number[];
}

export interface TokenInfo {
  token: string;
  token_id: number;
  position: number;
}

export type ComponentType =
  | 'embedding'
  | 'residual_pre'
  | 'residual_post'
  | 'attn_output'
  | 'attn_pattern'
  | 'mlp_gate'
  | 'mlp_output'
  | 'final_logits';

export type InterventionType = 'zero' | 'mean' | 'patch' | 'additive';

export const COMPONENT_LABELS: Record<ComponentType, string> = {
  embedding: 'Embedding',
  residual_pre: 'Residual (Pre)',
  residual_post: 'Residual (Post)',
  attn_output: 'Attention Output',
  attn_pattern: 'Attention Pattern',
  mlp_gate: 'MLP Gate',
  mlp_output: 'MLP Output',
  final_logits: 'Final Logits',
};

export const INTERVENTION_LABELS: Record<InterventionType, string> = {
  zero: 'Zero Ablation',
  mean: 'Mean Ablation',
  patch: 'Activation Patching',
  additive: 'Additive Perturbation',
};
