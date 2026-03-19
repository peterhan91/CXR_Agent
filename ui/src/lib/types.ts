// ── Study list (from /api/results) ───────────────────────────────────────────

export interface StudySummary {
  study_id: string;
  dataset: string;
  image_path: string;
  view_position: string;
  is_followup: boolean;
  has_prior: boolean;
  num_steps: number;
  wall_time_ms: number;
  has_ritl_rerun: boolean;
  has_ritl_checkpoint: boolean;
}

export interface StudyListResponse {
  studies: StudySummary[];
  total: number;
  datasets: string[];
  models: string[];
}

// ── Single study detail (from /api/results/:id) ─────────────────────────────

export interface TrajectoryStep {
  iteration: number;
  type: string;
  tool_name: string;
  tool_input: Record<string, unknown>;
  tool_output: string;
  duration_ms: number;
  parallel_count?: number;
}

export interface Trajectory {
  study_id: string;
  concept_prior: string;
  input_tokens: number;
  output_tokens: number;
  num_steps: number;
  wall_time_ms: number;
  groundings: unknown[];
  feedback?: string;
  checkpoint_after?: number;
  steps: TrajectoryStep[];
}

export interface Prediction {
  study_id: string;
  report_pred: string;
  report_pred_raw: string;
  groundings: unknown[];
  input_tokens: number;
  output_tokens: number;
  num_steps: number;
  wall_time_ms: number;
  unused_tools?: string[];
  base_model?: string;
  feedback?: string;
}

export interface PriorStudy {
  image_path: string;
  report: string;
  findings: string;
  impression: string;
  study_date: string;
  study_id: string;
}

export interface TestStudy {
  study_id: string;
  dataset: string;
  image_path: string;
  report_gt: string;
  findings: string;
  impression: string;
  is_followup: boolean;
  prior_study?: PriorStudy;
  metadata: {
    subject_id?: string;
    view_position?: string;
    chexpert_labels?: Record<string, string>;
    study_date?: string;
    admission_info?: Record<string, unknown>;
  };
}

export interface StudyDetailResponse {
  study: TestStudy;
  predictions: Record<string, Prediction>;
  trajectories: Record<string, Trajectory>;
}

// ── Scores ───────────────────────────────────────────────────────────────────

export interface ScoreEntry {
  model: string;
  section: string;
  dataset: string;
  n_studies: number;
  BLEU: number;
  BERT: number;
  Semb: number;
  RadG: number;
  RCliQ: number;
  RaTE: number;
}
