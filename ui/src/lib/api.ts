import type { StudyListResponse, StudyDetailResponse, ScoreEntry } from "./types";

const API_BASE = "/api";

async function fetchJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) {
    throw new Error(`API error ${res.status}: ${await res.text()}`);
  }
  return res.json();
}

export function getStudyList(params?: {
  dataset?: string;
  ritl_only?: boolean;
}): Promise<StudyListResponse> {
  const qs = new URLSearchParams();
  if (params?.dataset) qs.set("dataset", params.dataset);
  if (params?.ritl_only) qs.set("ritl_only", "true");
  const query = qs.toString();
  return fetchJSON(`/results${query ? `?${query}` : ""}`);
}

export function getStudyDetail(studyId: string): Promise<StudyDetailResponse> {
  return fetchJSON(`/results/${encodeURIComponent(studyId)}`);
}

export function getScores(): Promise<{ scores: ScoreEntry[]; models: string[] }> {
  return fetchJSON("/scores");
}

export function imageUrl(imagePath: string): string {
  return `${API_BASE}/image?path=${encodeURIComponent(imagePath)}`;
}

// ── Browse all studies (paginated) ───────────────────────────────────────────

export interface StudyBrowseItem {
  study_id: string;
  dataset: string;
  image_path: string;
  is_followup: boolean;
  has_results: boolean;
  num_steps: number | null;
  wall_time_ms: number | null;
  has_ritl: boolean;
}

export interface StudyBrowseResponse {
  studies: StudyBrowseItem[];
  total: number;
  page: number;
  per_page: number;
  total_pages: number;
  datasets: Record<string, number>;
}

export function browseStudies(params: {
  dataset?: string;
  page?: number;
  per_page?: number;
  search?: string;
  has_results?: boolean;
}): Promise<StudyBrowseResponse> {
  const qs = new URLSearchParams();
  if (params.dataset) qs.set("dataset", params.dataset);
  if (params.page) qs.set("page", String(params.page));
  if (params.per_page) qs.set("per_page", String(params.per_page));
  if (params.search) qs.set("search", params.search);
  if (params.has_results !== undefined) qs.set("has_results", String(params.has_results));
  const query = qs.toString();
  return fetchJSON(`/studies${query ? `?${query}` : ""}`);
}

// ── Run API ──────────────────────────────────────────────────────────────────

export interface RunStatus {
  run_id: string;
  study_id: string;
  mode: string;
  status: "queued" | "running" | "complete" | "awaiting_feedback" | "error";
  message: string;
  result: Record<string, unknown> | null;
  trajectory: Record<string, unknown> | null;
  ritl_result: Record<string, unknown> | null;
  ritl_trajectory_steps: unknown[] | null;
  elapsed_ms: number;
}

export async function startRun(
  studyId: string,
  mode: string
): Promise<{ run_id: string }> {
  const res = await fetch(`${API_BASE}/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ study_id: studyId, mode }),
  });
  if (!res.ok) throw new Error(`Start run failed: ${await res.text()}`);
  return res.json();
}

export function pollRun(runId: string): Promise<RunStatus> {
  return fetchJSON(`/run/${runId}`);
}

export async function submitFeedback(
  runId: string,
  feedback: string
): Promise<void> {
  const res = await fetch(`${API_BASE}/run/${runId}/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ feedback }),
  });
  if (!res.ok) throw new Error(`Feedback failed: ${await res.text()}`);
}

export function checkServers(): Promise<{
  servers: Record<string, boolean>;
  all_healthy: boolean;
}> {
  return fetchJSON("/servers/health");
}
