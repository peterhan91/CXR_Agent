"use client";

import { useEffect, useState, useCallback } from "react";
import { useRouter } from "next/navigation";
import { browseStudies } from "@/lib/api";
import type { StudyBrowseItem } from "@/lib/api";

const DATASET_LABELS: Record<string, string> = {
  mimic_cxr: "MIMIC-CXR",
  iu_xray: "IU X-ray",
  chexpert_plus: "CheXpert+",
  rexgradient: "ReXGradient",
};

const DATASET_COLORS: Record<string, string> = {
  mimic_cxr: "text-blue-400",
  iu_xray: "text-green-400",
  chexpert_plus: "text-purple-400",
  rexgradient: "text-orange-400",
};

export default function DashboardPage() {
  const router = useRouter();
  const [studies, setStudies] = useState<StudyBrowseItem[]>([]);
  const [datasetCounts, setDatasetCounts] = useState<Record<string, number>>({});
  const [total, setTotal] = useState(0);
  const [totalPages, setTotalPages] = useState(0);
  const [page, setPage] = useState(1);
  const [dataset, setDataset] = useState<string>("");
  const [search, setSearch] = useState("");
  const [searchInput, setSearchInput] = useState("");
  const [filterResults, setFilterResults] = useState<boolean | undefined>(undefined);
  const [loading, setLoading] = useState(true);

  const fetchStudies = useCallback(async () => {
    setLoading(true);
    try {
      const res = await browseStudies({
        dataset: dataset || undefined,
        page,
        per_page: 50,
        search: search || undefined,
        has_results: filterResults,
      });
      setStudies(res.studies);
      setTotal(res.total);
      setTotalPages(res.total_pages);
      setDatasetCounts(res.datasets);
    } finally {
      setLoading(false);
    }
  }, [dataset, page, search, filterResults]);

  useEffect(() => {
    fetchStudies();
  }, [fetchStudies]);

  // Reset page when filters change
  useEffect(() => {
    setPage(1);
  }, [dataset, search, filterResults]);

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    setSearch(searchInput);
  };

  const totalStudies = Object.values(datasetCounts).reduce((a, b) => a + b, 0);

  return (
    <div className="min-h-screen bg-bg">
      {/* Header */}
      <header className="border-b border-separator">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div>
            <h1 className="text-xl font-semibold text-text-primary">
              CXR Agent
            </h1>
            <p className="text-xs text-text-secondary mt-0.5">
              Agentic CXR report generation with radiologist-in-the-loop
            </p>
          </div>
          <div className="text-xs text-text-tertiary">
            {totalStudies.toLocaleString()} studies across {Object.keys(datasetCounts).length} datasets
          </div>
        </div>
      </header>

      {/* Dataset tabs */}
      <div className="max-w-7xl mx-auto px-6 pt-3">
        <div className="flex items-center gap-1">
          <button
            onClick={() => setDataset("")}
            className={`px-4 py-2 text-xs rounded-t-lg transition-colors ${
              !dataset
                ? "bg-bg-surface text-text-primary border-t border-x border-separator"
                : "text-text-secondary hover:text-text-primary"
            }`}
          >
            All ({totalStudies.toLocaleString()})
          </button>
          {Object.entries(datasetCounts).map(([ds, count]) => (
            <button
              key={ds}
              onClick={() => setDataset(ds)}
              className={`px-4 py-2 text-xs rounded-t-lg transition-colors ${
                dataset === ds
                  ? "bg-bg-surface text-text-primary border-t border-x border-separator"
                  : "text-text-secondary hover:text-text-primary"
              }`}
            >
              <span className={dataset === ds ? DATASET_COLORS[ds] : ""}>
                {DATASET_LABELS[ds] || ds}
              </span>
              <span className="ml-1.5 text-text-tertiary">
                {count.toLocaleString()}
              </span>
            </button>
          ))}
        </div>
      </div>

      {/* Filters row */}
      <div className="max-w-7xl mx-auto px-6 py-3 bg-bg-surface border-x border-b border-separator rounded-b-lg flex items-center gap-3">
        <form onSubmit={handleSearch} className="flex-1 max-w-xs">
          <input
            type="text"
            value={searchInput}
            onChange={(e) => setSearchInput(e.target.value)}
            placeholder="Search study ID..."
            className="w-full bg-bg-elevated text-text-primary text-xs rounded-lg px-3 py-1.5 border border-separator focus:outline-none focus:border-accent placeholder:text-text-tertiary"
          />
        </form>
        <div className="flex items-center gap-1.5">
          <button
            onClick={() => setFilterResults(undefined)}
            className={`px-3 py-1.5 text-xs rounded-lg border transition-colors ${
              filterResults === undefined
                ? "bg-accent/20 border-accent/40 text-accent"
                : "bg-bg-elevated border-separator text-text-secondary hover:text-text-primary"
            }`}
          >
            All
          </button>
          <button
            onClick={() => setFilterResults(true)}
            className={`px-3 py-1.5 text-xs rounded-lg border transition-colors ${
              filterResults === true
                ? "bg-semantic-green/20 border-semantic-green/40 text-semantic-green"
                : "bg-bg-elevated border-separator text-text-secondary hover:text-text-primary"
            }`}
          >
            Has Results
          </button>
          <button
            onClick={() => setFilterResults(false)}
            className={`px-3 py-1.5 text-xs rounded-lg border transition-colors ${
              filterResults === false
                ? "bg-semantic-orange/20 border-semantic-orange/40 text-semantic-orange"
                : "bg-bg-elevated border-separator text-text-secondary hover:text-text-primary"
            }`}
          >
            No Results
          </button>
        </div>
        <span className="text-xs text-text-tertiary ml-auto">
          {total.toLocaleString()} studies
          {search && ` matching "${search}"`}
        </span>
      </div>

      {/* Study list */}
      <div className="max-w-7xl mx-auto px-6 pt-4 pb-8">
        {loading ? (
          <div className="text-center py-20 text-text-tertiary">
            Loading...
          </div>
        ) : (
          <>
            <div className="rounded-panel border border-separator overflow-hidden">
              <table className="w-full">
                <thead>
                  <tr className="bg-bg-surface text-xs text-text-tertiary uppercase tracking-wider">
                    <th className="text-left px-4 py-2 w-1/2">Study ID</th>
                    <th className="text-left px-4 py-2">Dataset</th>
                    <th className="text-center px-4 py-2">Status</th>
                    <th className="text-right px-4 py-2">Steps</th>
                    <th className="text-right px-4 py-2">Time</th>
                    <th className="text-right px-4 py-2 w-8"></th>
                  </tr>
                </thead>
                <tbody>
                  {studies.map((s) => (
                    <tr
                      key={s.study_id}
                      onClick={() =>
                        router.push(
                          `/study/${encodeURIComponent(s.study_id)}`
                        )
                      }
                      className="border-t border-separator hover:bg-bg-surface/50 transition-colors cursor-pointer"
                    >
                      <td className="px-4 py-2.5">
                        <span className="text-sm text-text-primary font-medium font-mono">
                          {s.study_id
                            .replace("mimic_", "")
                            .replace("chexpert_", "")
                            .replace("iu_", "")
                            .replace("rexgrad_", "")
                            .slice(0, 40)}
                        </span>
                        {s.is_followup && (
                          <span className="ml-2 text-xs text-text-tertiary">
                            F/U
                          </span>
                        )}
                      </td>
                      <td className="px-4 py-2.5">
                        <span
                          className={`text-xs ${
                            DATASET_COLORS[s.dataset] || "text-text-secondary"
                          }`}
                        >
                          {DATASET_LABELS[s.dataset] || s.dataset}
                        </span>
                      </td>
                      <td className="px-4 py-2.5 text-center">
                        {s.has_results ? (
                          <span className="inline-flex items-center gap-1">
                            <span className="w-1.5 h-1.5 rounded-full bg-semantic-green" />
                            <span className="text-xs text-semantic-green">
                              Evaluated
                            </span>
                          </span>
                        ) : (
                          <span className="text-xs text-text-tertiary">
                            —
                          </span>
                        )}
                        {s.has_ritl && (
                          <span className="ml-1.5 w-1.5 h-1.5 rounded-full bg-semantic-orange inline-block" title="Has RITL" />
                        )}
                      </td>
                      <td className="px-4 py-2.5 text-xs text-text-secondary text-right">
                        {s.num_steps ?? "—"}
                      </td>
                      <td className="px-4 py-2.5 text-xs text-text-secondary text-right">
                        {s.wall_time_ms != null
                          ? `${(s.wall_time_ms / 1000).toFixed(1)}s`
                          : "—"}
                      </td>
                      <td className="px-4 py-2.5 text-right">
                        <span className="text-text-secondary text-sm">
                          &rarr;
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Pagination */}
            {totalPages > 1 && (
              <div className="flex items-center justify-center gap-2 mt-4">
                <button
                  onClick={() => setPage(Math.max(1, page - 1))}
                  disabled={page === 1}
                  className="px-3 py-1 text-xs rounded bg-bg-elevated text-text-secondary hover:text-text-primary disabled:opacity-30 disabled:cursor-not-allowed"
                >
                  Prev
                </button>
                <span className="text-xs text-text-secondary">
                  Page {page} of {totalPages}
                </span>
                <button
                  onClick={() => setPage(Math.min(totalPages, page + 1))}
                  disabled={page === totalPages}
                  className="px-3 py-1 text-xs rounded bg-bg-elevated text-text-secondary hover:text-text-primary disabled:opacity-30 disabled:cursor-not-allowed"
                >
                  Next
                </button>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
