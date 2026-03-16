"""Evidence board — local memory tool for tracking verified findings during ReAct loop."""

from tools.base import BaseCXRTool


class EvidenceBoard:
    """In-process storage for findings evidence. One instance per agent run."""

    def __init__(self):
        self.confirmed = {}   # finding -> {sources: [], grounding: str|None}
        self.rejected = {}    # finding -> {sources: [], reason: str}

    def add(self, finding: str, sources: list, grounding: str = None) -> str:
        finding = finding.lower().strip()
        if finding in self.rejected:
            del self.rejected[finding]
        if finding in self.confirmed:
            # Merge new sources
            existing = self.confirmed[finding]
            for s in sources:
                if s not in existing["sources"]:
                    existing["sources"].append(s)
            if grounding and not existing["grounding"]:
                existing["grounding"] = grounding
        else:
            self.confirmed[finding] = {
                "sources": list(sources),
                "grounding": grounding,
            }
        entry = self.confirmed[finding]
        g = "grounded" if entry["grounding"] else "NOT grounded"
        return f"Added '{finding}' ({len(entry['sources'])} sources, {g})"

    def reject(self, finding: str, reason: str = "") -> str:
        finding = finding.lower().strip()
        sources = []
        if finding in self.confirmed:
            sources = self.confirmed.pop(finding)["sources"]
        self.rejected[finding] = {"sources": sources, "reason": reason}
        return f"Rejected '{finding}': {reason}"

    def update_grounding(self, finding: str, grounding: str) -> str:
        finding = finding.lower().strip()
        if finding in self.confirmed:
            self.confirmed[finding]["grounding"] = grounding
            return f"Updated grounding for '{finding}'"
        return f"'{finding}' not in confirmed findings — add it first"

    def list_all(self) -> str:
        lines = []
        if self.confirmed:
            lines.append("CONFIRMED FINDINGS:")
            for f, info in self.confirmed.items():
                src_str = ", ".join(info["sources"])
                g = info["grounding"] or "NO GROUNDING"
                lines.append(f"  {f} | sources: [{src_str}] | {g}")
        else:
            lines.append("CONFIRMED FINDINGS: (none)")

        if self.rejected:
            lines.append("REJECTED FINDINGS:")
            for f, info in self.rejected.items():
                lines.append(f"  {f} | reason: {info['reason']}")

        n_confirmed = len(self.confirmed)
        n_grounded = sum(1 for v in self.confirmed.values() if v["grounding"])
        lines.append(f"SUMMARY: {n_confirmed} confirmed, {n_grounded} grounded, {len(self.rejected)} rejected")
        return "\n".join(lines)

    def reset(self):
        self.confirmed.clear()
        self.rejected.clear()


class EvidenceBoardTool(BaseCXRTool):
    """Tool wrapper so the agent can call evidence_board via the standard tool-use API."""

    def __init__(self):
        self.board = EvidenceBoard()

    @property
    def name(self) -> str:
        return "evidence_board"

    @property
    def description(self) -> str:
        return (
            "[MEMORY] "
            "Track and manage your verified findings during analysis. "
            "Use this to build a structured evidence list before writing the final report. "
            "Actions: "
            "'add' — record a confirmed finding with its supporting sources and optional grounding. "
            "'reject' — mark a finding as excluded with a reason. "
            "'update_grounding' — attach grounding (bbox/segmentation) to an already-confirmed finding. "
            "'list' — show all confirmed and rejected findings with their evidence. "
            "Call 'list' before writing your final report to see exactly what to include. "
            "EXAMPLE: "
            "Input: {action: 'add', finding: 'cardiomegaly', sources: ['chexzero: present', 'cxr_foundation: present'], grounding: 'bbox=[0.25, 0.30, 0.75, 0.82]'} → "
            "'Added cardiomegaly (2 sources, grounded)' "
            "EXAMPLE: "
            "Input: {action: 'reject', finding: 'pneumonia', reason: 'only chexzero positive, cxr_foundation and chexagent2 both negative'} → "
            "'Rejected pneumonia: only chexzero positive, cxr_foundation and chexagent2 both negative' "
            "EXAMPLE: "
            "Input: {action: 'list'} → "
            "'CONFIRMED FINDINGS:\n  cardiomegaly | sources: [chexzero: present, cxr_foundation: present] | bbox=[0.25, 0.30, 0.75, 0.82]\n"
            "REJECTED FINDINGS:\n  pneumonia | reason: only chexzero positive\n"
            "SUMMARY: 1 confirmed, 1 grounded, 1 rejected'"
        )

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["add", "reject", "update_grounding", "list"],
                    "description": "Action to perform on the evidence board.",
                },
                "finding": {
                    "type": "string",
                    "description": "The clinical finding (e.g., 'cardiomegaly', 'left pleural effusion'). Required for add/reject/update_grounding.",
                },
                "sources": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of evidence sources (e.g., ['chexzero: present', 'chexagent2_report: mentioned']). Required for add.",
                },
                "grounding": {
                    "type": "string",
                    "description": "Spatial grounding info (e.g., 'bbox=[0.25, 0.30, 0.75, 0.82]' or 'biomedparse: 4.7% coverage'). Optional for add, required for update_grounding.",
                },
                "reason": {
                    "type": "string",
                    "description": "Why this finding is being rejected. Required for reject.",
                },
            },
            "required": ["action"],
        }

    def run(self, action: str, finding: str = "", sources: list = None,
            grounding: str = None, reason: str = "") -> str:
        if action == "add":
            if not finding:
                return "Error: 'finding' is required for add action."
            if not sources:
                return "Error: 'sources' is required for add action."
            return self.board.add(finding, sources, grounding)
        elif action == "reject":
            if not finding:
                return "Error: 'finding' is required for reject action."
            return self.board.reject(finding, reason)
        elif action == "update_grounding":
            if not finding or not grounding:
                return "Error: 'finding' and 'grounding' are required for update_grounding action."
            return self.board.update_grounding(finding, grounding)
        elif action == "list":
            return self.board.list_all()
        else:
            return f"Error: Unknown action '{action}'. Use add, reject, update_grounding, or list."
