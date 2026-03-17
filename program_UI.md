# CXR Agent — Web UI Program

Interactive web UI for the CXR agentic workflow: live agent execution, grounded report with per-finding bbox toggles, and radiologist-in-the-loop editing. Apple design language (dark mode, frosted glass panels, SF typography, semantic colors).

Design informed by [TissueLab](https://github.com/zhihuanglab/TissueLab) (agentic AI + human-in-the-loop for pathology), [Annalise.ai](figures/GUI_Lancet.png) (findings sidebar + CXR overlay), and [Centaur Labs](figures/GUI_NEJM_AI.jpg) (bbox annotation on CXR).

## TissueLab Component Mapping

| TissueLab Component | Our CXR Agent Equivalent | Notes |
|---|---|---|
| `dashboard.tsx` + `WebFileManager` | Study browser (`page.tsx`) | File upload + study list |
| `OpenSeadragonContainer` | `CXRViewer.tsx` (HTML Canvas) | Plain PNG/JPEG — no DICOM, no Cornerstone needed |
| `DrawingOverlay` + `PatchOverlay` | Bbox + mask overlays on CXR | Per-finding toggle, color-coded |
| `SidebarChat` + `SidebarBotOnly` | `WorkflowPanel.tsx` | Agent trajectory stream instead of chat |
| `SidebarWorkflowOnly` | `LiveTimeline.tsx` | 6-phase pipeline visualization |
| `SidebarAnnotation` + `AnnotationPopup` | `FindingEditor.tsx` + `DrawBbox.tsx` | Inline edit + mouse-draw rectangle |
| `ActiveLearningPanel` + `CandidateGallery` | Radiologist review panel | Accept/reject/reclassify findings |
| `ProbabilityCurve` | Classifier agreement chart | CheXZero vs CXR Foundation vs CheXagent-2 |
| `ThresholdController` | Mask opacity slider | BiomedParse segmentation overlay |
| `AIModelZoo` | Tool config panel (Phase 4) | Enable/disable specific CXR tools |
| `SidebarPythonScripts` | Not needed | Our tools are HTTP servers |
| Electron desktop wrapper | Not needed | Web-only for research |

**Key architectural differences from TissueLab:**
- **Image viewer**: HTML Canvas with zoom/pan (PNG/JPEG inputs — no DICOM, no Cornerstone.js needed) instead of OpenSeadragon (pathology WSI)
- **Brightness/contrast**: CSS filters on `<img>` or Canvas pixel manipulation — not DICOM window/level, since inputs are 8-bit PNG/JPEG already processed by the agent's `_encode_image()` normalizer
- **Agent transparency**: We show full tool call trajectory with disagreement highlighting — TissueLab's workflow is more opaque
- **Structured edits**: Text/draw/voice input for finding-level edits instead of free-form chat
- **No desktop app**: Web-only, accessed via SSH tunnel or VPN (medical images stay on GPU server)

## Architecture

**Option B — doctor testing (recommended):** everything on GPU server, doctors access via VPN.

```
┌─ Doctor's Machine (on VPN) ──────────────────────────────────────┐
│  Browser → http://gpu-server-vpn-ip:3000                         │
└──────────────────────────┬───────────────────────────────────────┘
                           │ VPN network
┌─ GPU Server ─────────────┼───────────────────────────────────────┐
│  Next.js frontend :3000  ←┘  (bound to VPN IP, not 0.0.0.0)     │
│    ──→ FastAPI gateway :9000 (bound to 127.0.0.1)                │
│           ├→ Agent (react_agent.py)  ──→ tool servers :8001-:8010│
│           ├→ /api/run          POST   start agent run            │
│           ├→ /api/ws           WS     stream trajectory live     │
│           ├→ /api/results      GET    list past results          │
│           ├→ /api/results/:id  GET    single result + trajectory │
│           ├→ /api/edit         POST   radiologist overrides (L1)  │
│           ├→ /api/feedback     POST   feedback injection (L2)    │
│           ├→ /api/tools/:name  POST   re-run single tool         │
│           └→ /api/image/:path  GET    serve CXR image            │
│                                                                   │
│  Tool servers (unchanged, localhost only):                        │
│    :8001 CheXagent-2  :8002 CheXOne     :8004 MedVersa           │
│    :8005 BiomedParse  :8007 FactCheXcker :8008 CXR Foundation    │
│    :8009 CheXzero     :8010 MedGemma                              │
└───────────────────────────────────────────────────────────────────┘
```

**Option A — development:** frontend on local machine, SSH tunnel to gateway.

```
┌─ Local Machine (macOS) ──────────────────────────────────────────┐
│  Browser → http://localhost:3000  (Next.js dev server)           │
│  Next.js frontend ──→ http://localhost:9000/api/*                │
│                         ↑ SSH tunnel                             │
└─────────────────────────┼────────────────────────────────────────┘
                          │ ssh -L 9000:localhost:9000 gpu-server
┌─ GPU Server ────────────┼────────────────────────────────────────┐
│  FastAPI gateway :9000  ←┘  (bound to 127.0.0.1)                │
│    ├→ Agent + tool servers :8001-:8010 (localhost only)           │
└──────────────────────────────────────────────────────────────────┘
```

### Networking: How to access the UI

The model/tool servers run on a GPU server accessible only via VPN + SSH. For doctor testing, use Option B (simplest — doctors just open a URL on VPN). For development, use Option A or C.

**Option A: SSH tunnel (recommended for development)**

Run the Next.js frontend locally, tunnel only the API gateway:

```bash
# Terminal 1: SSH tunnel — forwards gateway port to localhost
ssh -L 9000:localhost:9000 gpu-server

# On GPU server: start the API gateway
conda run -n cxr_agent python ui/server/gateway.py --port 9000

# Terminal 2: local machine — start frontend
cd ui && npm run dev   # → http://localhost:3000
```

Frontend calls `http://localhost:9000/api/*` which tunnels to the GPU server.

**Option B: Run everything on GPU server (recommended for doctor testing)**

```bash
# On GPU server:
conda run -n cxr_agent python ui/server/gateway.py --port 9000  # binds 127.0.0.1 by default
cd ui && npm run build && node server.js                          # binds to VPN IP, proxies WS

# Doctors on VPN open: http://gpu-server-vpn-ip:3000
```

No tunnel needed — both frontend and backend on the same machine. Only the frontend (:3000) is exposed on VPN; the gateway (:9000) and tool servers (:8001-:8010) stay on localhost. Doctors only need VPN access, nothing else.

Next.js must proxy `/api/*` to the gateway. Two configs needed — HTTP rewrites for REST endpoints, and a custom server for WebSocket:

```js
// next.config.js — proxy REST API calls to gateway on localhost
module.exports = {
  async rewrites() {
    return [
      { source: '/api/:path*', destination: 'http://127.0.0.1:9000/api/:path*' },
    ];
  },
};
```

Next.js `rewrites()` does NOT proxy WebSocket upgrades. For live agent streaming (`/api/ws/{run_id}`), use a custom server wrapper:

```js
// server.js — custom Next.js server with WebSocket proxy
const { createServer } = require('http');
const { parse } = require('url');
const next = require('next');
const httpProxy = require('http-proxy');

const app = next({ dev: false });
const handle = app.getRequestHandler();
const proxy = httpProxy.createProxyServer({ target: 'http://127.0.0.1:9000', ws: true });

app.prepare().then(() => {
  const server = createServer((req, res) => handle(req, res, parse(req.url, true)));
  server.on('upgrade', (req, socket, head) => {
    if (req.url.startsWith('/api/ws')) {
      proxy.ws(req, socket, head);
    }
  });
  server.listen(3000, '10.x.x.x');  // bind to VPN IP
});
```

Start with: `node server.js` instead of `npm start`.

This way the doctor's browser only talks to `:3000`. REST requests are proxied via Next.js rewrites, WebSocket upgrades are proxied via `http-proxy` — the gateway is never exposed on the network.

**Option C: VS Code Remote + port forwarding (easiest for solo dev)**

```bash
# VS Code Remote-SSH auto-forwards ports. Just:
# 1. Connect to gpu-server via VS Code Remote
# 2. Start gateway + frontend on gpu-server
# 3. VS Code auto-forwards 3000 + 9000 to localhost
```

**Option D: Public domain (future — e.g. `https://app.cxragent.org/`)**

For later public/demo deployment. See appendix at end of this doc. Not needed for internal testing.

## Tech Stack

| Layer | Choice | Why |
|-------|--------|-----|
| Frontend | Next.js 14 + React + TypeScript | TissueLab-proven; App Router for layouts |
| CXR Viewer | HTML Canvas + `<img>` | Inputs are PNG/JPEG (not DICOM) — no Cornerstone.js needed |
| Brightness/Contrast | CSS `filter: brightness() contrast()` | 8-bit images, simple slider controls |
| Bbox overlay | Canvas 2D (`strokeRect`) | Draw/toggle colored rectangles per finding |
| Mask overlay | Canvas 2D (`drawImage` with alpha) | BiomedParse masks rendered as semi-transparent layer |
| Drawing tool | Canvas `mousedown/mousemove/mouseup` | Radiologist draws rectangle → normalized coords |
| Zoom/Pan | CSS `transform: scale() translate()` | Or `wheel` + drag handlers on a wrapper div |
| Styling | Tailwind CSS | Apple design tokens as Tailwind config |
| State | Zustand | Lightweight, minimal boilerplate |
| Real-time | WebSocket (native) | Stream agent trajectory steps live |
| Voice input | Web Speech API | Zero dependency, browser-native |
| API Gateway | FastAPI + WebSocket | Thin layer on GPU server; calls existing tool servers |
| Agent | Existing `react_agent.py` | Gateway calls `CXRReActAgent.run()`; L2 adds `continue_with_feedback()` |

## Pages (inspired by TissueLab's dashboard → imageViewer flow)

### Page 1: Dashboard (`/`)

Study browser + file manager. Like TissueLab's `dashboard.tsx` but for CXR studies.

```
┌──────────────────────────────────────────────────────────────────┐
│  CXR Agent                                    [Upload CXR] [Run]│
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Recent Studies                                   Filter: [All ▾]│
│  ┌────────┬──────────────┬──────────┬────────┬─────────────────┐│
│  │ Thumb  │ MIMIC/p10032 │ 8 tools  │ 34.2s  │ RadCliQ: 0.72  ││
│  │  [CXR] │ 3 findings   │ Phase 6✓ │ 2m ago │ [View →]       ││
│  ├────────┼──────────────┼──────────┼────────┼─────────────────┤│
│  │  [CXR] │ CheXpert/042 │ 6 tools  │ 28.1s  │ RadCliQ: 0.68  ││
│  │        │ 5 findings   │ Phase 6✓ │ 5m ago │ [View →]       ││
│  ├────────┼──────────────┼──────────┼────────┼─────────────────┤│
│  │  [CXR] │ IU-Xray/1234 │ running  │ ●●●    │                ││
│  │        │ Phase 3/6    │ 12.4s    │ live   │ [View →]       ││
│  └────────┴──────────────┴──────────┴────────┴─────────────────┘│
│                                                                  │
│  Batch: eval_20260316 (120 studies)              [Compare Mode] │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ MIMIC: 30 ✓  CheXpert: 30 ✓  RexGrad: 30 ✓  IU: 30 ✓     ││
│  │ Avg RadCliQ: 0.71  |  Avg GREEN: 0.68  |  Avg tools: 7.2  ││
│  └─────────────────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────────────────┘
```

### Page 2: Study Viewer (`/study/[id]`)

3-panel layout (described below). Click any study row to enter this view.

## UI Layout (Apple Design Language)

```
┌──────────────────────────────────────────────────────────────────┐
│  ◉ ◉ ◉   CXR Agent        Study: p10032546       ◐ Dark        │
├────────────────────┬─────────────────────┬───────────────────────┤
│                    │                     │                       │
│  [CXR Image]       │  FINDINGS           │  Agent Workflow       │
│                    │  ┌───────────────┐  │  ┌─────────────────┐  │
│  Canvas (PNG/JPEG) │  │ ■ Cardiomeg.  │←→│  │ Phase 1 ██████ ✓│  │
│  - brightness/ctr  │  │ ■ L effusion  │  │  │ Phase 2 ████░░ ●│  │
│  - bbox overlays   │  │ □ ETT in situ │  │  │ Phase 3 ░░░░░░  │  │
│  - mask overlays   │  │               │  │  │ ...              │  │
│  - draw rectangle  │  │ [+ Add]       │  │  │                  │  │
│  - zoom/pan        │  ├───────────────┤  │  │ Tool calls:      │  │
│  - prior compare   │  │ IMPRESSION    │  │  │ ▸ chexagent2_rpt │  │
│                    │  │ (editable)    │  │  │ ▸ chexzero_cls   │  │
│                    │  ├───────────────┤  │  │ ▸ ...            │  │
│  ☀ Brightness ──── │  │ 🎤 Voice  ✎  │  │  │                  │  │
│  ◐ Contrast ────── │  └───────────────┘  │  │ ⏱ 34s  🔧 8 calls│  │
│   ~50% width       │    ~25% width       │    ~25% width        │
└────────────────────┴─────────────────────┴───────────────────────┘
```

### Design Tokens (Tailwind config)

```js
// tailwind.config.js — Apple design language
colors: {
  bg:        { DEFAULT: '#000000', surface: '#1C1C1E', elevated: '#2C2C2E' },
  glass:     { DEFAULT: 'rgba(44,44,46,0.72)' },
  text:      { primary: '#F5F5F7', secondary: '#86868B', tertiary: '#48484A' },
  accent:    { DEFAULT: '#0071E3', hover: '#0077ED' },
  semantic:  { green: '#34C759', orange: '#FF9F0A', red: '#FF3B30' },
  separator: 'rgba(255,255,255,0.06)',
},
fontFamily: {
  sans: ['-apple-system', 'BlinkMacSystemFont', 'SF Pro Text', 'Inter', 'sans-serif'],
},
backdropBlur: { glass: '20px' },
borderRadius: { panel: '12px' },
```

## Development Phases

### Phase 0: API Gateway (GPU server)

Thin FastAPI app that wraps the existing agent and tool servers.

```
ui/server/
├── gateway.py              # FastAPI app, CORS, WebSocket
├── routes/
│   ├── run.py              # POST /api/run — start agent, return run_id
│   ├── ws.py               # WS /api/ws/:run_id — stream trajectory
│   ├── results.py          # GET /api/results, GET /api/results/:id
│   ├── edit.py             # POST /api/edit — apply radiologist overrides (L1)
│   ├── feedback.py         # POST /api/feedback — inject edits, agent re-reasons (L2)
│   └── tools.py            # POST /api/tools/:name — re-run single tool
└── requirements.txt        # fastapi, uvicorn, websockets (lightweight)
```

Key endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/run` | POST | `{ image_path, config?, prior_report? }` → start agent, return `run_id` |
| `/api/ws/{run_id}` | WS | Stream: `{ phase, tool_name, tool_output, duration_ms }` per step |
| `/api/results` | GET | List all past runs (from `results/` dir) |
| `/api/results/{run_id}` | GET | Full result JSON + trajectory |
| `/api/edit` | POST | `{ run_id, edits: [{action, finding, bbox?}] }` → updated report (L1: static) |
| `/api/feedback` | POST | `{ run_id, edits: [...] }` → agent re-reasons, returns updated report (L2) |
| `/api/tools/{tool_name}` | POST | `{ image_path, params }` → re-run one tool |
| `/api/image/{path}` | GET | Serve CXR image (PNG/JPEG) to frontend |

### Phase 1: Static Result Viewer (v0.1)

Load an existing result JSON, display CXR + report + bbox overlays.

```
ui/
├── app/
│   ├── layout.tsx          # Root layout, dark theme, font
│   ├── page.tsx            # Study list / file picker
│   └── study/[id]/
│       └── page.tsx        # 3-panel study viewer
├── components/
│   ├── CXRViewer.tsx       # Canvas-based PNG/JPEG viewer + bbox/mask overlay
│   ├── ReportPanel.tsx     # FINDINGS + IMPRESSION display
│   ├── FindingRow.tsx      # Single finding: checkbox, color dot, label
│   ├── WorkflowPanel.tsx   # Agent phase timeline (read-only)
│   └── ToolCallCard.tsx    # Expandable tool call detail
├── stores/
│   └── studyStore.ts       # Zustand: findings[], bboxVisible{}, activePhase
├── lib/
│   ├── api.ts              # Fetch wrapper for gateway
│   └── types.ts            # Result, Finding, Grounding, TrajectoryStep
├── tailwind.config.js
├── package.json
└── next.config.js
```

**Deliverables:**
- [ ] Load result JSON (from file or gateway API)
- [ ] CXR image display (PNG/JPEG) with brightness/contrast sliders
- [ ] Bbox overlays per finding, color-coded, individually togglable
- [ ] Segmentation mask overlay (BiomedParse) with opacity slider
- [ ] Findings list with confidence dots (green/orange/red based on classifier agreement)
- [ ] Agent timeline: 6 phases, tool calls expandable with raw output
- [ ] Dark mode only (radiologist default)

### Phase 2: Live Agent Execution (v0.2)

Run agent from UI, stream trajectory via WebSocket.

**New components:**
```
├── components/
│   ├── RunControls.tsx     # Image picker, config selector, [Run Agent] button
│   ├── LiveTimeline.tsx    # Animated phase progress, tool calls append in real-time
│   └── ConceptPrior.tsx    # Show CLEAR top-K concepts for current image
```

**Deliverables:**
- [ ] Upload or select CXR image → POST /api/run
- [ ] WebSocket connection streams each tool call as it happens
- [ ] Phase indicator animates through 6 phases
- [ ] Tool call cards appear one by one with duration
- [ ] Final report renders when agent completes
- [ ] CLEAR concept prior displayed as ranked list

### Phase 3: Radiologist-in-the-Loop (v0.3)

Three levels of doctor integration, from simple to deep:

#### Level 1: Post-hoc Override (v0.3)

Doctor edits the finished report directly. No agent re-reasoning — just modify the output JSON.

**New components:**
```
├── components/
│   ├── FindingEditor.tsx   # Inline text edit per finding
│   ├── DrawBbox.tsx        # Mouse rectangle tool on CXR canvas
│   ├── VoiceInput.tsx      # Web Speech API → text → parsed edits
│   ├── AddFinding.tsx      # "+ Add finding" with auto-grounding
│   └── RerunTool.tsx       # "Re-check with VQA" button per finding
```

**Interaction model:**

| Action | Input | Result |
|--------|-------|--------|
| Toggle finding | Click checkbox | Finding included/excluded; bbox shows/hides |
| Edit text | Click finding text → inline edit | Report text updates |
| Draw bbox | Select drawing tool → drag on image | New grounding attached to selected finding |
| Add finding | Click [+ Add] → type finding name | Auto-runs `chexagent2_grounding` to get bbox |
| Remove finding | Click ✗ → confirm | Soft delete (can undo) |
| Voice dictate | Click 🎤 → speak | Speech-to-text → NLU parses into structured edits |
| Re-run tool | Click tool name → "Re-run" | Sends request to gateway → tool result updates |

**Deliverables:**
- [ ] Inline finding text editing with live report preview
- [ ] Mouse-draw rectangle → normalized bbox → attach to finding
- [ ] Add new finding → auto-ground via tool call
- [ ] Remove finding with undo (30s)
- [ ] Voice input: browser Speech API, parsed into add/remove/edit actions
- [ ] Re-run single tool for a specific finding
- [ ] Edit history (who changed what, when)

#### Level 2: Feedback Injection — Agent Re-reasons (v0.5)

Doctor's edits are injected as a new user message into the agent's conversation history. The agent re-enters the ReAct loop, calls tools as needed, and produces an updated report. This leverages the existing `react_agent.py` architecture — tool results already come back as `role: "user"` messages, so doctor feedback uses the same mechanism.

**How it works:**

```
Agent run (Phase 1-6) → draft report
           ↓
Doctor reviews: removes "pleural effusion", adds "atelectasis"
           ↓
Gateway constructs a structured user message:
  "The radiologist has reviewed your report and made these corrections:
   REMOVED: pleural effusion (radiologist: not convincing on this image)
   ADDED: right lower lobe atelectasis
   KEPT: cardiomegaly, ETT in situ
   Please update your report: re-ground new findings, remove groundings
   for removed findings, and rewrite FINDINGS/IMPRESSION accordingly."
           ↓
Agent re-enters ReAct loop with full prior context:
  → calls chexagent2_grounding("atelectasis") for the new finding
  → calls biomedparse_segment("atelectasis") for coverage
  → removes pleural effusion grounding
  → rewrites report incorporating doctor's corrections
           ↓
Updated report + groundings returned to UI
```

**Implementation in `react_agent.py`:**

```python
# New method: continue an existing run with doctor feedback
def continue_with_feedback(
    self,
    trajectory: AgentTrajectory,
    messages: list,          # preserved from original run
    system_prompt: str,      # preserved from original run
    feedback: str,           # structured doctor feedback text
) -> AgentTrajectory:
    """Resume the ReAct loop with radiologist feedback injected."""
    messages.append({
        "role": "user",
        "content": feedback,
    })
    # Re-enter the same loop from react_agent.py:247
    # Agent sees full history + doctor's corrections → calls tools → updates report
    for iteration in range(self.max_iterations):
        # ... same loop as run(), appending to existing trajectory
```

**Gateway endpoint:**

```
POST /api/feedback
{
  "run_id": "abc123",
  "edits": [
    {"action": "remove", "finding": "pleural effusion", "reason": "not convincing"},
    {"action": "add",    "finding": "right lower lobe atelectasis"},
    {"action": "keep",   "finding": "cardiomegaly"},
    {"action": "edit",   "finding": "ETT in situ", "text": "ETT tip 4cm above carina"}
  ]
}
→ Gateway formats edits into natural language → calls agent.continue_with_feedback()
→ Returns updated report + trajectory (streamed via WebSocket)
```

**State persistence:** The gateway must preserve `messages` and `system_prompt` from the original run, keyed by `run_id`. On `/api/feedback`, it loads these, appends the doctor's feedback, and re-enters the loop. Simple in-memory dict works for testing (single server, few concurrent runs); persist to disk/Redis for production.

**Why this works well:**
- No changes to the core ReAct loop logic — just appending a user message and continuing
- Agent has full prior context (all tool outputs from the original run) so it doesn't re-call unnecessary tools
- Agent can reason about the doctor's feedback ("the radiologist says effusion is not convincing, and indeed CheXZero score was only 0.31...")
- New findings get properly grounded via tool calls
- Multiple rounds of feedback possible (doctor corrects → agent updates → doctor refines again)

#### Level 3: Checkpoint Review — Doctor Reviews Mid-Loop (v0.6)

Agent pauses at a configurable checkpoint (e.g., after Phase 3: findings confirmed but not yet grounded). Doctor reviews the intermediate state, accepts/rejects findings, then the agent continues from that point.

**How it works:**

```
Agent runs Phase 1-3 → pauses
  ↓
WebSocket sends checkpoint event to UI:
  { "type": "checkpoint", "phase": 3, "findings": [...], "classifier_scores": {...} }
  ↓
UI shows: findings list with classifier agreement scores
  Doctor: ✓ cardiomegaly, ✓ atelectasis, ✗ effusion (removes), + "rib fracture" (adds)
  ↓
Doctor clicks [Continue]
  ↓
Gateway injects doctor's decisions as user message → agent resumes at Phase 4
  Agent only grounds accepted findings → skips removed ones → grounds new ones
  → Phase 5 verify → Phase 6 write report
```

**Implementation — add checkpoint support to `react_agent.py`:**

```python
class CXRReActAgent:
    def __init__(self, ..., checkpoints: list[int] = None):
        # checkpoints = [3] means pause after Phase 3
        self.checkpoints = checkpoints or []
        self._pause_event = None   # asyncio.Event, set by gateway

    async def run_async(self, ..., on_checkpoint=None):
        """Async version of run() that supports mid-loop pausing."""
        for iteration in range(self.max_iterations):
            # ... normal loop ...

            # Detect phase transitions from tool calls
            current_phase = self._detect_phase(trajectory)
            if current_phase in self.checkpoints and on_checkpoint:
                # Pause: send intermediate state to UI, wait for doctor
                feedback = await on_checkpoint(current_phase, trajectory)
                if feedback:
                    messages.append({"role": "user", "content": feedback})
```

**Gateway WebSocket flow:**

```
Client                    Gateway                   Agent
  |                          |                         |
  |--- POST /api/run ------->|                         |
  |                          |--- run_async() -------->|
  |                          |                         | Phase 1-3 tools...
  |<-- WS: tool_call events --|<-- stream -------------|
  |                          |                         | checkpoint! Phase 3
  |<-- WS: checkpoint -------|<-- on_checkpoint() -----|
  |                          |                         | (agent paused)
  | Doctor reviews...        |                         |
  |--- WS: feedback -------->|                         |
  |                          |--- inject feedback ---->| (agent resumes)
  |                          |                         | Phase 4-6 tools...
  |<-- WS: tool_call events --|<-- stream -------------|
  |<-- WS: complete ---------|<-- final report --------|
```

**Checkpoint UI component:**

```
├── components/
│   └── CheckpointReview.tsx  # Modal overlay during agent pause
│       Shows: findings table with classifier scores
│       Actions: ✓ accept / ✗ reject / + add / edit per finding
│       Button: [Continue Agent] / [Abort]
```

### Summary: Three Levels

| Level | When | Doctor input | Agent response | Complexity |
|-------|------|-------------|----------------|------------|
| **L1: Override** (v0.3) | After agent finishes | Edit output directly | None — static edit | Low |
| **L2: Feedback** (v0.5) | After agent finishes | Structured edits → user message | Re-enters loop, calls tools, rewrites report | Medium |
| **L3: Checkpoint** (v0.6) | Mid-loop (after Phase 3) | Accept/reject findings before grounding | Continues loop with doctor's decisions | High |

**Plan:** L1 (v0.3) as stepping stone — get the UI editing interactions right. Then L2 (v0.3 → v0.5) is the target for doctor testing — doctors get real agent re-reasoning with minimal code change (`continue_with_feedback()` is just appending a user message and re-entering the existing loop). L3 is future work if checkpoint review proves valuable.

### Phase 4: Multi-Study & Comparison (v0.4)

For evaluation and research workflows.

**Deliverables:**
- [ ] Study list browser (filter by dataset, study ID)
- [ ] Side-by-side: agent report vs ground truth vs baseline
- [ ] Batch view: scroll through 120 studies with metrics per study
- [ ] Score overlay: RadCliQ, GREEN, RaTEScore per study
- [ ] Export edited reports as CSV (compatible with `eval_mimic.py --mode score`)

## File Structure (full)

```
ui/
├── server/                     # API gateway (runs on GPU server)
│   ├── gateway.py
│   ├── routes/
│   │   ├── run.py
│   │   ├── ws.py
│   │   ├── results.py
│   │   ├── edit.py
│   │   ├── feedback.py
│   │   └── tools.py
│   └── requirements.txt
├── app/                        # Next.js App Router
│   ├── layout.tsx
│   ├── page.tsx
│   ├── study/[id]/page.tsx
│   └── globals.css
├── components/
│   ├── CXRViewer.tsx
│   ├── ReportPanel.tsx
│   ├── FindingRow.tsx
│   ├── WorkflowPanel.tsx
│   ├── ToolCallCard.tsx
│   ├── RunControls.tsx
│   ├── LiveTimeline.tsx
│   ├── ConceptPrior.tsx
│   ├── FindingEditor.tsx
│   ├── DrawBbox.tsx
│   ├── VoiceInput.tsx
│   ├── AddFinding.tsx
│   ├── RerunTool.tsx
│   └── CheckpointReview.tsx    # L3 future: mid-loop review modal
├── stores/
│   └── studyStore.ts
├── lib/
│   ├── api.ts
│   └── types.ts
├── public/
├── tailwind.config.js
├── tsconfig.json
├── package.json
├── next.config.js
└── server.js                   # Custom server for WebSocket proxy (Option B)
```

## Quick Start

**For doctor testing (Option B — everything on GPU server):**

```bash
# On GPU server:

# 1. Start tool servers (same as program.md)
conda run -n cxr_chexagent2 python servers/chexagent2_server.py --port 8001 &
CUDA_VISIBLE_DEVICES=1 python servers/chexone_server.py --port 8002 &
# ... (all servers as in program.md)

# 2. Start API gateway (localhost only — frontend proxies to it)
conda run -n cxr_agent python ui/server/gateway.py --port 9000

# 3. Build + start frontend (bind to VPN IP so doctors can reach it)
cd ui && npm install && npm run build && node server.js

# Doctors on VPN open: http://gpu-server-vpn-ip:3000
```

**For development (Option A — SSH tunnel):**

```bash
# Terminal 1: SSH tunnel
ssh -L 9000:localhost:9000 gpu-server

# On GPU server: start gateway
conda run -n cxr_agent python ui/server/gateway.py --port 9000

# Terminal 2: local frontend
cd ui && npm install && npm run dev
# Open http://localhost:3000
```

## Setup Reference

Inherits from `program.md`:

| Port | Server | GPU |
|------|--------|-----|
| 8001 | CheXagent-2 | 0 |
| 8002 | CheXOne | 1 |
| 8004 | MedVersa | 1 |
| 8005 | BiomedParse | 1 |
| 8007 | FactCheXcker | 2 |
| 8008 | CXR Foundation | CPU |
| 8009 | CheXzero | 1 |
| 8010 | MedGemma | 2 |
| **9000** | **API Gateway (new)** | **CPU** |
| **3000** | **Next.js frontend** | **local or GPU server** |

## Security (internal testing with doctors)

For now we're VPN + SSH only. The GPU server is never directly exposed to the internet.

**What's already safe (Option B):**
- Tool servers (:8001-:8010) bind to `localhost` — only reachable from the GPU server itself
- Gateway (:9000) binds to `localhost` — only the frontend can reach it, not doctors directly
- Only the Next.js frontend (:3000) is exposed on VPN — it proxies API calls to the gateway internally
- No patient data leaves the GPU server — images served through the gateway, never copied to doctor's machine
- Doctors only need VPN access — no SSH, no tunnel, no setup on their end

**Minimum precautions for doctor testing:**

```python
# In gateway.py — always bind to localhost (gateway is internal only)
uvicorn.run(app, host="127.0.0.1", port=9000)
# Next.js frontend handles external access; it proxies /api/* to localhost:9000
```

| Risk | Mitigation |
|------|------------|
| Path traversal via `/api/image/` | Whitelist allowed directories (e.g., only `data/` and `results/`) |
| Unbounded agent runs eating GPU | Rate limit: 1 concurrent run per user, 60s timeout |
| Doctors uploading non-CXR files | Validate file type (PNG/JPEG only), max 10MB |
| Tool server crash from bad input | Gateway validates `image_path` exists before forwarding |
| Anthropic API key exposure | Key stays in server env, never sent to frontend |

**Gateway path validation (critical):**

```python
# gateway.py — prevent path traversal
import os

ALLOWED_ROOTS = ["/home/than/DeepLearning/CXR_Agent/data",
                 "/home/than/DeepLearning/CXR_Agent/results"]

def validate_image_path(path: str) -> str:
    resolved = os.path.realpath(path)
    if not any(resolved.startswith(root) for root in ALLOWED_ROOTS):
        raise HTTPException(403, "Access denied: path outside allowed directories")
    if not resolved.lower().endswith((".png", ".jpg", ".jpeg")):
        raise HTTPException(400, "Only PNG/JPEG files allowed")
    return resolved
```

**For later (public deployment):** See Appendix A.

---

## Appendix A: Public Domain Deployment (future)

For `https://app.cxragent.org/` — not needed for internal testing.

### Option 1: Cloudflare Tunnel (recommended — no public IP needed)

```
┌─ Browser ──────────────────────────────────────────────────────────┐
│  https://app.cxragent.org  →  Cloudflare Edge (TLS termination)   │
└───────────────────────────────┬────────────────────────────────────┘
                                │ encrypted tunnel (outbound-only)
┌─ GPU Server (behind VPN/NAT) ─┼───────────────────────────────────┐
│  cloudflared tunnel           ←┘                                   │
│    → localhost:3000 (Next.js)                                      │
│    → localhost:9000 (FastAPI gateway)                               │
│         → :8001-:8010 (tool servers)                               │
└────────────────────────────────────────────────────────────────────┘
```

```bash
# 1. Install cloudflared on GPU server
curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \
  -o /usr/local/bin/cloudflared && chmod +x /usr/local/bin/cloudflared

# 2. Authenticate + create tunnel
cloudflared tunnel login
cloudflared tunnel create cxr-agent
cloudflared tunnel route dns cxr-agent app.cxragent.org

# 3. Config: ~/.cloudflared/config.yml
cat > ~/.cloudflared/config.yml << 'EOF'
tunnel: <TUNNEL_UUID>
credentials-file: ~/.cloudflared/<TUNNEL_UUID>.json
ingress:
  - hostname: app.cxragent.org
    service: http://localhost:3000
  - hostname: api.cxragent.org
    service: http://localhost:9000
  - service: http_status:404
EOF

# 4. Run
cloudflared tunnel run cxr-agent
```

Add **Cloudflare Access** for authentication — restrict to `@your-institution.edu` emails. Free tier.

### Option 2: nginx + Let's Encrypt (if GPU server has public IP)

```bash
sudo apt install nginx certbot python3-certbot-nginx

# /etc/nginx/sites-available/cxragent
server {
    server_name app.cxragent.org;
    location / { proxy_pass http://localhost:3000; proxy_http_version 1.1;
                 proxy_set_header Upgrade $http_upgrade; proxy_set_header Connection "upgrade"; }
    location /api/ { proxy_pass http://localhost:9000; proxy_http_version 1.1;
                     proxy_set_header Upgrade $http_upgrade; proxy_set_header Connection "upgrade"; }
}

sudo ln -s /etc/nginx/sites-available/cxragent /etc/nginx/sites-enabled/
sudo certbot --nginx -d app.cxragent.org
```
