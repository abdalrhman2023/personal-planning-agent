# 🗓️ Personal Planning Agent (Agent 3)

> **Generative AI Personal Life & Study Coach** — An autonomous, multi-agent AI study planning system that builds conflict-free, fatigue-aware study schedules, syncs them live with **Google Calendar**, and coordinates with peer agents (AI Tutor, Life Coach, Adaptive Manager, and Assessment Agent).

---

## 🌟 Key Features

- **Hybrid AI Architecture**:
  - **Gemini LLM (Conversational & Deconstruction)**: Natural language parsing, conversational planning chat, and cognitive micro-task decomposition.
  - **Deterministic Optimizer (Constraint Satisfaction)**: Conflict-free time slot allocation considering university timetables, existing calendar events, cognitive load windows (peak vs. secondary), and daily burnout caps.
- **Two-Way LangGraph Workflow**:
  - **Conversational Chat Mode**: Direct student interactions via natural language (e.g., *"Add a high-priority study task for Deep Learning tomorrow for 45 minutes and optimize my schedule"*).
  - **Multi-Agent Control Signals Mode**: Programmatic signals from peer agents:
    - `PREEMPTIVE_SCHEDULE_RESTRUCTURE` (Adaptive Manager / Agent 5)
    - `MILESTONE_TASKS` (Life Coach / Agent 4)
    - `REMEDIATION_TASKS` (Assessment Agent / Agent 2)
- **Live Google Calendar Sync**:
  - Real-time OAuth2 synchronization to Google Calendar with timezone accuracy (`Africa/Cairo` by default).
  - Clean restructuring: automatically deletes previously created study events before pushing updated schedules to prevent duplicates.
  - Automatic mock mode fallback if credentials are not configured.
- **Outbound Telemetry & Multi-Agent Signals**:
  - **Workload Density**: Calculates daily operating density metrics.
  - **Velocity Slip Alert**: Triggers an alert when completion rate drops by >20% below expected pace.
  - **AI Tutor Study Invitations**: Emits targeted study invitations with curriculum topics and cognitive load for each block.

---

## 📂 Project Structure

```
personal-planning-agent/
├── app/
│   ├── config.py                  # Environment and model configurations
│   ├── schema.py                  # Pydantic data schemas (tasks, blocks, signals)
│   ├── api/
│   │   └── routes.py              # FastAPI endpoints (/optimize, /chat, /telemetry)
│   ├── agent/
│   │   ├── graph.py               # LangGraph state machine & router
│   │   ├── tools.py               # LangChain tools (create_task, reschedule, optimize)
│   │   └── prompts.py             # System and routing prompts
│   ├── engine/
│   │   ├── optimizer.py           # Deterministic interval scheduling engine
│   │   ├── deconstructor.py       # Task chunking & Gemini micro-task deconstruction
│   │   └── telemetry.py           # Workload density, velocity slip & signal generator
│   └── integrations/
│       └── gcal_service.py        # Google Calendar OAuth2 & Mock API service
├── tests/                         # Full pytest suite (Phases 1 through 5, 124 tests)
├── authenticate_gcal.py           # One-time Google Calendar OAuth authentication script
├── demo_simulation.py             # Multi-scenario end-to-end simulation
├── main.py                        # FastAPI application entrypoint
├── requirements.txt               # Dependencies
└── .env.example                   # Environment variable template
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+
- A Google Cloud Project with Google Calendar API enabled (optional for Live Calendar sync; mock mode works out of the box).
- A Gemini API Key from [Google AI Studio](https://aistudio.google.com/).

### 2. Installation

Clone this repository and install dependencies:
```bash
git clone <YOUR_GITHUB_REPO_URL>
cd personal-planning-agent
pip install -r requirements.txt
```

### 3. Environment Configuration

Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

Configure your variables in `.env`:
```env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL_NAME=gemini-3.5-flash
GOOGLE_CREDENTIALS_PATH=credentials.json
GOOGLE_TOKEN_PATH=token.json
TIMEZONE=Africa/Cairo
```

---

## 📅 Google Calendar Setup (Optional)

If you want to sync with real Google Calendar:
1. Place your OAuth `credentials.json` file in the project root directory.
2. Run the authentication script:
   ```bash
   python authenticate_gcal.py
   ```
3. A browser window will open. Sign in and grant calendar access. The token will be saved to `token.json`.

*(If `credentials.json` is not provided, the agent will automatically use built-in Mock Mode without failing).*

---

## 🖥️ Running the Application

### 1. FastAPI Server & Interactive Swagger UI
Start the API server:
```bash
python main.py
```
Open your browser at [http://localhost:8000/docs](http://localhost:8000/docs) to explore and execute the endpoints interactively:
- `POST /api/v1/planning/optimize`: Run schedule optimization with timetable constraints.
- `POST /api/v1/planning/chat`: Conversational agent interface and multi-agent control signals.
- `POST /api/v1/planning/telemetry`: Calculate workload density and emit tutor invitations.

### 2. End-to-End Simulation Script
Run the 3 multi-agent scenarios demo:
```bash
python demo_simulation.py
```
This demonstrates:
- **Scenario 1**: University timetable ingestion + Life Coach milestone deconstruction + GCal sync.
- **Scenario 2**: Preemptive schedule restructuring from Adaptive Manager applying fatigue factors.
- **Scenario 3**: Detecting >20% velocity slip and firing alerts to Adaptive Manager.

### 3. Running Unit & Integration Tests
Run the full test suite (124 tests):
```bash
pytest tests/ -v
```

---

## 👥 Multi-Agent Signal Contracts

| Signal Type | Sender / Source | Action / Payload |
|---|---|---|
| `MILESTONE_TASKS` | Life Coach (Agent 4) | Ingests goals and deconstructs tasks exceeding focus limit into micro-tasks. |
| `PREEMPTIVE_SCHEDULE_RESTRUCTURE` | Adaptive Manager (Agent 5) | Inflates task durations by fatigue factor and imposes daily study cap. |
| `REMEDIATION_TASKS` | Assessment Agent (Agent 2) | Schedules urgent remediation study blocks for struggling topics. |
| `VELOCITY_SLIP_ALERT` | Planning Agent &rarr; Agent 5 | Emitted when completion velocity deviates by >20%. |
| `STUDY_BLOCK_INVITATION` | Planning Agent &rarr; AI Tutor (Agent 1) | Dispatched for every scheduled study block with topic and cognitive load metadata. |
