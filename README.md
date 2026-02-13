<p align="center">
  <h1 align="center">🧠 Zenfa AI Engine</h1>
  <p align="center">
    <strong>Hybrid Knapsack + LLM Agentic PC Build Optimizer</strong><br>
    <em>Powering intelligent PC recommendations for the Bangladesh market</em>
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11+-blue?logo=python&logoColor=white" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/LLM-Gemini%20Flash-4285F4?logo=google&logoColor=white" alt="Gemini">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License">
</p>

---

## Overview

Zenfa AI Engine is the **build intelligence core** of the [PC Lagbe?](https://github.com/your-org/project-zenfa) platform. It combines two specialized agents in an **agentic negotiation loop** to produce the best possible PC builds:

| Agent | Role | Strength |
|-------|------|----------|
| **Knapsack Engine** | Deterministic constraint-satisfaction optimizer | Real BD prices, stock, compatibility — 100% accurate |
| **LLM Evaluator** | Intelligent advisor (Gemini Flash) | Global benchmarks, community sentiment, value analysis |

The two agents negotiate iteratively until they converge on a build that is both **mathematically optimal** and **real-world intelligent**.

## Architecture

```
User Request → Knapsack generates build → LLM evaluates & scores →
  If score < 8.5 → LLM suggests swaps → Knapsack validates against BD market →
    Loop continues until score ≥ 8.5 or time limit reached
```

## Project Structure

```
zenfa-ai/
├── zenfa_ai/                  # Main package
│   ├── engine/                # Knapsack optimizer + compatibility rules
│   ├── evaluator/             # LLM client, prompts, schemas
│   ├── orchestrator/          # Agentic negotiation loop + state management
│   ├── models/                # Pydantic models (build, components)
│   └── api/                   # FastAPI routes + auth
├── tests/                     # Test suite
├── docs/                      # Architecture docs
│   ├── Model pitch.md
│   ├── Agentic loop.md
│   └── AGENT_BRIEF.md
├── pyproject.toml             # Dependencies & tooling config
├── Dockerfile                 # Container build
├── docker-compose.yml         # Service orchestration
└── .env.example               # Environment variable template
```

## Getting Started

### Prerequisites

- Python 3.11+
- Redis (for build caching)
- A Gemini API key ([get one here](https://aistudio.google.com/apikey))

### Setup

```bash
# Clone the repo
git clone https://github.com/your-org/zenfa-ai.git
cd zenfa-ai

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -e ".[dev]"

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Run the service
uvicorn zenfa_ai.api.app:app --host 0.0.0.0 --port 8001 --reload
```

### Docker

```bash
docker compose up --build
```

The API will be available at `http://localhost:8001`.

## API Usage

### Generate a Build

```bash
POST /build
Content-Type: application/json

{
  "budget": 80000,
  "purpose": "gaming",
  "components": [ ... ]   # Component catalog with prices
}
```

### Response

```json
{
  "build": { "components": [...], "total_price": 79200, "remaining_budget": 800 },
  "quality": { "score": 9.1, "iterations_used": 2, "time_taken_seconds": 18.4 },
  "explanation": { "summary": "This build maximizes gaming performance..." },
  "metadata": { "engine_version": "0.1.0", "llm_model": "gemini-2.0-flash" }
}
```

## Development

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=zenfa_ai

# Lint
ruff check .

# Type check
mypy zenfa_ai/
```

## API Tiers (Future)

| Tier | Price | Features |
|------|-------|----------|
| **Basic** | Free (100/month) | Knapsack-only builds |
| **Pro** | $0.05/build | Full agentic loop + explanations |
| **Enterprise** | Custom | Vendor filtering, analytics, white-label |

## Documentation

- **[Model Pitch](docs/Model%20pitch.md)** — Architecture vision & business case
- **[Agentic Loop](docs/Agentic%20loop.md)** — Technical specification of the negotiation loop
- **[Agent Brief](docs/AGENT_BRIEF.md)** — Full implementation handoff document

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
