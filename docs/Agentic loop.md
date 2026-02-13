# 🔄 Agentic Loop — Technical Specification

**System:** Zenfa AI Engine  
**Component:** Build Negotiation Orchestrator  
**Version:** 1.0 Draft  
**Date:** February 2026

---

## Overview

The Agentic Loop is the core orchestration layer that manages the back-and-forth negotiation between the **Knapsack Engine** (deterministic optimizer) and the **LLM Evaluator** (intelligent advisor). It is responsible for:

- Managing iteration state
- Enforcing time and iteration limits
- Translating LLM suggestions into Knapsack constraints
- Translating Knapsack realities back to LLM context
- Deciding when a build is "good enough" to return
- Graceful degradation when components fail

---

## System Flow

```
┌──────────────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR (agentic_loop.py)                │
│                                                                  │
│  Input: BuildRequest { budget, purpose, preferences? }           │
│  Output: FinalBuild { components, score, explanation, metadata } │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │                    ITERATION LOOP                        │     │
│  │                                                         │     │
│  │  ┌──────────┐       ┌───────────┐       ┌──────────┐   │     │
│  │  │  PHASE 1 │──────▶│  PHASE 2  │──────▶│ PHASE 3  │   │     │
│  │  │ Knapsack │       │    LLM    │       │ Decision │   │     │
│  │  │ Generate │       │ Evaluate  │       │  Gate    │   │     │
│  │  └──────────┘       └───────────┘       └────┬─────┘   │     │
│  │       ▲                                       │         │     │
│  │       │            score < 8.5                │         │     │
│  │       └───────────────────────────────────────┘         │     │
│  │                                                         │     │
│  │  EXIT CONDITIONS:                                       │     │
│  │  ✅ score >= 8.5 → return build                         │     │
│  │  ⏱️ time > 120s → return best build so far              │     │
│  │  🔄 iterations > 5 → return best build so far           │     │
│  │  ❌ LLM failure → return knapsack-only build            │     │
│  └─────────────────────────────────────────────────────────┘     │
│                                                                  │
│  POST-LOOP: Generate explanation via LLM (optional final pass)   │
└──────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Knapsack Generation

### First Iteration (No LLM Context Yet)
```
Input:
  - budget: 80000
  - purpose: "gaming"
  - available_components: [all in-stock components from DB]
  - compatibility_rules: [socket, DDR, PSU wattage, form factor]

Process:
  1. Filter components by compatibility (hard constraints)
  2. Apply purpose-based weights:
     Gaming:  GPU=0.40, CPU=0.25, RAM=0.15, Storage=0.10, PSU=0.10
     Editing: CPU=0.35, RAM=0.30, GPU=0.20, Storage=0.10, PSU=0.05
     Office:  CPU=0.30, RAM=0.25, Storage=0.25, GPU=0.10, PSU=0.10
  3. Run weighted knapsack: maximize total_score where sum(prices) <= budget
  4. Attach real prices + vendor info + stock status

Output: CandidateBuild
  - components: [{name, category, price, vendor, in_stock, specs}]
  - total_price: 76400
  - remaining_budget: 3600
  - compatibility_verified: true
```

### Subsequent Iterations (With LLM Suggestions)
```
Input:
  - Same as above, PLUS:
  - llm_suggestions: [
      { action: "swap", component: "gpu", 
        suggested: ["RX 7700 XT", "RX 7800 XT"],
        reason: "better value" },
      { action: "downgrade", component: "ram_type", 
        suggested: ["DDR4"],
        reason: "save budget for GPU" }
    ]
  - previous_build: Build v(N-1)

Process:
  1. Attempt to apply each suggestion:
     - Check if suggested component exists in BD market
     - Check if it's in stock
     - Check if it's within budget after swap
     - Check compatibility with rest of build
  2. For each suggestion, record result:
     - APPLIED: successfully swapped
     - UNAVAILABLE: component not found in any BD vendor
     - OUT_OF_STOCK: exists but currently unavailable
     - OVER_BUDGET: swap would exceed budget by X amount
     - INCOMPATIBLE: fails compatibility check (explain why)
  3. Re-run knapsack with applied changes as preferences

Output: CandidateBuild + SuggestionResults
  - components: [updated build]
  - suggestion_results: [
      { suggestion: "swap gpu to RX 7700 XT", 
        status: "APPLIED", 
        note: "Available at StarTech for 58,000৳" },
      { suggestion: "downgrade to DDR4", 
        status: "APPLIED", 
        note: "Switched to B550 motherboard, saved 8,000৳" }
    ]
```

---

## Phase 2: LLM Evaluation

### System Prompt (Stable Across Iterations)
```
You are ZenFA, a PC hardware expert evaluator. You evaluate PC builds 
for the Bangladesh market.

SCORING RUBRIC (score each 0-10):
- Performance Match (weight: 3): Does the build match the stated 
  purpose? A gaming build should prioritize GPU FPS. An editing 
  build should prioritize multi-core + RAM.
- Value Score (weight: 3): Is each component the best 
  performance-per-taka? Avoid "value traps" where a cheaper older 
  part delivers 90%+ of performance at 50% cost.
- Build Balance (weight: 2): Is the build balanced? A top-tier GPU 
  with a budget CPU is a bottleneck. Flag imbalances.
- Future-Proofing (weight: 1): Will this build stay relevant for 
  2-3 years? Prefer current-gen platforms with upgrade paths.
- Community Trust (weight: 1): Are these components well-reviewed?
  Flag any known issues (thermal throttling, driver bugs, QC problems).

FINAL SCORE = weighted average of all criteria.

RULES:
1. You MUST return valid JSON matching the schema below.
2. If score < 8.5, you MUST provide at least one actionable suggestion.
3. Suggestions must be specific component names, not vague advice.
4. You do NOT know Bangladesh prices. Do not comment on pricing.
   The Knapsack engine handles pricing.
5. Focus on: real-world performance, value-for-money (globally), 
   community sentiment, and build balance.
```

### User Prompt (Changes Each Iteration)

#### First Iteration
```
Evaluate this PC build for {purpose} at {budget}৳ budget:

BUILD:
- CPU: {name} ({specs})
- GPU: {name} ({specs})
- RAM: {name} ({specs})
- Storage: {name} ({specs})
- PSU: {name} ({specs})
- Motherboard: {name} ({specs})

Total price: {total}৳
Remaining budget: {remaining}৳
```

#### Subsequent Iterations (With Context from Knapsack)
```
Evaluate this REVISED build. The user's local market has the 
following constraints that affected your previous suggestions:

SUGGESTION RESULTS:
{for each previous suggestion}
- "{suggestion}" → {APPLIED/UNAVAILABLE/OUT_OF_STOCK/OVER_BUDGET}
  Context: {knapsack's explanation}
{end for}

UPDATED BUILD:
- CPU: {name} ({specs})
...

Consider the market constraints above when making new suggestions.
Do NOT re-suggest components that were marked UNAVAILABLE or 
OUT_OF_STOCK.
```

### Expected LLM Response Schema
```json
{
  "scores": {
    "performance_match": 8,
    "value_score": 6,
    "build_balance": 9,
    "future_proofing": 7,
    "community_trust": 8
  },
  "final_score": 7.2,
  "reasoning": "The GPU is the weak point. RTX 4060 offers poor...",
  "suggestions": [
    {
      "action": "swap | upgrade | downgrade | remove | add",
      "component_category": "gpu",
      "current_component": "RTX 4060",
      "suggested_alternatives": ["RX 7700 XT", "RTX 4060 Ti"],
      "reason": "RX 7700 XT delivers 40% more FPS at similar cost",
      "priority": "high | medium | low"
    }
  ],
  "red_flags": [
    "RTX 4060 8GB VRAM may struggle with newer titles at 1440p"
  ],
  "approved": false
}
```

---

## Phase 3: Decision Gate

```python
def should_continue(state: LoopState) -> Decision:
    # Hard exits
    if state.elapsed_time > MAX_TIME_SECONDS:       # 120s
        return Decision.RETURN_BEST
    if state.iteration_count >= MAX_ITERATIONS:       # 5
        return Decision.RETURN_BEST
    if state.llm_failed:
        return Decision.RETURN_KNAPSACK_ONLY
    
    # Score-based exits
    if state.current_score >= TARGET_SCORE:            # 8.5
        if state.elapsed_time < OPTIMIZATION_WINDOW:   # 90s
            return Decision.CONTINUE_OPTIMIZING        # try for higher
        return Decision.RETURN_CURRENT
    
    if state.current_score >= MINIMUM_SCORE:           # 7.0
        if state.score_improving:                      # still getting better
            return Decision.CONTINUE
        return Decision.RETURN_CURRENT                 # plateaued
    
    # Score too low, keep trying
    return Decision.CONTINUE
```

### Decision Behaviors

| Decision | Action |
|---|---|
| `RETURN_CURRENT` | Return the current build as final |
| `RETURN_BEST` | Return the highest-scored build from any iteration |
| `RETURN_KNAPSACK_ONLY` | Return Knapsack build without LLM evaluation |
| `CONTINUE` | Run another Knapsack → LLM iteration |
| `CONTINUE_OPTIMIZING` | Score is good but time remains — try for even better |

---

## State Management

```python
@dataclass
class LoopState:
    # Request
    budget: int
    purpose: str
    
    # Timing
    start_time: float
    elapsed_time: float
    max_time: float = 120.0  # seconds
    
    # Iterations
    iteration_count: int = 0
    max_iterations: int = 5
    
    # Builds history (keep all for "return best")
    builds: list[CandidateBuild] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    
    # Current state
    current_build: CandidateBuild | None = None
    current_score: float = 0.0
    best_build: CandidateBuild | None = None
    best_score: float = 0.0
    
    # LLM state
    llm_suggestions: list[Suggestion] = field(default_factory=list)
    suggestion_history: list[SuggestionResult] = field(default_factory=list)
    unavailable_components: set[str] = field(default_factory=set)
    llm_failed: bool = False
    
    # Computed
    @property
    def score_improving(self) -> bool:
        if len(self.scores) < 2:
            return True
        return self.scores[-1] > self.scores[-2]
```

---

## Error Handling & Edge Cases

### LLM Failures
```
Scenario: LLM returns invalid JSON
Action: Retry once with stricter prompt. If fails again, mark llm_failed=True.
Result: Return knapsack-only build.

Scenario: LLM API timeout (>15s)
Action: Mark llm_failed=True immediately.
Result: Return knapsack-only build.

Scenario: LLM API rate limited
Action: Wait 2s, retry once. If still limited, mark llm_failed=True.
Result: Return knapsack-only build.
```

### Oscillation Prevention
```
Scenario: LLM suggests A→B in round 2, then B→A in round 3
Detection: Check if suggested component was the "current" in any previous iteration
Action: Add to "locked" list - component cannot be swapped again
Result: Forces convergence
```

### Budget Edge Cases
```
Scenario: Budget too low for any decent build (<25,000৳)
Action: Skip LLM loop entirely. Return knapsack-only + warning message.
Message: "Budget is very tight. This is the best configuration possible, 
          but we recommend increasing to at least X৳ for a better experience."

Scenario: Budget extremely high (>500,000৳)
Action: Normal loop, but LLM prompt includes "diminishing returns" guidance.
Note: Prevent the build from just picking the most expensive everything.
```

---

## API Response Format

### Final Response to Client
```json
{
  "build": {
    "components": [
      {
        "category": "cpu",
        "name": "AMD Ryzen 5 7600",
        "price": 22500,
        "vendor": "StarTech",
        "vendor_url": "https://startech.com.bd/...",
        "in_stock": true,
        "specs": { "cores": 6, "threads": 12, "socket": "AM5" }
      }
    ],
    "total_price": 79200,
    "remaining_budget": 800
  },
  "quality": {
    "score": 9.1,
    "scores_breakdown": {
      "performance_match": 9,
      "value_score": 9,
      "build_balance": 10,
      "future_proofing": 8,
      "community_trust": 9
    },
    "iterations_used": 2,
    "time_taken_seconds": 18.4
  },
  "explanation": {
    "summary": "This build maximizes gaming performance for your budget...",
    "per_component": {
      "cpu": "Ryzen 5 7600 offers the best single-core gaming perf...",
      "gpu": "RX 7700 XT was chosen over RTX 4060 for 40% more FPS..."
    },
    "trade_offs": "We chose DDR4 over DDR5 to allocate more budget to GPU...",
    "upgrade_path": "When budget allows, upgrade to 32GB RAM and NVMe Gen4 SSD."
  },
  "metadata": {
    "engine_version": "1.0",
    "llm_model": "gemini-2.0-flash",
    "fallback_used": false
  }
}
```

---

## Performance Targets

| Metric | Target | Notes |
|---|---|---|
| Average iterations | 2-3 | Most builds converge quickly |
| Average response time | 15-30s | With LLM loop |
| Knapsack-only fallback time | < 2s | When LLM is unavailable |
| Maximum response time | 120s | Hard cutoff |
| LLM cost per build | $0.003-$0.015 | Gemini Flash, 3 iterations avg |
| Target score | ≥ 8.5/10 | Minimum acceptable: 7.0 |
| Cache hit rate | ~30% | Common budget+purpose combos |

---

## Caching Strategy

```
Cache Key: build:{budget_range}:{purpose}:{timestamp_bucket}
Example:   build:75k-85k:gaming:2026-02-13

Rules:
- Budget is bucketed to nearest 5k range (exact budget still used in calc)
- Cache TTL: 1 hour (prices may change)
- Cache invalidated when: new scrape completes, component stock changes
- Cached builds skip the loop entirely → instant response
- Only builds with score >= 8.5 are cached
```

---

## File Structure (When Implemented)

```
zenfa-ai/                     # Separate repo
├── zenfa_ai/
│   ├── __init__.py
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── knapsack.py       # Knapsack optimization algorithm
│   │   ├── compatibility.py  # Hardware compatibility rules
│   │   └── weights.py        # Purpose-based scoring weights
│   ├── evaluator/
│   │   ├── __init__.py
│   │   ├── llm_client.py     # Gemini/OpenAI API wrapper
│   │   ├── prompts.py        # System & user prompt templates
│   │   └── schemas.py        # Pydantic models for LLM I/O
│   ├── orchestrator/
│   │   ├── __init__.py
│   │   ├── loop.py           # The agentic negotiation loop
│   │   ├── state.py          # LoopState management
│   │   └── decisions.py      # Decision gate logic
│   ├── models/
│   │   ├── __init__.py
│   │   ├── build.py          # Build request/response models
│   │   └── components.py     # Component data models
│   └── api/
│       ├── __init__.py
│       ├── router.py         # FastAPI routes
│       └── auth.py           # API key management
├── tests/
│   ├── test_knapsack.py
│   ├── test_compatibility.py
│   ├── test_loop.py
│   └── test_evaluator.py
├── pyproject.toml
├── Dockerfile
└── README.md
```
