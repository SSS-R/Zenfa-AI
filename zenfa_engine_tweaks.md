# Zenfa AI Engine (Current Repo) - Required Tweaks

To support the B2B and B2C platforms outlined in the implementation plans, the core Zenfa AI engine needs the following updates:

## 1. API Schema Updates (B2C & B2B Input Handling)
The current Zenfa engine primarily takes `purpose` and `budget_max`/`budget_min`. To support the new B2C UI and the B2B API integrations, the `BuildPreferences` model in `models/components.py` and the knapsack engine must be upgraded to accept:
- `prefer_cpu_brand` / `prefer_gpu_brand`: Separate brand preferences for CPU (Intel/AMD) and GPU (NVIDIA/AMD), replacing the single generic `prefer_brand`.
- `include_monitor`: Boolean flag for whether to include a monitor in the build.
- `monitor_budget_pct`: Optional percentage of total budget to allocate for the monitor.
- `form_factor`: Preferred case/motherboard form factor (ATX/mATX/ITX).
- `rgb_priority`: Tiered RGB preference (high/medium/low) replacing the current boolean `prefer_rgb`.
- The Knapsack engine's `_filter_by_preferences()` must be updated to respect these new constraints before sending the build to the LLM.

> **Design Principle:** All preferences remain as typed Pydantic Literal/Enum fields — NO free-text inputs allowed. This prevents LLM hallucination on user intent.

## 2. RAG via LLM Grounding (Community Knowledge)
Instead of building a custom web scraper or Reddit API client, leverage the LLM's **built-in web search/grounding capabilities** (e.g., Gemini's Google Search grounding):
- **Prompt Engineering:** Update the system prompt in `prompts.py` to explicitly instruct the LLM to search for and consider recent community data — Reddit threads, YouTube reviews, tech forum posts — when evaluating components.
- **Grounding Configuration:** Enable search grounding in the LLM client (`llm_client.py`) so the model can pull real-time community sentiment during evaluation.
- **Structured Output:** The LLM's evaluation response should surface any community-sourced red flags (thermal issues, driver bugs, bad batches) in the existing `red_flags` field of `EvaluationResponse`.

> **Why not a custom scraper?** The LLM already has the power to search through websites, articles, Reddit, and X. We just need to prompt it in a structured way. This avoids the complexity of maintaining scrapers, handling rate limits, and filtering noisy data.

## 3. API Gateway & Authentication Updates (B2B)
- **API Key Validation:** Upgrade `zenfa_ai/api/auth.py` to validate API keys against a database (or Redis cache) instead of the current env-var approach.
- **Redis-Based Rate Limiting:** Replace the current in-memory sliding window rate limiter with a Redis-backed implementation to work correctly across multiple server instances.
- **Domain Verification:** Enforce Origin header checks for vendor API keys to prevent unauthorized browser-based usage. Note: this is a soft check (server-to-server calls can spoof Origin), so it should be combined with key-domain binding in the database.

## 4. Telemetry and Analytics
- **Structured Logging Middleware:** Add a FastAPI middleware that logs every build request as structured JSON (purpose, budget, vendor key, latency, LLM model used, LLM cost estimate, cache hit/miss, final score).
- **Log Format:** JSON to stdout — can be piped to any log aggregator (CloudWatch, Datadog, ELK, etc.) without coupling the engine to a specific database.
- **Analytics Database:** The aggregation and dashboard queries happen outside the engine — the B2B vendor portal and admin panel consume these logs from the log aggregator.

> **Why not direct DB writes?** Keeps the engine stateless, fast, and deployment-flexible. The engine's job is to generate builds, not manage analytics storage.

---

## Out of Scope for This Repo

### Session & State Management (B2C)
Session tracking, build state persistence, and token deduction logic belong in the **PC Lagbe? (B2C) website backend**, NOT in this engine. The engine remains **stateless per-request** — the B2C app handles:
- Storing the current build and remaining tweaks in its own Redis/DB
- Sending "tweak" requests as new `BuildRequest` calls to the engine
- Tracking and deducting from the user's purchased token balance

See `b2c_website_implementation.md` for details.

### Vendor Dashboard & Analytics Storage (B2B)
The vendor dashboard, subscription management, and analytics database belong in the **Zenfa AI Vendor Portal (B2B) website backend**. The engine only produces structured logs — the portal consumes and visualizes them.

See `b2b_website_implementation.md` for details.
