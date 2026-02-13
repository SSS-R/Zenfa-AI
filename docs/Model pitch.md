# 🧠 Zenfa AI Engine — Model Architecture Pitch

**Project:** PC Lagbe? — AI-Driven PC Builder  
**Component:** Zenfa AI Engine (The Build Intelligence Core)  
**Version:** 1.0 Draft  
**Date:** February 2026

---

## 🎯 What Is This?

Zenfa AI Engine is a **hybrid intelligence system** that combines a **deterministic optimization engine** (Knapsack) with a **Large Language Model evaluator** (Gemini/OpenAI) in an **agentic negotiation loop** to produce the **mathematically optimal AND real-world intelligent** PC builds for any budget.

This is **not** a chatbot wrapper. This is **not** a filtered database query. This is a genuine AI system where two specialized agents negotiate until they reach consensus on the best possible build.

---

## 🏗️ The Two Agents

### Agent 1: The Knapsack Engine — "The Local Expert"

| Attribute | Detail |
|---|---|
| **Type** | Deterministic constraint-satisfaction + optimization algorithm |
| **Knows** | Real-time BD prices, stock availability, hardware compatibility rules |
| **Doesn't Know** | Real-world performance, community sentiment, value-for-money analysis |
| **Speed** | Milliseconds |
| **Accuracy** | 100% for compatibility and pricing (it reads directly from our DB) |

**What it does:**
- Applies hard compatibility rules (socket match, DDR type, PSU wattage)
- Fetches real, live prices from Bangladeshi vendors (StarTech, Ryans, Skyland, etc.)
- Runs a weighted Knapsack algorithm to maximize a score within the user's budget
- Validates that every suggested component is **in stock** and **purchasable today**

### Agent 2: The LLM Evaluator — "The Global Advisor"

| Attribute | Detail |
|---|---|
| **Type** | Large Language Model (Gemini Flash / GPT-4o-mini) |
| **Knows** | Global benchmarks, Reddit reviews, YouTube consensus, tech journalist opinions, historical pricing trends |
| **Doesn't Know** | BD-specific pricing, local stock availability |
| **Speed** | 1-3 seconds per evaluation |
| **Accuracy** | High for relative performance comparisons, unreliable for exact prices |

**What it does:**
- Evaluates each component for real-world performance (not just specs on paper)
- Detects "value traps" (e.g., RTX 5090 when a 3090 Ti does the job at 40% cost)
- Flags community red flags (known thermal issues, driver problems, etc.)
- Suggests better alternatives based on global market knowledge
- Scores the build on a structured rubric (0-10)

---

## 🔄 How They Work Together: The Negotiation Loop

```
User: "80,000৳ gaming build"
         │
         ▼
    ┌─────────┐    Build v1    ┌─────────┐
    │Knapsack │ ──────────────▶│  LLM    │
    │ Engine  │                │Evaluator│
    │         │◀──────────────│         │
    └─────────┘  "Swap GPU,   └─────────┘
         │        score: 7.2"       
         │                          
    ┌─────────┐    Build v2    ┌─────────┐
    │Knapsack │ ──────────────▶│  LLM    │
    │ Engine  │  + context:    │Evaluator│
    │         │  "RX 7700 XT   │         │
    │         │◀── available   │         │
    └─────────┘   at 58,000৳"  └─────────┘
         │          score: 9.1 ✅
         ▼
    Return final build to user
```

Each round:
1. **Knapsack** generates the best build it can with current constraints
2. **LLM** evaluates using global knowledge and scores it
3. If score < 8.5 → LLM provides structured suggestions
4. **Knapsack** applies suggestions, but validates against **real BD data**
5. If a suggestion is impossible (out of stock, too expensive), Knapsack tells the LLM **why**
6. LLM adapts its evaluation with this new context
7. Loop continues until **score ≥ 8.5** or **time limit reached**

---

## 📊 Scoring Rubric (Given to LLM)

| Criteria | Weight | Description |
|---|---|---|
| **Performance Match** | 3 pts | Does the build match the stated purpose (gaming/editing/office)? |
| **Value Score** | 3 pts | Is each component the best performance-per-taka available? |
| **Build Balance** | 2 pts | Is the build balanced? (No RTX 5090 paired with a Celeron) |
| **Future-Proofing** | 1 pt | Will this build stay relevant for 2-3 years? |
| **Community Trust** | 1 pt | Are these components well-reviewed with no known issues? |

**Minimum acceptable score:** 7.0/10 (fallback)  
**Target score:** 8.5/10 (loop continues until this is reached)  
**Time budget:** 90-120 seconds maximum

---

## 💰 Why This Is Sellable as an API

### For Our Platform (PC Lagbe?)
- Users get **genuinely intelligent** build recommendations
- Not just "cheapest parts that fit" — actual expert-level analysis
- Build explanations that educate users ("We picked X because...")

### For Vendor API Customers
```
POST https://api.zenfa.ai/v1/build/generate
Headers: X-API-Key: vendor_startech_abc123
Body: { "budget": 80000, "purpose": "gaming", "vendor_filter": "startech" }
```

| API Tier | Price | Features |
|---|---|---|
| **Basic** | Free (100/month) | Knapsack-only builds (no LLM loop) |
| **Pro** | $0.05/build | Full agentic loop, build explanations |
| **Enterprise** | Custom | Vendor-filtered builds, analytics dashboard, white-label |

### Cost Structure
- LLM cost per build: **~$0.003-$0.015** (Gemini Flash, 3 iterations avg)
- Revenue per Pro API call: **$0.05**
- **Gross margin: ~70-95%** per build

---

## 🆚 Competitive Advantage

| Competitor Approach | Our Approach |
|---|---|
| Static filters ("sort by price") | Dynamic optimization with real AI |
| Chatbot wrappers (GPT answers questions) | Structured agent loop with real data validation |
| Manual builds by shopkeepers | Automated, unbiased, best-value recommendations |
| No BD market awareness | Built on live BD pricing + stock data |

**Our moat:** The combination of **live BD market data** + **intelligent optimization** + **LLM evaluation**. Competitors would need to build all three layers AND the scraping infrastructure to compete.

---

## 🛡️ Risk Mitigation

| Risk | Mitigation |
|---|---|
| LLM API goes down | Knapsack works standalone as fallback (still good builds) |
| LLM hallucinates a product | Knapsack validates every suggestion against real DB — hallucinations die here |
| LLM suggests unavailable parts | Knapsack reports back "unavailable" + LLM adapts |
| High LLM costs at scale | Cache common builds; identical requests return cached results |
| Loop doesn't converge | Hard limits: max 5 iterations, 120s timeout, return best-so-far |

---

## 🗺️ Development Phases

| Phase | Deliverable | Dependency |
|---|---|---|
| **3A** | Knapsack Engine + Compatibility Rules | Phase 2 (scraper data) |
| **3B** | Benchmark Score Database (curated) | Manual data entry |
| **3C** | LLM Evaluator Service | Gemini API key |
| **3D** | Agentic Negotiation Loop (Orchestrator) | 3A + 3C |
| **3E** | Build Explanation Generator | 3D |
| **3F** | API Key Management + Vendor Filtering | 3D |
| **3G** | Community Sentiment Pipeline (Reddit/forums) | Future enhancement |
