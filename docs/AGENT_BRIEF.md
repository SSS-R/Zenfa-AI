# 📋 Zenfa AI Engine — Agent Handoff Brief

> **Purpose:** This document provides all the context an AI coding agent needs to initialize and build the `zenfa-ai` repo from scratch. Read this file ALONG WITH `Model pitch.md` (the architecture vision) and `Agentic loop.md` (the technical spec).

---

## 1. Project Context

**"PC Lagbe?"** is an AI-driven PC builder & price aggregator for Bangladesh. It has two repos:

| Repo | Purpose | Status |
|---|---|---|
| `project-zenfa` (main) | Frontend (Next.js) + Backend (FastAPI) + Scrapers | Active, Phase 2 complete |
| **`zenfa-ai` (this repo)** | The AI build engine — knapsack optimizer + LLM evaluator + agentic loop | **NEW — build from scratch** |

The main repo scrapes real-time prices from Bangladeshi hardware vendors (StarTech, Ryans, Skyland, etc.) and stores them in PostgreSQL. **This repo** (`zenfa-ai`) receives component + pricing data and generates optimized PC builds using the hybrid knapsack + LLM architecture described in the companion docs.

---

## 2. Integration Method

`zenfa-ai` should be built as a **standalone FastAPI microservice** with its own `Dockerfile`. The main `project-zenfa` backend will call it via internal HTTP.

### Communication Flow
```
project-zenfa backend                     zenfa-ai service
─────────────────────                     ────────────────
POST /api/v1/build/generate ──┐
                               │   HTTP (internal)
                               └──▶ POST /build
                                    Body: BuildRequest
                                           │
                                    ┌──────▼──────┐
                                    │ Agentic Loop │
                                    │ (Knapsack ⇄  │
                                    │     LLM)     │
                                    └──────┬──────┘
                                           │
                                    ◀──────┘
                               Response: BuildResponse
```

### Key Design Constraint
**`zenfa-ai` does NOT connect to the database directly.** It receives component + pricing data as input in the API request. This keeps it:
- Fully decoupled from the main DB
- Testable with mock data
- Sellable to external vendors (they send their own product data)

---

## 3. Existing Database Schema (From `project-zenfa`)

These are the exact models the main backend uses. The `zenfa-ai` engine must understand this data shape because it will receive it as input.

### 3.1 Enums (Shared Vocabulary)

```python
class ComponentType(str, Enum):
    CPU = "cpu"
    MOTHERBOARD = "motherboard"
    RAM = "ram"
    GPU = "gpu"
    STORAGE = "storage"
    PSU = "psu"
    CASE = "case"
    COOLER = "cooler"
    CASE_FAN = "case_fan"
    MONITOR = "monitor"
    # Non-build peripherals: LAPTOP, KEYBOARD, MOUSE, HEADPHONE, UPS, ACCESSORY

class SocketType(str, Enum):
    AM5 = "AM5"
    AM4 = "AM4"
    LGA1700 = "LGA1700"
    LGA1200 = "LGA1200"

class FormFactor(str, Enum):
    ATX = "ATX"
    MICRO_ATX = "mATX"
    MINI_ITX = "ITX"
    E_ATX = "E-ATX"

class RAMType(str, Enum):
    DDR4 = "DDR4"
    DDR5 = "DDR5"

class StorageType(str, Enum):
    NVME = "NVMe"
    SATA = "SATA"
    HDD = "HDD"

class PSUkb(str, Enum):   # PSU Efficiency Rating
    BRONZE = "80+ Bronze"
    GOLD = "80+ Gold"
    PLATINUM = "80+ Platinum"
    TITANIUM = "80+ Titanium"
    WHITE = "80+ White"
    NONE = "None"

class CoolerType(str, Enum):
    AIR = "Air"
    LIQUID = "Liquid"
```

### 3.2 Component Models

#### Base Component (all components share these fields)
```python
# Table: "components"
id: int                          # Primary key
name: str                        # e.g., "AMD Ryzen 5 7600"
slug: str                        # e.g., "amd-ryzen-5-7600" (unique)
image_url: Optional[str]
component_type: ComponentType    # "cpu", "gpu", etc.
brand: Optional[str]             # e.g., "AMD", "NVIDIA"
performance_score: int           # 0-100, used by AI builder (DEFAULT 50)
created_at: datetime
updated_at: datetime
```

#### CPU (Table: "cpus")
```python
component_id: int                # FK → components.id
socket: SocketType               # AM5, AM4, LGA1700, LGA1200
core_count: int
thread_count: int
base_clock_ghz: float
boost_clock_ghz: Optional[float]
tdp: int                         # Watts
integrated_graphics: bool
igpu_name: Optional[str]
```

#### GPU (Table: "gpus")
```python
component_id: int                # FK → components.id
vram_gb: int
length_mm: int                   # For case compatibility check
recommended_psu_wattage: int     # Minimum PSU needed
```

#### Motherboard (Table: "motherboards")
```python
component_id: int                # FK → components.id
socket: SocketType               # Must match CPU socket
form_factor: FormFactor          # ATX, mATX, ITX, E-ATX
ram_type: RAMType                # DDR4 or DDR5
ram_slots: int                   # Default 4
max_ram_gb: int                  # Default 128
chipset: Optional[str]           # e.g., "B650", "Z790"
pcie_x16_slots: int              # Default 1
m2_slots: int                    # Default 1
```

#### RAM (Table: "rams")
```python
component_id: int                # FK → components.id
ram_type: RAMType                # Must match motherboard ram_type
capacity_gb: int
speed_mhz: int
modules: int                     # 1 = single stick, 2 = kit of 2
cas_latency: Optional[int]       # CL number
rgb: bool
```

#### Storage (Table: "storages")
```python
component_id: int                # FK → components.id
storage_type: StorageType        # NVMe, SATA, HDD
capacity_gb: int
```

#### PSU (Table: "psus")
```python
component_id: int                # FK → components.id
wattage: int                     # Must be >= GPU's recommended_psu_wattage
efficiency_rating: PSUkb         # 80+ Bronze/Gold/etc.
modular: bool
```

#### Casing (Table: "casings")
```python
component_id: int                # FK → components.id
max_gpu_length_mm: int           # Must be >= GPU's length_mm
max_cpu_cooler_height_mm: int    # Must be >= cooler height
psu_support: Optional[str]       # e.g., "ATX"
form_factor_support: List[str]   # ["ATX", "mATX"] — must include mobo form factor
```

#### CPU Cooler (Table: "cpu_coolers")
```python
component_id: int                # FK → components.id
cooler_type: CoolerType          # Air or Liquid
fan_size_mm: Optional[int]
radiator_size_mm: Optional[int]  # For liquid coolers
tdp_capacity_watts: Optional[int] # Must be >= CPU TDP
socket_support: List[str]         # ["AM5", "LGA1700"] — must include CPU socket
```

#### Case Fan (Table: "case_fans")
```python
component_id: int                # FK → components.id
size_mm: int                     # 120, 140, etc.
rpm: Optional[int]
rgb: bool
noise_level_db: Optional[float]
```

### 3.3 Pricing Model

```python
# Table: "vendor_prices"
id: int
component_id: int                # FK → components.id
vendor_name: VendorName          # "StarTech", "Ryans", "Skyland", etc.
price_bdt: int                   # Price in Bangladeshi Taka (৳)
url: str                         # Direct link to product page
in_stock: bool
raw_data: Optional[Dict]         # Raw scraped data for debugging
last_updated: datetime
# Unique constraint: (component_id, vendor_name)
```

### 3.4 Supported Vendors
```python
class VendorName(str, Enum):
    STARTECH = "StarTech"
    RYANS = "Ryans"
    TECHLAND = "TechLand"
    UCC = "UCC"
    SKYLAND = "Skyland"
    NEXUS = "Nexus"
```

---

## 4. Input/Output Contract

### 4.1 Build Request (What zenfa-ai receives)

```python
class BuildRequest(BaseModel):
    """Sent by project-zenfa to zenfa-ai."""
    budget: int                        # In BDT (e.g., 80000)
    purpose: Literal["gaming", "editing", "office", "general"]
    
    # Component catalog — the entire available market
    components: List[ComponentWithPrice]  # All in-stock components + cheapest price
    
    # Optional vendor filtering (for API customers)
    vendor_filter: Optional[str] = None  # e.g., "StarTech" — only use their inventory
    
    # Optional preferences from user
    preferences: Optional[BuildPreferences] = None

class ComponentWithPrice(BaseModel):
    """Flattened component with its cheapest available price."""
    id: int
    name: str
    slug: str
    component_type: str              # "cpu", "gpu", "motherboard", etc.
    brand: Optional[str]
    performance_score: int           # 0-100
    
    # Cheapest available price
    price_bdt: int
    vendor_name: str
    vendor_url: str
    in_stock: bool
    
    # Type-specific specs (varies by component_type)
    specs: Dict[str, Any]
    # CPU example: {"socket": "AM5", "core_count": 6, "tdp": 65, ...}
    # GPU example: {"vram_gb": 8, "length_mm": 240, "recommended_psu_wattage": 550}
    # etc.

class BuildPreferences(BaseModel):
    """Optional user preferences."""
    prefer_brand: Optional[str] = None       # "AMD" or "Intel" or "NVIDIA"
    prefer_rgb: bool = False
    min_storage_gb: int = 256
    prefer_wifi: bool = False
```

### 4.2 Build Response (What zenfa-ai returns)

```python
class BuildResponse(BaseModel):
    """Returned to project-zenfa."""
    build: FinalBuild
    quality: BuildQuality
    explanation: BuildExplanation
    metadata: BuildMetadata

class FinalBuild(BaseModel):
    components: List[SelectedComponent]
    total_price: int
    remaining_budget: int

class SelectedComponent(BaseModel):
    id: int
    name: str
    component_type: str
    price_bdt: int
    vendor_name: str
    vendor_url: str
    specs: Dict[str, Any]

class BuildQuality(BaseModel):
    score: float                     # 0.0 - 10.0
    scores_breakdown: Dict[str, int] # performance_match, value_score, etc.
    iterations_used: int
    time_taken_seconds: float

class BuildExplanation(BaseModel):
    summary: str                     # "This build maximizes gaming perf..."
    per_component: Dict[str, str]    # {"cpu": "Ryzen 5 7600 was chosen because..."}
    trade_offs: str                  # "We chose DDR4 over DDR5 to save for GPU..."
    upgrade_path: str                # "When budget allows, upgrade to..."

class BuildMetadata(BaseModel):
    engine_version: str
    llm_model: str                   # "gemini-2.0-flash" or "knapsack-only"
    fallback_used: bool              # True if LLM was unavailable
    cached: bool                     # True if this was a cache hit
```

---

## 5. Compatibility Rules (Hard Constraints)

These are **boolean pass/fail** rules. A build MUST satisfy ALL of them.

```
RULE 1: CPU.socket == Motherboard.socket
RULE 2: RAM.ram_type == Motherboard.ram_type
RULE 3: PSU.wattage >= GPU.recommended_psu_wattage
RULE 4: Casing.max_gpu_length_mm >= GPU.length_mm
RULE 5: Motherboard.form_factor IN Casing.form_factor_support
RULE 6: CPU.socket IN CPUCooler.socket_support
RULE 7: CPUCooler.tdp_capacity_watts >= CPU.tdp (if both values exist)
RULE 8: RAM.modules * RAM.capacity_gb <= Motherboard.max_ram_gb
RULE 9: RAM.modules <= Motherboard.ram_slots
```

---

## 6. Required Components Per Build

Every PC build MUST include exactly one of each:

| Slot | Required | Notes |
|---|---|---|
| CPU | ✅ Yes | Always |
| Motherboard | ✅ Yes | Always |
| RAM | ✅ Yes | Always |
| GPU | ⚠️ Conditional | Skip if CPU has integrated_graphics AND purpose is "office" |
| Storage | ✅ Yes | Always (at least 1) |
| PSU | ✅ Yes | Always |
| Casing | ✅ Yes | Always |
| CPU Cooler | ⚠️ Optional | Recommended for high-TDP CPUs (>65W) |
| Case Fan | ❌ Optional | Extra, not required |
| Monitor | ❌ Not in build | Separate purchase, not part of the builder |

---

## 7. LLM Configuration

### Recommended: Google Gemini Flash
- **Model:** `gemini-2.0-flash` (cheapest, fastest, sufficient for evaluation)
- **Fallback:** `gpt-4o-mini` via OpenAI API
- **Structured output:** Use Gemini's `response_schema` / OpenAI's `response_format` for guaranteed JSON

### Environment Variables
```env
# Primary LLM
GEMINI_API_KEY=your_key_here
LLM_MODEL=gemini-2.0-flash

# Fallback LLM (optional)
OPENAI_API_KEY=your_key_here
FALLBACK_LLM_MODEL=gpt-4o-mini

# Loop Configuration
MAX_LOOP_ITERATIONS=5
MAX_LOOP_TIME_SECONDS=120
TARGET_SCORE=8.5
MIN_ACCEPTABLE_SCORE=7.0

# Caching
REDIS_URL=redis://localhost:6379/1
BUILD_CACHE_TTL_SECONDS=3600
```

---

## 8. Tech Stack for This Repo

```
Python 3.11+
FastAPI          — API framework
Pydantic v2      — Request/response validation & structured LLM output
google-genai     — Gemini API client (or google-generativeai)
openai           — Fallback LLM client
redis            — Build result caching
uvicorn          — ASGI server
httpx            — HTTP client (if needed)
pytest           — Testing
```

---

## 9. Build Order (Implement in This Sequence)

```
Step 1: Project scaffolding
        - pyproject.toml / requirements.txt
        - FastAPI app stub
        - Pydantic models (input/output contracts from Section 4)
        - Dockerfile

Step 2: Compatibility engine (compatibility.py)
        - Implement the 9 hard rules from Section 5
        - Unit tests for each rule

Step 3: Knapsack optimizer (knapsack.py)
        - Purpose-based weight allocation
        - Budget constraint solver
        - Returns CandidateBuild

Step 4: LLM evaluator service (evaluator.py)
        - System prompt with scoring rubric
        - Structured JSON output via Gemini
        - Timeout handling + retry logic

Step 5: Agentic loop orchestrator (loop.py)
        - Wire knapsack ⇄ LLM negotiation
        - State management, decision gate
        - Oscillation prevention
        - Graceful degradation (LLM down → knapsack-only)

Step 6: Build explanation generator
        - Final LLM pass to generate human-readable explanation
        - Per-component reasoning / trade-offs / upgrade path

Step 7: API endpoints
        - POST /build — main build generation
        - POST /build/explain — standalone explanation
        - POST /compatibility/check — compatibility validator
        - Health check endpoint

Step 8: Caching layer
        - Redis cache for common builds
        - Cache invalidation logic

Step 9: API key auth (for vendor API sales)
        - X-API-Key header validation
        - Rate limiting per key
        - Usage tracking
```

---

## 10. Sample Test Data

Use this data to test the engine. These are real components available in BD as of early 2026.

```json
[
  {
    "id": 1, "name": "AMD Ryzen 5 7600", "component_type": "cpu",
    "brand": "AMD", "performance_score": 72,
    "price_bdt": 22500, "vendor_name": "StarTech", "in_stock": true,
    "specs": {"socket": "AM5", "core_count": 6, "thread_count": 12,
              "base_clock_ghz": 3.8, "boost_clock_ghz": 5.1, "tdp": 65,
              "integrated_graphics": true, "igpu_name": "Radeon Graphics"}
  },
  {
    "id": 2, "name": "Intel Core i5-13400F", "component_type": "cpu",
    "brand": "Intel", "performance_score": 70,
    "price_bdt": 19500, "vendor_name": "StarTech", "in_stock": true,
    "specs": {"socket": "LGA1700", "core_count": 10, "thread_count": 16,
              "base_clock_ghz": 2.5, "boost_clock_ghz": 4.6, "tdp": 65,
              "integrated_graphics": false}
  },
  {
    "id": 3, "name": "MSI B650M MORTAR WIFI", "component_type": "motherboard",
    "brand": "MSI", "performance_score": 65,
    "price_bdt": 18500, "vendor_name": "StarTech", "in_stock": true,
    "specs": {"socket": "AM5", "form_factor": "mATX", "ram_type": "DDR5",
              "ram_slots": 4, "max_ram_gb": 128, "chipset": "B650",
              "pcie_x16_slots": 1, "m2_slots": 2}
  },
  {
    "id": 4, "name": "Gigabyte B550M DS3H", "component_type": "motherboard",
    "brand": "Gigabyte", "performance_score": 55,
    "price_bdt": 9200, "vendor_name": "StarTech", "in_stock": true,
    "specs": {"socket": "AM4", "form_factor": "mATX", "ram_type": "DDR4",
              "ram_slots": 2, "max_ram_gb": 64, "chipset": "B550",
              "pcie_x16_slots": 1, "m2_slots": 1}
  },
  {
    "id": 5, "name": "XFX Speedster SWFT 210 RX 7700 XT", "component_type": "gpu",
    "brand": "AMD", "performance_score": 82,
    "price_bdt": 58000, "vendor_name": "StarTech", "in_stock": true,
    "specs": {"vram_gb": 12, "length_mm": 322, "recommended_psu_wattage": 600}
  },
  {
    "id": 6, "name": "MSI GeForce RTX 4060 VENTUS 2X", "component_type": "gpu",
    "brand": "NVIDIA", "performance_score": 68,
    "price_bdt": 37000, "vendor_name": "StarTech", "in_stock": true,
    "specs": {"vram_gb": 8, "length_mm": 240, "recommended_psu_wattage": 550}
  },
  {
    "id": 7, "name": "G.Skill Trident Z5 RGB 16GB DDR5 6000MHz", "component_type": "ram",
    "brand": "G.Skill", "performance_score": 75,
    "price_bdt": 5800, "vendor_name": "StarTech", "in_stock": true,
    "specs": {"ram_type": "DDR5", "capacity_gb": 16, "speed_mhz": 6000,
              "modules": 1, "cas_latency": 36, "rgb": true}
  },
  {
    "id": 8, "name": "Corsair Vengeance LPX 16GB DDR4 3200MHz", "component_type": "ram",
    "brand": "Corsair", "performance_score": 60,
    "price_bdt": 3600, "vendor_name": "StarTech", "in_stock": true,
    "specs": {"ram_type": "DDR4", "capacity_gb": 16, "speed_mhz": 3200,
              "modules": 2, "cas_latency": 16, "rgb": false}
  },
  {
    "id": 9, "name": "Samsung 980 PRO 1TB NVMe", "component_type": "storage",
    "brand": "Samsung", "performance_score": 80,
    "price_bdt": 8500, "vendor_name": "StarTech", "in_stock": true,
    "specs": {"storage_type": "NVMe", "capacity_gb": 1000}
  },
  {
    "id": 10, "name": "Corsair RM750e 750W 80+ Gold", "component_type": "psu",
    "brand": "Corsair", "performance_score": 70,
    "price_bdt": 9500, "vendor_name": "StarTech", "in_stock": true,
    "specs": {"wattage": 750, "efficiency_rating": "80+ Gold", "modular": true}
  },
  {
    "id": 11, "name": "NZXT H5 Flow", "component_type": "case",
    "brand": "NZXT", "performance_score": 65,
    "price_bdt": 8500, "vendor_name": "StarTech", "in_stock": true,
    "specs": {"max_gpu_length_mm": 365, "max_cpu_cooler_height_mm": 165,
              "psu_support": "ATX", "form_factor_support": ["ATX", "mATX", "ITX"]}
  },
  {
    "id": 12, "name": "DeepCool AK400", "component_type": "cooler",
    "brand": "DeepCool", "performance_score": 60,
    "price_bdt": 2800, "vendor_name": "StarTech", "in_stock": true,
    "specs": {"cooler_type": "Air", "fan_size_mm": 120,
              "tdp_capacity_watts": 220, "socket_support": ["AM5", "AM4", "LGA1700"]}
  }
]
```

### Expected Test Result
With data above, a request for `{ budget: 80000, purpose: "gaming" }` should:
- Pick AMD Ryzen 5 7600 (AM5 platform)
- Pick RX 7700 XT over RTX 4060 (better value for gaming)
- Pick compatible DDR5 RAM + B650 motherboard
- Total should be under 80,000৳
- Score should be ≥ 8.0 after LLM evaluation

---

## 11. Repo vs Package? → **Separate Repo, Microservice**

Build as a **standalone Docker microservice** in its own repo (`zenfa-ai`). Reasons:
- Clean separation from scraper/frontend code
- Can scale independently
- Sellable to external vendors without exposing main codebase
- Independent CI/CD pipeline

Connect to main backend via:
```yaml
# In project-zenfa's docker-compose.yml, add:
  zenfa-ai:
    build: ../zenfa-ai
    ports:
      - "8001:8001"
    environment:
      - GEMINI_API_KEY=${GEMINI_API_KEY}
      - REDIS_URL=redis://redis:6379/1
```

Main backend calls it at `http://zenfa-ai:8001/build`.
