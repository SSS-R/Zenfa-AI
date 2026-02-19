"""Shared enums, component models, and user preference models for Zenfa AI Engine."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────
# Enums — Shared Vocabulary
# ──────────────────────────────────────────────


class ComponentType(str, Enum):
    """Hardware component categories used in PC builds."""

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


class SocketType(str, Enum):
    """CPU / motherboard socket types."""

    AM5 = "AM5"
    AM4 = "AM4"
    LGA1700 = "LGA1700"
    LGA1200 = "LGA1200"


class FormFactor(str, Enum):
    """Motherboard / case form factors."""

    ATX = "ATX"
    MICRO_ATX = "mATX"
    MINI_ITX = "ITX"
    E_ATX = "E-ATX"


class RAMType(str, Enum):
    """RAM generation types."""

    DDR4 = "DDR4"
    DDR5 = "DDR5"


class StorageType(str, Enum):
    """Storage interface types."""

    NVME = "NVMe"
    SATA = "SATA"
    HDD = "HDD"


class PSUEfficiency(str, Enum):
    """PSU 80 PLUS efficiency ratings."""

    BRONZE = "80+ Bronze"
    GOLD = "80+ Gold"
    PLATINUM = "80+ Platinum"
    TITANIUM = "80+ Titanium"
    WHITE = "80+ White"
    NONE = "None"


class CoolerType(str, Enum):
    """CPU cooler types."""

    AIR = "Air"
    LIQUID = "Liquid"


class VendorName(str, Enum):
    """Supported Bangladeshi hardware vendors."""

    STARTECH = "StarTech"
    RYANS = "Ryans"
    TECHLAND = "TechLand"
    UCC = "UCC"
    SKYLAND = "Skyland"
    NEXUS = "Nexus"


class Purpose(str, Enum):
    """Build purpose — selectable, not free-text."""

    GAMING = "gaming"
    EDITING = "editing"
    OFFICE = "office"
    GENERAL = "general"


# ──────────────────────────────────────────────
# Component Data Models
# ──────────────────────────────────────────────


class ComponentWithPrice(BaseModel):
    """Flattened component with its cheapest available price.

    Sent by the main backend as part of the BuildRequest.
    The `specs` dict contains type-specific fields (socket, vram_gb, etc.).
    """

    id: int
    name: str
    slug: str
    component_type: ComponentType
    brand: Optional[str] = None
    performance_score: int = Field(default=50, ge=0, le=100)

    # Cheapest available price
    price_bdt: int = Field(ge=0, description="Price in Bangladeshi Taka (৳)")
    vendor_name: str
    vendor_url: str = ""
    in_stock: bool = True

    # Type-specific specs — schema varies by component_type
    # CPU: {"socket": "AM5", "core_count": 6, "tdp": 65, ...}
    # GPU: {"vram_gb": 8, "length_mm": 240, "recommended_psu_wattage": 550}
    specs: Dict[str, Any] = Field(default_factory=dict)


# ──────────────────────────────────────────────
# User Preferences — All Predefined Options
# ──────────────────────────────────────────────

# Allowed storage tier values (GB)
StorageTier = Literal[256, 512, 1000, 2000]

# Allowed brand preferences
BrandPreference = Literal["AMD", "Intel", "NVIDIA"]


class BuildPreferences(BaseModel):
    """Optional user preferences — all predefined selectable options.

    Every field is an enum, literal, or bool. NO free-text fields allowed.
    This prevents LLM hallucination on user intent.
    """

    prefer_brand: Optional[BrandPreference] = None
    prefer_rgb: bool = False
    min_storage_gb: StorageTier = 256
    prefer_wifi: bool = False

    # ── Future selectable options go here ──
    # prefer_architecture: Optional[Literal["AMD", "Intel"]] = None
    # prefer_gpu_series: Optional[Literal["RTX 40", "RTX 50", "RX 7000"]] = None
