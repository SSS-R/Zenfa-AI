"""Pydantic models for build requests, responses, and components."""

from zenfa_ai.models.build import (
    BuildExplanation,
    BuildMetadata,
    BuildQuality,
    BuildRequest,
    BuildResponse,
    CandidateBuild,
    FinalBuild,
    SelectedComponent,
)
from zenfa_ai.models.components import (
    BrandPreference,
    BuildPreferences,
    ComponentType,
    ComponentWithPrice,
    CoolerType,
    FormFactor,
    PSUEfficiency,
    Purpose,
    RAMType,
    SocketType,
    StorageTier,
    StorageType,
    VendorName,
)

__all__ = [
    # Components & enums
    "BrandPreference",
    "BuildPreferences",
    "ComponentType",
    "ComponentWithPrice",
    "CoolerType",
    "FormFactor",
    "PSUEfficiency",
    "Purpose",
    "RAMType",
    "SocketType",
    "StorageTier",
    "StorageType",
    "VendorName",
    # Build models
    "BuildExplanation",
    "BuildMetadata",
    "BuildQuality",
    "BuildRequest",
    "BuildResponse",
    "CandidateBuild",
    "FinalBuild",
    "SelectedComponent",
]
