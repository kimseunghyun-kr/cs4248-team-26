"""
Personalization-first modules for the `vlm-personalization-pgd` branch.

This package keeps the reusable ideas from the earlier codebase:

- frozen transformer loading from `encoder.py`
- adaptable dataset loading for instance/class data
- upstream-style PGD direction discovery
"""

from .data import (
    ConceptMediaRecord,
    PersonalizationDataset,
    build_dataloaders,
    build_media_records,
)
from .iccv import ImageICCVArtifacts, MultimodalEmbedder, TextICCVArtifacts
from .projection import project_out
from .prompts import NuisanceAxis, PersonalizationPromptBank, build_prompt_bank
from .trainer import PersonalizationConfig, PersonalizationTrainer

__all__ = [
    "ConceptMediaRecord",
    "ImageICCVArtifacts",
    "MultimodalEmbedder",
    "NuisanceAxis",
    "PersonalizationConfig",
    "PersonalizationDataset",
    "PersonalizationPromptBank",
    "PersonalizationTrainer",
    "TextICCVArtifacts",
    "build_dataloaders",
    "build_media_records",
    "build_prompt_bank",
    "project_out",
]
