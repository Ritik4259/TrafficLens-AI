"""
Reproducible Sequence-Level Dataset Curator for UA-DETRAC.

Selects a balanced, curated subset of 5–10 representative sequences across
environmental conditions (night, cloudy, rainy, sunny) and camera states (stable, unstable),
producing deterministic train/val/test splits while preserving all original metadata.
"""

from __future__ import annotations
import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set

from .parser import SequenceAnnotation


# Canonical 8-sequence provisional benchmark subset covering all operational regimes
DEFAULT_CURATED_SEQUENCES: List[Dict[str, str]] = [
    {"name": "MVI_39801", "condition": "cloudy", "camera_state": "stable", "split": "train"},
    {"name": "MVI_40141", "condition": "rainy", "camera_state": "stable", "split": "train"},
    {"name": "MVI_63521", "condition": "sunny", "camera_state": "stable", "split": "train"},
    {"name": "MVI_63561", "condition": "sunny", "camera_state": "unstable", "split": "train"},
    {"name": "MVI_20012", "condition": "night", "camera_state": "unstable", "split": "val"},
    {"name": "MVI_39811", "condition": "cloudy", "camera_state": "stable", "split": "val"},
    {"name": "MVI_20011", "condition": "night", "camera_state": "unstable", "split": "test"},
    {"name": "MVI_40131", "condition": "rainy", "camera_state": "unstable", "split": "test"},
]


@dataclass
class CuratedManifest:
    """Manifest describing the curated sequence subset and split configuration."""
    version: str
    status: str
    description: str
    total_sequences: int
    splits: Dict[str, List[str]]
    split_conditions: Dict[str, List[str]]
    sequence_metadata: Dict[str, Dict[str, str]]
    seed: int

    def validate_split_integrity(self) -> bool:
        """Verifies that all sequence IDs are strictly unique across splits."""
        all_seqs: List[str] = []
        for split_name, seq_list in self.splits.items():
            all_seqs.extend(seq_list)
        return len(all_seqs) == len(set(all_seqs)) == self.total_sequences


class SequenceCurator:
    """Manages reproducible sequence curation and split generation."""

    def __init__(self, seed: int = 42):
        self.seed = seed

    def get_default_curated_subset(self) -> CuratedManifest:
        """Returns the canonical 8-sequence balanced provisional benchmark subset."""
        splits: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
        split_conditions: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
        metadata: Dict[str, Dict[str, str]] = {}

        for item in DEFAULT_CURATED_SEQUENCES:
            seq_name = item["name"]
            split = item["split"]
            splits[split].append(seq_name)
            cond_desc = f"{item['condition']} ({item['camera_state']})"
            split_conditions[split].append(cond_desc)
            metadata[seq_name] = {
                "condition": item["condition"],
                "camera_state": item["camera_state"],
            }

        manifest = CuratedManifest(
            version="1.0.0",
            status="provisional",
            description="Provisional 8-sequence benchmark subset for local development and rapid evaluation.",
            total_sequences=len(DEFAULT_CURATED_SEQUENCES),
            splits=splits,
            split_conditions=split_conditions,
            sequence_metadata=metadata,
            seed=self.seed,
        )
        assert manifest.validate_split_integrity(), "Duplicate sequence IDs detected in default curation!"
        return manifest

    def curate_from_available(
        self,
        available_sequences: Dict[str, SequenceAnnotation],
        target_count: int = 8,
        train_ratio: float = 0.5,
        val_ratio: float = 0.25,
        test_ratio: float = 0.25,
    ) -> CuratedManifest:
        """
        Deterministically selects a balanced subset from available sequences with reproducible seed.
        """
        rng = random.Random(self.seed)
        seq_names = sorted(available_sequences.keys())

        # Group by condition
        by_condition: Dict[str, List[str]] = {}
        for name in seq_names:
            cond = available_sequences[name].scene_condition
            by_condition.setdefault(cond, []).append(name)

        # Select balanced sequences across conditions
        selected: List[str] = []
        conditions = sorted(by_condition.keys())
        while len(selected) < min(target_count, len(seq_names)):
            added = False
            for cond in conditions:
                pool = [s for s in by_condition[cond] if s not in selected]
                if pool:
                    chosen = rng.choice(pool)
                    selected.append(chosen)
                    added = True
                    if len(selected) >= target_count:
                        break
            if not added:
                break

        # Deterministically assign splits using RNG
        rng.shuffle(selected)
        n = len(selected)
        n_train = max(1, int(round(n * train_ratio)))
        n_val = max(1, int(round(n * val_ratio))) if n > 2 else 0

        train_seqs = sorted(selected[:n_train])
        val_seqs = sorted(selected[n_train:n_train + n_val])
        test_seqs = sorted(selected[n_train + n_val:])

        splits = {"train": train_seqs, "val": val_seqs, "test": test_seqs}
        split_conditions = {"train": [], "val": [], "test": []}
        metadata = {}

        for split_name, seqs in splits.items():
            for name in seqs:
                seq = available_sequences[name]
                cond_desc = f"{seq.scene_condition} ({seq.camera_state})"
                split_conditions[split_name].append(cond_desc)
                metadata[name] = {
                    "condition": seq.scene_condition,
                    "camera_state": seq.camera_state,
                    "fog_status": seq.fog_status,
                    "total_frames": str(seq.total_frames),
                    "unique_vehicles": str(len(seq.get_all_target_ids())),
                }

        manifest = CuratedManifest(
            version="1.0.0",
            status="provisional",
            description="Provisional curated benchmark subset for local development and rapid evaluation.",
            total_sequences=len(selected),
            splits=splits,
            split_conditions=split_conditions,
            sequence_metadata=metadata,
            seed=self.seed,
        )
        assert manifest.validate_split_integrity(), "Curated splits must not have overlapping sequence IDs!"
        return manifest

    def save_manifest(self, manifest: CuratedManifest, output_path: Path | str) -> None:
        """Serializes curated manifest to a JSON file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(manifest), f, indent=2)

    @staticmethod
    def load_manifest(manifest_path: Path | str) -> CuratedManifest:
        """Loads curated manifest from a JSON file."""
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return CuratedManifest(**data)
