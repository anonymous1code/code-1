"""
Stratified sampler for fine-tuning data.

Selects N image pairs per rejection reason category + N accepted pairs
from each dataset split (vcape-r-20k and vcape-s-20k).

Saves a JSON manifest of the original row indices used for training so
they can be excluded during evaluation.

Usage:
    python scripts/finetune_sampler.py
    python scripts/finetune_sampler.py --samples-per-category 5 --seed 123 --force
"""

import argparse
import json
import os
import random
from collections import defaultdict

from datasets import Dataset, concatenate_datasets, load_from_disk

# Expected rejection reasons per split
VCAPE_R_REJECTION_REASONS = [
    "Irrelevant Objects",
    "Geometry Artifacts",
    "Texture/Lighting/Color Issues",
    "Wrong Orientation",
    "Generated-Main Image Mismatch (Color/Material/Pattern)",
    "Background Issues",
]

VCAPE_S_REJECTION_REASONS = [
    "wrong_orientation",
    "wrong_product_same_pt",
    "wrong_product_different_pt",
]

# vcape-r uses "accepted" as rejection_reason for accepted pairs
# vcape-s uses "" (empty string) for accepted pairs
_VCAPE_R_ACCEPTED_REASONS = {"accepted"}
_VCAPE_S_ACCEPTED_REASONS = {"", "accept"}


def _is_accepted(rejection_reason: str, split: str) -> bool:
    """Check if a row is accepted based on its rejection_reason and split."""
    if split == "vcape-r-20k":
        return rejection_reason in _VCAPE_R_ACCEPTED_REASONS
    else:
        return rejection_reason in _VCAPE_S_ACCEPTED_REASONS


def _group_by_reason(ds: Dataset, split: str) -> dict[str, list[int]]:
    """Group row indices by rejection_reason category.

    Returns a dict mapping each rejection reason to a list of row indices,
    plus an "accepted" key for accepted pairs.
    """
    groups: dict[str, list[int]] = defaultdict(list)
    for i, reason in enumerate(ds["rejection_reason"]):
        if _is_accepted(reason, split):
            groups["accepted"].append(i)
        else:
            groups[reason].append(i)
    return dict(groups)


def _sample_n_per_group(
    ds: Dataset,
    groups: dict[str, list[int]],
    rng: random.Random,
    split: str,
    n: int,
) -> tuple[Dataset, dict[str, list[int]]]:
    """Sample *n* rows per group and add a 'split' column.

    Returns the sampled Dataset and a dict mapping each category to the
    original row indices that were selected.
    """
    sampled_indices: list[int] = []
    index_manifest: dict[str, list[int]] = {}

    for reason, indices in sorted(groups.items()):
        k = min(n, len(indices))
        chosen = rng.sample(indices, k)
        sampled_indices.extend(chosen)
        index_manifest[reason] = chosen

    subset = ds.select(sampled_indices)
    # Add split column
    subset = subset.add_column("split", [split] * len(subset))
    return subset, index_manifest


def sample_finetune_data(
    vcape_r_path: str = "data/vcape-r-20k",
    vcape_s_path: str = "data/vcape-s-20k",
    output_path: str = "data/finetune_samples",
    seed: int = 42,
    samples_per_category: int = 1,
    force: bool = False,
) -> Dataset:
    """Sample N image pairs per rejection reason + N accepted per split.

    Parameters
    ----------
    vcape_r_path : str
        Path to the vcape-r-20k dataset on disk.
    vcape_s_path : str
        Path to the vcape-s-20k dataset on disk.
    output_path : str
        Directory to save the sampled dataset and index manifest.
    seed : int
        Random seed for reproducible sampling.
    samples_per_category : int
        Number of samples to draw per rejection reason category (and
        per accepted category).  Capped at the available count if a
        category has fewer rows.
    force : bool
        If *True*, re-sample even when *output_path* already exists.

    Returns
    -------
    Dataset
        A HuggingFace Dataset containing all original columns plus a
        ``split`` column indicating the source dataset.
    """
    # Reuse cached dataset if available
    if os.path.isdir(output_path) and not force:
        print(f"Reusing cached finetune samples at {output_path}")
        return load_from_disk(output_path)

    rng = random.Random(seed)

    # --- vcape-r-20k ---
    ds_r = load_from_disk(vcape_r_path)
    groups_r = _group_by_reason(ds_r, "vcape-r-20k")
    samples_r, manifest_r = _sample_n_per_group(
        ds_r, groups_r, rng, "vcape-r-20k", samples_per_category,
    )
    print(f"vcape-r-20k: sampled {len(samples_r)} rows "
          f"({samples_per_category}/category, "
          f"{len(groups_r) - 1} rejected + 1 accepted)")

    # --- vcape-s-20k ---
    ds_s = load_from_disk(vcape_s_path)
    groups_s = _group_by_reason(ds_s, "vcape-s-20k")
    samples_s, manifest_s = _sample_n_per_group(
        ds_s, groups_s, rng, "vcape-s-20k", samples_per_category,
    )
    print(f"vcape-s-20k: sampled {len(samples_s)} rows "
          f"({samples_per_category}/category, "
          f"{len(groups_s) - 1} rejected + 1 accepted)")

    # Concatenate
    combined = concatenate_datasets([samples_r, samples_s])
    print(f"Total finetune samples: {len(combined)}")

    # Save dataset to disk
    combined.save_to_disk(output_path)
    print(f"Saved finetune samples to {output_path}")

    # Save index manifest — maps (split, category) → list of original
    # row indices.  This lets the eval pipeline exclude training rows.
    manifest = {
        "seed": seed,
        "samples_per_category": samples_per_category,
        "vcape-r-20k": manifest_r,
        "vcape-s-20k": manifest_s,
    }
    # Flatten all indices for quick lookup during eval
    manifest["all_indices"] = {
        "vcape-r-20k": sorted(
            idx for indices in manifest_r.values() for idx in indices
        ),
        "vcape-s-20k": sorted(
            idx for indices in manifest_s.values() for idx in indices
        ),
    }
    manifest_path = os.path.join(output_path, "train_indices.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved training index manifest to {manifest_path}")

    return combined


def load_train_indices(manifest_path: str) -> dict[str, list[int]]:
    """Load the flat index manifest for excluding training rows at eval.

    Parameters
    ----------
    manifest_path : str
        Path to ``train_indices.json`` (inside the sampled dataset dir).

    Returns
    -------
    dict[str, list[int]]
        Mapping from dataset name (``"vcape-r-20k"``, ``"vcape-s-20k"``)
        to sorted list of row indices used for training.
    """
    with open(manifest_path) as f:
        manifest = json.load(f)
    return manifest["all_indices"]


def main():
    parser = argparse.ArgumentParser(
        description="Sample stratified fine-tuning data from vcape datasets."
    )
    parser.add_argument("--vcape-r", type=str, default="data/vcape-r-20k")
    parser.add_argument("--vcape-s", type=str, default="data/vcape-s-20k")
    parser.add_argument("--output", type=str, default="data/finetune_samples")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples-per-category", type=int, default=1,
                        help="Number of samples per rejection reason + accepted")
    parser.add_argument("--force", action="store_true",
                        help="Force re-sampling even if cached data exists")
    args = parser.parse_args()

    ds = sample_finetune_data(
        vcape_r_path=args.vcape_r,
        vcape_s_path=args.vcape_s,
        output_path=args.output,
        seed=args.seed,
        samples_per_category=args.samples_per_category,
        force=args.force,
    )
    print(f"\nColumns: {ds.column_names}")
    print(f"Splits:  {set(ds['split'])}")
    print(f"Total:   {len(ds)} samples")

    # Summary by category
    from collections import Counter
    reasons = Counter(ds["rejection_reason"])
    print(f"Reasons: {dict(reasons)}")


if __name__ == "__main__":
    main()
