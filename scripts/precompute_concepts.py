#!/usr/bin/env python3
"""
Batch-precompute CLEAR concept scores for a dataset.

Adapted from cxr_concept/reports/01_compute_concept_scores.py.
Computes concept scores for all images in an HDF5 file and saves
as a new HDF5 file for fast lookup during agent inference.

Usage:
    python scripts/precompute_concepts.py \
        --h5_path data/mimic.h5 \
        --output data/mimic_concept_scores.h5 \
        --batch_size 32
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import h5py

sys.path.insert(0, str(Path(__file__).parent.parent))

from clear.concept_scorer import CLEARConceptScorer


def main():
    parser = argparse.ArgumentParser(description="Precompute CLEAR concept scores")
    parser.add_argument("--h5_path", type=str, required=True, help="Path to CXR HDF5 file")
    parser.add_argument("--output", type=str, required=True, help="Output HDF5 path")
    parser.add_argument("--model_path", type=str, default=None, help="CLEAR model weights")
    parser.add_argument("--concepts_path", type=str, default=None, help="Concepts JSON path")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    print("=== Precomputing CLEAR Concept Scores ===")

    scorer = CLEARConceptScorer(
        model_path=args.model_path,
        concepts_path=args.concepts_path,
    )

    print("Loading CLEAR model...")
    scorer.load()
    print(f"Model loaded. {len(scorer.concepts)} concepts encoded.")

    print(f"Scoring images from {args.h5_path}...")
    scores_matrix = scorer.score_batch_h5(args.h5_path, batch_size=args.batch_size)
    print(f"Scores matrix shape: {scores_matrix.shape}")

    print(f"Saving to {args.output}...")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.output, "w") as f:
        f.create_dataset("concept_scores", data=scores_matrix, compression="gzip")
        f.create_dataset("concepts", data=[c.encode("utf-8") for c in scorer.concepts])
        f.create_dataset("original_indices", data=np.arange(scores_matrix.shape[0]))

        f.attrs["num_images"] = scores_matrix.shape[0]
        f.attrs["num_concepts"] = scores_matrix.shape[1]
        f.attrs["normalization"] = "L2 normalized (cosine similarity)"
        f.attrs["score_range_min"] = float(scores_matrix.min())
        f.attrs["score_range_max"] = float(scores_matrix.max())

    print(f"Done. Saved {scores_matrix.shape[0]} images x {scores_matrix.shape[1]} concepts")


if __name__ == "__main__":
    main()
