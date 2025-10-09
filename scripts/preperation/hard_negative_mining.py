#!/usr/bin/env python3
"""
🎯 Hard Negative Mining Script for ESCO Skill Retrieval

This script performs hard negative mining using an existing model to find
challenging negatives that will improve training effectiveness.

Usage:
    python hard_negative_mining.py --model_path intfloat/multilingual-e5-base --train_data data/eval_split/train_dataset_remaining.jsonl
    python hard_negative_mining.py --model_path ./models/finetuned_model --train_data data/combined.jsonl --top_k 10

Features:
    - Uses existing model to find hard negatives
    - Configurable number of hard negatives per query
    - Avoids including existing positives as negatives
    - Preserves original data structure
    - Creates augmented training dataset
"""

import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
import numpy as np
from tqdm import tqdm
import time
import random

try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print(
        "❌ sentence-transformers not available. Install with: pip install sentence-transformers"
    )

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Default thresholds (will be overridden by calibration)
DEFAULT_MIN_QUERY_SIMILARITY = 0.4  # Minimum similarity to query (must be challenging)
DEFAULT_MAX_POSITIVE_SIMILARITY = (
    0.7  # Maximum similarity to any positive (avoid synonyms)
)


def load_jsonl_data(file_path: Path) -> List[Dict]:
    """Load JSONL training data."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON on line {line_num}: {e}")
                    continue

    logger.info(f"Loaded {len(data)} samples from {file_path}")
    return data


def save_jsonl_data(data: List[Dict], file_path: Path):
    """Save data to JSONL format."""
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(data)} samples to {file_path}")


def extract_all_labels(data: List[Dict]) -> List[str]:
    """Extract all unique labels from the dataset."""
    all_labels = set()

    for sample in data:
        pos_labels = sample.get("pos", [])
        neg_labels = sample.get("neg", [])

        for label in pos_labels + neg_labels:
            label = label.strip()
            if label:
                all_labels.add(label)

    unique_labels = list(all_labels)
    logger.info(f"Extracted {len(unique_labels)} unique labels")
    return unique_labels


def calibrate_similarity_thresholds(
    model: SentenceTransformer, data: List[Dict], num_calibration_samples: int = 50
) -> Tuple[float, float]:
    """
    Fast calibration to determine appropriate similarity thresholds.

    Samples representative query-positive and query-negative pairs to understand
    the model's similarity score distribution for this dataset.

    Returns:
        (min_query_similarity, max_positive_similarity) thresholds
    """
    logger.info(
        f"🔬 Calibrating similarity thresholds on {num_calibration_samples} samples..."
    )

    query_pos_similarities = []
    query_neg_similarities = []
    pos_pos_similarities = []  # Similarity between different positives for same query

    # Sample random queries for calibration
    calibration_samples = random.sample(data, min(num_calibration_samples, len(data)))

    for sample in tqdm(calibration_samples, desc="Calibration"):
        query = sample.get("query", "").strip()
        pos_labels = sample.get("pos", [])
        neg_labels = sample.get("neg", [])

        if not query or not pos_labels or not neg_labels:
            continue

        # Encode query
        query_emb = model.encode([query], convert_to_tensor=True)

        # Encode positives
        pos_embs = model.encode(pos_labels, convert_to_tensor=True)

        # Encode negatives (sample if too many)
        sample_negs = random.sample(neg_labels, min(10, len(neg_labels)))
        neg_embs = model.encode(sample_negs, convert_to_tensor=True)

        # Query-Positive similarities
        q_pos_sims = model.similarity(query_emb, pos_embs)[0]
        query_pos_similarities.extend(q_pos_sims.cpu().numpy().tolist())

        # Query-Negative similarities
        q_neg_sims = model.similarity(query_emb, neg_embs)[0]
        query_neg_similarities.extend(q_neg_sims.cpu().numpy().tolist())

        # Positive-Positive similarities (if multiple positives)
        if len(pos_labels) >= 2:
            for i in range(len(pos_embs)):
                for j in range(i + 1, len(pos_embs)):
                    sim = model.similarity(pos_embs[i : i + 1], pos_embs[j : j + 1])[
                        0, 0
                    ]
                    pos_pos_similarities.append(sim.item())

    # Calculate statistics
    q_pos_mean = np.mean(query_pos_similarities)
    q_pos_std = np.std(query_pos_similarities)
    q_pos_25th = np.percentile(query_pos_similarities, 25)
    q_pos_50th = np.percentile(query_pos_similarities, 50)

    q_neg_mean = np.mean(query_neg_similarities)
    q_neg_std = np.std(query_neg_similarities)
    q_neg_75th = np.percentile(query_neg_similarities, 75)
    q_neg_90th = np.percentile(query_neg_similarities, 90)

    logger.info(f"📊 Calibration Results:")
    logger.info(f"   Query-Positive similarities:")
    logger.info(f"      Mean: {q_pos_mean:.4f} ± {q_pos_std:.4f}")
    logger.info(f"      25th percentile: {q_pos_25th:.4f}")
    logger.info(f"      50th percentile (median): {q_pos_50th:.4f}")
    logger.info(f"   Query-Negative similarities:")
    logger.info(f"      Mean: {q_neg_mean:.4f} ± {q_neg_std:.4f}")
    logger.info(f"      75th percentile: {q_neg_75th:.4f}")
    logger.info(f"      90th percentile: {q_neg_90th:.4f}")

    if pos_pos_similarities:
        pp_mean = np.mean(pos_pos_similarities)
        pp_50th = np.percentile(pos_pos_similarities, 50)
        logger.info(f"   Positive-Positive similarities (synonyms):")
        logger.info(f"      Mean: {pp_mean:.4f}")
        logger.info(f"      Median: {pp_50th:.4f}")

    # Set thresholds based on calibration
    # For avoiding false negatives: exclude candidates too similar to positives
    # We want to filter out synonyms while keeping challenging negatives
    # Use 50th percentile of query-positive (median) as the cutoff
    # This means: exclude top 50% most similar to positives
    max_positive_similarity = q_pos_50th

    # No minimum query similarity - we want exactly top_k negatives
    # Candidates are sorted by query similarity, so most similar are picked first
    # Only filter is positive similarity to avoid false negatives (synonyms)
    min_query_similarity = 0.0

    logger.info(f"✅ Calibrated Thresholds:")
    logger.info(
        f"   min_query_similarity: {min_query_similarity:.4f} (disabled - accept all)"
    )
    logger.info(f"      (All candidates accepted, sorted by query similarity)")
    logger.info(f"   max_positive_similarity: {max_positive_similarity:.4f}")
    logger.info(f"      (Candidates scoring >= this with any positive are excluded)")

    # Explain the strategy
    logger.info(f"   💡 Strategy:")
    logger.info(
        f"      - Accept negatives sorted by query similarity (no minimum threshold)"
    )
    logger.info(
        f"      - Reject only if too similar to positives (top {100*(1-np.searchsorted(np.sort(query_pos_similarities), max_positive_similarity)/len(query_pos_similarities)):.0f}% of query-pos similarities)"
    )

    return min_query_similarity, max_positive_similarity


def load_calibration_from_analysis(
    analysis_path: Path,
) -> Optional[Tuple[float, float]]:
    """
    Load pre-computed calibration thresholds from analyze_similarity_distribution.py results.

    Returns:
        (min_query_similarity, max_positive_similarity) or None if file not found
    """
    if not analysis_path.exists():
        logger.warning(f"Calibration file not found: {analysis_path}")
        return None

    try:
        with open(analysis_path, "r", encoding="utf-8") as f:
            analysis = json.load(f)

        recommendations = analysis.get("recommendations", {})

        # Use conservative threshold from analysis
        # This is (1 - similarity) for distance, so we need to convert
        distance_threshold = recommendations.get("conservative_distance_threshold")

        if distance_threshold is not None:
            max_positive_similarity = 1 - distance_threshold

            # For min query similarity, use median of same_occupation pairs
            same_occ_stats = analysis.get("statistics", {}).get("same_occupation", {})
            min_query_similarity = same_occ_stats.get(
                "percentile_75", DEFAULT_MIN_QUERY_SIMILARITY
            )

            logger.info(f"📁 Loaded calibration from {analysis_path}")
            logger.info(f"   min_query_similarity: {min_query_similarity:.4f}")
            logger.info(f"   max_positive_similarity: {max_positive_similarity:.4f}")

            return min_query_similarity, max_positive_similarity

    except Exception as e:
        logger.warning(f"Error loading calibration: {e}")
        return None

    return None


def mine_hard_negatives(
    model: SentenceTransformer,
    data: List[Dict],
    all_labels: List[str],
    top_k: int = 10,
    batch_size: int = 32,
    min_query_similarity: float = DEFAULT_MIN_QUERY_SIMILARITY,
    max_positive_similarity: float = DEFAULT_MAX_POSITIVE_SIMILARITY,
) -> List[Dict]:
    """
    Mine hard negatives using the provided model with false negative filtering.

    Selects top_k labels most similar to the query, filtering only to exclude:
    1. Labels already in positive or negative sets
    2. Labels too similar to any positive (similarity >= max_positive_similarity)

    This ensures we always get top_k negatives while preventing false negatives (synonyms).
    The min_query_similarity parameter is ignored (kept for API compatibility).
    """
    logger.info(f"🔍 Mining hard negatives with top_k={top_k}")
    logger.info(f"   Strategy: Select top {top_k} most similar to query")
    logger.info(
        f"   Filter: Exclude if similarity to any positive >= {max_positive_similarity:.4f}"
    )

    # Encode all labels once
    logger.info(f"🧮 Encoding {len(all_labels)} labels...")
    label_embeddings = model.encode(
        all_labels,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_tensor=True,
    )

    # Create label lookup
    label_to_idx = {label: idx for idx, label in enumerate(all_labels)}

    augmented_data = []

    # Statistics
    total_candidates = 0
    excluded_by_query_sim = 0
    excluded_by_positive_sim = 0
    excluded_already_used = 0
    total_added = 0

    logger.info(f"⛏️  Mining hard negatives for {len(data)} samples...")
    for sample in tqdm(data, desc="Mining negatives"):
        query = sample.get("query", "").strip()
        pos_labels = list(sample.get("pos", []))
        pos_labels_set = set(pos_labels)
        existing_neg_labels = set(sample.get("neg", []))

        if not query or not pos_labels:
            augmented_data.append(sample)
            continue

        # Encode query
        query_embedding = model.encode([query], convert_to_tensor=True)

        # Encode positives for similarity checking
        pos_embeddings = model.encode(pos_labels, convert_to_tensor=True)

        # Calculate similarities to all labels
        query_label_similarities = model.similarity(query_embedding, label_embeddings)[
            0
        ]

        # Get top similar labels (sorted by similarity to query)
        top_indices = query_label_similarities.argsort(descending=True)

        # Find hard negatives with filtering
        hard_negatives = []
        sample_candidates = 0
        sample_excluded_query = 0
        sample_excluded_positive = 0
        sample_excluded_used = 0

        for idx in top_indices:
            candidate_label = all_labels[idx.item()]
            query_sim = query_label_similarities[idx].item()

            sample_candidates += 1

            # Skip if it's already a positive or existing negative
            if (
                candidate_label in pos_labels_set
                or candidate_label in existing_neg_labels
            ):
                sample_excluded_used += 1
                continue

            # Only filter: Must NOT be too similar to any positive (avoid false negatives/synonyms)
            candidate_idx = label_to_idx[candidate_label]
            candidate_embedding = label_embeddings[candidate_idx : candidate_idx + 1]

            # Check similarity to all positives
            pos_similarities = model.similarity(candidate_embedding, pos_embeddings)[0]
            max_pos_sim = pos_similarities.max().item()

            if max_pos_sim >= max_positive_similarity:
                sample_excluded_positive += 1
                continue

            # This is a valid hard negative
            hard_negatives.append(candidate_label)
            if len(hard_negatives) >= top_k:
                break

        # Update statistics
        total_candidates += sample_candidates
        excluded_by_query_sim += sample_excluded_query
        excluded_by_positive_sim += sample_excluded_positive
        excluded_already_used += sample_excluded_used
        total_added += len(hard_negatives)

        # Create augmented sample
        augmented_sample = sample.copy()

        # Combine existing negatives with hard negatives
        all_negatives = list(existing_neg_labels) + hard_negatives
        augmented_sample["neg"] = all_negatives

        # Add metadata about mining
        if "meta" not in augmented_sample:
            augmented_sample["meta"] = {}
        augmented_sample["meta"]["hard_negatives_added"] = len(hard_negatives)
        augmented_sample["meta"]["total_negatives"] = len(all_negatives)
        augmented_sample["meta"][
            "excluded_by_positive_similarity"
        ] = sample_excluded_positive

        augmented_data.append(augmented_sample)

    logger.info(f"✅ Hard negative mining completed")
    logger.info(f"📊 Filtering Statistics:")
    logger.info(f"   Total candidates considered: {total_candidates}")
    logger.info(f"   Excluded (already used): {excluded_already_used}")
    logger.info(f"   Excluded (too similar to positives): {excluded_by_positive_sim}")
    logger.info(f"   Successfully added: {total_added}")
    logger.info(f"   Average per sample: {total_added / len(data):.2f}")
    logger.info(f"   Target per sample: {top_k}")

    if excluded_by_positive_sim > 0:
        logger.info(
            f"   🎯 Prevented {excluded_by_positive_sim} potential false negatives!"
        )

    return augmented_data


def analyze_mining_results(original_data: List[Dict], augmented_data: List[Dict]):
    """Analyze the results of hard negative mining."""
    original_neg_counts = []
    augmented_neg_counts = []
    hard_neg_added = []

    for orig, aug in zip(original_data, augmented_data):
        orig_negs = len(orig.get("neg", []))
        aug_negs = len(aug.get("neg", []))
        added = aug.get("meta", {}).get("hard_negatives_added", 0)

        original_neg_counts.append(orig_negs)
        augmented_neg_counts.append(aug_negs)
        hard_neg_added.append(added)

    logger.info(f"📊 Mining Analysis:")
    logger.info(f"   Average original negatives: {np.mean(original_neg_counts):.2f}")
    logger.info(f"   Average augmented negatives: {np.mean(augmented_neg_counts):.2f}")
    logger.info(f"   Average hard negatives added: {np.mean(hard_neg_added):.2f}")
    logger.info(
        f"   Samples with hard negatives: {sum(1 for x in hard_neg_added if x > 0)}"
    )


def main():
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        logger.error("sentence-transformers is required!")
        logger.error("Install with: pip install sentence-transformers")
        return 1

    parser = argparse.ArgumentParser(
        description="Mine hard negatives for embedding training with false negative filtering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # With automatic calibration (recommended):
  python hard_negative_mining.py -m intfloat/multilingual-e5-base -t train.jsonl --calibrate
  
  # Using pre-computed calibration from analyze_similarity_distribution.py:
  python hard_negative_mining.py -m ./models/finetuned -t train.jsonl --calibration-file analysis_results.json
  
  # Manual thresholds:
  python hard_negative_mining.py -m ./models/finetuned -t train.jsonl --min-query-sim 0.4 --max-pos-sim 0.7
        """,
    )
    parser.add_argument(
        "--model_path",
        "-m",
        type=str,
        required=True,
        help="Path to model for mining (local path or HuggingFace identifier)",
    )
    parser.add_argument(
        "--train_data",
        "-t",
        type=str,
        required=True,
        help="Path to training data (JSONL format)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output file path (default: auto-generated)",
    )
    parser.add_argument(
        "--top_k",
        "-k",
        type=int,
        default=8,
        help="Number of hard negatives to mine per query (default: 8)",
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        type=int,
        default=32,
        help="Batch size for encoding (default: 32)",
    )

    # Calibration options
    calibration_group = parser.add_argument_group(
        "calibration", "Options for similarity threshold calibration"
    )
    calibration_group.add_argument(
        "--calibrate",
        action="store_true",
        help="Run automatic calibration on training data (recommended)",
    )
    calibration_group.add_argument(
        "--calibration-samples",
        type=int,
        default=50,
        help="Number of samples for calibration (default: 50)",
    )
    calibration_group.add_argument(
        "--calibration-file",
        type=str,
        default=None,
        help="Path to pre-computed calibration JSON from analyze_similarity_distribution.py",
    )

    # Manual threshold options
    threshold_group = parser.add_argument_group(
        "thresholds", "Manual similarity thresholds (if not calibrating)"
    )
    threshold_group.add_argument(
        "--min-query-sim",
        type=float,
        default=None,
        help=f"Minimum similarity to query (default: {DEFAULT_MIN_QUERY_SIMILARITY})",
    )
    threshold_group.add_argument(
        "--max-pos-sim",
        type=float,
        default=None,
        help=f"Maximum similarity to positives (default: {DEFAULT_MAX_POSITIVE_SIMILARITY})",
    )

    args = parser.parse_args()

    # Load model
    logger.info(f"🤖 Loading model: {args.model_path}")
    try:
        model = SentenceTransformer(args.model_path)
        logger.info(f"✅ Model loaded successfully")
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        return 1

    # Load training data
    train_path = Path(args.train_data)
    if not train_path.exists():
        logger.error(f"Training data {train_path} does not exist!")
        return 1

    original_data = load_jsonl_data(train_path)
    if not original_data:
        logger.error("No valid training data found!")
        return 1

    # Extract all labels for candidate pool
    all_labels = extract_all_labels(original_data)
    if not all_labels:
        logger.error("No labels found in training data!")
        return 1

    # Determine similarity thresholds
    min_query_similarity = DEFAULT_MIN_QUERY_SIMILARITY
    max_positive_similarity = DEFAULT_MAX_POSITIVE_SIMILARITY

    if args.calibrate:
        # Run automatic calibration
        logger.info("🔬 Running automatic calibration...")
        min_query_similarity, max_positive_similarity = calibrate_similarity_thresholds(
            model=model,
            data=original_data,
            num_calibration_samples=args.calibration_samples,
        )
    elif args.calibration_file:
        # Load from pre-computed analysis
        calibration_path = Path(args.calibration_file)
        thresholds = load_calibration_from_analysis(calibration_path)
        if thresholds:
            min_query_similarity, max_positive_similarity = thresholds
        else:
            logger.warning("Could not load calibration file, using defaults")
    else:
        # Use manual thresholds if provided
        if args.min_query_sim is not None:
            min_query_similarity = args.min_query_sim
            logger.info(
                f"Using manual min_query_similarity: {min_query_similarity:.4f}"
            )
        if args.max_pos_sim is not None:
            max_positive_similarity = args.max_pos_sim
            logger.info(
                f"Using manual max_positive_similarity: {max_positive_similarity:.4f}"
            )

        if args.min_query_sim is None and args.max_pos_sim is None:
            logger.warning(f"⚠️  No calibration requested, using default thresholds:")
            logger.warning(f"   min_query_similarity: {min_query_similarity:.4f}")
            logger.warning(f"   max_positive_similarity: {max_positive_similarity:.4f}")
            logger.warning(f"   Consider using --calibrate for better results!")

    # Mine hard negatives
    logger.info(f"🚀 Starting hard negative mining with false negative filtering...")
    start_time = time.time()

    augmented_data = mine_hard_negatives(
        model=model,
        data=original_data,
        all_labels=all_labels,
        top_k=args.top_k,
        batch_size=args.batch_size,
        min_query_similarity=min_query_similarity,
        max_positive_similarity=max_positive_similarity,
    )

    mining_time = time.time() - start_time
    logger.info(f"⏱️  Mining completed in {mining_time:.2f}s")

    # Analyze results
    analyze_mining_results(original_data, augmented_data)

    # Save augmented data
    if args.output:
        output_path = Path(args.output)
    else:
        # Create output filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        train_stem = train_path.stem
        output_path = (
            train_path.parent / f"{train_stem}_hard_negatives_{timestamp}.jsonl"
        )

    save_jsonl_data(augmented_data, output_path)

    logger.info(f"✨ Hard negative mining completed successfully!")
    logger.info(f"   Original samples: {len(original_data)}")
    logger.info(f"   Augmented file: {output_path}")
    logger.info(f"   Next step: Use {output_path} for training")

    return 0


if __name__ == "__main__":
    exit(main())
