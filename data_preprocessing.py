#!/usr/bin/env python

"""
CLI utility to convert gzipped FCS data into gzipped CSV outputs for GateMeClass.

This version:
1. Performs sample-level train/test splitting
2. Outputs numeric label IDs (not cell type names)
3. Creates a mapping file (numeric ID <-> cell type name) for GateMeClass to use

Args:
    --data.raw      Path to a gz-compressed FCS file.
    --data.labels   Path to a gz-compressed labels file (optional, for renaming columns).
    --output_dir    Directory where the matrix/label CSV files will be written.
    --name          Dataset name used for the output filenames.
    --seed          Random seed for reproducible train/test split.
    --train_ratio   Proportion of samples to use for training (default: 0.5).
    --min_train_samples  Minimum number of samples in training set (default: 2).
"""

import pandas as pd
import numpy as np
import argparse
import gzip
import os
import sys
import fcsparser
import tempfile
from pathlib import Path
from typing import Iterable, List, Tuple


def read_bytes_handling_gzip(path: str) -> bytes:
    """
    Return file contents, transparently handling gzip-compressed files.

    Some inputs may have a .gz suffix even when they are plain text; fall back to
    normal reads if gzip decompression fails.
    """
    try:
        with gzip.open(path, "rb") as fh:
            return fh.read()
    except (OSError, gzip.BadGzipFile):
        with open(path, "rb") as fh:
            return fh.read()


def parse_fcs_to_dataframe(raw_gz_path: str):
    """Parse FCS file and return as DataFrame."""
    data_bytes = read_bytes_handling_gzip(raw_gz_path)

    # fcsparser.parse expects a file path; use a temporary file
    with tempfile.NamedTemporaryFile(suffix=".fcs", delete=False) as tmp:
        tmp.write(data_bytes)
        tmp_path = tmp.name

    try:
        _, data = fcsparser.parse(tmp_path, reformat_meta=True)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    return data


def parse_label_lines(label_text: str, expected_count: int, source: str) -> List[str]:
    """Parse label file lines and validate count."""
    labels = [line.strip() for line in label_text.splitlines() if line.strip()]
    if not labels:
        raise ValueError(f"No labels found in {source}.")

    if len(labels) != expected_count:
        raise ValueError(
            f"Label count ({len(labels)}) does not match number of columns ({expected_count})."
        )
    return labels


def detect_label_format(label_path: str, label_text: str) -> str:
    """Return 'txt' or 'xml' based on path suffix or content."""
    suffixes = [s.lower() for s in Path(label_path).suffixes if s.lower() != ".gz"]
    if ".xml" in suffixes:
        return "xml"
    if ".txt" in suffixes:
        return "txt"

    stripped = label_text.lstrip()
    if stripped.startswith("<"):
        return "xml"

    return "txt"


def apply_labels(label_gz_path: str, df):
    """Apply labels to DataFrame columns (marker names) if provided."""
    if not label_gz_path or not os.path.exists(label_gz_path):
        print("Warning: No label file provided or file not found. Using original column names.", 
              file=sys.stderr)
        return df
    
    try:
        label_text = read_bytes_handling_gzip(label_gz_path).decode("utf-8")
    except UnicodeDecodeError as exc:
        print(f"Warning: Could not decode label file: {exc}. Using original column names.",
              file=sys.stderr)
        return df

    if not label_text.strip():
        print("Warning: Label file is empty. Using original column names.", file=sys.stderr)
        return df

    label_format = detect_label_format(label_gz_path, label_text)

    if label_format == "xml":
        print("Warning: XML label format not implemented. Using original column names.",
              file=sys.stderr)
        return df
    
    if label_format != "txt":
        print("Warning: Unknown label format. Using original column names.", file=sys.stderr)
        return df

    try:
        labels = parse_label_lines(label_text, expected_count=df.shape[1], source=label_gz_path)
        df.columns = labels
        print(f"Applied {len(labels)} column labels from {label_gz_path}", file=sys.stderr)
    except ValueError as exc:
        print(f"Warning: {exc} Using original column names.", file=sys.stderr)
    
    return df


def replace_NAs(df):
    """Replace NAs in label column with empty string."""
    if "label" in df.columns:
        df["label"] = df["label"].fillna("")
    return df


def get_unique_samples(df):
    """Get unique sample identifiers from the dataframe."""
    if "sample" not in df.columns:
        raise ValueError("No 'sample' column found in the data. Cannot perform sample-level split.")
    
    samples_unique = df["sample"].unique()
    return samples_unique


def create_label_mapping(df):
    """
    Create a mapping between numeric IDs and cell type names from the label column.
    
    Returns:
        dict: {numeric_id: cell_type_name}
        pd.Series: labels converted to numeric IDs
    """
    if "label" not in df.columns:
        return None, None
    
    # Get unique cell types (excluding empty strings)
    unique_labels = df["label"][df["label"] != ""].unique()
    
    # Create mapping: numeric ID (1, 2, 3, ...) -> cell type name
    label_mapping = {i+1: label for i, label in enumerate(sorted(unique_labels))}
    
    # Create reverse mapping for conversion
    reverse_mapping = {label: id for id, label in label_mapping.items()}
    
    # Convert labels to numeric IDs (empty strings become 0)
    numeric_labels = df["label"].apply(lambda x: reverse_mapping.get(x, 0) if x != "" else 0)
    
    return label_mapping, numeric_labels


def train_test_sample_split(df, samples_unique, seed=0, train_ratio=0.5, min_train_samples=2):
    """
    Split features and labels into train/test subsets based on sample column.
    
    Args:
        df: DataFrame with a 'sample' column
        samples_unique: Array of unique sample identifiers
        seed: Random seed for reproducibility
        train_ratio: Proportion of samples to use for training (0.0 to 1.0)
        min_train_samples: Minimum number of samples required in training set
    
    Returns:
        train_set, test_set: DataFrames with sample column removed
    """
    n_samples = len(samples_unique)
    
    # Validate inputs
    if n_samples < 2:
        raise ValueError(f"Need at least 2 samples for train/test split, found {n_samples}")
    
    if train_ratio <= 0 or train_ratio >= 1:
        raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}")
    
    # Calculate number of training samples
    n_train = max(min_train_samples, int(np.ceil(n_samples * train_ratio)))
    
    # Ensure we have at least 1 test sample
    if n_train >= n_samples:
        n_train = n_samples - 1
        print(
            f"Warning: Adjusted training samples to {n_train} to ensure at least 1 test sample.",
            file=sys.stderr,
        )
    
    # Randomly select training samples
    rng = np.random.RandomState(seed)
    train_samples = rng.choice(samples_unique, size=n_train, replace=False)
    
    # Split the data
    train_set = df[df["sample"].isin(train_samples)].copy()
    test_set = df[~df["sample"].isin(train_samples)].copy()
    
    # Validate split
    nrow_df = df.shape[0]
    nrow_train = train_set.shape[0]
    nrow_test = test_set.shape[0]
    
    if nrow_train + nrow_test != nrow_df:
        raise ValueError(
            f"Rows in train ({nrow_train}) + test ({nrow_test}) don't match original ({nrow_df})"
        )
    
    # Report split statistics
    print(f"\n=== Split Summary ===", file=sys.stderr)
    print(f"Total samples: {n_samples}", file=sys.stderr)
    print(f"Training samples ({len(train_samples)}): {sorted(train_samples)}", file=sys.stderr)
    print(f"Test samples ({n_samples - len(train_samples)}): {sorted(set(samples_unique) - set(train_samples))}", file=sys.stderr)
    print(f"Training cells: {nrow_train:,} ({100*nrow_train/nrow_df:.1f}%)", file=sys.stderr)
    print(f"Test cells: {nrow_test:,} ({100*nrow_test/nrow_df:.1f}%)", file=sys.stderr)
    
    # Check label distribution if labels exist
    if "label" in df.columns:
        print(f"\nLabel distribution:", file=sys.stderr)
        train_labels = train_set["label"].value_counts()
        test_labels = test_set["label"].value_counts()
        all_labels = sorted(set(train_labels.index) | set(test_labels.index))
        
        for label in all_labels:
            train_count = train_labels.get(label, 0)
            test_count = test_labels.get(label, 0)
            print(f"  {label}: train={train_count:,}, test={test_count:,}", file=sys.stderr)
    
    print("=" * 20 + "\n", file=sys.stderr)
    
    # Remove sample column
    train_set = train_set.drop("sample", axis=1)
    test_set = test_set.drop("sample", axis=1)
    
    return train_set, test_set


def split_features_and_labels(df) -> Tuple:
    """
    Split the loaded dataframe into features and labels if a label column exists.

    The column named 'label' (case-insensitive) is treated as the target vector.
    Returns (features_df, labels_series_or_None).
    """
    label_col = next((c for c in df.columns if c.lower() == "label"), None)
    if label_col is None:
        print("Warning: no label column found; writing all data as features.", file=sys.stderr)
        return df, None

    labels = df[label_col]
    features = df.drop(columns=[label_col])
    return features, labels


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preprocess gzipped FCS data into CSV for GateMeClass."
    )
    parser.add_argument(
        "--data.raw",
        type=str,
        required=True,
        help="Gz-compressed FCS data file.",
    )
    parser.add_argument(
        "--data.labels",
        type=str,
        default=None,
        help="Gz-compressed labels file for marker names (optional).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to write the resulting CSV files.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="dataset",
        help="Dataset name used for output filename.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducible train/test split.",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.5,
        help="Proportion of samples to use for training (default: 0.5).",
    )
    parser.add_argument(
        "--min_train_samples",
        type=int,
        default=2,
        help="Minimum number of samples in training set (default: 2).",
    )

    return parser


def main(argv: Iterable[str] = None):
    parser = parse_args()
    args = parser.parse_args(argv)

    raw_path = getattr(args, "data.raw")
    label_path = getattr(args, "data.labels")
    output_dir = args.output_dir
    name = args.name
    seed = args.seed
    train_ratio = args.train_ratio
    min_train_samples = args.min_train_samples

    # Parse FCS file
    print(f"Loading FCS file: {raw_path}", file=sys.stderr)
    data_df = parse_fcs_to_dataframe(raw_path)
    print(f"Loaded {data_df.shape[0]:,} cells with {data_df.shape[1]} features", file=sys.stderr)
    
    # Apply marker name labels if provided (this renames the columns, not the cell type labels)
    if label_path:
        data_df = apply_labels(label_path, data_df)
    
    # Handle missing labels in the 'label' column
    data_df = replace_NAs(data_df)
    
    # Create label mapping BEFORE splitting (we need consistent IDs across train/test)
    label_mapping, numeric_labels = create_label_mapping(data_df)
    
    if label_mapping:
        # Replace the label column with numeric IDs
        data_df["label"] = numeric_labels
        print(f"\nCreated label mapping with {len(label_mapping)} cell types:", file=sys.stderr)
        for id, name in sorted(label_mapping.items()):
            print(f"  {id}: {name}", file=sys.stderr)
    
    # Get unique samples and perform split
    samples_unique = get_unique_samples(data_df)
    train_set, test_set = train_test_sample_split(
        data_df, 
        samples_unique, 
        seed=seed,
        train_ratio=train_ratio,
        min_train_samples=min_train_samples
    )

    # Split features and labels
    features_train, labels_train = split_features_and_labels(train_set)
    features_test, labels_test = split_features_and_labels(test_set)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save training set
    train_matrix_path = os.path.join(output_dir, f"{name}.train.matrix.gz")
    features_train.to_csv(train_matrix_path, index=False, compression="gzip")
    print(f"Saved training features: {train_matrix_path}", file=sys.stderr)
    
    if labels_train is not None:
        train_labels_path = os.path.join(output_dir, f"{name}.train.labels.gz")
        labels_train.to_csv(
            train_labels_path,
            index=False,
            header=False,
            compression="gzip",
        )
        print(f"Saved training labels: {train_labels_path}", file=sys.stderr)

    # Save test set
    test_matrix_path = os.path.join(output_dir, f"{name}.test.matrix.gz")
    features_test.to_csv(test_matrix_path, index=False, compression="gzip")
    print(f"Saved test features: {test_matrix_path}", file=sys.stderr)
    
    if labels_test is not None:
        test_labels_path = os.path.join(output_dir, f"{name}.test.labels.gz")
        labels_test.to_csv(
            test_labels_path,
            index=False,
            header=False,
            compression="gzip",
        )
        print(f"Saved test labels: {test_labels_path}", file=sys.stderr)
    
    # Save label mapping for GateMeClass to use
    if label_mapping:
        mapping_path = os.path.join(output_dir, f"{name}.label_mapping.gz")
        mapping_df = pd.DataFrame(list(label_mapping.items()), columns=["id", "cell_type"])
        mapping_df.to_csv(mapping_path, index=False, header=False, compression="gzip")
        print(f"Saved label mapping: {mapping_path}", file=sys.stderr)

    print("\nProcessing complete!", file=sys.stderr)


if __name__ == "__main__":
    main()
    
