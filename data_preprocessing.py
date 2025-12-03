#!/usr/bin/env python

"""
CLI utility to convert gzipped FCS data into gzipped CSV outputs with optional column relabeling.

NOW WITH TRAIN/TEST SPLIT (70/30 stratified)

Args:
    --data.raw      Path to a gz-compressed FCS file.
    --data.labels   Path to a gz-compressed labels file.
    --output_dir    Directory where the matrix/label CSV files will be written.
    --name          Dataset name used for the output filenames.
    --train_frac    Fraction of data to use for training (default: 0.7)
    --random_seed   Random seed for reproducible splits (default: 42)
"""

import argparse
import gzip
import os
import sys
import fcsparser
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Iterable, List, Tuple
from sklearn.model_selection import train_test_split


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
    data_bytes = read_bytes_handling_gzip(raw_gz_path)

    # fcsparser.parse expects a file path; use a temporary file to avoid keeping data on disk.
    with tempfile.NamedTemporaryFile(suffix=".fcs", delete=False) as tmp:
        tmp.write(data_bytes)
        tmp_path = tmp.name

    try:
        _, data = fcsparser.parse(tmp_path, reformat_meta=True)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass  # If cleanup fails, we still want to return the parsed data/error.

    return data


def parse_label_lines(label_text: str, expected_count: int, source: str) -> List[str]:
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
    """Apply labels to DataFrame columns according to the provided rules."""
    try:
        label_text = read_bytes_handling_gzip(label_gz_path).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Unexpected label file format: unable to decode as UTF-8 text.") from exc

    if not label_text.strip():
        raise ValueError("Unexpected label file format: file is empty after decompression.")

    label_format = detect_label_format(label_gz_path, label_text)

    if label_format == "xml":
        raise NotImplementedError("XML label handling not implemented.")
    if label_format != "txt":
        raise ValueError("Unexpected label file format.")

    try:
        labels = parse_label_lines(label_text, expected_count=df.shape[1], source=label_gz_path)
    except ValueError as exc:
        print(
            f"Warning: {exc} Column relabeling skipped; keeping original headers.",
            file=sys.stderr,
        )
        return df
    df.columns = labels
    return df


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


def create_train_test_split(features_df, labels, train_frac=0.7, random_seed=42):
    """
    Create stratified train/test split.
    
    Returns:
        train_features, test_features, train_labels, test_labels
    """
    if labels is None:
        print("Warning: No labels found, cannot create stratified split.", file=sys.stderr)
        return features_df, None, labels, None
    
    # Remove unassigned cells for stratification
    assigned_mask = labels != "unassigned"
    features_assigned = features_df[assigned_mask]
    labels_assigned = labels[assigned_mask]
    
    print(f"\n{'='*70}", file=sys.stderr)
    print("TRAIN/TEST SPLIT", file=sys.stderr)
    print(f"{'='*70}", file=sys.stderr)
    print(f"Total cells: {len(features_df)}", file=sys.stderr)
    print(f"Assigned cells: {len(features_assigned)}", file=sys.stderr)
    print(f"Unassigned cells: {sum(~assigned_mask)}", file=sys.stderr)
    print(f"Unique cell types: {len(labels_assigned.unique())}", file=sys.stderr)
    print(f"\nCell type distribution:", file=sys.stderr)
    print(labels_assigned.value_counts().to_string(), file=sys.stderr)
    
    # Stratified split
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            features_assigned,
            labels_assigned,
            test_size=(1 - train_frac),
            random_state=random_seed,
            stratify=labels_assigned
        )
        
        print(f"\n✓ Split created successfully:", file=sys.stderr)
        print(f"  Training: {len(X_train)} cells ({100*train_frac:.1f}%)", file=sys.stderr)
        print(f"  Test: {len(X_test)} cells ({100*(1-train_frac):.1f}%)", file=sys.stderr)
        print(f"{'='*70}\n", file=sys.stderr)
        
        return X_train, X_test, y_train, y_test
        
    except ValueError as e:
        print(f"Warning: Stratified split failed ({e}). Using random split without stratification.", file=sys.stderr)
        X_train, X_test, y_train, y_test = train_test_split(
            features_assigned,
            labels_assigned,
            test_size=(1 - train_frac),
            random_state=random_seed,
            stratify=None
        )
        return X_train, X_test, y_train, y_test


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess gzipped FCS data into CSV with train/test split.")
    parser.add_argument(
        "--data.raw",
        type=str,
        required=True,
        help="Gz-compressed FCS data file.",
    )
    parser.add_argument(
        "--data.labels",
        type=str,
        required=True,
        help="Gz-compressed labels file. Text replaces FCS headers; XML is not supported.",
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
        "--train_frac",
        type=float,
        default=0.7,
        help="Fraction of data to use for training (default: 0.7 = 70%%).",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits (default: 42).",
    )
    return parser


def main(argv: Iterable[str] = None):
    parser = parse_args()
    args = parser.parse_args(argv)

    raw_path = getattr(args, "data.raw")
    label_path = getattr(args, "data.labels")
    output_dir = args.output_dir
    name = args.name
    train_frac = args.train_frac
    random_seed = args.random_seed

    # Parse FCS and apply labels
    data_df = parse_fcs_to_dataframe(raw_path)
    data_df = apply_labels(label_path, data_df)
    features_df, labels = split_features_and_labels(data_df)

    os.makedirs(output_dir, exist_ok=True)

    # Create train/test split if labels exist
    if labels is not None:
        X_train, X_test, y_train, y_test = create_train_test_split(
            features_df, labels, train_frac, random_seed
        )
        
        # Save TRAINING files
        train_matrix_path = os.path.join(output_dir, f"{name}_train.matrix.gz")
        X_train.to_csv(train_matrix_path, index=False, compression="gzip")
        print(f"✓ Saved training matrix: {train_matrix_path}", file=sys.stderr)
        
        train_labels_path = os.path.join(output_dir, f"{name}_train.true_labels.gz")
        y_train.to_csv(train_labels_path, index=False, header=False, compression="gzip")
        print(f"✓ Saved training labels: {train_labels_path}", file=sys.stderr)
        
        # Save TEST files
        test_matrix_path = os.path.join(output_dir, f"{name}_test.matrix.gz")
        X_test.to_csv(test_matrix_path, index=False, compression="gzip")
        print(f"✓ Saved test matrix: {test_matrix_path}", file=sys.stderr)
        
        test_labels_path = os.path.join(output_dir, f"{name}_test.true_labels.gz")
        y_test.to_csv(test_labels_path, index=False, header=False, compression="gzip")
        print(f"✓ Saved test labels: {test_labels_path}", file=sys.stderr)
    
    # Save FULL dataset (for backward compatibility or reference)
    full_matrix_path = os.path.join(output_dir, f"{name}.matrix.gz")
    features_df.to_csv(full_matrix_path, index=False, compression="gzip")
    print(f"✓ Saved full matrix: {full_matrix_path}", file=sys.stderr)
    
    if labels is not None:
        full_labels_path = os.path.join(output_dir, f"{name}.true_labels.gz")
        labels.to_csv(full_labels_path, index=False, header=False, compression="gzip")
        print(f"✓ Saved full labels: {full_labels_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
