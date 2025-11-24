#!/usr/bin/env python

"""
CLI utility to convert gzipped FCS data into CSV with optional column relabeling.

Args:
    --data.raw      Path to a gz-compressed FCS file.
    --data.labels   Path to a gz-compressed labels file.
    --output_dir    Directory where the CSV will be written.
    --name          Dataset name used for the output filename.
"""

import argparse
import gzip
import os
import sys
import fcsparser
import tempfile
from pathlib import Path
from typing import Iterable, List


def read_gzip_bytes(path: str) -> bytes:
    """Return the raw bytes stored inside a gzip-compressed file."""
    with gzip.open(path, "rb") as fh:
        return fh.read()


def parse_fcs_to_dataframe(raw_gz_path: str):
    data_bytes = read_gzip_bytes(raw_gz_path)

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
        label_text = read_gzip_bytes(label_gz_path).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Unexpected label file format: unable to decode as UTF-8 text.") from exc

    if not label_text.strip():
        raise ValueError("Unexpected label file format: file is empty after decompression.")

    label_format = detect_label_format(label_gz_path, label_text)

    if label_format == "xml":
        raise NotImplementedError("XML label handling not implemented.")
    if label_format != "txt":
        raise ValueError("Unexpected label file format.")

    labels = parse_label_lines(label_text, expected_count=df.shape[1], source=label_gz_path)
    df.columns = labels
    return df


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess gzipped FCS data into CSV.")
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
        help="Directory to write the resulting CSV file.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="dataset",
        help="Dataset name used for output filename.",
    )
    return parser


def main(argv: Iterable[str] = None):
    parser = parse_args()
    args = parser.parse_args(argv)

    raw_path = getattr(args, "data.raw")
    label_path = getattr(args, "data.labels")
    output_dir = args.output_dir
    name = args.name

    data_df = parse_fcs_to_dataframe(raw_path)
    data_df = apply_labels(label_path, data_df)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{name}.csv")
    data_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
    
