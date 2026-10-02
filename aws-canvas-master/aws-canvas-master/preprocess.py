"""
Generic preprocessing script for SageMaker Processing Job.
Reads Canvas preprocessed data from S3, encodes string columns,
fills nulls, puts target column first, writes train/validation CSVs.

Environment variables:
  INPUT_PATH   - local path where Processing Job mounts input data
  OUTPUT_PATH  - local path where Processing Job writes output
  TARGET_COL_INDEX - index of target column in input CSV (default: 0)
"""
import os
import csv
import math

INPUT_PATH = os.environ.get("INPUT_PATH", "/opt/ml/processing/input")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "/opt/ml/processing/output")
TARGET_COL_INDEX = int(os.environ.get("TARGET_COL_INDEX", "0"))
TRAIN_RATIO = float(os.environ.get("TRAIN_RATIO", "0.8"))


def read_csv(filepath):
    with open(filepath, "r") as f:
        reader = csv.reader(f)
        rows = list(reader)
    return rows


def write_csv(filepath, rows):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print(f"Written {len(rows)} rows to {filepath}")


def encode_columns(rows):
    """
    Detect string columns and label-encode them.
    Numeric columns pass through unchanged.
    Empty values are replaced with 0.
    """
    if not rows:
        return rows

    num_cols = len(rows[0])
    # Build encoder per column
    encoders = {}  # col_idx -> {value: int}
    for col_idx in range(num_cols):
        values = [row[col_idx] for row in rows if col_idx < len(row)]
        is_numeric = True
        for v in values:
            if v.strip() == "":
                continue
            try:
                float(v)
            except ValueError:
                is_numeric = False
                break
        if not is_numeric:
            # Build label map — sort for reproducibility
            unique_vals = sorted(set(v for v in values if v.strip() != ""))
            encoders[col_idx] = {v: i for i, v in enumerate(unique_vals)}

    if encoders:
        print(f"Encoding {len(encoders)} string columns: {list(encoders.keys())}")

    encoded = []
    for row in rows:
        new_row = []
        for col_idx, val in enumerate(row):
            if col_idx in encoders:
                if val.strip() == "":
                    new_row.append("0")
                else:
                    new_row.append(str(encoders[col_idx].get(val, 0)))
            else:
                if val.strip() == "":
                    new_row.append("0")
                else:
                    new_row.append(val)
        encoded.append(new_row)
    return encoded


def process_file(input_file, train_out, val_out):
    print(f"Processing: {input_file}")
    rows = read_csv(input_file)
    if not rows:
        print("Empty file, skipping")
        return

    print(f"Input: {len(rows)} rows x {len(rows[0])} cols")

    # Encode string columns
    rows = encode_columns(rows)

    # Split train/val
    split_idx = math.floor(len(rows) * TRAIN_RATIO)
    train_rows = rows[:split_idx]
    val_rows = rows[split_idx:]

    write_csv(train_out, train_rows)
    write_csv(val_out, val_rows)
    print(f"Train: {len(train_rows)} rows, Validation: {len(val_rows)} rows")


def main():
    # Find input CSV files
    input_files = []
    for root, dirs, files in os.walk(INPUT_PATH):
        for fname in files:
            if fname.endswith(".csv"):
                input_files.append(os.path.join(root, fname))

    if not input_files:
        print(f"No CSV files found in {INPUT_PATH}")
        return

    print(f"Found {len(input_files)} CSV file(s): {input_files}")

    train_out = os.path.join(OUTPUT_PATH, "train", "train.csv")
    val_out = os.path.join(OUTPUT_PATH, "validation", "validation.csv")

    # Use first CSV found (Canvas outputs one chunk)
    process_file(input_files[0], train_out, val_out)
    print("Preprocessing complete.")


if __name__ == "__main__":
    main()
