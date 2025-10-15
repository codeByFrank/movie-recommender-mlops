# src/dataops/split_csv.py
# Purpose: split ratings_sample.csv into small shuffled batches to simulate arrivals.

import os
import math
import argparse
import pandas as pd
from datetime import datetime

def main(input_csv, output_dir, batch_size, shuffle):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(input_csv)
    if shuffle:
        df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    total = len(df)
    num_batches = math.ceil(total / batch_size)

    print(f"Total rows: {total}, batch_size: {batch_size}, batches: {num_batches}")

    for i in range(num_batches):
        start = i * batch_size
        end = min(start + batch_size, total)
        batch_df = df.iloc[start:end].copy()

        # Timestamped filename so each is unique
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        out_path = os.path.join(output_dir, f"ratings_batch_{i+1:03d}_{ts}.csv")

        batch_df.to_csv(out_path, index=False)
        print(f"Wrote: {out_path} ({len(batch_df)} rows)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split a CSV into small batches.")
    parser.add_argument("--input_csv", default="data/sample/ratings_sample.csv",
                        help="Path to the input ratings CSV.")
    parser.add_argument("--output_dir", default="data/landing",
                        help="Directory where batch files will be written.")
    parser.add_argument("--batch_size", type=int, default=1000,
                        help="Number of rows per batch file.")
    parser.add_argument("--shuffle", action="store_true",
                        help="Shuffle rows before splitting.")
    args = parser.parse_args()

    main(args.input_csv, args.output_dir, args.batch_size, args.shuffle)
