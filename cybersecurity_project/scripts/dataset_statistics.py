"""
Reproduce the dataset record counts reported in the project report:

    - rows read from the nine selected CSV files
    - repeated header rows removed
    - cleaned working records
    - Benign / Attack class counts
    - 80/20 training and test partition sizes (seed 42)

Run after data_preprocessing.py. Output: results/dataset_statistics.json
"""

import json

from pyspark.sql import functions as F

from common import (
    DATA_DIR,
    GOOD_FILES,
    RESULTS_DIR,
    HDFS_PROCESSED,
    create_spark,
)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    spark = create_spark("CyberSentinel-Dataset-Statistics")

    try:
        local_files = [DATA_DIR / name for name in GOOD_FILES]

        missing = [str(p) for p in local_files if not p.is_file()]
        if missing:
            raise FileNotFoundError(
                "Required dataset files are missing:\n" + "\n".join(missing)
            )

        # Raw counts: read the same files the same way as preprocessing.
        raw = (
            spark.read
            .option("header", "true")
            .option("inferSchema", "true")
            .option("mode", "PERMISSIVE")
            .csv([p.as_uri() for p in local_files])
        )

        raw_rows = raw.count()
        header_rows = raw.filter(
            F.trim(F.col("Label")) == F.lit("Label")
        ).count()

        # Cleaned counts: read the processed Parquet stored in HDFS.
        cleaned = spark.read.parquet(HDFS_PROCESSED)
        cleaned_rows = cleaned.count()

        class_counts = {
            int(row["label"]): int(row["count"])
            for row in cleaned.groupBy("label").count().collect()
        }
        benign = class_counts.get(0, 0)
        attack = class_counts.get(1, 0)

        # Same split call as train_model.py / evaluate_model.py.
        train, test = cleaned.randomSplit([0.8, 0.2], seed=42)
        train_rows = train.count()
        test_rows = test.count()

        stats = {
            "selected_files": len(GOOD_FILES),
            "raw_rows_before_header_filter": raw_rows,
            "repeated_header_rows_removed": header_rows,
            "cleaned_rows": cleaned_rows,
            "benign_rows": benign,
            "attack_rows": attack,
            "benign_percent": round(100 * benign / cleaned_rows, 2),
            "attack_percent": round(100 * attack / cleaned_rows, 2),
            "training_rows": train_rows,
            "test_rows": test_rows,
        }

        output_path = RESULTS_DIR / "dataset_statistics.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

        print("\nDataset statistics")
        for key, value in stats.items():
            shown = f"{value:,}" if isinstance(value, int) else value
            print(f"  {key}: {shown}")
        print("Saved:", output_path)

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
