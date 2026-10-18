import json

from common import (
    BASE_DIR,
    RESULTS_DIR,
    HDFS_PROCESSED,
    create_spark,
    load_and_clean_local_dataset,
)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    spark = create_spark("CyberSentinel-Data-Preprocessing")

    try:
        print("Loading and cleaning CSE-CIC-IDS2018 files...")

        cleaned_df, feature_columns = load_and_clean_local_dataset(
            spark
        )

        print("Numeric feature columns:", len(feature_columns))

        # Store the cleaned dataset in HDFS as Parquet.
        print("Writing cleaned dataset to HDFS...")
        (
            cleaned_df.write
            .mode("overwrite")
            .parquet(HDFS_PROCESSED)
        )

        # Preserve the exact input-feature order.
        feature_file = RESULTS_DIR / "feature_columns.json"

        with open(feature_file, "w", encoding="utf-8") as f:
            json.dump(feature_columns, f, indent=2)

        # Count the processed rows for the run summary.
        row_count = cleaned_df.count()

        print("\nPreprocessing completed successfully.")
        print("Cleaned records:", f"{row_count:,}")
        print("Feature columns:", len(feature_columns))
        print("HDFS output:", HDFS_PROCESSED)
        print("Feature configuration:", feature_file)

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
