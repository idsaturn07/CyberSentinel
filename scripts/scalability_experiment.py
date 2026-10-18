import csv
import json
import statistics
import time

from pyspark import StorageLevel
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.feature import VectorAssembler

from common import (
    RESULTS_DIR,
    MODELS_DIR,
    HDFS_PROCESSED,
    HDFS_MODEL,
    create_spark,
)

# Number of timed runs per batch size.
REPEATS = 3


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    feature_file = RESULTS_DIR / "feature_columns.json"

    with open(feature_file, "r", encoding="utf-8") as f:
        feature_columns = json.load(f)

    spark = create_spark("CyberSentinel-Scalability")

    try:
        local_model_path = MODELS_DIR / "random_forest_model"

        if local_model_path.exists():
            print("Loading the local Random Forest model...")
            model = RandomForestClassificationModel.load(
                local_model_path.as_uri()
            )
        else:
            print("Loading the model from HDFS...")
            model = RandomForestClassificationModel.load(HDFS_MODEL)

        print("Loading the cleaned dataset from HDFS...")

        cleaned_df = spark.read.parquet(HDFS_PROCESSED)

        assembler = VectorAssembler(
            inputCols=feature_columns,
            outputCol="features"
        )

        data = assembler.transform(cleaned_df).select(
            "features", "label"
        )

        # Use the same fixed maximum dataset for all measurements.
        benchmark_data = data.limit(1_000_000).persist(
            StorageLevel.DISK_ONLY
        )

        available_records = benchmark_data.count()

        if available_records == 0:
            raise ValueError("No records are available for benchmarking.")

        print("Benchmark records available:", f"{available_records:,}")

        # Warm up Spark and the prediction pipeline before measurement.
        print("Warming up the prediction pipeline...")

        (
            model.transform(benchmark_data.limit(10_000))
            .groupBy("prediction")
            .count()
            .collect()
        )

        sizes = [
            size for size in [100_000, 250_000, 500_000, 1_000_000]
            if size <= available_records
        ]

        # Each size is measured REPEATS times. Sizes are run in rotation
        # (100k, 250k, 500k, 1M, 100k, ...) so that any slow-down or
        # speed-up of the VM over time affects every size equally.
        runs = []

        for repeat in range(1, REPEATS + 1):
            for size in sizes:
                print(f"\nRun {repeat}/{REPEATS}: benchmarking {size:,} records...",
                      flush=True)

                start_time = time.perf_counter()

                prediction_counts = (
                    model.transform(benchmark_data.limit(size))
                    .groupBy("prediction")
                    .count()
                    .collect()
                )

                elapsed = time.perf_counter() - start_time

                processed = sum(
                    int(row["count"]) for row in prediction_counts
                )

                run = {
                    "Run": repeat,
                    "Records": processed,
                    "Time_seconds": round(elapsed, 4),
                    "Records_per_second": round(processed / elapsed, 2)
                }

                runs.append(run)
                print(run, flush=True)

        # Every individual run.
        with open(
            RESULTS_DIR / "scalability_runs.csv",
            "w",
            newline="",
            encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["Run", "Records", "Time_seconds", "Records_per_second"]
            )
            writer.writeheader()
            writer.writerows(runs)

        # Mean and sample standard deviation per size.
        measurements = []

        for size in sizes:
            times = [r["Time_seconds"] for r in runs if r["Records"] == size]
            rates = [r["Records_per_second"] for r in runs if r["Records"] == size]

            measurements.append({
                "Records": size,
                "Runs": len(times),
                "Time_seconds": round(statistics.mean(times), 2),
                "Time_std": round(statistics.stdev(times), 2) if len(times) > 1 else 0.0,
                "Records_per_second": round(statistics.mean(rates), 2),
                "Throughput_std": round(statistics.stdev(rates), 2) if len(rates) > 1 else 0.0
            })

        output_path = RESULTS_DIR / "scalability_results.csv"

        with open(
            output_path,
            "w",
            newline="",
            encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "Records",
                    "Runs",
                    "Time_seconds",
                    "Time_std",
                    "Records_per_second",
                    "Throughput_std"
                ]
            )

            writer.writeheader()
            writer.writerows(measurements)

        print("\nMean (± standard deviation) over", REPEATS, "runs:")
        for m in measurements:
            print(
                f"  {m['Records']:>9,} records: "
                f"{m['Time_seconds']:.2f} ± {m['Time_std']:.2f} s, "
                f"{m['Records_per_second']:,.0f} ± {m['Throughput_std']:,.0f} records/s"
            )

        print("\nScalability experiment completed.")
        print("Results saved to:", output_path)

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
