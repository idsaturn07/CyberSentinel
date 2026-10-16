import json

from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

from common import (
    RESULTS_DIR,
    HDFS_PROCESSED,
    create_spark,
)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    feature_file = RESULTS_DIR / "feature_columns.json"

    with open(feature_file, "r", encoding="utf-8") as f:
        feature_columns = json.load(f)

    spark = create_spark("CyberSentinel-CrossValidation")

    try:
        print("Loading cleaned data from HDFS...")

        cleaned_df = spark.read.parquet(HDFS_PROCESSED)

        assembler = VectorAssembler(
            inputCols=feature_columns,
            outputCol="features"
        )

        data = assembler.transform(cleaned_df).select(
            "features", "label"
        )

        train, test = data.randomSplit(
            [0.8, 0.2],
            seed=42
        )

        training_count = train.count()

        if training_count == 0:
            raise ValueError("The training split contains no records.")

        # Match the approximately 500,000-record CV sample
        # used in the original experiment.
        fraction = min(1.0, 500000.0 / training_count)

        cv_data = train.sample(
            withReplacement=False,
            fraction=fraction,
            seed=42
        )

        sample_count = cv_data.count()

        print("Cross-validation sample:", f"{sample_count:,}")
        print("Number of folds: 3")

        rf = RandomForestClassifier(
            featuresCol="features",
            labelCol="label",
            numTrees=20,
            maxDepth=10,
            seed=42
        )

        evaluator = MulticlassClassificationEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="accuracy"
        )

        # An empty parameter grid evaluates the configured model
        # without searching additional hyperparameter combinations.
        param_grid = ParamGridBuilder().build()

        cv = CrossValidator(
            estimator=rf,
            estimatorParamMaps=param_grid,
            evaluator=evaluator,
            numFolds=3,
            seed=42,
            parallelism=1
        )

        print("Running 3-fold cross-validation...")

        cv_model = cv.fit(cv_data)

        fold_accuracies = [
            float(score) for score in cv_model.avgMetrics
        ]

        result = {
            "algorithm": "Random Forest",
            "folds": 3,
            "sample_records": sample_count,
            "num_trees": 20,
            "max_depth": 10,
            "seed": 42,
            "fold_accuracies": fold_accuracies,
            "mean_cv_accuracy": (
                sum(fold_accuracies) / len(fold_accuracies)
            )
        }

        output_path = RESULTS_DIR / "cross_validation_results.json"

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        # Update the existing project metrics without discarding
        # the other saved evaluation results.
        metrics_path = RESULTS_DIR / "project_metrics.json"

        metrics = {}

        if metrics_path.exists():
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)

        metrics["cross_validation_accuracy"] = (
            result["mean_cv_accuracy"]
        )

        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        print("\nCross-validation completed.")
        print(
            "Mean CV accuracy:",
            f"{result['mean_cv_accuracy']:.4%}"
        )
        print("Saved:", output_path)

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
