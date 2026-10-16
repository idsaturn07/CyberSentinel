import json

from pyspark import StorageLevel
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler

from common import (
    RESULTS_DIR,
    MODELS_DIR,
    HDFS_PROCESSED,
    HDFS_MODEL,
    create_spark,
)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    feature_file = RESULTS_DIR / "feature_columns.json"

    with open(feature_file, "r", encoding="utf-8") as f:
        feature_columns = json.load(f)

    spark = create_spark("CyberSentinel-Model-Evaluation")

    try:
        local_model_path = MODELS_DIR / "random_forest_model"

        if local_model_path.exists():
            print("Loading model from local models folder...")
            model = RandomForestClassificationModel.load(
                local_model_path.as_uri()
            )
        else:
            print("Loading model from HDFS...")
            model = RandomForestClassificationModel.load(HDFS_MODEL)

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

        print("Generating test predictions...")
        predictions = model.transform(test).persist(
            StorageLevel.DISK_ONLY
        )

        evaluator = MulticlassClassificationEvaluator(
            labelCol="label",
            predictionCol="prediction"
        )

        accuracy = evaluator.setMetricName(
            "accuracy"
        ).evaluate(predictions)

        precision = evaluator.setMetricName(
            "weightedPrecision"
        ).evaluate(predictions)

        recall = evaluator.setMetricName(
            "weightedRecall"
        ).evaluate(predictions)

        f1 = evaluator.setMetricName(
            "f1"
        ).evaluate(predictions)

        counts = {
            (0, 0): 0,
            (0, 1): 0,
            (1, 0): 0,
            (1, 1): 0
        }

        for row in (
            predictions
            .groupBy("label", "prediction")
            .count()
            .collect()
        ):
            key = (int(row["label"]), int(row["prediction"]))
            counts[key] = int(row["count"])

        tn = counts[(0, 0)]
        fp = counts[(0, 1)]
        fn = counts[(1, 0)]
        tp = counts[(1, 1)]

        attack_precision = tp / (tp + fp) if tp + fp else 0.0
        attack_recall = tp / (tp + fn) if tp + fn else 0.0

        attack_f1 = (
            2 * attack_precision * attack_recall
            / (attack_precision + attack_recall)
            if attack_precision + attack_recall
            else 0.0
        )

        false_negative_rate = (
            fn / (fn + tp)
            if fn + tp
            else 0.0
        )

        # Calculate training accuracy for comparison.
        print("Checking training accuracy...")
        train_predictions = model.transform(train)

        training_accuracy = MulticlassClassificationEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="accuracy"
        ).evaluate(train_predictions)

        # Preserve the previously recorded CV result, if available.
        metrics_path = RESULTS_DIR / "project_metrics.json"

        if metrics_path.exists():
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
        else:
            metrics = {}

        metrics.update({
            "training_accuracy": float(training_accuracy),
            "test_accuracy": float(accuracy),
            "weighted_precision": float(precision),
            "weighted_recall": float(recall),
            "weighted_f1": float(f1),
            "attack_precision": float(attack_precision),
            "attack_recall": float(attack_recall),
            "attack_f1": float(attack_f1),
            "false_negative_rate": float(false_negative_rate),
            "confusion_matrix": {
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "TP": tp
            }
        })

        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        print("\nEvaluation completed.")
        print(f"Training accuracy: {training_accuracy:.4%}")
        print(f"Test accuracy: {accuracy:.4%}")
        print(f"Weighted precision: {precision:.4%}")
        print(f"Weighted recall: {recall:.4%}")
        print(f"Weighted F1-score: {f1:.4%}")
        print(f"Attack precision: {attack_precision:.4%}")
        print(f"Attack recall: {attack_recall:.4%}")
        print(f"Attack F1-score: {attack_f1:.4%}")
        print(f"False-negative rate: {false_negative_rate:.4%}")
        print("Confusion matrix counts:", metrics["confusion_matrix"])
        print("Saved metrics:", metrics_path)

        predictions.unpersist()

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
