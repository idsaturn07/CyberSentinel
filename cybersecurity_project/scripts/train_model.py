import json

import pandas as pd
from pyspark.ml.classification import RandomForestClassifier
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
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    feature_file = RESULTS_DIR / "feature_columns.json"

    if not feature_file.exists():
        raise FileNotFoundError(
            f"Feature configuration not found: {feature_file}. "
            "Run data_preprocessing.py first."
        )

    with open(feature_file, "r", encoding="utf-8") as f:
        feature_columns = json.load(f)

    spark = create_spark("CyberSentinel-RandomForest-Training")

    try:
        print("Loading cleaned dataset from HDFS...")
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

        print("Training Random Forest...")
        print("Trees: 20 | Maximum depth: 10")

        rf = RandomForestClassifier(
            featuresCol="features",
            labelCol="label",
            numTrees=20,
            maxDepth=10,
            seed=42
        )

        model = rf.fit(train)

        # Save the model locally for the GitHub project.
        local_model_path = MODELS_DIR / "random_forest_model"

        model.write().overwrite().save(local_model_path.as_uri())

        # Save a copy in HDFS.
        model.write().overwrite().save(HDFS_MODEL)

        # Save feature importances.
        importance_df = pd.DataFrame({
            "Feature": feature_columns,
            "Importance": model.featureImportances.toArray()
        }).sort_values(
            "Importance",
            ascending=False
        )

        importance_df.to_csv(
            RESULTS_DIR / "feature_importance.csv",
            index=False
        )

        # Record the configuration used for this training run.
        summary = {
            "algorithm": "Random Forest",
            "features": len(feature_columns),
            "num_trees": 20,
            "max_depth": 10,
            "random_seed": 42,
            "split": {
                "training_fraction": 0.8,
                "test_fraction": 0.2
            },
            "local_model_path": local_model_path.as_uri(),
            "hdfs_model_path": HDFS_MODEL
        }

        with open(
            RESULTS_DIR / "training_summary.json",
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(summary, f, indent=2)

        print("\nTraining and model saving completed.")
        print("Local model:", local_model_path)
        print("HDFS model:", HDFS_MODEL)
        print("Feature importances saved.")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
