from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import NumericType


# Project directories
BASE_DIR = Path.home() / "cybersecurity_project"
DATA_DIR = BASE_DIR / "data" / "CSE_CIC_IDS2018"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"
PLOTS_DIR = BASE_DIR / "plots"

# Hadoop/HDFS configuration
HDFS_BASE = "hdfs://localhost:9000/cybersecurity/CSE_CIC_IDS2018"
HDFS_PROCESSED = f"{HDFS_BASE}/processed/cleaned_flows"
HDFS_MODEL = f"{HDFS_BASE}/results/random_forest_model"

# Use the same nine schema-consistent files as the experiment.
# The large Tuesday file is excluded because its header/schema
# did not match the other processed CSV files.
GOOD_FILES = [
    "Friday-02-03-2018_TrafficForML_CICFlowMeter.csv",
    "Friday-16-02-2018_TrafficForML_CICFlowMeter.csv",
    "Friday-23-02-2018_TrafficForML_CICFlowMeter.csv",
    "Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv",
    "Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv",
    "Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv",
    "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv",
    "Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv",
    "Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv",
]


def create_spark(app_name):
    """Create a local Spark session configured to access HDFS."""
    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        # The default 1 GB driver heap is too small for model training here.
        .config("spark.driver.memory", "3g")
        .config(
            "spark.hadoop.fs.defaultFS",
            "hdfs://localhost:9000"
        )
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")
    return spark


def quote_column(name):
    """Safely reference feature names containing spaces or symbols."""
    return F.col("`" + name.replace("`", "``") + "`")


def load_and_clean_local_dataset(spark, keep_attack_type=False):
    """
    Load the nine schema-consistent CSV files, remove repeated headers,
    convert labels to binary classes, and clean invalid numeric values.

    If keep_attack_type is True, the original label text (for example
    "DoS attacks-Hulk") is also kept in an 'attack_type' column. It is used
    only for analysis, never as a model input.

    Returns:
        cleaned_df: numeric features plus the binary 'label' column
        feature_columns: feature names in their original column order
    """
    local_files = [DATA_DIR / name for name in GOOD_FILES]

    missing = [str(path) for path in local_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Required dataset files are missing:\n" + "\n".join(missing)
        )

    raw = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .option("mode", "PERMISSIVE")
        # file:// is required: fs.defaultFS points at HDFS, so a bare
        # path would otherwise be looked up inside HDFS, not on disk.
        .csv([path.as_uri() for path in local_files])
    )

    if "Label" not in raw.columns:
        raise ValueError(
            "The CSV files must contain a 'Label' column."
        )

    # Remove the repeated header rows found inside some CSV files.
    raw = raw.filter(
        F.trim(F.col("Label")) != F.lit("Label")
    )

    # Keep numeric feature columns in the dataset's original order.
    feature_columns = [
        field.name
        for field in raw.schema.fields
        if isinstance(field.dataType, NumericType)
        and field.name not in ("Label", "label")
    ]

    if not feature_columns:
        raise ValueError(
            "No numeric feature columns were found."
        )

    # Keep the original label text before 'label' is created: Spark column
    # names are case-insensitive, so 'label' replaces the text 'Label' column.
    if keep_attack_type:
        raw = raw.withColumn("attack_type", F.trim(F.col("Label")))

    # Benign = 0; all other attack labels = 1.
    cleaned = raw.withColumn(
        "label",
        F.when(
            F.trim(F.col("Label")) == F.lit("Benign"),
            F.lit(0.0)
        ).otherwise(F.lit(1.0))
    )

    # Replace missing, NaN, and infinite numeric values with zero.
    # Cast features to double so VectorAssembler receives numeric values.
    for name in feature_columns:
        value = quote_column(name).cast("double")

        invalid = (
            value.isNull()
            | F.isnan(value)
            | (value == float("inf"))
            | (value == float("-inf"))
        )

        cleaned = cleaned.withColumn(
            name,
            F.when(invalid, F.lit(0.0)).otherwise(value)
        )

    # The model uses only numeric features and the binary target.
    extra = [F.col("attack_type")] if keep_attack_type else []

    cleaned = cleaned.select(
        *[quote_column(name) for name in feature_columns],
        F.col("label"),
        *extra
    )

    return cleaned, feature_columns
