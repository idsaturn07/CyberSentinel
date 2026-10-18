import os
import json
import tempfile
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path.home() / "cybersecurity_project"
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = BASE_DIR / "plots"

METRICS_FILE = RESULTS_DIR / "project_metrics.json"
FEATURES_FILE = RESULTS_DIR / "feature_columns.json"
IMPORTANCE_FILE = RESULTS_DIR / "feature_importance.csv"
SCALABILITY_FILE = RESULTS_DIR / "scalability_results.csv"

# Results of the extended experiments (optional: pages show a hint if missing)
UNSEEN_FILE = RESULTS_DIR / "unseen_attack_results.csv"
UNSEEN_SUMMARY_FILE = RESULTS_DIR / "unseen_attack_summary.json"
FAMILY_COUNTS_FILE = RESULTS_DIR / "attack_family_counts.csv"
AUDIT_FILE = RESULTS_DIR / "data_quality_audit.csv"
AUDIT_SUMMARY_FILE = RESULTS_DIR / "data_quality_summary.json"
INVALID_FILE = RESULTS_DIR / "invalid_values_by_column.csv"
LABELS_FILE = RESULTS_DIR / "labels_by_file.csv"

MODEL_HDFS_PATH = (
    "hdfs://localhost:9000/"
    "cybersecurity/CSE_CIC_IDS2018/results/random_forest_model"
)

st.set_page_config(
    page_title="CyberSentinel | Cybersecurity Analytics",
    page_icon="🛡️",
    layout="wide"
)


# ============================================================
# LOAD SAVED RESULTS
# ============================================================

if not METRICS_FILE.exists():
    st.error(f"Metrics file not found: {METRICS_FILE}")
    st.stop()

if not FEATURES_FILE.exists():
    st.error(f"Feature configuration not found: {FEATURES_FILE}")
    st.stop()

with open(METRICS_FILE, "r") as f:
    metrics = json.load(f)

with open(FEATURES_FILE, "r") as f:
    feature_columns = json.load(f)


# ============================================================
# SPARK MODEL LOADER
# The Spark JVM and model are started only when needed.
# ============================================================

@st.cache_resource(show_spinner=False)
def load_spark_model():
    from pyspark.sql import SparkSession
    from pyspark.ml.classification import RandomForestClassificationModel

    spark = (
        SparkSession.builder
        .appName("CyberSentinelDashboard")
        .master("local[*]")
        .config(
            "spark.hadoop.fs.defaultFS",
            "hdfs://localhost:9000"
        )
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")

    model = RandomForestClassificationModel.load(
        MODEL_HDFS_PATH
    )

    return spark, model


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("🛡️ CyberSentinel")

st.sidebar.caption(
    "Big Data-Based Cybersecurity Anomaly Detection"
)

page = st.sidebar.radio(
    "Navigate",
    [
        "Dashboard Overview",
        "Model Evaluation",
        "Performance Analysis",
        "Data Quality Audit",
        "Known vs Unseen Attacks",
        "CSV Anomaly Detection"
    ]
)

st.sidebar.divider()

st.sidebar.markdown("### Technology Stack")

st.sidebar.markdown(
    """
    - Hadoop HDFS
    - Apache Spark
    - PySpark MLlib
    - Random Forest
    - Streamlit
    """
)

st.sidebar.caption("Dataset: CSE-CIC-IDS2018")


# ============================================================
# DASHBOARD OVERVIEW
# ============================================================

if page == "Dashboard Overview":

    st.title("🛡️ CyberSentinel")
    st.subheader("Big Data Cybersecurity Analytics Platform")

    st.markdown(
        """
        CyberSentinel uses a Hadoop-based storage layer and
        Apache Spark machine-learning pipeline to analyze
        network traffic and classify records as benign or attack.
        """
    )

    st.info(
        "This dashboard presents the results of the trained "
        "Random Forest model. The CSV Anomaly Detection page "
        "can also classify compatible uploaded traffic CSV files."
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Test Accuracy",
        f"{metrics['test_accuracy']:.2%}"
    )

    col2.metric(
        "Attack Precision",
        f"{metrics['attack_precision']:.2%}"
    )

    col3.metric(
        "Attack Recall",
        f"{metrics['attack_recall']:.2%}"
    )

    col4.metric(
        "Attack F1-Score",
        f"{metrics['attack_f1']:.2%}"
    )

    st.subheader("Model Summary")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            **Machine Learning Model:** Random Forest

            **Processing Framework:** Apache Spark

            **Storage:** Hadoop Distributed File System (HDFS)

            **Prediction Classes:**
            - `0` — Benign traffic
            - `1` — Attack traffic
            """
        )

    with col2:
        st.metric(
            "Training Accuracy",
            f"{metrics['training_accuracy']:.2%}"
        )

        st.metric(
            "3-Fold CV Accuracy",
            f"{metrics['cross_validation_accuracy']:.2%}",
            help=(
                "Cross-validation was performed on a "
                "sample of approximately 500,000 training records."
            )
        )

        st.metric(
            "False Negative Rate",
            f"{metrics['false_negative_rate']:.2%}"
        )

    st.subheader("System Workflow")

    st.markdown(
        """
        **Network Traffic → HDFS → PySpark Preprocessing → 
        Random Forest → Benign / Attack Classification**
        """
    )

    st.caption(
        "The experiment used nine schema-consistent processed "
        "CSE-CIC-IDS2018 CSV files. One malformed large file "
        "was excluded from the modelling dataset."
    )


# ============================================================
# MODEL EVALUATION
# ============================================================

elif page == "Model Evaluation":

    st.title("📊 Model Evaluation")

    st.markdown(
        "Evaluation results calculated using the held-out test set."
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Accuracy", f"{metrics['test_accuracy']:.2%}")
    col2.metric("Precision", f"{metrics['weighted_precision']:.2%}")
    col3.metric("Recall", f"{metrics['weighted_recall']:.2%}")
    col4.metric("F1-Score", f"{metrics['weighted_f1']:.2%}")

    st.subheader("Confusion Matrix")

    cm_path = PLOTS_DIR / "confusion_matrix.png"

    if cm_path.exists():
        st.image(
            str(cm_path),
            caption="Random Forest confusion matrix on the test set",
            use_container_width=True
        )
    else:
        st.warning("Confusion matrix graph not found.")

    cm = metrics["confusion_matrix"]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("True Negatives", f"{cm['TN']:,}")
    col2.metric("False Positives", f"{cm['FP']:,}")
    col3.metric("False Negatives", f"{cm['FN']:,}")
    col4.metric("True Positives", f"{cm['TP']:,}")

    st.subheader("Attack-Specific Performance")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Attack Precision",
        f"{metrics['attack_precision']:.2%}"
    )

    col2.metric(
        "Attack Recall",
        f"{metrics['attack_recall']:.2%}"
    )

    col3.metric(
        "Attack F1-Score",
        f"{metrics['attack_f1']:.2%}"
    )

    st.warning(
        f"The model's false-negative rate is "
        f"{metrics['false_negative_rate']:.2%}. "
        "These are attacks incorrectly classified as benign."
    )


# ============================================================
# PERFORMANCE AND SCALABILITY
# ============================================================

elif page == "Performance Analysis":

    st.title("📈 Performance & Scalability Analysis")

    st.markdown(
        """
        These graphs show the controlled prediction benchmarks
        and feature importance calculated during experimentation.
        """
    )

    st.subheader("Processing Time vs Dataset Size")

    processing_path = (
        PLOTS_DIR / "scalability_processing_time.png"
    )

    if processing_path.exists():
        st.image(
            str(processing_path),
            use_container_width=True
        )
    else:
        st.warning("Processing-time graph not found.")

    st.subheader("Throughput vs Dataset Size")

    throughput_path = PLOTS_DIR / "scalability_throughput.png"

    if throughput_path.exists():
        st.image(
            str(throughput_path),
            use_container_width=True
        )
    else:
        st.warning("Throughput graph not found.")

    st.subheader("Feature Importance")

    importance_path = PLOTS_DIR / "feature_importance.png"

    if importance_path.exists():
        st.image(
            str(importance_path),
            use_container_width=True
        )
    else:
        st.warning("Feature-importance graph not found.")

    if IMPORTANCE_FILE.exists():

        importance_df = pd.read_csv(IMPORTANCE_FILE)

        st.markdown("### Top 15 Features")

        st.dataframe(
            importance_df.head(15),
            use_container_width=True,
            hide_index=True
        )

    if SCALABILITY_FILE.exists():

        scalability_df = pd.read_csv(SCALABILITY_FILE)

        st.markdown("### Benchmark Measurements")

        st.dataframe(
            scalability_df,
            use_container_width=True,
            hide_index=True
        )

    st.caption(
        "The scalability experiment measures warmed-up prediction "
        "and aggregation time in a single-machine Spark environment. "
        "It is not a distributed multi-node cluster benchmark."
    )


# ============================================================
# DATA QUALITY AUDIT
# ============================================================

elif page == "Data Quality Audit":

    st.title("🧹 Data Quality Audit")

    st.markdown(
        """
        Every one of the ten CSE-CIC-IDS2018 processed CSV files was
        audited before modelling: header schema, repeated header rows,
        missing / NaN / Infinity values and the traffic labels present.
        This makes each data-cleaning decision explicit and reproducible.
        """
    )

    if not AUDIT_FILE.exists():
        st.info(
            "Audit results not found. Run "
            "`python scripts/data_quality_audit.py` to create them."
        )
    else:
        audit = pd.read_csv(AUDIT_FILE)
        summary = {}
        if AUDIT_SUMMARY_FILE.exists():
            with open(AUDIT_SUMMARY_FILE, "r") as f:
                summary = json.load(f)

        included = audit[audit["decision"] == "Included"]
        excluded = audit[audit["decision"] == "Excluded"]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Files audited", f"{len(audit)}")
        col2.metric("Files used for modelling", f"{len(included)}")
        col3.metric(
            "Repeated header rows removed",
            f"{int(included['repeated_header_rows'].sum()):,}"
        )
        col4.metric(
            "NaN + Infinity values (used files)",
            f"{int(included['nan_values'].sum() + included['infinity_values'].sum()):,}"
        )

        st.subheader("Per-file audit")

        show = audit[[
            "file", "size_gb", "header_columns", "matches_shared_schema",
            "data_rows_read", "repeated_header_rows", "missing_values",
            "nan_values", "infinity_values", "decision"
        ]].rename(columns={
            "file": "File",
            "size_gb": "Size (GB)",
            "header_columns": "Columns",
            "matches_shared_schema": "Matches shared schema",
            "data_rows_read": "Rows read",
            "repeated_header_rows": "Repeated headers",
            "missing_values": "Missing",
            "nan_values": "NaN",
            "infinity_values": "Infinity",
            "decision": "Decision",
        })

        st.dataframe(show, use_container_width=True, hide_index=True)

        for _, row in excluded.iterrows():
            extra = row["extra_columns"] if isinstance(row["extra_columns"], str) else ""
            missing = row["missing_columns"] if isinstance(row["missing_columns"], str) else ""
            st.warning(
                f"**Excluded:** {row['file']} — {row['reason']}. "
                f"Columns: {row['header_columns']}. "
                + (f"Extra columns: {extra}. " if extra else "")
                + (f"Missing columns: {missing}." if missing else "")
            )

        if INVALID_FILE.exists():
            invalid = pd.read_csv(INVALID_FILE)
            invalid = invalid[invalid["total_invalid"] > 0]
            st.subheader("Features containing invalid values (used files)")
            if invalid.empty:
                st.success("No missing, NaN or Infinity values were found.")
            else:
                st.dataframe(invalid, use_container_width=True, hide_index=True)
                st.caption(
                    "These values are replaced with 0 during preprocessing "
                    "so that Spark can build feature vectors."
                )

        if LABELS_FILE.exists():
            labels = pd.read_csv(LABELS_FILE)
            st.subheader("Traffic labels in each file")
            table = (
                labels.pivot_table(
                    index="file", columns="label", values="count",
                    aggfunc="sum", fill_value=0
                )
            )
            st.dataframe(table, use_container_width=True)
            st.caption(
                "Each capture day contains different attack types, which is "
                "why a random train/test split mixes days."
            )


# ============================================================
# KNOWN VS UNSEEN ATTACKS
# ============================================================

elif page == "Known vs Unseen Attacks":

    st.title("🧪 Known vs Unseen Attack Families")

    st.markdown(
        """
        The headline accuracy measures attacks the model **was trained on**.
        This experiment asks a harder question: can the detector catch an
        attack family it has **never seen**?

        - **Seen:** Random Forest trained on all attack families.
        - **Unseen:** for each family, a Random Forest trained *without*
          that family is tested on it.
        - **Anomaly detector:** K-Means trained only on **benign** traffic;
          flows far from normal traffic are flagged. Its threshold was set
          on separate benign calibration data (1 % or 0.1 % false alarms).
        - **Hybrid:** flagged if the Random Forest **or** the anomaly
          detector flags it.
        """
    )

    if not UNSEEN_FILE.exists():
        st.info(
            "Results not found. Run "
            "`python scripts/unseen_attack_experiment.py` to create them."
        )
    else:
        res = pd.read_csv(UNSEEN_FILE)
        seen = res[res["kind"].str.startswith("seen")].set_index("evaluated_family")
        # Prefer the like-for-like rows (held-out family, same test records as
        # "seen"); older result files only have the pool + test rows.
        if (res["kind"] == "unseen (test split)").any():
            unseen = res[res["kind"] == "unseen (test split)"].set_index("evaluated_family")
        else:
            unseen = res[res["kind"].str.startswith("unseen")].set_index("evaluated_family")
        ref = res[res["model"] == "RF (all families)"].iloc[0]

        families = list(unseen.index)

        # Missed attacks of the model trained on all families
        if "detected_rf" in seen.columns:
            seen_missed = seen["records"] - seen["detected_rf"]
        else:
            seen_missed = (seen["records"] * (1 - seen["rf_recall"])).round()
        total_missed = seen_missed.sum()
        top_family = seen_missed.idxmax() if total_missed else "-"
        top_share = seen_missed.max() / total_missed if total_missed else 0.0

        extra_fa = (
            ref["hybrid_p99.9_benign_false_alarm_rate"]
            - ref["rf_benign_false_alarm_rate"]
        ) * 1e6

        col1, col2, col3, col4 = st.columns(4)
        col1.metric(
            "Avg. detection, seen families",
            f"{seen.loc[families, 'rf_recall'].mean():.1%}"
        )
        col2.metric(
            "Avg. detection, unseen families (RF)",
            f"{unseen['rf_recall'].mean():.1%}"
        )
        col3.metric(
            "Avg. unseen detection, hybrid (0.1 % FA)",
            f"{unseen['hybrid_p99.9_recall'].mean():.1%}"
        )
        col4.metric(
            "Extra false alarms / million benign (hybrid 0.1 %)",
            f"{extra_fa:,.0f}"
        )
        st.caption("Averages are unweighted means over the attack families.")

        st.subheader("Detection rate per attack family")

        table = pd.DataFrame({
            "Family": families,
            "Records": [int(unseen.loc[f, "records"]) for f in families],
            "RF – seen": [seen["rf_recall"].get(f, float("nan")) for f in families],
            "RF – unseen": [unseen.loc[f, "rf_recall"] for f in families],
            "Anomaly (1 % FA)": [unseen.loc[f, "ad_p99_recall"] for f in families],
            "Hybrid (1 % FA)": [unseen.loc[f, "hybrid_p99_recall"] for f in families],
            "Anomaly (0.1 % FA)": [unseen.loc[f, "ad_p99.9_recall"] for f in families],
            "Hybrid (0.1 % FA)": [unseen.loc[f, "hybrid_p99.9_recall"] for f in families],
        })
        pct_cols = [c for c in table.columns if c not in ("Family", "Records")]
        shown = table.copy()
        for c in pct_cols:
            shown[c] = shown[c].map(lambda v: f"{v:.1%}" if pd.notna(v) else "-")
        shown["Records"] = shown["Records"].map(lambda v: f"{v:,}")
        st.dataframe(shown, use_container_width=True, hide_index=True)

        chart_path = PLOTS_DIR / "unseen_family_recall.png"
        if chart_path.exists():
            st.image(
                str(chart_path),
                caption="Detection of seen vs unseen attack families",
                use_container_width=True
            )

        st.subheader("Where the missed attacks come from (seen model)")
        st.bar_chart(seen_missed.rename("Missed attacks"))
        if total_missed:
            st.warning(
                f"Of about {int(total_missed):,} attacks missed on the test "
                f"set, **{top_share:.0%} are {top_family}**. Even when it is "
                f"included in training, {top_family} is detected only "
                f"{seen.loc[top_family, 'rf_recall']:.2%} of the time."
            )

        st.subheader("False-alarm cost on benign test traffic")
        fa = pd.DataFrame({
            "Detector": [
                "Random Forest (all families)",
                "Anomaly detector (1 % threshold)",
                "Hybrid (1 % threshold)",
                "Anomaly detector (0.1 % threshold)",
                "Hybrid (0.1 % threshold)",
            ],
            "False-alarm rate": [
                ref["rf_benign_false_alarm_rate"],
                ref["ad_p99_benign_false_alarm_rate"],
                ref["hybrid_p99_benign_false_alarm_rate"],
                ref["ad_p99.9_benign_false_alarm_rate"],
                ref["hybrid_p99.9_benign_false_alarm_rate"],
            ],
        })
        fa["False alarms per million benign flows"] = (
            fa["False-alarm rate"] * 1e6
        ).map(lambda v: f"{v:,.0f}")
        fa["False-alarm rate"] = fa["False-alarm rate"].map(lambda v: f"{v:.3%}")
        st.dataframe(fa, use_container_width=True, hide_index=True)

        st.caption(
            "A withheld family in a public benchmark simulates an unseen "
            "attack type; it is not proof of real-world zero-day detection. "
            "Each Random Forest here was trained on a sample of at most "
            "1,000,000 rows to fit the 8 GB virtual machine."
        )


# ============================================================
# LIVE ANOMALY DETECTION
# ============================================================

elif page == "CSV Anomaly Detection":

    st.title("🔍 CSV Anomaly Detection")

    st.markdown(
        """
        Upload a network traffic CSV with the same feature headers
        as the CSE-CIC-IDS2018 processed dataset.

        The application will load your saved Random Forest model
        from HDFS and classify the uploaded records.
        """
    )

    st.info(
        "For a successful prediction, the CSV must contain all "
        f"{len(feature_columns)} required numeric feature columns. "
        "A Label column is not required."
    )

    uploaded_file = st.file_uploader(
        "Upload network traffic CSV",
        type=["csv"]
    )

    max_rows = st.slider(
        "Maximum number of records to classify",
        min_value=100,
        max_value=100000,
        value=10000,
        step=100
    )

    if uploaded_file is not None:

        st.write("**Uploaded file:**", uploaded_file.name)
        st.write(
            "**File size:**",
            f"{uploaded_file.size / (1024 * 1024):.2f} MB"
        )

        if st.button(
            "🚀 Run Anomaly Detection",
            type="primary"
        ):

            tmp_path = None

            try:
                from pyspark.sql import functions as F
                from pyspark.ml.feature import VectorAssembler

                with st.spinner(
                    "Loading Spark and the trained model..."
                ):
                    spark, model = load_spark_model()

                # Save the uploaded file temporarily so Spark can read it.
                tmp_path = os.path.join(
                    tempfile.gettempdir(),
                    f"cybersentinel_{uuid.uuid4().hex}.csv"
                )

                with open(tmp_path, "wb") as f:
                    f.write(uploaded_file.getvalue())

                with st.spinner(
                    "Reading the CSV and validating its feature columns..."
                ):
                    raw = (
                        spark.read
                        .option("header", "true")
                        .option("mode", "PERMISSIVE")
                        .csv("file://" + tmp_path)
                    )

                # Remove extra surrounding whitespace and BOM from headers.
                cleaned_headers = [
                    c.strip().lstrip("\ufeff")
                    for c in raw.columns
                ]

                raw = raw.toDF(*cleaned_headers)

                # Match source headers case-insensitively.
                column_lookup = {
                    c.strip().lstrip("\ufeff").casefold(): c
                    for c in raw.columns
                }

                missing_columns = [
                    c for c in feature_columns
                    if c.strip().casefold() not in column_lookup
                ]

                if missing_columns:
                    st.error(
                        "The uploaded CSV is missing required feature "
                        "columns. This may not be a compatible dataset."
                    )

                    st.write("Missing columns:")
                    st.code(", ".join(missing_columns))

                    st.stop()

                # Select the exact model feature order and cast columns.
                expressions = []

                for feature in feature_columns:
                    source_column = column_lookup[
                        feature.strip().casefold()
                    ]

                    quoted_source = (
                        "`" + source_column.replace("`", "``") + "`"
                    )

                    expressions.append(
                        F.col(quoted_source)
                        .cast("double")
                        .alias(feature)
                    )

                selected = (
                    raw.limit(int(max_rows))
                    .select(*expressions)
                )

                # Replace null, NaN and infinite feature values.
                clean = selected

                for feature in feature_columns:

                    c = F.col("`" + feature.replace("`", "``") + "`")

                    invalid = (
                        c.isNull()
                        | F.isnan(c)
                        | (c == float("inf"))
                        | (c == float("-inf"))
                    )

                    clean = clean.withColumn(
                        feature,
                        F.when(invalid, 0.0).otherwise(c)
                    )

                with st.spinner(
                    "Building feature vectors and predicting traffic..."
                ):

                    assembler = VectorAssembler(
                        inputCols=feature_columns,
                        outputCol="features"
                    )

                    feature_data = assembler.transform(clean)

                    predictions = model.transform(feature_data)

                    predictions = predictions.withColumn(
                        "predicted_class",
                        F.when(
                            F.col("prediction") == 0.0,
                            F.lit("Benign")
                        ).otherwise(F.lit("Attack"))
                    )

                    prediction_counts = (
                        predictions
                        .groupBy("predicted_class")
                        .count()
                        .collect()
                    )

                    counts = {
                        row["predicted_class"]: int(row["count"])
                        for row in prediction_counts
                    }

                    benign_count = counts.get("Benign", 0)
                    attack_count = counts.get("Attack", 0)
                    total_count = benign_count + attack_count

                if total_count == 0:
                    st.warning("No records were available to classify.")
                else:
                    st.success(
                        f"Classification completed for "
                        f"{total_count:,} records."
                    )

                    col1, col2, col3 = st.columns(3)

                    col1.metric(
                        "Records Analyzed",
                        f"{total_count:,}"
                    )

                    col2.metric(
                        "Benign Records",
                        f"{benign_count:,}",
                        f"{benign_count / total_count:.1%} of records"
                    )

                    col3.metric(
                        "Attack Records",
                        f"{attack_count:,}",
                        f"{attack_count / total_count:.1%} of records"
                    )

                    st.subheader("Prediction Summary")

                    summary_df = pd.DataFrame({
                        "Classification": ["Benign", "Attack"],
                        "Records": [benign_count, attack_count]
                    })

                    st.bar_chart(
                        summary_df.set_index("Classification")
                    )

                    st.subheader("Prediction Preview")

                    # Select useful features for the on-screen preview.
                    preview_features = [
                        c for c in [
                            "Dst Port",
                            "Protocol",
                            "Flow Duration",
                            "Pkt Len Max",
                            "Flow Pkts/s"
                        ]
                        if c in feature_columns
                    ]

                    # Prepare the complete prediction dataset for download.
                    # The uploaded file is limited by the max_rows slider.
                    with st.spinner(
                        "Preparing downloadable predictions for all records..."
                    ):
                        export_df = (
                            predictions
                            .select(
                                *feature_columns,
                                "prediction",
                                "predicted_class"
                            )
                            .toPandas()
                        )

                    # Show only a small preview in the dashboard.
                    preview_df = export_df[
                        preview_features + [
                            "prediction",
                            "predicted_class"
                        ]
                    ].head(100)

                    st.dataframe(
                        preview_df,
                        use_container_width=True,
                        hide_index=True
                    )

                    # Download EVERY processed record, not only the preview.
                    download_data = export_df.to_csv(
                        index=False
                    ).encode("utf-8")

                    st.download_button(
                        "Download ALL prediction results",
                        data=download_data,
                        file_name="cybersentinel_all_predictions.csv",
                        mime="text/csv"
                    )

                    st.success(
                        f"Complete results ready: {len(export_df):,} records, "
                        f"{len(feature_columns)} input features, "
                        "and prediction columns."
                    )

                    st.caption(
                        "The downloadable CSV contains all records processed "
                        "in this run, all 78 model input features, the numeric "
                        "prediction, and its Benign/Attack classification."
                    )

            except Exception as e:

                st.error(
                    "Prediction could not be completed. "
                    "Check that HDFS is running, the model exists, "
                    "and the CSV has compatible feature columns."
                )

                st.exception(e)

            finally:

                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass


# ============================================================
# FOOTER
# ============================================================

st.sidebar.divider()

st.sidebar.caption(
    "CyberSentinel | Big Data Project"
)
