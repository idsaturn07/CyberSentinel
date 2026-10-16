"""
Data-quality audit of the CSE-CIC-IDS2018 processed CSV files.

For every CSV file in data/CSE_CIC_IDS2018/ (all ten, including the one that
is excluded from modelling) this records:

    - file size and number of columns in the header
    - whether the header matches the shared schema of the nine model files
      (same column names in the same order), and which columns differ
    - data rows and repeated header rows (rows whose Label is "Label")
    - missing, NaN and +/-Infinity values in the 78 model feature columns
    - the traffic labels present in the file and their counts
    - the inclusion / exclusion decision used by the pipeline

Every file is read on its own with its own header, so the excluded file can
be audited even though its schema differs. Values are read as text (no type
inference) so that "NaN" and "Infinity" strings are counted exactly as they
appear in the files.

Run after data_preprocessing.py (needs results/feature_columns.json).

Outputs:
    results/data_quality_audit.csv           one row per file
    results/invalid_values_by_column.csv     invalid counts per feature (included files)
    results/labels_by_file.csv               label counts per file
    results/data_quality_summary.json        totals
"""

import csv
import json

import pandas as pd
from pyspark.sql import functions as F

from common import DATA_DIR, GOOD_FILES, RESULTS_DIR, create_spark

NAN_STRINGS = ["NaN", "nan"]
INF_STRINGS = ["Infinity", "-Infinity", "inf", "-inf", "Inf", "-Inf"]


def read_header(path):
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        return [c.strip().lstrip("\ufeff") for c in next(csv.reader(f))]


def qcol(name):
    return F.col("`" + name.replace("`", "``") + "`")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(RESULTS_DIR / "feature_columns.json", "r", encoding="utf-8") as f:
        feature_columns = json.load(f)

    files = sorted(DATA_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {DATA_DIR}")

    # Reference schema = header of the first included file.
    reference = read_header(DATA_DIR / GOOD_FILES[0])

    spark = create_spark("CyberSentinel-Data-Quality-Audit")

    audit_rows, label_rows, column_rows = [], [], []

    try:
        for path in files:
            name = path.name
            header = read_header(path)
            included = name in GOOD_FILES
            extra = [c for c in header if c not in reference]
            missing = [c for c in reference if c not in header]
            same_schema = header == reference

            print(f"\nAuditing {name} ({path.stat().st_size / 1e9:.2f} GB, "
                  f"{len(header)} columns)...", flush=True)

            df = (
                spark.read
                .option("header", "true")
                .option("inferSchema", "false")
                .option("mode", "PERMISSIVE")
                .csv(path.as_uri())
            )

            if "Label" in df.columns:
                is_header_row = F.trim(F.col("Label")) == F.lit("Label")
            else:
                is_header_row = F.lit(False)

            present = [c for c in feature_columns if c in df.columns]

            # One pass: row counts plus missing / NaN / Infinity per feature.
            exprs = [
                F.count(F.lit(1)).alias("rows_read"),
                F.sum(is_header_row.cast("long")).alias("header_rows"),
            ]
            for i, c in enumerate(present):
                v = F.trim(qcol(c))
                exprs += [
                    F.sum((v.isNull() | (v == "")).cast("long")).alias(f"m{i}"),
                    F.sum(v.isin(*NAN_STRINGS).cast("long")).alias(f"n{i}"),
                    F.sum(v.isin(*INF_STRINGS).cast("long")).alias(f"i{i}"),
                ]
            agg = df.agg(*exprs).collect()[0]

            rows_read = int(agg["rows_read"])
            header_rows = int(agg["header_rows"] or 0)

            missing_total = nan_total = inf_total = 0
            for i, c in enumerate(present):
                m, n, inf = (int(agg[f"m{i}"] or 0), int(agg[f"n{i}"] or 0),
                             int(agg[f"i{i}"] or 0))
                missing_total += m
                nan_total += n
                inf_total += inf
                if included and (m or n or inf):
                    column_rows.append({"file": name, "feature": c,
                                        "missing": m, "nan": n, "infinity": inf})

            # Labels present in the file (header rows excluded).
            n_labels = 0
            if "Label" in df.columns:
                for r in (df.filter(~is_header_row)
                          .groupBy(F.trim(F.col("Label")).alias("label"))
                          .count().collect()):
                    label_rows.append({"file": name, "label": r["label"],
                                       "count": int(r["count"])})
                    n_labels += 1

            if included:
                decision = "Included"
                reason = "Matches the shared 80-column schema"
            else:
                decision = "Excluded"
                reason = ("Header does not match the shared schema"
                          if not same_schema else "Not in GOOD_FILES list")

            audit_rows.append({
                "file": name,
                "size_gb": round(path.stat().st_size / 1e9, 3),
                "header_columns": len(header),
                "matches_shared_schema": same_schema,
                "extra_columns": "; ".join(extra),
                "missing_columns": "; ".join(missing),
                "data_rows_read": rows_read,
                "repeated_header_rows": header_rows,
                "rows_after_header_removal": rows_read - header_rows,
                "missing_values": missing_total,
                "nan_values": nan_total,
                "infinity_values": inf_total,
                "distinct_labels": n_labels,
                "decision": decision,
                "reason": reason,
            })
            print({k: audit_rows[-1][k] for k in (
                "header_columns", "matches_shared_schema", "data_rows_read",
                "repeated_header_rows", "nan_values", "infinity_values",
                "decision")}, flush=True)

        audit = pd.DataFrame(audit_rows)
        labels = pd.DataFrame(label_rows)
        columns = pd.DataFrame(column_rows,
                               columns=["file", "feature", "missing", "nan", "infinity"])

        audit.to_csv(RESULTS_DIR / "data_quality_audit.csv", index=False)
        labels.to_csv(RESULTS_DIR / "labels_by_file.csv", index=False)

        by_feature = (columns.groupby("feature")[["missing", "nan", "infinity"]]
                      .sum().reset_index())
        by_feature["total_invalid"] = by_feature[["missing", "nan", "infinity"]].sum(axis=1)
        by_feature = by_feature.sort_values("total_invalid", ascending=False)
        by_feature.to_csv(RESULTS_DIR / "invalid_values_by_column.csv", index=False)

        inc = audit[audit["decision"] == "Included"]
        summary = {
            "files_audited": int(len(audit)),
            "files_included": int(len(inc)),
            "files_excluded": audit.loc[audit["decision"] == "Excluded", "file"].tolist(),
            "all_files_rows_read": int(audit["data_rows_read"].sum()),
            "included_rows_read": int(inc["data_rows_read"].sum()),
            "included_repeated_header_rows": int(inc["repeated_header_rows"].sum()),
            "included_rows_after_header_removal": int(inc["rows_after_header_removal"].sum()),
            "included_missing_values": int(inc["missing_values"].sum()),
            "included_nan_values": int(inc["nan_values"].sum()),
            "included_infinity_values": int(inc["infinity_values"].sum()),
            "features_with_invalid_values": by_feature.loc[
                by_feature["total_invalid"] > 0, "feature"].tolist(),
            "shared_schema_columns": len(reference),
        }
        with open(RESULTS_DIR / "data_quality_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        pd.set_option("display.width", 200)
        pd.set_option("display.max_colwidth", 60)
        print("\n=== DATA-QUALITY AUDIT (per file) ===")
        print(audit[["file", "size_gb", "header_columns", "matches_shared_schema",
                     "data_rows_read", "repeated_header_rows", "nan_values",
                     "infinity_values", "decision"]].to_string(index=False))
        excluded = audit[audit["decision"] == "Excluded"]
        for _, r in excluded.iterrows():
            print(f"\nExcluded: {r['file']}")
            print(f"  extra columns  : {r['extra_columns'] or '-'}")
            print(f"  missing columns: {r['missing_columns'] or '-'}")
        print("\n=== Features with invalid values (included files) ===")
        print(by_feature[by_feature["total_invalid"] > 0].to_string(index=False))
        print("\n=== Labels by file ===")
        if not labels.empty:
            print(labels.sort_values(["file", "count"], ascending=[True, False])
                  .to_string(index=False))
        print("\n=== SUMMARY ===")
        print(json.dumps(summary, indent=2))
        print("\nSaved to", RESULTS_DIR)

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
