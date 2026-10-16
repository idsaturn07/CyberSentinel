"""
Known vs unseen attacks: attack-family holdout with a supervised, an
anomaly-based and a hybrid detector.

Question: the Random Forest scores well on attack types it was trained on,
but how well does it detect an attack FAMILY that was completely left out of
training, and does a detector trained only on BENIGN traffic help?

Data split (seed 42), made once and shared by every model:
    70 % train pool  - supervised training (sampled, see TRAIN_CAP)
                       and benign-only K-Means training
    10 % calibration - BENIGN rows only, used to choose the anomaly
                       threshold (never used for evaluation)
    20 % test        - benign false-alarm rate and known-attack recall

Detectors:
    RF       Random Forest (20 trees, depth 10, seed 42), like train_model.py
    AD       Benign-only anomaly detector: standardised features, K-Means
             trained on benign traffic only; a flow is flagged when its
             distance to the nearest cluster centre is above the 99th (or
             99.9th) percentile of distances in the benign calibration set
    Hybrid   flagged if RF OR AD flags it

Experiments:
    Reference  RF trained on all families -> recall per family on the test
               split ("seen" recall)
    Holdout    for each attack family F: RF trained WITHOUT any F records.
               Its "unseen" recall (RF, AD and Hybrid) is reported twice:
                 - on F's records in the TEST split: the same records on
                   which the reference "seen" recall is measured, so seen
                   and unseen are compared like for like (main result);
                 - on every F record in train pool + test (none were used
                   in training): a larger sample, as a robustness check.
               Exact detected counts are saved next to every recall.

Note: a held-out family in a public benchmark only simulates an unseen
attack type; it is not evidence of real-world zero-day detection. In this
dataset each day contains different attacks, so family differences are partly
also differences in traffic over time.

Run after data_preprocessing.py (needs results/feature_columns.json and the
nine CSV files in data/CSE_CIC_IDS2018/).

Outputs:
    results/attack_family_counts.csv
    results/unseen_attack_results.csv
    results/unseen_attack_summary.json
    plots/unseen_family_recall.png
"""

import json
import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from pyspark import StorageLevel
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F

from common import (
    RESULTS_DIR,
    PLOTS_DIR,
    create_spark,
    load_and_clean_local_dataset,
)

SEED = 42
RF_SETTINGS = {"numTrees": 20, "maxDepth": 10, "seed": SEED}
TRAIN_CAP = 1_000_000        # max rows used to train each Random Forest
BENIGN_KMEANS_CAP = 200_000  # max benign rows used to train K-Means
K_CLUSTERS = 10
ANOMALY_PERCENTILES = [0.99, 0.999]   # 1 % and 0.1 % benign false-alarm targets
MIN_FAMILY_RECORDS = 100     # families smaller than this are not held out


def family_of(label_col):
    """Map the original label text to an attack family."""
    s = F.lower(F.regexp_replace(label_col, r"\s+", " "))
    return (
        F.when(s == "benign", "Benign")
        .when(s.contains("ddos"), "DDoS")
        .when(s.contains("dos"), "DoS")
        .when(s.contains("ftp-bruteforce") | s.contains("ssh-bruteforce"), "Brute-force")
        .when(s.contains("web") | s.contains("xss") | s.contains("sql"), "Web")
        .when(s.contains("bot"), "Bot")
        .when(s.contains("infil"), "Infiltration")
        .otherwise("Other")
    )


def sample_cap(df, n_rows, cap):
    fraction = min(1.0, cap / n_rows) if n_rows else 1.0
    return df if fraction >= 1.0 else df.sample(False, fraction, seed=SEED)


def add_distance(km_model, df, k):
    """Squared distance of each row to its nearest K-Means centre (JVM only)."""
    centres = F.array(*[
        F.array(*[F.lit(float(v)) for v in centre])
        for centre in km_model.clusterCenters()
    ])
    out = km_model.transform(df)
    x = vector_to_array("scaled")
    c = F.element_at(centres, F.col("cluster") + 1)
    sq = F.zip_with(x, c, lambda a, b: (a - b) * (a - b))
    return out.withColumn("distance", F.aggregate(sq, F.lit(0.0), lambda acc, v: acc + v))


def rate(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    spark = create_spark("CyberSentinel-Unseen-Attack-Experiment")

    try:
        print("Loading and cleaning the nine CSV files (keeping attack names)...", flush=True)
        cleaned, feature_columns = load_and_clean_local_dataset(spark, keep_attack_type=True)

        data = (
            VectorAssembler(inputCols=feature_columns, outputCol="features")
            .transform(cleaned.withColumn("family", family_of(F.col("attack_type"))))
            .select("features", "label", "attack_type", "family")
        )

        pool, calib, test = data.randomSplit([0.7, 0.1, 0.2], seed=SEED)
        pool = pool.persist(StorageLevel.DISK_ONLY)
        test = test.persist(StorageLevel.DISK_ONLY)
        calib_benign = calib.filter(F.col("label") == 0.0).persist(StorageLevel.DISK_ONLY)

        # ---- family counts ------------------------------------------------
        counts = {}
        for split_name, df in [("train_pool", pool), ("test", test)]:
            for r in df.groupBy("family", "attack_type").count().collect():
                key = (r["family"], r["attack_type"])
                counts.setdefault(key, {"train_pool": 0, "test": 0})[split_name] = int(r["count"])
        counts_df = pd.DataFrame([
            {"family": f, "attack_type": a, **v} for (f, a), v in counts.items()
        ]).sort_values(["family", "attack_type"])
        counts_df["total"] = counts_df["train_pool"] + counts_df["test"]
        counts_df.to_csv(RESULTS_DIR / "attack_family_counts.csv", index=False)
        print("\n=== Labels and families (train pool + test) ===")
        print(counts_df.to_string(index=False), flush=True)

        fam_totals = counts_df.groupby("family")[["train_pool", "test", "total"]].sum()
        pool_rows = int(fam_totals["train_pool"].sum())
        families = [
            f for f in fam_totals.index
            if f not in ("Benign", "Other") and fam_totals.loc[f, "total"] >= MIN_FAMILY_RECORDS
        ]
        skipped = [f for f in fam_totals.index
                   if f not in ("Benign",) and f not in families]
        print("\nFamilies held out one at a time:", families)
        if skipped:
            print("Not held out (too small or unmapped):", skipped)

        n_cal = calib_benign.count()
        test_benign_n = int(fam_totals.loc["Benign", "test"]) if "Benign" in fam_totals.index else 0

        # ---- benign-only anomaly detector ------------------------------------
        print("\nTraining benign-only anomaly detector (StandardScaler + K-Means)...", flush=True)
        benign_pool = pool.filter(F.col("label") == 0.0)
        benign_rows = int(fam_totals.loc["Benign", "train_pool"])
        benign_train = sample_cap(benign_pool, benign_rows, BENIGN_KMEANS_CAP)

        scaler = StandardScaler(inputCol="features", outputCol="scaled",
                                withMean=True, withStd=True).fit(benign_train)
        # Persisted on disk so K-Means does not cache its input in memory.
        km_input = scaler.transform(benign_train).select("scaled").persist(
            StorageLevel.DISK_ONLY)
        km = KMeans(featuresCol="scaled", predictionCol="cluster", k=K_CLUSTERS,
                    seed=SEED, maxIter=20).fit(km_input)
        km_input.unpersist()

        cal_dist = add_distance(km, scaler.transform(calib_benign), K_CLUSTERS)
        thresholds = cal_dist.approxQuantile("distance", ANOMALY_PERCENTILES, 0.0001)
        thresholds = dict(zip(ANOMALY_PERCENTILES, thresholds))
        print(f"Benign calibration rows: {n_cal:,}")
        for p, t in thresholds.items():
            print(f"  threshold at {p:.1%} of benign calibration distances: {t:.4f}")

        def score_ad(df):
            """Add the anomaly distance; flags are derived per threshold later."""
            return add_distance(km, scaler.transform(df), K_CLUSTERS).drop("scaled", "cluster")

        # ---- evaluation helper -------------------------------------------------
        def evaluate(rf_model, held_out=None):
            """Returns per-family recall and benign false-alarm rates for RF, AD, Hybrid."""
            test_scored = score_ad(rf_model.transform(test)).select(
                "label", "family", "prediction", "distance")

            exprs = [F.count(F.lit(1)).alias("n")]
            exprs.append(F.sum((F.col("prediction") == 1.0).cast("long")).alias("rf"))
            for i, p in enumerate(ANOMALY_PERCENTILES):
                ad = F.col("distance") > F.lit(thresholds[p])
                exprs.append(F.sum(ad.cast("long")).alias(f"ad{i}"))
                exprs.append(F.sum((ad | (F.col("prediction") == 1.0)).cast("long")).alias(f"hy{i}"))

            rows = {r["family"]: r for r in test_scored.groupBy("family").agg(*exprs).collect()}

            unseen = None
            if held_out is not None:
                # Every record of the held-out family was excluded from training,
                # so all of them (train pool + test) are unseen.
                fam_df = pool.filter(F.col("family") == held_out).unionByName(
                    test.filter(F.col("family") == held_out))
                unseen = score_ad(rf_model.transform(fam_df)).agg(*exprs).collect()[0]
            return rows, unseen

        def summarise(rows, held_out, unseen, model_name):
            b = rows.get("Benign")
            out = []
            # benign false-alarm rates on the test split
            fa = {"rf": rate(b["rf"], b["n"]) if b else 0.0}
            for i, p in enumerate(ANOMALY_PERCENTILES):
                fa[f"ad{i}"] = rate(b[f"ad{i}"], b["n"]) if b else 0.0
                fa[f"hy{i}"] = rate(b[f"hy{i}"], b["n"]) if b else 0.0

            known = [f for f in rows if f not in ("Benign", "Other", held_out)]
            k_n = sum(rows[f]["n"] for f in known)
            known_rf = rate(sum(rows[f]["rf"] for f in known), k_n)

            def record(family, kind, r):
                rec = {
                    "model": model_name,
                    "held_out_family": held_out or "-",
                    "evaluated_family": family,
                    "kind": kind,
                    "records": int(r["n"]),
                    "rf_recall": rate(r["rf"], r["n"]),
                    "rf_benign_false_alarm_rate": fa["rf"],
                    "known_families_rf_recall": known_rf,
                }
                for i, p in enumerate(ANOMALY_PERCENTILES):
                    tag = f"p{round(p * 100, 1):g}"
                    rec[f"ad_{tag}_recall"] = rate(r[f"ad{i}"], r["n"])
                    rec[f"hybrid_{tag}_recall"] = rate(r[f"hy{i}"], r["n"])
                    rec[f"ad_{tag}_benign_false_alarm_rate"] = fa[f"ad{i}"]
                    rec[f"hybrid_{tag}_benign_false_alarm_rate"] = fa[f"hy{i}"]
                    rec[f"hybrid_{tag}_false_alarms_per_million"] = fa[f"hy{i}"] * 1e6
                # exact counts, so missed attacks need no rounding
                rec["detected_rf"] = int(r["rf"])
                for i, p in enumerate(ANOMALY_PERCENTILES):
                    tag = f"p{round(p * 100, 1):g}"
                    rec[f"detected_ad_{tag}"] = int(r[f"ad{i}"])
                    rec[f"detected_hybrid_{tag}"] = int(r[f"hy{i}"])
                return rec

            if held_out is None:
                for fam in sorted(f for f in rows if f not in ("Benign", "Other")):
                    out.append(record(fam, "seen (test split)", rows[fam]))
            else:
                # Like-for-like: the held-out family's TEST records, the same
                # records used for the reference model's "seen" recall.
                out.append(record(held_out, "unseen (test split)", rows[held_out]))
                # Robustness check: all of the family's pool + test records.
                out.append(record(held_out, "unseen (pool + test)", unseen))
            return out

        results = []

        # ---- reference: RF trained on all families ---------------------------------
        print(f"\n[Reference] Training RF on all families (cap {TRAIN_CAP:,} rows)...", flush=True)
        t0 = time.perf_counter()
        ref_rf = RandomForestClassifier(featuresCol="features", labelCol="label",
                                        **RF_SETTINGS).fit(sample_cap(pool, pool_rows, TRAIN_CAP))
        rows, _ = evaluate(ref_rf)
        results += summarise(rows, None, None, "RF (all families)")
        print(f"  done in {time.perf_counter() - t0:.0f} s", flush=True)

        # ---- holdout: one family left out at a time ---------------------------------
        for fam in families:
            print(f"\n[Holdout] Training RF WITHOUT family '{fam}'...", flush=True)
            t0 = time.perf_counter()
            remaining = pool_rows - int(fam_totals.loc[fam, "train_pool"])
            train_df = sample_cap(pool.filter(F.col("family") != fam), remaining, TRAIN_CAP)
            rf = RandomForestClassifier(featuresCol="features", labelCol="label",
                                        **RF_SETTINGS).fit(train_df)
            rows, unseen = evaluate(rf, held_out=fam)
            results += summarise(rows, fam, unseen, f"RF without {fam}")
            for last in results[-2:]:
                print(f"  {last['kind']:<22} RF {last['rf_recall']:.4f} | "
                      f"AD(99%) {last['ad_p99_recall']:.4f} | "
                      f"Hybrid(99%) {last['hybrid_p99_recall']:.4f}", flush=True)
            print(f"  done in {time.perf_counter() - t0:.0f} s", flush=True)

        res = pd.DataFrame(results)
        res.to_csv(RESULTS_DIR / "unseen_attack_results.csv", index=False)

        seen = res[res["kind"] == "seen (test split)"].set_index("evaluated_family")
        unseen_df = res[res["kind"] == "unseen (test split)"].set_index("evaluated_family")
        unseen_all = res[res["kind"] == "unseen (pool + test)"].set_index("evaluated_family")

        # ---- plot --------------------------------------------------------------
        fams = list(unseen_df.index)
        if fams:
            series = [
                ("RF, family seen in training", [seen["rf_recall"].get(f, float("nan")) for f in fams]),
                ("RF, family unseen", list(unseen_df["rf_recall"])),
                ("Anomaly detector (1% FA)", list(unseen_df["ad_p99_recall"])),
                ("Hybrid RF + AD (1% FA)", list(unseen_df["hybrid_p99_recall"])),
            ]
            fig, ax = plt.subplots(figsize=(11, 6))
            width = 0.2
            for j, (name, vals) in enumerate(series):
                xs = [i + (j - 1.5) * width for i in range(len(fams))]
                ax.bar(xs, [v * 100 for v in vals], width=width, label=name)
            ax.set_xticks(range(len(fams)), fams)
            ax.set_ylabel("Attack family detected (%)")
            ax.set_ylim(0, 105)
            ax.set_title("Detection of Seen vs Unseen Attack Families (same test records)", pad=34)
            ax.legend(fontsize=9, loc="lower center", bbox_to_anchor=(0.5, 1.0),
                      ncol=4, frameon=False)
            ax.grid(axis="y", alpha=0.3)
            fig.tight_layout()
            fig.savefig(PLOTS_DIR / "unseen_family_recall.png", dpi=200)
            plt.close(fig)

        summary = {
            "split": {"train_pool": 0.7, "benign_calibration": 0.1, "test": 0.2, "seed": SEED},
            "train_cap_rows": TRAIN_CAP,
            "kmeans": {"k": K_CLUSTERS, "benign_training_cap": BENIGN_KMEANS_CAP,
                       "benign_calibration_rows": n_cal},
            "anomaly_thresholds": {f"p{round(p * 100, 1):g}": t for p, t in thresholds.items()},
            "test_benign_rows": test_benign_n,
            "families_held_out": families,
            "families_not_held_out": skipped,
            "runtime_minutes": round((time.perf_counter() - started) / 60, 1),
        }
        with open(RESULTS_DIR / "unseen_attack_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        pd.set_option("display.width", 200)
        cols = ["rf_recall", "ad_p99_recall", "hybrid_p99_recall",
                "ad_p99.9_recall", "hybrid_p99.9_recall"]
        print("\n=== SEEN: RF trained on all families, test split ===")
        seen_out = seen[["records", "detected_rf", "rf_recall"]].copy()
        seen_out["missed_rf"] = seen_out["records"] - seen_out["detected_rf"]
        print(seen_out.round(4).to_string())
        print("Total missed attacks (exact):", int(seen_out["missed_rf"].sum()))
        print("\n=== UNSEEN (main result): family held out, SAME test records as SEEN ===")
        print(unseen_df[["records"] + cols + ["known_families_rf_recall"]].round(4).to_string())
        print("\n=== UNSEEN (robustness check): family held out, all pool + test records ===")
        print(unseen_all[["records"] + cols].round(4).to_string())
        print("\n=== Benign false-alarm rate on the test split ===")
        fa_cols = ["rf_benign_false_alarm_rate", "ad_p99_benign_false_alarm_rate",
                   "hybrid_p99_benign_false_alarm_rate", "ad_p99.9_benign_false_alarm_rate",
                   "hybrid_p99.9_benign_false_alarm_rate"]
        print(res.set_index("model")[fa_cols].drop_duplicates().round(5).to_string())
        print("\n=== SUMMARY ===")
        print(json.dumps(summary, indent=2))
        print("Saved results to", RESULTS_DIR, "and plot to", PLOTS_DIR)

        for df in (pool, test, calib_benign):
            df.unpersist()

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
