"""
Regenerate the four project figures from the saved result files.

Inputs (written by the earlier pipeline stages):
    results/project_metrics.json      -> confusion_matrix.png
    results/feature_importance.csv    -> feature_importance.png
    results/scalability_results.csv   -> scalability_processing_time.png
                                         scalability_throughput.png

It does not start Spark or contact HDFS; it only reads local result files.
"""

import json

import matplotlib

matplotlib.use("Agg")  # render to files without a display

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import RESULTS_DIR, PLOTS_DIR


def plot_confusion_matrix():
    with open(RESULTS_DIR / "project_metrics.json", "r", encoding="utf-8") as f:
        cm = json.load(f)["confusion_matrix"]

    matrix = np.array([
        [cm["TN"], cm["FP"]],
        [cm["FN"], cm["TP"]],
    ])

    fig, ax = plt.subplots(figsize=(10, 8.5))
    image = ax.imshow(matrix, cmap="viridis")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Number of Records")

    labels = ["Benign", "Attack"]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predicted Class", labelpad=12)
    ax.set_ylabel("Actual Class", labelpad=12)
    ax.set_title("Confusion Matrix - Random Forest Anomaly Detection", pad=16)

    threshold = matrix.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, f"{matrix[i, j]:,}",
                ha="center", va="center",
                fontsize=16, fontweight="bold",
                color="white" if matrix[i, j] > threshold else "black",
            )

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "confusion_matrix.png", dpi=240)
    plt.close(fig)


def plot_feature_importance(top_n=15):
    importance = (
        pd.read_csv(RESULTS_DIR / "feature_importance.csv")
        .sort_values("Importance", ascending=False)
        .head(top_n)
        .iloc[::-1]  # largest bar at the top
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(importance["Feature"], importance["Importance"])
    ax.set_xlabel("Feature Importance")
    ax.set_ylabel("Network Traffic Feature")
    ax.set_title(f"Top {top_n} Features - Random Forest Anomaly Detection")

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "feature_importance.png", dpi=300)
    plt.close(fig)


def plot_scalability():
    results = pd.read_csv(RESULTS_DIR / "scalability_results.csv")
    size_thousands = results["Records"] / 1000

    charts = [
        (
            "Time_seconds",
            "Processing Time (Seconds)",
            "Spark Prediction Processing Time vs Dataset Size",
            "scalability_processing_time.png",
        ),
        (
            "Records_per_second",
            "Throughput (Records per Second)",
            "Spark Prediction Throughput vs Dataset Size",
            "scalability_throughput.png",
        ),
    ]

    # Repeated runs (scalability_experiment.py with REPEATS > 1) add standard
    # deviation columns; they are drawn as error bars.
    std_columns = {"Time_seconds": "Time_std",
                   "Records_per_second": "Throughput_std"}

    for column, y_label, title, filename in charts:
        fig, ax = plt.subplots(figsize=(10, 5.5))
        std = std_columns[column]
        if std in results.columns:
            runs = int(results["Runs"].iloc[0]) if "Runs" in results.columns else None
            ax.errorbar(size_thousands, results[column], yerr=results[std],
                        marker="o", linewidth=2, capsize=5)
            if runs:
                title = f"{title} (mean ± SD, {runs} runs)"
        else:
            ax.plot(size_thousands, results[column], marker="o", linewidth=2)
        ax.set_xlabel("Dataset Size (Thousands of Records)")
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.grid(True)

        fig.tight_layout()
        fig.savefig(PLOTS_DIR / filename, dpi=267)
        plt.close(fig)


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    plot_confusion_matrix()
    plot_feature_importance()
    plot_scalability()

    print("Plots saved to:", PLOTS_DIR)
    for path in sorted(PLOTS_DIR.glob("*.png")):
        print("  -", path.name)


if __name__ == "__main__":
    main()
