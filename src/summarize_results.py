import csv
from pathlib import Path
from collections import defaultdict


def get_project_root() -> Path:
    """
    Return project root directory.
    This assumes summarize_results.py is inside src/.
    """
    return Path(__file__).resolve().parents[1]


def read_results(csv_path: Path):
    """
    Read per-image evaluation results from CSV.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Cannot find result CSV: {csv_path}")

    data = defaultdict(lambda: {
        "EPE": [],
        "Bad3": [],
        "Runtime": [],
        "ValidRatio": []
    })

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            method = row["method"]

            data[method]["EPE"].append(float(row["EPE"]))
            data[method]["Bad3"].append(float(row["Bad-3px (%)"]))
            data[method]["Runtime"].append(float(row["Runtime (ms)"]))
            data[method]["ValidRatio"].append(float(row["Valid prediction ratio (%)"]))

    return data


def compute_average_results(data):
    """
    Compute average metrics for each method.
    """
    rows = []

    for method, values in data.items():
        n = len(values["EPE"])

        avg_epe = sum(values["EPE"]) / n
        avg_bad3 = sum(values["Bad3"]) / n
        avg_runtime = sum(values["Runtime"]) / n
        avg_valid = sum(values["ValidRatio"]) / n

        rows.append({
            "method": method,
            "num_samples": n,
            "average_EPE": avg_epe,
            "average_Bad-3px (%)": avg_bad3,
            "average_Runtime (ms)": avg_runtime,
            "average_Valid prediction ratio (%)": avg_valid
        })

    return rows


def print_summary(rows):
    """
    Print average results in terminal.
    """
    print("\nAverage results")
    print("-" * 105)
    print(
        f"{'Method':<15} "
        f"{'Samples':<10} "
        f"{'EPE':<12} "
        f"{'Bad-3px (%)':<15} "
        f"{'Runtime (ms)':<15} "
        f"{'Valid Ratio (%)':<18}"
    )
    print("-" * 105)

    for row in rows:
        print(
            f"{row['method']:<15} "
            f"{row['num_samples']:<10} "
            f"{row['average_EPE']:<12.3f} "
            f"{row['average_Bad-3px (%)']:<15.2f} "
            f"{row['average_Runtime (ms)']:<15.2f} "
            f"{row['average_Valid prediction ratio (%)']:<18.2f}"
        )


def save_summary(rows, summary_path: Path):
    """
    Save average results to CSV.
    """
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "method",
        "num_samples",
        "average_EPE",
        "average_Bad-3px (%)",
        "average_Runtime (ms)",
        "average_Valid prediction ratio (%)"
    ]

    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    root = get_project_root()

    csv_path = root / "outputs" / "metrics" / "bm_sgbm_kitti_results.csv"
    summary_path = root / "outputs" / "summaries" / "average_results.csv"

    data = read_results(csv_path)
    rows = compute_average_results(data)

    print_summary(rows)
    save_summary(rows, summary_path)

    print(f"\nSaved summary to: {summary_path}")
    print("summarize_results.py completed successfully.")


if __name__ == "__main__":
    main()