"""Render the committed comparison JSON; no hand-entered benchmark values."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quest", type=Path, default=Path("docs/reports/layout-quest.json"))
    parser.add_argument("--token", type=Path, default=Path("docs/reports/layout-token.json"))
    parser.add_argument("--output", type=Path, default=Path("docs/assets/layout-comparison.svg"))
    args = parser.parse_args()
    reports = [json.loads(p.read_text()) for p in (args.quest, args.token)]
    assert reports[0]["trace_sha256"] == reports[1]["trace_sha256"]
    assert reports[0]["events_per_run"] == reports[1]["events_per_run"]
    assert reports[0]["repeats"] == reports[1]["repeats"]
    plt.rcParams.update({"font.size": 11, "svg.fonttype": "none", "svg.hashsalt": "mini-hisparse"})
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.6), layout="constrained")
    colors = {"resident": "#64748b", "pages": "#2563eb", "hybrid": "#0f766e"}
    titles = [
        f"Quest: {reports[0]['selection_budget']} pages / head",
        f"Offline oracle: {reports[1]['selection_budget'] * reports[1]['page_size']} tokens / head",
    ]
    events = reports[0]["events_per_run"]
    for row, (report, title) in enumerate(zip(reports, titles)):
        cases = report["cases"]
        names = [c["layout"] for c in cases]
        labels = [f"{c['layout']}\n{c['gpu_pages']} blocks" for c in cases]
        times = [c["median_of_run_medians_ms"] for c in cases]
        error = [
            [t - c["run_median_range_ms"][0] for c, t in zip(cases, times)],
            [c["run_median_range_ms"][1] - t for c, t in zip(cases, times)],
        ]
        for col, (values, unit) in enumerate(
            [
                ([c["h2d_bytes"] / 2**20 for c in cases], f"H2D payload / {events} events (MiB)"),
                (times, "Median layer append + attention (ms)"),
            ]
        ):
            ax = axes[row, col]
            bars = ax.bar(
                labels,
                values,
                color=[colors[n] for n in names],
                width=0.56,
                yerr=error if col else None,
                capsize=4,
            )
            ax.bar_label(bars, labels=[f"{v:.3f}" for v in values], padding=8)
            ax.set(title=title, ylabel=unit, ylim=(0, max(max(values), 0.1) * 1.28))
            ax.set_axisbelow(True)
            ax.grid(axis="y", alpha=0.2)
            ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Same selection within each row; resident uses extra KV memory", fontsize=15)
    gpu = reports[0]["gpu"] or "CPU"
    dtype = reports[0]["dtype"].removeprefix("torch.")
    fig.supxlabel(
        f"{gpu} / {dtype} / {reports[0]['repeats']} repeats. "
        "Whiskers: min-max run medians, not confidence intervals.\n"
        "JIT, prefill and selection excluded. Resident uses page-reference kernels; this is not serving throughput.",
        fontsize=10,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, metadata={"Date": None})
    if args.output.suffix.lower() == ".svg":
        # Matplotlib emits trailing spaces in SVG path data; keep diffs clean.
        args.output.write_text(
            "\n".join(line.rstrip() for line in args.output.read_text().splitlines()) + "\n"
        )
    plt.close(fig)


if __name__ == "__main__":
    main()
