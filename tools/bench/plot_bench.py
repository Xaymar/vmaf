#!/usr/bin/env python3
"""Plot docs/benchmarks/*.json (written by vmaf_bench.py) into dark and light PNGs.

Labels are "<machine> · <device>". CPU runs ("<machine> · CPU") are drawn in every panel of that machine.
One panel per machine and GPU, one row per resolution and model count, log fps axis, a marker per engine at the
mean with a whisker from the slowest to the fastest run.

  python plot_bench.py docs/benchmarks docs/benchmark-v1   (reads docs/benchmarks/<machine>/*.json)
"""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

THEMES = {
    "dark": dict(bg="#1a1a1a", fg="#e6e6e6", grid="#3a3a3a", sub="#9a9a9a"),
    "light": dict(bg="#ffffff", fg="#1f2328", grid="#d8dee4", sub="#59636e"),
}
ENGINES = [("cpu", "libvmaf CPU", "o", "#e8833a"), ("fast", "libvmaf-fast", "D", "#2ea67a"), ("vulkan", "this fork (Vulkan)", "s", "#4a8fe7")]
ROWS = [(r, n) for r in ("1080", "4k") for n in (1, 2, 4)]


def load(d):
    runs = [json.loads(p.read_text()) for p in sorted(Path(d).glob("*/*.json"))]
    panels = {}
    for r in runs:
        m = r["meta"]
        machine, dev = m["label"].split(" · ", 1)
        if m["engine"] != "cpu":
            panels.setdefault((machine, dev), {})[m["engine"]] = (r, m)
    for r in runs:
        m = r["meta"]
        machine = m["label"].split(" · ", 1)[0]
        if m["engine"] == "cpu":
            for (mc, _), e in panels.items():
                if mc == machine:
                    e["cpu"] = (r, m)
    order = sorted(panels, key=lambda k: (k[0], any(x in k[1] for x in ("iGPU", "Vega", "780M")), k[1]))
    return runs, {k: panels[k] for k in order}


def draw(panels, theme, out, foot):
    t = THEMES[theme]
    n = len(panels)
    fig, axes = plt.subplots(n, 1, figsize=(8.4, 1.1 + 3.0 * n), facecolor=t["bg"], squeeze=False)
    fig.suptitle("Scoring speed, frames per second (log scale, higher is better)", x=0.02, ha="left", color=t["fg"], fontsize=13, fontweight="bold")
    for ax, ((machine, dev), eng) in zip(axes[:, 0], panels.items()):
        ax.set_facecolor(t["bg"])
        ax.set_title(f"{machine} · {dev}", loc="left", color=t["fg"], fontsize=10.5, fontweight="bold")
        for k, (res, nm) in enumerate(ROWS):
            y = len(ROWS) - 1 - k
            for j, (key, _, mk, col) in enumerate(ENGINES):
                if key not in eng:
                    continue
                row = next((x for x in eng[key][0]["results"] if x["res"] == res and x["models"] == nm), None)
                if not row:
                    continue
                yy = y + (j - 1) * 0.2
                ax.plot([row["min"], row["max"]], [yy, yy], color=col, lw=1.6, alpha=0.8, solid_capstyle="round")
                ax.plot([row["min"], row["max"]], [yy, yy], "|", color=col, ms=6)
                ax.plot(row["avg"], yy, mk, color=col, ms=6.5, mec=t["bg"], mew=0.8)
        ax.set_xscale("log")
        ax.set_yticks(range(len(ROWS)))
        ax.set_yticklabels([f"{'4K' if r == '4k' else '1080p'} · {m} model{'s' if m > 1 else ''}" for r, m in reversed(ROWS)], color=t["fg"], fontsize=8.5)
        ax.set_ylim(-0.6, len(ROWS) - 0.4)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}" if v / 10 ** int(__import__("math").floor(__import__("math").log10(v))) in (2, 5) else ""))
        for k in range(1, len(ROWS)):
            if k % 3 == 0:
                ax.axhline(k - 0.5, color=t["grid"], lw=1.0)
        ax.tick_params(which="both", colors=t["sub"], labelsize=8.5, length=0)
        ax.grid(True, axis="x", color=t["grid"], lw=0.7, which="both", alpha=0.7)
        for s in ax.spines.values():
            s.set_visible(False)
    handles = [Line2D([0], [0], marker=mk, color=col, lw=1.6, ms=6.5, label=lab) for _, lab, mk, col in ENGINES]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 0.955), ncol=3, frameon=False, labelcolor=t["fg"], fontsize=9)
    fig.text(0.02, 0.008, __import__("textwrap").fill(foot, 125), color=t["sub"], fontsize=7.5, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    fig.savefig(out, dpi=160, facecolor=t["bg"])
    plt.close(fig)


def write_csv(runs, path):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "engine", "resolution", "models", "fps_min", "fps_avg", "fps_max", "runs"])
        for r in runs:
            for x in r["results"]:
                w.writerow([r["meta"]["label"], r["meta"]["engine"], x["res"], x["models"], f"{x['min']:.2f}", f"{x['avg']:.2f}", f"{x['max']:.2f}", len(x["runs"])])


def write_md(panels, path):
    """Markdown tables: mean fps (min-max) per machine/device and engine."""
    cols = [(r, n) for r in ("1080", "4k") for n in (1, 2, 4)]
    out = ["# Results (fps, mean over runs, slowest to fastest in brackets)\n"]
    for (machine, dev), eng in panels.items():
        out.append(f"\n## {machine} · {dev}\n")
        out.append("| engine | " + " | ".join(f"{'4K' if r == '4k' else '1080p'} · {n}" for r, n in cols) + " |")
        out.append("|---|" + "--:|" * len(cols))
        for key, lab, _, _ in ENGINES:
            if key not in eng:
                continue
            cells = []
            for r, n in cols:
                x = next(x for x in eng[key][0]["results"] if x["res"] == r and x["models"] == n)
                cells.append(f"{x['avg']:.1f} ({x['min']:.0f}-{x['max']:.0f})")
            out.append(f"| {lab} | " + " | ".join(cells) + " |")
    Path(path).write_text("\n".join(out) + "\n")


def main():
    d, prefix = sys.argv[1], sys.argv[2]
    runs, panels = load(d)
    write_csv(runs, Path(d) / "results.csv")
    write_md(panels, Path(d) / "results.md")
    m = runs[0]["meta"]
    foot = (f"Marker = mean, whisker = slowest to fastest of {m['runs']} runs. {m['frames']}-frame 8-bit 4:2:0 clips, v1.0.16 models "
            f"(1d5h_2160, +3d0h, +3d0h_2160, +5d0h), start-up excluded. Method and versions in README.")
    for th in THEMES:
        draw(panels, th, f"{prefix}-{th}.png", foot)


if __name__ == "__main__":
    main()
