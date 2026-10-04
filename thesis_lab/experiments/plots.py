"""
Figures for the thesis from results/*.csv (run after the experiments).

Each function reads one CSV and skips silently if that experiment has not been
run yet, so you can re-run this script at any time.

Figures -> results/figures/*.png

Run:  python -m experiments.plots
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # no display needed (servers, SSH)
import matplotlib.pyplot as plt
import pandas as pd

from kvlab.io import RESULTS_DIR

FIG_DIR = RESULTS_DIR / "figures"


def _read(name: str) -> pd.DataFrame | None:
    path = RESULTS_DIR / name
    if not path.exists():
        print(f"[skip] {name} not found")
        return None
    return pd.read_csv(path)


def _save(fig, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=150)
    plt.close(fig)
    print(f"[fig] {FIG_DIR / name}")


def _ok(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["status"] == "ok"] if "status" in df else df


def e01() -> None:
    df = _read("e01_kv_vs_context.csv")
    if df is not None:
        fig, axes = plt.subplots(1, df["model"].nunique(), figsize=(11, 4), squeeze=False)
        for ax, (model, g) in zip(axes[0], df.groupby("model")):
            for col in [c for c in g.columns if c.startswith("kv_GiB")]:
                ax.plot(g["context"], g[col], marker="o", label=col.replace("kv_GiB_", ""))
            ax.set(xscale="log", yscale="log", xlabel="context tokens", ylabel="KV cache (GiB)", title=model)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
        _save(fig, "e01_kv_vs_context.png")

    df = _read("e01_decode_roofline.csv")
    if df is not None:
        fig, ax = plt.subplots(figsize=(6, 4))
        for (model, batch), g in df.groupby(["model", "batch"]):
            ax.plot(g["context"], g["t_lower_bound_ms"], marker="o", label=f"{model}, batch {batch}")
        ax.set(xscale="log", yscale="log", xlabel="context tokens", ylabel="decode step lower bound (ms)",
               title="Roofline: one decode step (A100-80GB)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
        _save(fig, "e01_roofline.png")


def e02() -> None:
    df = _read("e02_scaling.csv")
    if df is not None:
        df = _ok(df)
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        axes[0].plot(df["context"], df["ttft_s"], marker="o")
        axes[0].set(title="Prefill (TTFT)", ylabel="seconds")
        axes[1].plot(df["context"], df["decode_ms_mean"], marker="o")
        axes[1].set(title="Decode latency", ylabel="ms / token")
        axes[2].plot(df["context"], df["cache_held_mib"], marker="o", label="measured cache")
        axes[2].plot(df["context"], df["analytic_kv_mib"], "--", label="analytic 2*L*Hkv*Dh*T*2B")
        axes[2].set(title="KV cache", ylabel="MiB")
        axes[2].legend()
        for ax in axes:
            ax.set(xscale="log", xlabel="context tokens")
            ax.grid(alpha=0.3)
        _save(fig, "e02_scaling.png")

    df = _read("e02_backends.csv")
    if df is not None:
        fig, ax = plt.subplots(figsize=(6, 4))
        for backend, g in _ok(df).groupby("backend"):
            ax.plot(g["context"], g["peak_gib"], marker="o", label=backend)
        ax.set(xscale="log", xlabel="context tokens", ylabel="peak GPU memory (GiB)", title="Prefill memory by backend")
        ax.grid(alpha=0.3)
        ax.legend()
        _save(fig, "e02_backends.png")


def e03() -> None:
    df = _read("e03_chunk_sizes.csv")
    if df is not None:
        df = _ok(df).sort_values("chunk")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(df["chunk"], df["peak_gib"], marker="o", color="C0")
        ax.set(xscale="log", xlabel="prefill chunk size", ylabel="peak memory (GiB)", title="Chunked prefill")
        ax2 = ax.twinx()
        ax2.plot(df["chunk"], df["prefill_s"], marker="s", color="C1")
        ax2.set_ylabel("prefill time (s)", color="C1")
        _save(fig, "e03_chunk_sizes.png")

    df = _read("e03_bounded.csv")
    if df is not None:
        fig, ax = plt.subplots(figsize=(6, 4))
        for method, g in _ok(df).groupby("method"):
            ax.plot(g["context"], g["peak_gib"], marker="o", label=method)
        ax.set(xscale="log", xlabel="prompt tokens", ylabel="peak memory (GiB)", title="Bounded-memory prefill")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        _save(fig, "e03_bounded.png")


def e04() -> None:
    df = _read("e04_quality.csv")
    if df is None:
        return
    df["kv_mib"] = df["cache_logical_bytes"] / 2**20
    for metric in ("bpb", "ppl"):
        langs = df["lang"].unique()
        fig, axes = plt.subplots(1, len(langs), figsize=(6 * len(langs), 4.5), squeeze=False)
        for ax, lang in zip(axes[0], langs):
            for policy, g in df[df["lang"] == lang].groupby("policy"):
                g = g.sort_values("kv_mib")
                ax.plot(g["kv_mib"], g[metric], marker="o", label=policy)
            ax.set(xscale="log", xlabel="KV cache memory (MiB, logical)", ylabel=metric,
                   title=f"{metric.upper()} vs memory ({lang})")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7, ncol=2)
        _save(fig, f"e04_{metric}_vs_memory.png")


def e05() -> None:
    df = _read("e05_passkey.csv")
    if df is None:
        return
    pols = df["policy"].unique()
    fig, axes = plt.subplots(1, len(pols), figsize=(3.2 * len(pols), 3.2), squeeze=False)
    for ax, policy in zip(axes[0], pols):
        pivot = df[df["policy"] == policy].pivot(index="depth", columns="context", values="accuracy")
        ax.imshow(pivot.values, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto", origin="lower")
        ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=45, fontsize=7)
        ax.set_yticks(range(len(pivot.index)), pivot.index, fontsize=7)
        ax.set(title=policy, xlabel="context")
    axes[0][0].set_ylabel("needle depth")
    _save(fig, "e05_passkey.png")


def e06() -> None:
    df = _read("e06_efficiency.csv")
    if df is None:
        return
    df = _ok(df)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col, title in zip(axes, ("ttft_s", "decode_ms_mean", "peak_decode_gib"),
                              ("TTFT (s)", "decode ms/token", "peak decode memory (GiB)")):
        df.pivot(index="policy", columns="context", values=col).plot.bar(ax=ax)
        ax.set(title=title, xlabel="")
        ax.grid(alpha=0.3, axis="y")
    _save(fig, "e06_efficiency.png")


if __name__ == "__main__":
    for fn in (e01, e02, e03, e04, e05, e06):
        fn()
