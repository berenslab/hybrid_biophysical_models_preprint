"""
Timing experiments for HH-K channel architecture and multicompartment models.
Saves runtime dataframes as CSV and prints LaTeX tables to console.

Usage:
    python scripts/evaluate_runtimes.py [--quick]

    --quick: run with a single experiment file per loop (for testing)
"""

import argparse
import os
import sys
import json
import time
import re

# Must be set before JAX is imported. Serializes XLA's LLVM compilation threads,
# preventing "Cannot allocate memory" crashes on memory-constrained machines where
# many parallel compile threads exhaust available RAM.
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")

import numpy as np
import pandas as pd

from jax import config as jax_config
jax_config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from hybrid_models.hh import *
from hybrid_models.hh import integrate
from hybrid_models.utils import Dataset
from scripts.configs import *

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

rmse = lambda x, y: jnp.sqrt(jnp.mean((x - y) ** 2))


def _block_until_ready(tree):
    jax.tree.map(
        lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
        tree,
    )


def _time_integrate(model, ts, init_state, num_repeats=10):
    out = integrate(model, ts, init_state)
    _block_until_ready(out)
    times = []
    for _ in range(num_repeats):
        start = time.perf_counter()
        out = integrate(model, ts, init_state)
        _block_until_ready(out)
        times.append(time.perf_counter() - start)
    return out, float(np.mean(times)), float(np.std(times))


def _time_forwardpass(model, init_state, num_repeats=10):
    out = model(0.0, init_state)
    _block_until_ready(out)
    times = []
    for _ in range(num_repeats):
        start = time.perf_counter()
        out = model(0.0, init_state)
        _block_until_ready(out)
        times.append(time.perf_counter() - start)
    return out, float(np.mean(times)), float(np.std(times))


@eqx.filter_vmap(in_axes=(None, 0, None, None))
def _v_integrate(model, params, u0, ts):
    _model = model.set(params)
    t, x = integrate(_model, ts, u0)
    return t, x


@eqx.filter_vmap(in_axes=(None, 0, None))
def _v_forwardpass(model, params, u0):
    _model = model.set(params)
    return _model(0.0, u0)


def _time_vmap_integrate(model, batch_params, u0, ts, num_repeats=10):
    out = _v_integrate(model, batch_params, u0, ts)
    _block_until_ready(out)
    times = []
    for _ in range(num_repeats):
        start = time.perf_counter()
        out = _v_integrate(model, batch_params, u0, ts)
        _block_until_ready(out)
        times.append(time.perf_counter() - start)
    return out, float(np.mean(times)), float(np.std(times))


def _time_vmap_forwardpass(model, batch_params, u0, num_repeats=10):
    out = _v_forwardpass(model, batch_params, u0)
    _block_until_ready(out)
    times = []
    for _ in range(num_repeats):
        start = time.perf_counter()
        out = _v_forwardpass(model, batch_params, u0)
        _block_until_ready(out)
        times.append(time.perf_counter() - start)
    return out, float(np.mean(times)), float(np.std(times))


# ---------------------------------------------------------------------------
# latex formatting helpers (from notebook)
# ---------------------------------------------------------------------------

def df_to_latex(
    df,
    highlight_vals=None,
    col_fmt=None,
    include_index=False,
    hlines=True,
    float_format="%.2f",
    mean_pm_std=False,
):
    df2 = df.copy()
    original_cols = list(df2.columns)
    ncols = df2.shape[1]

    if highlight_vals is None:
        highlight_vals = [None] * ncols
    if len(highlight_vals) != ncols:
        raise ValueError("highlight_vals must have one entry per column")

    def _fmt_val(v):
        if pd.isna(v):
            return ""
        try:
            if callable(float_format):
                return float_format(float(v))
            if isinstance(float_format, str) and "%" in float_format:
                return float_format % float(v)
            return format(float(v), float_format)
        except Exception:
            return str(v)

    merged_means_map = {}
    if mean_pm_std:
        mean_pref = "mean_"
        std_pref = "std_"
        merged_values = {}
        modes_map = {}

        for i, col in enumerate(original_cols):
            if col.startswith(mean_pref):
                base = col[len(mean_pref):]
                mean_ser = pd.to_numeric(df2[col], errors="coerce")
                std_col = std_pref + base

                if std_col in original_cols:
                    std_ser = pd.to_numeric(df2[std_col], errors="coerce")
                    std_idx = original_cols.index(std_col)
                    modes_map[base] = (
                        highlight_vals[i]
                        if highlight_vals[i] is not None
                        else highlight_vals[std_idx]
                    )
                else:
                    std_ser = pd.Series([pd.NA] * len(df2), index=df2.index)
                    modes_map[base] = highlight_vals[i]

                merged_means_map[base] = mean_ser
                merged_values[base] = pd.Series(
                    [
                        (
                            f"{_fmt_val(m)} $\\pm$ {_fmt_val(s)}"
                            if not pd.isna(s)
                            else _fmt_val(m)
                        )
                        if not pd.isna(m)
                        else ""
                        for m, s in zip(mean_ser, std_ser)
                    ],
                    index=df2.index,
                )

        out_cols = []
        for col in original_cols:
            if col.startswith(mean_pref):
                out_cols.append(col[len(mean_pref):])
            elif col.startswith(std_pref):
                base = col[len(std_pref):]
                if base in out_cols:
                    continue
                out_cols.append(col)
            else:
                out_cols.append(col)

        for base, ser in merged_values.items():
            df2[base] = ser

        to_drop = [
            c for c in df2.columns if c.startswith(mean_pref) or c.startswith(std_pref)
        ]
        df2 = df2.drop(columns=to_drop, errors="ignore")

        final_cols = [c for c in out_cols if c in df2.columns]
        df2 = df2[final_cols]

        new_highlight = []
        for col in df2.columns:
            if col in modes_map:
                new_highlight.append(modes_map[col])
            elif col in original_cols:
                new_highlight.append(highlight_vals[original_cols.index(col)])
            else:
                new_highlight.append(None)
        highlight_vals = new_highlight
        ncols = df2.shape[1]

    for i, col in enumerate(df2.columns):
        if col in merged_means_map:
            continue
        try:
            if pd.api.types.is_float_dtype(df2[col]):
                ser_num = pd.to_numeric(df2[col], errors="coerce")
                df2[col] = ser_num.map(lambda x: _fmt_val(x) if not pd.isna(x) else "")
        except Exception:
            pass

    for i, mode in enumerate(highlight_vals):
        if mode is None:
            continue
        mode_l = str(mode).lower()
        if mode_l not in {"max", "min"}:
            continue

        col = df2.columns[i]
        if col in merged_means_map:
            ser_num = merged_means_map[col]
        else:
            ser = df2[col]
            ser_num = pd.to_numeric(ser, errors="coerce")
        if ser_num.dropna().empty:
            continue

        target = ser_num.max() if mode_l == "max" else ser_num.min()
        mask = ser_num == target

        df2[col] = df2[col].astype(object)
        df2.loc[mask, col] = df2.loc[mask, col].map(lambda x: f"\\textbf{{{x}}}")
        df2[col] = df2[col].astype(str)

    tabular_str = df2.to_latex(
        index=include_index,
        escape=False,
        column_format=col_fmt,
        float_format=float_format,
    )

    if hlines:
        lines = tabular_str.splitlines()
        out = []
        in_tabular = False
        for line in lines:
            s = line.strip()
            if s.startswith(r"\begin{tabular"):
                in_tabular = True
                out.append(line)
                out.append(r"\hline")
                continue
            if s.startswith(r"\end{tabular"):
                in_tabular = False
                out.append(line)
                continue
            if s in {r"\toprule", r"\midrule", r"\bottomrule"}:
                continue
            out.append(line)
            if in_tabular and s.endswith(r"\\"):
                out.append(r"\hline")
        tabular_str = "\n".join(out)

    latex_str = "\\begin{table}[ht]\n\\centering\n"
    latex_str += tabular_str + "\n"
    latex_str += "\\end{table}"
    return latex_str


def latex_to_str(latex_table):
    m = re.search(r"\\begin{tabular}{.*?}(.*?)\\end{tabular}", latex_table, re.S)
    if not m:
        raise ValueError("no tabular found in input")
    inner = m.group(1)
    inner = re.sub(r"\\hline", "", inner)
    inner = re.sub(r"\\toprule|\\midrule|\\bottomrule", "", inner)
    parts = re.split(r"\\\\\s*\n?", inner)
    rows = []
    for part in parts:
        s = part.strip()
        if not s:
            continue
        s = s.strip("& ")
        cols = [c.strip() for c in s.split("&")]

        def _clean(cell: str) -> str:
            cell = cell.strip()
            cell = re.sub(r"\\textbf\{([^}]*)\}", r"**\1**", cell)
            cell = cell.replace(r"\pm", "±")
            cell = cell.replace("{", "").replace("}", "")
            cell = cell.replace("$", "")
            cell = re.sub(r"\s+", " ", cell).strip()
            return cell

        cols = [_clean(c) for c in cols]
        rows.append(cols)

    if not rows:
        return ""

    header = rows[0]
    data = rows[1:]
    for i, r in enumerate(data):
        if len(r) < len(header):
            data[i] = r + [""] * (len(header) - len(r))
        elif len(r) > len(header):
            data[i] = r[: len(header)]

    md_lines = []
    md_lines.append("| " + " | ".join(header) + " |")
    md_lines.append("|" + "|".join(["---:"] * len(header)) + "|")
    for r in data:
        md_lines.append("| " + " | ".join(r) + " |")

    return "\n".join(md_lines)


# ---------------------------------------------------------------------------
# HH-K channel architecture runtimes
# ---------------------------------------------------------------------------

def run_hh_k_runtimes(exp_dir, config, csv_path, quick=False):
    json_files = [f for f in os.listdir(exp_dir) if f.endswith(".json") and "tanh" in f]

    filtered_experiments = pd.DataFrame(columns=["json_file", "width", "depth", "rmse"])
    for json_file in json_files:
        with open(os.path.join(exp_dir, json_file), "r") as f:
            loaded_data = json.load(f)
            width = loaded_data["channel_model_kwargs"]["width_size"]
            depth = loaded_data["channel_model_kwargs"]["depth_size"]
            val_data = loaded_data["validation_data"]
            v_val = np.array(val_data["val"]["v"][0])
            v_pred = np.array(val_data["pred"]["v"][0])
            rmse_val = rmse(v_val, v_pred)
            filtered_experiments.loc[len(filtered_experiments)] = [json_file, width, depth, rmse_val]

    filtered_experiments = (
        filtered_experiments
        .sort_values("rmse")
        .drop_duplicates(subset=["width", "depth"], keep="first")
        .reset_index(drop=True)
    )

    if quick:
        filtered_experiments = filtered_experiments.iloc[:1]

    # Resume from existing CSV if present.
    if os.path.exists(csv_path):
        df_k_runtimes = pd.read_csv(csv_path)
        done = set(zip(df_k_runtimes["width"].astype(int), df_k_runtimes["depth"].astype(int)))
        print(f"Resuming from {csv_path}: {len(done)} rows already cached.")
    else:
        df_k_runtimes = pd.DataFrame(columns=[
            "width", "depth",
            "mean_integrate_time", "std_integrate_time",
            "mean_forward_time", "std_forward_time",
        ])
        done = set()

    dt = 0.1
    ts = jnp.arange(0, 50.0, dt)
    hh_true = HH([Na(), K(), Leak()], [StepCurrent(amp=15.0, start=5.0, end=45.0)])
    u0 = hh_true.init(0.0, config.u_init)

    if (0, 0) not in done:
        _, hh_k_mean_time, hh_k_std_time = _time_integrate(hh_true, ts, u0)
        _, hh_k_fwd_time, hh_k_fwd_std = _time_forwardpass(hh_true, u0)
        print(f"hh_true integrate:    {hh_k_mean_time:.6f} ± {hh_k_std_time:.6f} s")
        print(f"hh_true forward pass: {hh_k_fwd_time:.6f} ± {hh_k_fwd_std:.6f} s")
        row = {"width": 0, "depth": 0, "mean_integrate_time": hh_k_mean_time,
               "std_integrate_time": hh_k_std_time, "mean_forward_time": hh_k_fwd_time,
               "std_forward_time": hh_k_fwd_std}
        df_k_runtimes = pd.concat([df_k_runtimes, pd.DataFrame([row])], ignore_index=True)
        df_k_runtimes.to_csv(csv_path, index=False)
    else:
        print("Skipping hh_true baseline (already cached).")

    hh = config.hh_model
    for json_file in filtered_experiments["json_file"]:
        json_path = os.path.join(exp_dir, json_file)
        with open(json_path, "r") as f:
            loaded_data = json.load(f)
            kwargs = loaded_data["channel_model_kwargs"]
            width = kwargs["width_size"]
            depth = kwargs["depth_size"]

        if (int(width), int(depth)) in done:
            print(f"Skipping ({width}x{depth}) (already cached).")
            continue

        kwargs["activation"] = jax.nn.tanh
        kwargs["last_layer_initializer"] = config.channel_model_kwargs["last_layer_initializer"]
        node = config.ChannelModel(key=jr.key(0), **kwargs)
        init_model = hh.insert(node).set(config.init_params)
        loaded_model = eqx.tree_deserialise_leaves(
            os.path.join(exp_dir, json_file.replace(".json", ".eqx")), init_model
        )

        _, hybrid_k_mean_time, hybrid_k_std_time = _time_integrate(loaded_model, ts, u0, num_repeats=50)
        _, hybrid_k_fwd_time, hybrid_k_fwd_std = _time_forwardpass(loaded_model, u0, num_repeats=50)
        print(f"hh_hybrid integrate ({width}x{depth}):    {hybrid_k_mean_time:.6f} ± {hybrid_k_std_time:.6f} s")
        print(f"hh_hybrid forward pass ({width}x{depth}): {hybrid_k_fwd_time:.6f} ± {hybrid_k_fwd_std:.6f} s")

        row = {"width": width, "depth": depth, "mean_integrate_time": hybrid_k_mean_time,
               "std_integrate_time": hybrid_k_std_time, "mean_forward_time": hybrid_k_fwd_time,
               "std_forward_time": hybrid_k_fwd_std}
        df_k_runtimes = pd.concat([df_k_runtimes, pd.DataFrame([row])], ignore_index=True)
        df_k_runtimes.to_csv(csv_path, index=False)

    df_k_runtimes["speedup_integrate"] = (
        df_k_runtimes["mean_integrate_time"].iloc[0] / df_k_runtimes["mean_integrate_time"]
    ).astype(float)
    df_k_runtimes["speedup_forward"] = (
        df_k_runtimes["mean_forward_time"].iloc[0] / df_k_runtimes["mean_forward_time"]
    ).astype(float)

    df_k_runtimes = df_k_runtimes.sort_values(["width", "depth"]).reset_index(drop=True)
    df_k_runtimes.to_csv(csv_path, index=False)
    return df_k_runtimes


# ---------------------------------------------------------------------------
# multicompartment runtimes
# ---------------------------------------------------------------------------

def run_multicomp_runtimes(exp_dir, config, num_comps, csv_path, quick=False):
    json_files = [f for f in os.listdir(exp_dir) if f.endswith(".json")]

    filtered_experiments = pd.DataFrame(columns=["json_file", "width", "depth", "num_latents", "rmse"])
    for json_file in json_files:
        with open(os.path.join(exp_dir, json_file), "r") as f:
            loaded_data = json.load(f)
            has_weights = os.path.exists(os.path.join(exp_dir, json_file.replace(".json", ".eqx")))
            if "validation_data" in loaded_data and has_weights:
                width = loaded_data["channel_model_kwargs"]["width_size"]
                depth = loaded_data["channel_model_kwargs"]["depth_size"]
                num_latents = len(loaded_data["channel_model_kwargs"]["latent_states"])
                val_data = loaded_data["validation_data"]
                v_val = np.array(val_data["val"]["v"][0])
                v_pred = np.array(val_data["pred"]["v"][0])
                rmse_val = rmse(v_val, v_pred)
            else:
                if not has_weights:
                    print(f"model weights not found for {json_file}, skipping for runtime eval")
                else:
                    print(f"validation data not found in {json_file}, skipping for runtime eval")
                width = np.nan
                depth = np.nan
                num_latents = np.nan
                rmse_val = np.nan
            filtered_experiments.loc[len(filtered_experiments)] = [json_file, width, depth, num_latents, rmse_val]

    filtered_experiments = (
        filtered_experiments
        .sort_values("rmse")
        .drop_duplicates(subset=["width", "depth", "num_latents"], keep="first")
        .reset_index(drop=True)
        .dropna(subset=["width", "depth", "num_latents"])
        .reset_index(drop=True)
    )

    if quick:
        filtered_experiments = filtered_experiments.iloc[:1]

    # Resume from existing CSV if present.
    ncomp_col = r"$N_{comp}$"
    if os.path.exists(csv_path):
        df_multicomp_runtimes = pd.read_csv(csv_path)
        done = set(zip(
            df_multicomp_runtimes["width"].astype(int),
            df_multicomp_runtimes["depth"].astype(int),
            df_multicomp_runtimes["num_latents"].astype(int),
            df_multicomp_runtimes[ncomp_col].astype(int),
        ))
        print(f"Resuming from {csv_path}: {len(done)} rows already cached.")
    else:
        df_multicomp_runtimes = pd.DataFrame(columns=[
            "width", "depth", "num_latents", ncomp_col,
            "mean_integrate_time", "std_integrate_time",
            "mean_forward_time", "std_forward_time",
        ])
        done = set()

    dt = 0.1
    ts = jnp.arange(0, 70.0, dt)
    hh_true = config.hh_model
    u0_base = {k: v for k, v in config.u_init.items() if not k.startswith("z")}

    batch_params = {
        ".l": jnp.array([hh_true.l] * num_comps) + 0.001 * jr.normal(jr.key(0), (num_comps,))
    }

    if (0, 0, 0, 1) not in done:
        _, hh_single_mean_time, hh_single_std_time = _time_integrate(hh_true, ts, u0_base)
        _, hh_single_fwd_time, hh_single_fwd_std = _time_forwardpass(hh_true, u0_base)
        print(f"hh_true integrate:    {hh_single_mean_time:.6f} ± {hh_single_std_time:.6f} s")
        print(f"hh_true forward pass: {hh_single_fwd_time:.6f} ± {hh_single_fwd_std:.6f} s")
        row = {"width": 0, "depth": 0, "num_latents": 0, ncomp_col: 1,
               "mean_integrate_time": hh_single_mean_time, "std_integrate_time": hh_single_std_time,
               "mean_forward_time": hh_single_fwd_time, "std_forward_time": hh_single_fwd_std}
        df_multicomp_runtimes = pd.concat([df_multicomp_runtimes, pd.DataFrame([row])], ignore_index=True)
        df_multicomp_runtimes.to_csv(csv_path, index=False)
    else:
        print("Skipping hh_true single-compartment baseline (already cached).")

    # Vmapped baseline: num_comps single-compartment models solved in parallel.
    if (0, 0, 0, num_comps) not in done:
        _, hh_vmap_mean_time, hh_vmap_std_time = _time_vmap_integrate(hh_true, batch_params, u0_base, ts)
        _, hh_vmap_fwd_time, hh_vmap_fwd_std = _time_vmap_forwardpass(hh_true, batch_params, u0_base)
        print(f"hh_true vmap ({num_comps} comps) integrate:    {hh_vmap_mean_time:.6f} ± {hh_vmap_std_time:.6f} s")
        print(f"hh_true vmap ({num_comps} comps) forward pass: {hh_vmap_fwd_time:.6f} ± {hh_vmap_fwd_std:.6f} s")
        row = {"width": 0, "depth": 0, "num_latents": 0, ncomp_col: num_comps,
               "mean_integrate_time": hh_vmap_mean_time, "std_integrate_time": hh_vmap_std_time,
               "mean_forward_time": hh_vmap_fwd_time, "std_forward_time": hh_vmap_fwd_std}
        df_multicomp_runtimes = pd.concat([df_multicomp_runtimes, pd.DataFrame([row])], ignore_index=True)
        df_multicomp_runtimes.to_csv(csv_path, index=False)
    else:
        print("Skipping hh_true vmapped baseline (already cached).")

    hh = config.hh_model
    files_to_run = filtered_experiments["json_file"].tolist()
    for json_file in files_to_run:
        json_path = os.path.join(exp_dir, json_file)
        with open(json_path, "r") as f:
            loaded_data = json.load(f)
            kwargs = loaded_data["channel_model_kwargs"]
            width = kwargs["width_size"]
            depth = kwargs["depth_size"]
            model_latents = loaded_data["channel_model_kwargs"]["latent_states"]
            num_latents = len(model_latents)

        if (int(width), int(depth), int(num_latents), 1) in done:
            print(f"Skipping ({width}x{depth}x{num_latents}) (already cached).")
            continue

        kwargs["activation"] = jax.nn.tanh
        kwargs["last_layer_initializer"] = config.channel_model_kwargs["last_layer_initializer"]
        node = config.ChannelModel(key=jr.key(0), **kwargs)
        init_model = hh.insert(node).set(config.init_params)
        loaded_model = eqx.tree_deserialise_leaves(
            os.path.join(exp_dir, json_file.replace(".json", ".eqx")), init_model
        )
        u0 = loaded_model.init(0.0, config.u_init)
        u0 = {k: v for k, v in u0.items() if not k.startswith("z") or k in model_latents}

        _, hybrid_multi_mean_time, hybrid_multi_std_time = _time_integrate(loaded_model, ts, u0, num_repeats=50)
        _, hybrid_multi_fwd_time, hybrid_multi_fwd_std = _time_forwardpass(loaded_model, u0, num_repeats=50)
        print(f"hh_hybrid integrate ({width}x{depth}x{num_latents}):    {hybrid_multi_mean_time:.6f} ± {hybrid_multi_std_time:.6f} s")
        print(f"hh_hybrid forward pass ({width}x{depth}x{num_latents}): {hybrid_multi_fwd_time:.6f} ± {hybrid_multi_fwd_std:.6f} s")

        row = {"width": width, "depth": depth, "num_latents": num_latents, ncomp_col: 1,
               "mean_integrate_time": hybrid_multi_mean_time, "std_integrate_time": hybrid_multi_std_time,
               "mean_forward_time": hybrid_multi_fwd_time, "std_forward_time": hybrid_multi_fwd_std}
        df_multicomp_runtimes = pd.concat([df_multicomp_runtimes, pd.DataFrame([row])], ignore_index=True)
        df_multicomp_runtimes.to_csv(csv_path, index=False)

    multi_comp_row = df_multicomp_runtimes[df_multicomp_runtimes[ncomp_col] == num_comps].iloc[0]

    df_multicomp_runtimes[r"$\text{speedup}_{\text{integrate}}$"] = (
        float(multi_comp_row["mean_integrate_time"]) / df_multicomp_runtimes["mean_integrate_time"]
    ).astype(float)
    df_multicomp_runtimes[r"$\text{speedup}_{\text{forward}}$"] = (
        float(multi_comp_row["mean_forward_time"]) / df_multicomp_runtimes["mean_forward_time"]
    ).astype(float)

    df_multicomp_runtimes = df_multicomp_runtimes.sort_values(["num_latents", "width", "depth"]).reset_index(drop=True)
    return df_multicomp_runtimes


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate model runtimes and print LaTeX tables.")
    parser.add_argument("--quick", action="store_true", help="Run with a single experiment file per loop (for testing)")
    parser.add_argument("--output-dir", default="results", help="Directory to save CSV files (default: results/)")
    args = parser.parse_args()

    root = os.path.join(os.path.dirname(__file__), "..")
    output_dir = os.path.join(root, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # ---- HH-K channel architecture runtimes --------------------------------
    print("\n" + "="*60)
    print("HH-K channel architecture runtimes")
    print("="*60)

    config_k = train_hybrid_on_hh_k()
    exp_dir_k = os.path.join(root, "results/sweeps/hh_channel_arch_sweep3/")

    csv_path_k = os.path.join(output_dir, "runtimes_hh_k.csv")
    df_k_runtimes = run_hh_k_runtimes(exp_dir_k, config_k, csv_path_k, quick=args.quick)
    print(f"\nSaved: {csv_path_k}")

    latex_table_k = df_to_latex(
        df_k_runtimes,
        highlight_vals=[None, None, "min", None, "min", None, "max", "max"],
        mean_pm_std=True,
        float_format="%.3f",
    )
    print("\n--- LaTeX table (HH-K runtimes) ---")
    print(latex_table_k)
    print("\n--- Formatted table (HH-K runtimes) ---")
    print(latex_to_str(latex_table_k))

    # ---- Multicompartment runtimes -----------------------------------------
    print("\n" + "="*60)
    print("Multicompartment runtimes")
    print("="*60)

    num_comps = 154

    config_mc = train_hybrid_on_multicomp()
    exp_dir_mc = os.path.join(root, "results/sweeps/multicomp_sweep2/")

    csv_path_mc = os.path.join(output_dir, "runtimes_multicomp.csv")
    df_multicomp_runtimes = run_multicomp_runtimes(exp_dir_mc, config_mc, num_comps, csv_path_mc, quick=args.quick)
    print(f"\nSaved: {csv_path_mc}")

    latex_table_multicomp = df_to_latex(
        df_multicomp_runtimes,
        highlight_vals=[None, None, None, "min", None, "min", None, None, "max", "max"],
        mean_pm_std=True,
        float_format="%.3f",
    )
    print("\n--- LaTeX table (multicomp runtimes) ---")
    print(latex_table_multicomp)
    print("\n--- Formatted table (multicomp runtimes) ---")
    print(latex_to_str(latex_table_multicomp))


if __name__ == "__main__":
    main()
